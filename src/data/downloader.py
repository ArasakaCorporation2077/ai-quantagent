"""Download historical kline data from data.binance.vision."""

import asyncio
import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

from src.config import get_data_dir

logger = logging.getLogger(__name__)


class BinanceDataDownloader:
    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config['data']
        self.base_url = self.data_cfg['base_url']
        self.data_dir = get_data_dir(config)
        self.raw_dir = self.data_dir / 'raw' / 'klines'
        self.timeout = self.data_cfg.get('download_timeout', 60)
        self.retries = self.data_cfg.get('download_retries', 3)
        self.max_concurrent = self.data_cfg.get('max_concurrent', 20)

    async def download_all(self, symbols: list[str], frequencies: list[str]):
        """Download kline data for all symbol/frequency combinations."""
        start = pd.Timestamp(self.data_cfg['start_date'])
        end = pd.Timestamp(self.data_cfg['end_date'])

        # Build list of (symbol, freq, year, month) tasks
        tasks = []
        for symbol in symbols:
            for freq in frequencies:
                current = start
                while current <= end:
                    tasks.append((symbol, freq, current.year, current.month))
                    # Advance to next month
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)

        logger.info(f'Downloading {len(tasks)} kline files for {len(symbols)} symbols x {len(frequencies)} frequencies')

        semaphore = asyncio.Semaphore(self.max_concurrent)
        success = 0
        skipped = 0
        failed = 0

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            async def _download_one(symbol, freq, year, month):
                nonlocal success, skipped, failed
                async with semaphore:
                    result = await self._download_file(client, symbol, freq, year, month)
                    if result == 'success':
                        success += 1
                    elif result == 'skipped':
                        skipped += 1
                    else:
                        failed += 1

            await asyncio.gather(*[
                _download_one(s, f, y, m) for s, f, y, m in tasks
            ])

        logger.info(f'Download complete: {success} success, {skipped} skipped, {failed} failed')

    async def _download_file(
        self, client: httpx.AsyncClient,
        symbol: str, freq: str, year: int, month: int
    ) -> str:
        """Download a single kline ZIP, extract CSV, return status."""
        out_dir = self.raw_dir / symbol / freq
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f'{symbol}-{freq}-{year}-{month:02d}.csv'
        out_path = out_dir / filename

        if out_path.exists():
            return 'skipped'

        zip_name = f'{symbol}-{freq}-{year}-{month:02d}.zip'
        url = f'{self.base_url}/monthly/klines/{symbol}/{freq}/{zip_name}'

        for attempt in range(self.retries):
            try:
                resp = await client.get(url)
                if resp.status_code == 404:
                    logger.debug(f'Not found (expected for newer symbols): {zip_name}')
                    return 'skipped'
                resp.raise_for_status()

                # Extract CSV from ZIP
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
                    if not csv_names:
                        logger.warning(f'No CSV in ZIP: {zip_name}')
                        return 'failed'
                    with zf.open(csv_names[0]) as csv_file:
                        out_path.write_bytes(csv_file.read())

                logger.debug(f'Downloaded: {filename}')
                return 'success'

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return 'skipped'
                logger.warning(f'HTTP error {e.response.status_code} for {zip_name}, attempt {attempt+1}')
            except Exception as e:
                logger.warning(f'Error downloading {zip_name}: {e}, attempt {attempt+1}')

            if attempt < self.retries - 1:
                await asyncio.sleep(2 ** attempt)

        return 'failed'
