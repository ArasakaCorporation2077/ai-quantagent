"""Process raw kline CSVs into enriched parquet files."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

from src.config import get_data_dir
from src.data.schema import KLINE_COLUMNS, KLINE_DTYPES, PROCESSED_COLUMNS

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = get_data_dir(config)
        self.raw_dir = self.data_dir / 'raw' / 'klines'
        self.processed_dir = self.data_dir / 'processed'

    def process_all(self, symbols: list[str], frequencies: list[str]):
        """Process all symbol/frequency combinations."""
        total = len(symbols) * len(frequencies)
        pairs = [(s, f) for s in symbols for f in frequencies]

        for symbol, freq in track(pairs, description='Processing data...'):
            try:
                self.process_symbol_frequency(symbol, freq)
            except Exception as e:
                logger.warning(f'Failed to process {symbol}/{freq}: {e}')

    def process_symbol_frequency(self, symbol: str, freq: str) -> pd.DataFrame | None:
        """Process a single symbol/frequency combination."""
        out_dir = self.processed_dir / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{freq}.parquet'

        df = self._load_raw_csvs(symbol, freq)
        if df is None or df.empty:
            logger.warning(f'No data for {symbol}/{freq}')
            return None

        df = self._compute_derived_fields(df)
        df = self._validate_and_clean(df)

        df[PROCESSED_COLUMNS].to_parquet(out_path, engine='pyarrow')
        logger.info(f'Saved {symbol}/{freq}: {len(df)} rows -> {out_path}')
        return df

    def _load_raw_csvs(self, symbol: str, freq: str) -> pd.DataFrame | None:
        """Load and concatenate all monthly CSV files for a symbol/frequency."""
        csv_dir = self.raw_dir / symbol / freq
        if not csv_dir.exists():
            return None

        csv_files = sorted(csv_dir.glob('*.csv'))
        if not csv_files:
            return None

        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
                # Some files may have a header row
                if df.iloc[0]['open_time'] == 'open_time':
                    df = df.iloc[1:]
                dfs.append(df)
            except Exception as e:
                logger.warning(f'Error reading {f}: {e}')

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)

        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='ms')
        for col, dtype in KLINE_DTYPES.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)

        df = df.set_index('open_time').sort_index()
        df = df[~df.index.duplicated(keep='first')]

        return df

    def _compute_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived fields from raw kline data."""
        df['total_volume'] = df['volume']
        df['buy_volume'] = df['taker_buy_base_asset_volume']
        df['sell_volume'] = df['volume'] - df['taker_buy_base_asset_volume']

        # VWAP: quote_asset_volume / volume (exact for bar interval)
        df['vwap'] = df['quote_asset_volume'] / df['volume'].replace(0, np.nan)

        df['total_trades_count'] = df['number_of_trades']

        # Approximate buy/sell trade split by volume ratio
        buy_ratio = df['taker_buy_base_asset_volume'] / df['volume'].replace(0, np.nan)
        buy_ratio = buy_ratio.fillna(0.5).clip(0, 1)
        df['buy_trades_count'] = (df['number_of_trades'] * buy_ratio).round().astype('Int64')
        df['sell_trades_count'] = df['number_of_trades'] - df['buy_trades_count']

        return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle gaps, NaNs, and data quality issues."""
        # Forward-fill small gaps (up to 5 bars)
        df = df.ffill(limit=5)

        # Drop rows where OHLC is all NaN
        ohlc_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=ohlc_cols, how='all')

        # Fill remaining NaN in volume/trade columns with 0
        fill_zero = ['buy_volume', 'sell_volume', 'total_volume',
                      'buy_trades_count', 'sell_trades_count', 'total_trades_count']
        df[fill_zero] = df[fill_zero].fillna(0)

        # Fill vwap NaN with close price
        df['vwap'] = df['vwap'].fillna(df['close'])

        return df
