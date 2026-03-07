"""Live signal generator.

Fetches latest data from Binance API, evaluates the combined alpha portfolio,
and outputs today's position weights for each symbol.
"""

import logging
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.alpha.evaluator import AlphaEvaluator, EvaluationError
from src.backtest.engine import Backtester
from src.backtest.position import normalize_positions
from src.config import get_db_path
from src.data.schema import PROCESSED_COLUMNS
from src.storage.database import Database

logger = logging.getLogger(__name__)
console = Console()

# Binance API interval mapping
_INTERVAL_MAP = {
    '1m': '1m', '5m': '5m', '15m': '15m',
    '1h': '1h', '4h': '4h', '12h': '12h', '1d': '1d',
}


def _fetch_recent_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame | None:
    """Fetch recent klines from Binance Futures REST API (no auth needed)."""
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {
        'symbol': symbol,
        'interval': _INTERVAL_MAP.get(interval, interval),
        'limit': limit,
    }
    try:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        logger.warning(f'Failed to fetch klines for {symbol}: {e}')
        return None

    if not rows:
        return None

    # Binance kline format: [open_time, O, H, L, C, volume, close_time,
    #   quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote, ignore]
    df = pd.DataFrame(rows, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore',
    ])

    df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume',
                'quote_asset_volume', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['number_of_trades'] = pd.to_numeric(df['number_of_trades'], errors='coerce').astype('Int64')

    df = df.set_index('open_time').sort_index()

    # Drop the last row if it's an unclosed candle (close_time in the future)
    if len(df) > 0:
        last_close_time = pd.to_datetime(float(df['close_time'].iloc[-1]), unit='ms')
        if last_close_time > pd.Timestamp.utcnow():
            df = df.iloc[:-1]

    if len(df) == 0:
        return None

    # Compute derived fields (same as processor.py)
    df['total_volume'] = df['volume']
    df['buy_volume'] = df['taker_buy_base_asset_volume']
    df['sell_volume'] = df['volume'] - df['taker_buy_base_asset_volume']
    df['vwap'] = df['quote_asset_volume'] / df['volume'].replace(0, np.nan)
    df['vwap'] = df['vwap'].fillna(df['close'])
    df['total_trades_count'] = df['number_of_trades']
    buy_ratio = (df['taker_buy_base_asset_volume'] / df['volume'].replace(0, np.nan)).fillna(0.5).clip(0, 1)
    df['buy_trades_count'] = (df['number_of_trades'] * buy_ratio).round().astype('Int64')
    df['sell_trades_count'] = df['number_of_trades'] - df['buy_trades_count']

    return df[PROCESSED_COLUMNS]


class LiveSignalGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.bt_cfg = config['backtest']
        self.backtester = Backtester(config)
        self.db = Database(get_db_path(config))

    def run(self, method: str = 'sharpe', min_sharpe: float = 0.5,
            max_alphas: int = 20, frequency: str = '1d',
            capital: float | None = None) -> dict | None:
        """Generate today's live trading signal."""
        exec_cfg = self.config.get('execution', {})
        capital = capital or exec_cfg.get('capital', self.bt_cfg.get('capital', 10000))

        # 1. Query good alphas
        raw_alphas = self.db.get_top_alphas(min_sharpe=min_sharpe, limit=max_alphas)
        raw_alphas = [a for a in raw_alphas if a['frequency'] == frequency]
        if not raw_alphas:
            console.print('[red]No alphas found in DB[/red]')
            return None
        console.print(f'Loaded {len(raw_alphas)} alphas (Sharpe >= {min_sharpe})')

        # 2. Load historical data + fetch latest from Binance API
        data = self._load_updated_data(frequency)
        if not data:
            console.print('[red]No data available[/red]')
            return None

        # 3. Evaluate each alpha, compute weights, combine
        evaluator = AlphaEvaluator(data)
        alpha_records = []

        for a in raw_alphas:
            try:
                signal = evaluator.evaluate(a['expression'])
                alpha_records.append({
                    'expression': a['expression'],
                    'sharpe_ratio': a['sharpe_ratio'],
                    'signal': signal,
                })
            except (EvaluationError, Exception) as e:
                logger.warning(f'Skipping {a["expression"][:50]}: {e}')

        if not alpha_records:
            console.print('[red]No alphas could be evaluated[/red]')
            return None

        # 4. Compute weights
        weights = self._compute_weights(alpha_records, method)

        # 5. Combine signals (cross-sectional z-score + weighted sum)
        combined = self._combine_signals(alpha_records, weights)

        # 6. Get the latest row = today's signal
        latest_signal = combined.iloc[[-1]]
        latest_date = latest_signal.index[0]

        # 7. Position normalization (drop NaN symbols)
        latest_signal = latest_signal.dropna(axis=1)
        positions = normalize_positions(latest_signal, capital)
        pos_row = positions.iloc[0].sort_values()

        # 8. Print report
        self._print_report(pos_row, latest_date, method, capital, len(alpha_records))

        return {
            'date': str(latest_date),
            'positions': pos_row.to_dict(),
            'method': method,
            'capital': capital,
            'n_alphas': len(alpha_records),
        }

    def _load_updated_data(self, frequency: str) -> dict[str, pd.DataFrame]:
        """Load parquet data and supplement with latest Binance API data."""
        data = self.backtester.load_data(frequency)
        exec_cfg = self.config.get('execution', {})
        symbols = exec_cfg.get('symbols', self.config['symbols'])

        console.print(f'Fetching latest {frequency} candles from Binance...')
        updated = {}
        for sym in symbols:
            historical = data.get(sym)
            live = _fetch_recent_klines(sym, frequency, limit=100)

            if historical is not None and live is not None:
                combined = pd.concat([historical, live])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                updated[sym] = combined
            elif historical is not None:
                updated[sym] = historical
            elif live is not None:
                updated[sym] = live
            else:
                logger.warning(f'No data for {sym}')

        last_dates = {s: df.index[-1].strftime('%Y-%m-%d') for s, df in updated.items() if len(df) > 0}
        if last_dates:
            sample = list(last_dates.values())[0]
            console.print(f'Data updated to: {sample} ({len(updated)} symbols)')

        return updated

    def _compute_weights(self, alphas: list[dict], method: str) -> list[float]:
        n = len(alphas)
        if method == 'equal':
            return [1.0 / n] * n
        elif method == 'sharpe':
            sharpes = [max(a['sharpe_ratio'], 0) for a in alphas]
            total = sum(sharpes)
            if total == 0:
                return [1.0 / n] * n
            return [s / total for s in sharpes]
        elif method == 'inverse_vol':
            # Fallback to equal if no vol data available in signal-only mode
            return [1.0 / n] * n
        else:
            return [1.0 / n] * n

    def _combine_signals(self, alphas: list[dict], weights: list[float]) -> pd.DataFrame:
        """Weighted sum of cross-sectionally z-scored alpha signals."""
        signals = [a['signal'] for a in alphas]
        common_idx = signals[0].index
        common_cols = signals[0].columns
        for s in signals[1:]:
            common_idx = common_idx.intersection(s.index)
            common_cols = common_cols.intersection(s.columns)

        combined = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
        for sig, w in zip(signals, weights):
            aligned = sig.loc[common_idx, common_cols]
            row_mean = aligned.mean(axis=1)
            row_std = aligned.std(axis=1).replace(0, 1)
            normalized = aligned.sub(row_mean, axis=0).div(row_std, axis=0)
            combined += w * normalized

        return combined

    def _print_report(self, positions: pd.Series, date, method: str,
                      capital: float, n_alphas: int):
        console.print(f'\n[bold]Live Signal ({date.strftime("%Y-%m-%d")})[/bold]')
        console.print(f'Method: {method} | Capital: ${capital:,.0f} | Alphas: {n_alphas}\n')

        t = Table()
        t.add_column('Symbol', style='cyan')
        t.add_column('Direction', justify='center')
        t.add_column('Weight', justify='right')
        t.add_column('Amount', justify='right')
        t.add_column('Score', justify='right', style='dim')

        for sym in positions.index:
            pos = positions[sym]
            pct = pos / capital * 100
            direction = '[green]LONG[/green]' if pos > 0 else '[red]SHORT[/red]' if pos < 0 else 'FLAT'
            t.add_row(
                sym,
                direction,
                f'{pct:+.1f}%',
                f'${pos:+,.0f}',
                f'{pos / capital:+.3f}',
            )

        console.print(t)

        long_total = positions[positions > 0].sum()
        short_total = positions[positions < 0].sum()
        net = long_total + short_total
        console.print(f'\n  Long exposure:  {long_total/capital:+.1%} (${long_total:+,.0f})')
        console.print(f'  Short exposure: {short_total/capital:+.1%} (${short_total:+,.0f})')
        console.print(f'  Net exposure:   {net/capital:+.1%} (${net:+,.0f})\n')
