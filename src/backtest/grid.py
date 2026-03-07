"""Grid search for optimal portfolio parameters.

Varies symbol count, rebalancing frequency, and weighting method
to find the sweet spot for a given set of alphas.
"""

import logging
from itertools import product

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.alpha.evaluator import AlphaEvaluator, EvaluationError
from src.alpha.pruner import prune_alphas
from src.backtest.engine import Backtester, _oos_mask
from src.backtest.metrics import compute_all_metrics
from src.backtest.position import normalize_positions, compute_forward_returns, apply_transaction_costs
from src.config import get_db_path
from src.storage.database import Database

logger = logging.getLogger(__name__)
console = Console()


def _resample_positions(positions: pd.DataFrame, rebalance_bars: int) -> pd.DataFrame:
    """Hold positions for N bars before rebalancing.

    Only updates positions every `rebalance_bars` bars.
    Between rebalance points, the previous position is held.
    """
    if rebalance_bars <= 1:
        return positions

    result = positions.copy()
    rebal_indices = list(range(0, len(positions), rebalance_bars))

    for i in range(len(positions)):
        if i not in rebal_indices:
            # Find the most recent rebalance point
            prev_rebal = max(r for r in rebal_indices if r <= i)
            result.iloc[i] = result.iloc[prev_rebal]

    return result


def _compute_turnover(positions: pd.DataFrame, capital: float) -> float:
    """Annualized turnover as fraction of capital."""
    daily_turnover = positions.diff().abs().sum(axis=1)
    return float(daily_turnover.mean() / capital) if capital > 0 else 0.0


def run_grid(config: dict, alphas: list[dict],
             symbol_counts: list[int] = None,
             rebalance_bars_list: list[int] = None,
             methods: list[str] = None,
             frequency: str = '1d',
             corr_threshold: float = 0.85) -> pd.DataFrame:
    """Run grid search over portfolio parameters.

    Args:
        config: full config dict
        alphas: list of alpha dicts from DB
        symbol_counts: list of symbol counts to test (e.g. [5, 10, 15])
        rebalance_bars_list: list of rebalance periods in bars (e.g. [1, 3, 7])
        methods: weighting methods (e.g. ['equal', 'sharpe'])
        frequency: data frequency
        corr_threshold: pruning threshold

    Returns:
        DataFrame with columns: symbols, rebal_bars, method, sharpe, sharpe_oos,
        return, max_dd, turnover, win_rate, sortino
    """
    bt_cfg = config['backtest']
    capital = bt_cfg.get('capital', 10000)
    cost_bps = bt_cfg.get('transaction_cost_bps', 5)
    lookahead = bt_cfg.get('lookahead', 1)
    sampling = bt_cfg.get('sampling', 'quarterly')
    all_symbols = config['symbols']

    if symbol_counts is None:
        symbol_counts = [5, 10, 15]
    if rebalance_bars_list is None:
        rebalance_bars_list = [1, 3, 7]
    if methods is None:
        methods = ['equal', 'sharpe']

    bt = Backtester(config)
    results = []

    for n_symbols in symbol_counts:
        symbols = all_symbols[:n_symbols]
        console.print(f'\n[bold]Loading data for {n_symbols} symbols...[/bold]')

        data = bt.load_data(frequency, symbols)
        if len(data) < 2:
            console.print(f'  [red]Only {len(data)} symbols have data, skipping[/red]')
            continue

        # Prune alphas with this data subset
        if corr_threshold < 1.0:
            kept, _ = prune_alphas(alphas, data, threshold=corr_threshold)
        else:
            kept = alphas

        if len(kept) < 2:
            console.print(f'  [red]Only {len(kept)} alphas after pruning, skipping[/red]')
            continue

        # Evaluate all alpha signals once for this symbol set
        evaluator = AlphaEvaluator(data)
        alpha_signals = []
        alpha_sharpes = []

        for a in kept:
            try:
                sig = evaluator.evaluate(a['expression'])
                alpha_signals.append(sig)
                alpha_sharpes.append(a.get('sharpe_ratio', 0))
            except (EvaluationError, Exception):
                pass

        if len(alpha_signals) < 2:
            continue

        fwd_returns = compute_forward_returns(data, frequency, lookahead)

        for rebal_bars, method in product(rebalance_bars_list, methods):
            # Compute weights
            n = len(alpha_signals)
            if method == 'equal':
                weights = [1.0 / n] * n
            elif method == 'sharpe':
                total = sum(max(s, 0) for s in alpha_sharpes)
                if total == 0:
                    weights = [1.0 / n] * n
                else:
                    weights = [max(s, 0) / total for s in alpha_sharpes]
            else:
                weights = [1.0 / n] * n

            # Combine signals (cross-sectional z-score + weighted sum)
            common_idx = alpha_signals[0].index
            common_cols = alpha_signals[0].columns
            for sig in alpha_signals[1:]:
                common_idx = common_idx.intersection(sig.index)
                common_cols = common_cols.intersection(sig.columns)

            combined = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
            for sig, w in zip(alpha_signals, weights):
                aligned = sig.loc[common_idx, common_cols]
                row_mean = aligned.mean(axis=1)
                row_std = aligned.std(axis=1).replace(0, 1)
                normalized = aligned.sub(row_mean, axis=0).div(row_std, axis=0)
                combined += w * normalized

            # Positions with rebalancing
            positions = normalize_positions(combined, capital)
            positions = _resample_positions(positions, rebal_bars)

            # PnL
            ci = positions.index.intersection(fwd_returns.index)
            cc = positions.columns.intersection(fwd_returns.columns)
            pos = positions.loc[ci, cc]
            fwd = fwd_returns.loc[ci, cc]

            bar_pnl = (pos * fwd).sum(axis=1)
            costs = apply_transaction_costs(pos, cost_bps)
            pnl_returns = (bar_pnl - costs) / capital
            pnl_returns = pnl_returns.dropna()

            if len(pnl_returns) < 30:
                continue

            # Metrics
            metrics = compute_all_metrics(pnl_returns, frequency)
            turnover = _compute_turnover(pos, capital)

            # OOS
            try:
                oos_mask = _oos_mask(pnl_returns.index, sampling)
                pnl_oos = pnl_returns[oos_mask]
                metrics_oos = compute_all_metrics(pnl_oos, frequency) if len(pnl_oos) > 10 else {}
            except Exception:
                metrics_oos = {}

            results.append({
                'symbols': n_symbols,
                'rebal_bars': rebal_bars,
                'method': method,
                'sharpe': round(metrics.get('sharpe_ratio', 0), 3),
                'sharpe_oos': round(metrics_oos.get('sharpe_ratio', 0), 3),
                'return': round(metrics.get('annualized_return', 0), 4),
                'max_dd': round(metrics.get('max_drawdown', 0), 4),
                'turnover': round(turnover, 4),
                'win_rate': round(metrics.get('win_rate', 0), 4),
                'sortino': round(metrics.get('sortino_ratio', 0), 3),
                'n_alphas': len(alpha_signals),
            })

            label = f'{n_symbols}sym / {rebal_bars}bar / {method}'
            console.print(f'  {label}: Sharpe={metrics.get("sharpe_ratio",0):.2f}  '
                          f'OOS={metrics_oos.get("sharpe_ratio",0):.2f}  '
                          f'Return={metrics.get("annualized_return",0):.1%}  '
                          f'MaxDD={metrics.get("max_drawdown",0):.1%}')

    if not results:
        console.print('[red]No valid results from grid search[/red]')
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def print_grid_report(df: pd.DataFrame, frequency: str = '1d'):
    """Print a formatted grid search report."""
    if df.empty:
        return

    # Rebalance bars to human-readable labels
    freq_hours = {'1h': 1, '4h': 4, '12h': 12, '1d': 24}
    base_hours = freq_hours.get(frequency, 24)

    def bars_to_label(bars):
        hours = bars * base_hours
        if hours < 24:
            return f'{hours}H'
        elif hours == 24:
            return '1D'
        else:
            return f'{hours // 24}D'

    console.print(f'\n[bold]Grid Search Results (frequency={frequency})[/bold]\n')

    t = Table(title='Parameter Grid Comparison')
    t.add_column('Rank', style='dim')
    t.add_column('Symbols', justify='center')
    t.add_column('Rebalance', justify='center')
    t.add_column('Method', justify='center')
    t.add_column('Sharpe', justify='right', style='green')
    t.add_column('Sharpe OOS', justify='right', style='cyan')
    t.add_column('Return', justify='right')
    t.add_column('MaxDD', justify='right', style='red')
    t.add_column('Turnover', justify='right')
    t.add_column('Win Rate', justify='right')
    t.add_column('Alphas', justify='right', style='dim')

    # Sort by OOS Sharpe (most realistic), then by full Sharpe
    df_sorted = df.sort_values(['sharpe_oos', 'sharpe'], ascending=False)

    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        rebal_label = bars_to_label(int(row['rebal_bars']))
        style = 'bold' if i == 1 else ''

        t.add_row(
            str(i),
            str(int(row['symbols'])),
            rebal_label,
            row['method'],
            f'{row["sharpe"]:.2f}',
            f'{row["sharpe_oos"]:.2f}',
            f'{row["return"]:.1%}',
            f'{row["max_dd"]:.1%}',
            f'{row["turnover"]:.2%}',
            f'{row["win_rate"]:.1%}',
            str(int(row['n_alphas'])),
            style=style,
        )

    console.print(t)

    # Best config
    best = df_sorted.iloc[0]
    rebal_label = bars_to_label(int(best['rebal_bars']))
    console.print(f'\n[bold green]Best config (by OOS Sharpe):[/bold green]')
    console.print(f'  Symbols: {int(best["symbols"])}')
    console.print(f'  Rebalance: {rebal_label}')
    console.print(f'  Method: {best["method"]}')
    console.print(f'  Sharpe: {best["sharpe"]:.2f} (OOS: {best["sharpe_oos"]:.2f})')
    console.print(f'  Return: {best["return"]:.1%}')
    console.print(f'  MaxDD: {best["max_dd"]:.1%}')
    console.print()
