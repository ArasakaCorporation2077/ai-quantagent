"""Correlation-based alpha pruning.

Removes economically redundant alphas by computing pairwise signal correlations
and using greedy pruning to keep only diverse, uncorrelated alphas.
"""

import logging

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.alpha.evaluator import AlphaEvaluator, EvaluationError

logger = logging.getLogger(__name__)
console = Console()


def _flatten_signal(signal: pd.DataFrame) -> pd.Series:
    """Cross-sectional z-score then stack into a single vector for correlation."""
    row_mean = signal.mean(axis=1)
    row_std = signal.std(axis=1).replace(0, 1)
    normalized = signal.sub(row_mean, axis=0).div(row_std, axis=0)
    return normalized.stack().dropna()


def compute_signal_correlation(sig_a: pd.DataFrame, sig_b: pd.DataFrame) -> float:
    """Compute correlation between two alpha signals.

    Aligns on common (timestamp, symbol) pairs, z-scores cross-sectionally,
    then computes Pearson correlation of the flattened vectors.
    """
    common_idx = sig_a.index.intersection(sig_b.index)
    common_cols = sig_a.columns.intersection(sig_b.columns)
    if len(common_idx) < 30 or len(common_cols) < 2:
        return 0.0

    flat_a = _flatten_signal(sig_a.loc[common_idx, common_cols])
    flat_b = _flatten_signal(sig_b.loc[common_idx, common_cols])

    common = flat_a.index.intersection(flat_b.index)
    if len(common) < 30:
        return 0.0

    return flat_a.loc[common].corr(flat_b.loc[common])


def greedy_prune(alphas: list[dict], signals: dict[str, pd.DataFrame],
                 threshold: float = 0.85, max_per_cluster: int = 1) -> list[dict]:
    """Greedy correlation pruning.

    Args:
        alphas: list of alpha dicts (must have 'expression', 'sharpe_ratio'),
                pre-sorted by Sharpe descending.
        signals: {expression: signal_DataFrame} for each alpha.
        threshold: |correlation| >= this means same cluster.
        max_per_cluster: max alphas to keep per cluster (default 1).

    Returns:
        Pruned list of alpha dicts.
    """
    if not alphas:
        return []

    selected = []
    removed = []

    for alpha in alphas:
        expr = alpha['expression']
        if expr not in signals:
            continue

        sig = signals[expr]
        is_redundant = False

        for kept in selected:
            kept_sig = signals[kept['expression']]
            corr = compute_signal_correlation(sig, kept_sig)
            if abs(corr) >= threshold:
                is_redundant = True
                removed.append((alpha, kept, corr))
                break

        if not is_redundant:
            selected.append(alpha)

    return selected


def prune_alphas(alphas: list[dict], data: dict[str, pd.DataFrame],
                 threshold: float = 0.85) -> tuple[list[dict], list[tuple]]:
    """Full pruning pipeline: evaluate signals then greedy prune.

    Args:
        alphas: list of alpha dicts from DB (with 'expression', 'sharpe_ratio', 'frequency').
        data: {symbol: DataFrame} loaded market data.
        threshold: correlation threshold for pruning.

    Returns:
        (kept_alphas, removed_info) where removed_info is list of
        (removed_alpha, kept_alpha, correlation).
    """
    # Sort by Sharpe descending
    alphas = sorted(alphas, key=lambda a: a.get('sharpe_ratio', 0), reverse=True)

    # Evaluate all signals
    evaluator = AlphaEvaluator(data)
    signals = {}
    valid_alphas = []

    for a in alphas:
        try:
            sig = evaluator.evaluate(a['expression'])
            signals[a['expression']] = sig
            valid_alphas.append(a)
        except (EvaluationError, Exception) as e:
            logger.warning(f'Prune skip {a["expression"][:50]}: {e}')

    # Greedy prune
    removed_info = []
    selected = []

    for alpha in valid_alphas:
        expr = alpha['expression']
        sig = signals[expr]
        is_redundant = False

        for kept in selected:
            corr = compute_signal_correlation(sig, signals[kept['expression']])
            if abs(corr) >= threshold:
                is_redundant = True
                removed_info.append((alpha, kept, corr))
                break

        if not is_redundant:
            selected.append(alpha)

    return selected, removed_info


def print_prune_report(kept: list[dict], removed: list[tuple], threshold: float):
    """Print a summary of pruning results."""
    total = len(kept) + len(removed)
    console.print(f'\n[bold]Correlation Pruning Report[/bold]')
    console.print(f'Threshold: |corr| >= {threshold}')
    console.print(f'Before: {total} alphas -> After: {len(kept)} alphas ({len(removed)} removed)\n')

    if removed:
        t = Table(title='Removed (Redundant) Alphas')
        t.add_column('#', style='dim')
        t.add_column('Removed Expression', max_width=50)
        t.add_column('Sharpe', justify='right')
        t.add_column('Corr', justify='right', style='red')
        t.add_column('Kept Instead', max_width=50)

        for i, (rm, kept_a, corr) in enumerate(removed, 1):
            rm_expr = rm['expression'][:47] + '...' if len(rm['expression']) > 47 else rm['expression']
            kept_expr = kept_a['expression'][:47] + '...' if len(kept_a['expression']) > 47 else kept_a['expression']
            t.add_row(
                str(i), rm_expr,
                f'{rm.get("sharpe_ratio", 0):.2f}',
                f'{corr:.3f}',
                kept_expr,
            )
        console.print(t)

    console.print()
    t = Table(title='Kept (Diverse) Alphas')
    t.add_column('#', style='dim')
    t.add_column('Expression', max_width=65)
    t.add_column('Sharpe', justify='right', style='green')

    for i, a in enumerate(kept, 1):
        expr = a['expression'][:62] + '...' if len(a['expression']) > 62 else a['expression']
        t.add_row(str(i), expr, f'{a.get("sharpe_ratio", 0):.2f}')
    console.print(t)
    console.print()
