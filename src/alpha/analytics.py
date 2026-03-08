"""Alpha signal analytics: IC decay, spread decay, and half-life estimation.

Measures how quickly an alpha signal loses predictive power over time,
which informs optimal rebalancing frequency.
"""

import logging

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.alpha.evaluator import AlphaEvaluator, EvaluationError

logger = logging.getLogger(__name__)
console = Console()


def _build_close_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a close price matrix from loaded data. rows=date, cols=symbol."""
    closes = {}
    for symbol, df in data.items():
        closes[symbol] = df['close']
    return pd.DataFrame(closes)


def _forward_returns(close_df: pd.DataFrame, max_lag: int = 10) -> dict[int, pd.DataFrame]:
    """Compute cumulative forward returns for each lag."""
    fwd = {}
    for lag in range(1, max_lag + 1):
        fwd[lag] = close_df.shift(-lag) / close_df - 1.0
    return fwd


def cross_sectional_ic(alpha_df: pd.DataFrame, fwd_ret_df: pd.DataFrame,
                       method: str = 'spearman') -> pd.Series:
    """Per-date cross-sectional correlation between signal and forward return."""
    common_idx = alpha_df.index.intersection(fwd_ret_df.index)
    alpha_aligned = alpha_df.loc[common_idx]
    ret_aligned = fwd_ret_df.loc[common_idx]

    ic_values = []
    for dt in common_idx:
        x = alpha_aligned.loc[dt]
        y = ret_aligned.loc[dt]
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            ic_values.append(np.nan)
            continue
        corr = x[valid].corr(y[valid], method=method)
        ic_values.append(corr)

    return pd.Series(ic_values, index=common_idx)


def compute_ic_decay(alpha_df: pd.DataFrame, close_df: pd.DataFrame,
                     max_lag: int = 10, demean: bool = True) -> pd.DataFrame:
    """Compute IC decay across lags.

    Args:
        alpha_df: signal matrix (rows=date, cols=symbol)
        close_df: close price matrix
        max_lag: max forward lag to test
        demean: if True, subtract cross-sectional mean from returns

    Returns:
        DataFrame with lag, mean_ic, std_ic, ir_ic, n_dates
    """
    fwd_returns = _forward_returns(close_df, max_lag)

    rows = []
    for lag, fwd_df in fwd_returns.items():
        if demean:
            fwd_df = fwd_df.sub(fwd_df.mean(axis=1), axis=0)

        ic_series = cross_sectional_ic(alpha_df, fwd_df)
        ic_clean = ic_series.dropna()

        rows.append({
            'lag': lag,
            'mean_ic': ic_clean.mean() if len(ic_clean) > 0 else np.nan,
            'std_ic': ic_clean.std() if len(ic_clean) > 0 else np.nan,
            'ir_ic': ic_clean.mean() / ic_clean.std() if len(ic_clean) > 0 and ic_clean.std() > 0 else np.nan,
            'n_dates': len(ic_clean),
        })

    return pd.DataFrame(rows)


def quantile_spread_return(alpha_df: pd.DataFrame, close_df: pd.DataFrame,
                           lag: int = 1, top_q: float = 0.2,
                           bottom_q: float = 0.2) -> pd.Series:
    """Per-date top-bottom quantile spread return."""
    fwd_ret = close_df.shift(-lag) / close_df - 1.0
    common_idx = alpha_df.index.intersection(fwd_ret.index)

    spread_returns = []
    for dt in common_idx:
        signal = alpha_df.loc[dt]
        ret = fwd_ret.loc[dt]
        valid = signal.notna() & ret.notna()
        if valid.sum() < 5:
            spread_returns.append(np.nan)
            continue

        signal = signal[valid]
        ret = ret[valid]
        n = len(signal)
        top_n = max(1, int(n * top_q))
        bottom_n = max(1, int(n * bottom_q))

        ranked = signal.sort_values()
        short_names = ranked.index[:bottom_n]
        long_names = ranked.index[-top_n:]

        long_ret = ret[long_names].mean()
        short_ret = ret[short_names].mean()
        spread_returns.append(long_ret - short_ret)

    return pd.Series(spread_returns, index=common_idx)


def compute_spread_decay(alpha_df: pd.DataFrame, close_df: pd.DataFrame,
                         max_lag: int = 10) -> pd.DataFrame:
    """Compute long-short spread decay across lags."""
    rows = []
    for lag in range(1, max_lag + 1):
        spread = quantile_spread_return(alpha_df, close_df, lag=lag)
        spread_clean = spread.dropna()

        rows.append({
            'lag': lag,
            'mean_spread': spread_clean.mean() if len(spread_clean) > 0 else np.nan,
            'std_spread': spread_clean.std() if len(spread_clean) > 0 else np.nan,
            'ir_spread': spread_clean.mean() / spread_clean.std() if len(spread_clean) > 0 and spread_clean.std() > 0 else np.nan,
            'n_dates': len(spread_clean),
        })

    return pd.DataFrame(rows)


def estimate_half_life(decay_df: pd.DataFrame, metric_col: str = 'mean_ic') -> float | None:
    """Estimate half-life: first lag where metric drops below 50% of lag-1 value."""
    df = decay_df.sort_values('lag').copy()
    df = df[df[metric_col].notna()]

    if df.empty:
        return None

    first_val = df.iloc[0][metric_col]
    if pd.isna(first_val) or first_val <= 0:
        return None

    threshold = first_val * 0.5
    below = df[df[metric_col] <= threshold]

    if below.empty:
        return None

    return float(below.iloc[0]['lag'])


def analyze_alpha_halflife(expression: str, data: dict[str, pd.DataFrame],
                           max_lag: int = 10) -> dict:
    """Full half-life analysis for a single alpha expression.

    Returns dict with ic_decay, spread_decay, half_life_ic, half_life_spread.
    """
    evaluator = AlphaEvaluator(data)
    alpha_signal = evaluator.evaluate(expression)
    close_df = _build_close_matrix(data)

    # Align
    common_idx = alpha_signal.index.intersection(close_df.index)
    common_cols = alpha_signal.columns.intersection(close_df.columns)
    alpha_df = alpha_signal.loc[common_idx, common_cols]
    close_aligned = close_df.loc[common_idx, common_cols]

    ic_decay = compute_ic_decay(alpha_df, close_aligned, max_lag)
    spread_decay = compute_spread_decay(alpha_df, close_aligned, max_lag)

    hl_ic = estimate_half_life(ic_decay, 'mean_ic')
    hl_spread = estimate_half_life(spread_decay, 'mean_spread')

    return {
        'expression': expression,
        'ic_decay': ic_decay,
        'spread_decay': spread_decay,
        'half_life_ic': hl_ic,
        'half_life_spread': hl_spread,
    }


def analyze_combined_halflife(alphas: list[dict], data: dict[str, pd.DataFrame],
                              max_lag: int = 10, method: str = 'sharpe') -> dict:
    """Half-life analysis for combined alpha signal."""
    evaluator = AlphaEvaluator(data)
    close_df = _build_close_matrix(data)

    signals = []
    sharpes = []
    for a in alphas:
        try:
            sig = evaluator.evaluate(a['expression'])
            signals.append(sig)
            sharpes.append(a.get('sharpe_ratio', 0))
        except (EvaluationError, Exception):
            pass

    if len(signals) < 2:
        return {'error': 'Need at least 2 valid alphas'}

    # Combine: cross-sectional z-score + weighted sum
    common_idx = signals[0].index
    common_cols = signals[0].columns
    for sig in signals[1:]:
        common_idx = common_idx.intersection(sig.index)
        common_cols = common_cols.intersection(sig.columns)

    n = len(signals)
    if method == 'sharpe':
        total = sum(max(s, 0) for s in sharpes)
        weights = [max(s, 0) / total for s in sharpes] if total > 0 else [1.0 / n] * n
    else:
        weights = [1.0 / n] * n

    combined = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
    for sig, w in zip(signals, weights):
        aligned = sig.loc[common_idx, common_cols]
        row_mean = aligned.mean(axis=1)
        row_std = aligned.std(axis=1).replace(0, 1)
        normalized = aligned.sub(row_mean, axis=0).div(row_std, axis=0)
        combined += w * normalized

    close_aligned = close_df.loc[common_idx, common_cols]

    ic_decay = compute_ic_decay(combined, close_aligned, max_lag)
    spread_decay = compute_spread_decay(combined, close_aligned, max_lag)

    hl_ic = estimate_half_life(ic_decay, 'mean_ic')
    hl_spread = estimate_half_life(spread_decay, 'mean_spread')

    return {
        'expression': f'COMBINED ({n} alphas, {method})',
        'ic_decay': ic_decay,
        'spread_decay': spread_decay,
        'half_life_ic': hl_ic,
        'half_life_spread': hl_spread,
    }


def print_halflife_report(result: dict):
    """Print formatted half-life analysis report."""
    expr = result.get('expression', '?')
    if len(expr) > 70:
        expr = expr[:67] + '...'

    console.print(f'\n[bold]Half-Life Analysis[/bold]')
    console.print(f'Alpha: {expr}\n')

    # IC Decay table
    ic_df = result['ic_decay']
    t = Table(title='IC Decay (Spearman, cross-sectional demeaned)')
    t.add_column('Lag', justify='center')
    t.add_column('Mean IC', justify='right')
    t.add_column('Std IC', justify='right')
    t.add_column('IR (IC/std)', justify='right')
    t.add_column('Bar', min_width=20)

    max_ic = ic_df['mean_ic'].abs().max()
    for _, row in ic_df.iterrows():
        ic = row['mean_ic']
        bar_len = int(abs(ic) / max_ic * 20) if max_ic > 0 and not pd.isna(ic) else 0
        bar_char = '[green]' + '█' * bar_len + '[/green]' if ic > 0 else '[red]' + '█' * bar_len + '[/red]'

        ic_style = 'green' if ic > 0 else 'red'
        t.add_row(
            str(int(row['lag'])),
            f'[{ic_style}]{ic:.4f}[/{ic_style}]' if not pd.isna(ic) else 'N/A',
            f'{row["std_ic"]:.4f}' if not pd.isna(row['std_ic']) else 'N/A',
            f'{row["ir_ic"]:.3f}' if not pd.isna(row['ir_ic']) else 'N/A',
            bar_char,
        )
    console.print(t)

    # Spread Decay table
    sp_df = result['spread_decay']
    t2 = Table(title='Long-Short Spread Decay (top/bottom 20%)')
    t2.add_column('Lag', justify='center')
    t2.add_column('Mean Spread', justify='right')
    t2.add_column('Std', justify='right')
    t2.add_column('IR', justify='right')
    t2.add_column('Bar', min_width=20)

    max_sp = sp_df['mean_spread'].abs().max()
    for _, row in sp_df.iterrows():
        sp = row['mean_spread']
        bar_len = int(abs(sp) / max_sp * 20) if max_sp > 0 and not pd.isna(sp) else 0
        bar_char = '[green]' + '█' * bar_len + '[/green]' if sp > 0 else '[red]' + '█' * bar_len + '[/red]'

        sp_style = 'green' if sp > 0 else 'red'
        t2.add_row(
            str(int(row['lag'])),
            f'[{sp_style}]{sp:.4f}[/{sp_style}]' if not pd.isna(sp) else 'N/A',
            f'{row["std_spread"]:.4f}' if not pd.isna(row['std_spread']) else 'N/A',
            f'{row["ir_spread"]:.3f}' if not pd.isna(row['ir_spread']) else 'N/A',
            bar_char,
        )
    console.print(t2)

    # Summary
    hl_ic = result.get('half_life_ic')
    hl_spread = result.get('half_life_spread')

    console.print(f'\n[bold]Half-Life Estimates[/bold]')
    max_lag_ic = int(ic_df['lag'].max())
    max_lag_sp = int(sp_df['lag'].max())
    ic_str = f'{hl_ic:.0f} bars' if hl_ic else f'>{max_lag_ic} bars (signal persists)'
    sp_str = f'{hl_spread:.0f} bars' if hl_spread else f'>{max_lag_sp} bars (signal persists)'
    console.print(f'  IC-based:     {ic_str}')
    console.print(f'  Spread-based: {sp_str}')

    # Recommendation
    hl = hl_spread or hl_ic
    if hl:
        if hl <= 2:
            rec = 'every bar (aggressive rebalancing recommended)'
        elif hl <= 5:
            rec = f'every {max(1, int(hl * 0.7))}-{int(hl)} bars'
        else:
            rec = f'every {int(hl * 0.5)}-{int(hl)} bars (signal is persistent)'
        console.print(f'\n  [bold cyan]Recommended rebalance:[/bold cyan] {rec}')
    console.print()
