"""Position sizing and signal generation.

Matches the Quant Arb blog's cross-sectional normalization:
    position = capital * (x - x.mean()) / sum(abs(x - x.mean()))
This is dollar-neutral and sums to `capital` in absolute value.
"""

import numpy as np
import pandas as pd


def normalize_positions(alpha_values: pd.DataFrame, capital: float = 10000) -> pd.DataFrame:
    """Cross-sectional normalization of alpha values into positions.

    Exact formula from the blog:
        position = capital * (x - mean(x)) / sum(|x - mean(x)|)

    This is equivalent to scale(x - cs_mean(x)) * capital.
    Dollar-neutral: positions sum to zero at each timestamp.
    Leverage-normalized: sum(|position|) = capital at each timestamp.

    Args:
        alpha_values: DataFrame with columns=symbols, index=timestamps
        capital: total absolute notional (default $10,000)

    Returns:
        DataFrame of same shape with dollar positions per symbol.
    """
    def _normalize_row(row):
        valid = row.dropna()
        if len(valid) < 2:
            return row * 0
        demeaned = row - valid.mean()
        total_abs = demeaned.abs().sum()
        if total_abs == 0:
            return row * 0
        return capital * demeaned / total_abs

    return alpha_values.apply(_normalize_row, axis=1)


def compute_forward_returns(data: dict[str, pd.DataFrame], frequency: str, lookahead: int = 1) -> pd.DataFrame:
    """Compute forward returns for all symbols.

    Matches blog: close.pct_change(periods=lookahead).shift(-lookahead)

    Returns DataFrame: index=timestamps, columns=symbols, values=forward return.
    """
    returns = {}
    for symbol, df in data.items():
        close = df['close']
        fwd = close.pct_change(periods=lookahead).shift(-lookahead)
        returns[symbol] = fwd
    return pd.DataFrame(returns)


def apply_transaction_costs(
    positions: pd.DataFrame,
    cost_bps: float = 5
) -> pd.Series:
    """Compute transaction cost series from position changes.

    cost_bps: cost in basis points per side (half-spread).
    Returns: Series of costs per timestamp (positive values to subtract).
    """
    cost_rate = cost_bps / 10000
    turnover = positions.diff().abs().sum(axis=1)
    return turnover * cost_rate
