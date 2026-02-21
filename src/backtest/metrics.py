"""Performance metrics for backtesting."""

import numpy as np
import pandas as pd

from src.data.schema import ANNUALIZATION_FACTORS


def compute_all_metrics(pnl: pd.Series, frequency: str) -> dict:
    """Compute all performance metrics for a PnL series."""
    ann = ANNUALIZATION_FACTORS.get(frequency, 365)
    return {
        'sharpe_ratio': sharpe_ratio(pnl, ann),
        'sortino_ratio': sortino_ratio(pnl, ann),
        'annualized_return': annualized_return(pnl, ann),
        'max_drawdown': max_drawdown(pnl),
        'calmar_ratio': calmar_ratio(pnl, ann),
        'win_rate': win_rate(pnl),
        'profit_factor': profit_factor(pnl),
        'total_return': total_return(pnl),
        'num_periods': len(pnl),
        'avg_daily_pnl': pnl.mean(),
        'pnl_std': pnl.std(),
    }


def sharpe_ratio(pnl: pd.Series, periods_per_year: int = 365) -> float:
    if pnl.std() == 0 or len(pnl) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * pnl.mean() / pnl.std())


def sortino_ratio(pnl: pd.Series, periods_per_year: int = 365) -> float:
    downside = pnl[pnl < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * pnl.mean() / downside.std())


def annualized_return(pnl: pd.Series, periods_per_year: int = 365) -> float:
    if len(pnl) == 0:
        return 0.0
    total = (1 + pnl).prod()
    n_years = len(pnl) / periods_per_year
    if n_years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def max_drawdown(pnl: pd.Series) -> float:
    if len(pnl) == 0:
        return 0.0
    cumulative = (1 + pnl).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(pnl: pd.Series, periods_per_year: int = 365) -> float:
    mdd = max_drawdown(pnl)
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return(pnl, periods_per_year)
    return float(ann_ret / abs(mdd))


def win_rate(pnl: pd.Series) -> float:
    if len(pnl) == 0:
        return 0.0
    return float((pnl > 0).sum() / len(pnl))


def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


def total_return(pnl: pd.Series) -> float:
    if len(pnl) == 0:
        return 0.0
    return float((1 + pnl).prod() - 1)
