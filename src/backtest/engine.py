"""Core backtesting engine.

Follows the Quant Arb blog's approach:
- Cross-sectional position normalization (dollar-neutral, scaled to capital)
- Quarterly alternating out-of-sample sampling
- Forward returns with configurable lookahead
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.alpha.evaluator import AlphaEvaluator, EvaluationError
from src.alpha.validator import AlphaValidator, ValidationError
from src.backtest.metrics import compute_all_metrics
from src.backtest.position import normalize_positions, compute_forward_returns, apply_transaction_costs
from src.config import get_data_dir

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    metrics: dict = field(default_factory=dict)
    metrics_oos: dict = field(default_factory=dict)
    pnl_series: pd.Series | None = None
    error: str | None = None


def _oos_mask(index: pd.DatetimeIndex, sampling: str = 'quarterly') -> pd.Series:
    """Create out-of-sample mask using the blog's alternating scheme.

    Quarterly: quarter % 2 == year % 2  (alternating quarters)
    Monthly:   month % 2 == year % 2    (alternating months)

    Returns boolean Series: True = OOS, False = IS.
    """
    ts = pd.Series(index, index=index)
    if sampling == 'quarterly':
        return (ts.dt.quarter % 2) == (ts.dt.year % 2)
    elif sampling == 'monthly':
        return (ts.dt.month % 2) == (ts.dt.year % 2)
    else:
        raise ValueError(f'Unsupported sampling: {sampling}')


class Backtester:
    def __init__(self, config: dict):
        self.config = config
        self.bt_cfg = config['backtest']
        self.cost_bps = self.bt_cfg.get('transaction_cost_bps', 5)
        self.lookahead = self.bt_cfg.get('lookahead', 1)
        self.capital = self.bt_cfg.get('capital', 10000)
        self.sampling = self.bt_cfg.get('sampling', 'quarterly')
        self._data_cache: dict[tuple, dict[str, pd.DataFrame]] = {}

    def load_data(self, frequency: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Load processed parquet data for all symbols at a given frequency."""
        if symbols is None:
            symbols = self.config['symbols']

        cache_key = (frequency, tuple(sorted(symbols)))
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        data_dir = get_data_dir(self.config) / 'processed'

        data = {}
        for sym in symbols:
            path = data_dir / sym / f'{frequency}.parquet'
            if path.exists():
                try:
                    data[sym] = pd.read_parquet(path)
                except Exception as e:
                    logger.warning(f'Error reading {path}: {e}')
            else:
                logger.warning(f'No data file: {path}')

        self._data_cache[cache_key] = data
        return data

    def run(self, expression: str, frequency: str,
            data: dict[str, pd.DataFrame] | None = None) -> BacktestResult:
        """Run a full backtest for one alpha expression.

        Steps:
        1. Validate expression
        2. Load data
        3. Evaluate alpha on all symbols
        4. Normalize positions cross-sectionally (blog formula)
        5. Compute forward returns
        6. PnL = position * target (blog formula)
        7. Split IS/OOS using alternating quarters
        8. Compute metrics for both IS and OOS
        """
        # 1. Validate
        validator = AlphaValidator()
        try:
            validator.validate(expression)
        except ValidationError as e:
            return BacktestResult(error=f'Validation: {e}')

        # 2. Load data
        if data is None:
            data = self.load_data(frequency)
        if not data:
            return BacktestResult(error='No data available')

        # 3. Evaluate alpha
        evaluator = AlphaEvaluator(data)
        try:
            alpha_panel = evaluator.evaluate(expression)
        except (EvaluationError, Exception) as e:
            return BacktestResult(error=f'Evaluation: {e}')

        if alpha_panel.empty:
            return BacktestResult(error='Alpha evaluation produced empty result')

        # 4. Normalize positions (blog formula: demean then scale)
        positions = normalize_positions(alpha_panel, self.capital)

        # 5. Forward returns
        fwd_returns = compute_forward_returns(data, frequency, self.lookahead)

        # Align indices
        common_idx = positions.index.intersection(fwd_returns.index)
        common_cols = positions.columns.intersection(fwd_returns.columns)
        positions = positions.loc[common_idx, common_cols]
        fwd_returns = fwd_returns.loc[common_idx, common_cols]

        # 6. PnL = position * target (per bar, summed across symbols)
        bar_pnl = (positions * fwd_returns).sum(axis=1)

        # Transaction costs
        costs = apply_transaction_costs(positions, self.cost_bps)
        net_pnl = bar_pnl - costs

        # Normalize to returns
        pnl_returns = net_pnl / self.capital
        pnl_returns = pnl_returns.dropna()

        if len(pnl_returns) < 30:
            return BacktestResult(error=f'Too few periods: {len(pnl_returns)}')

        # 7. Split IS/OOS
        try:
            oos_mask = _oos_mask(pnl_returns.index, self.sampling)
            pnl_oos = pnl_returns[oos_mask]
            pnl_is = pnl_returns[~oos_mask]
        except Exception:
            pnl_oos = pd.Series(dtype=float)
            pnl_is = pnl_returns

        # 8. Compute metrics
        metrics_is = compute_all_metrics(pnl_is, frequency) if len(pnl_is) > 10 else {}
        metrics_oos = compute_all_metrics(pnl_oos, frequency) if len(pnl_oos) > 10 else {}

        # Combined metrics (full sample) for ranking
        metrics_full = compute_all_metrics(pnl_returns, frequency)
        metrics_full['sharpe_is'] = metrics_is.get('sharpe_ratio', 0)
        metrics_full['sharpe_oos'] = metrics_oos.get('sharpe_ratio', 0)
        metrics_full['n_is'] = len(pnl_is)
        metrics_full['n_oos'] = len(pnl_oos)

        return BacktestResult(
            metrics=metrics_full,
            metrics_oos=metrics_oos,
            pnl_series=pnl_returns,
        )
