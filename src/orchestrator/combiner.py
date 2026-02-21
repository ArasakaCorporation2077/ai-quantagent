"""Alpha combination and portfolio construction.

Combines multiple discovered alpha signals into a single composite portfolio.
Supports equal-weight, Sharpe-weighted, and inverse-volatility weighting.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.alpha.evaluator import AlphaEvaluator, EvaluationError
from src.backtest.engine import Backtester, BacktestResult, _oos_mask
from src.backtest.metrics import compute_all_metrics
from src.backtest.position import normalize_positions, compute_forward_returns, apply_transaction_costs
from src.config import get_db_path
from src.storage.database import Database

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class AlphaRecord:
    expression: str
    frequency: str
    sharpe_ratio: float
    signal: pd.DataFrame | None = None
    pnl_std: float = 0.0


@dataclass
class CombineResult:
    method: str
    weights: dict = field(default_factory=dict)
    combined_metrics: dict = field(default_factory=dict)
    combined_metrics_oos: dict = field(default_factory=dict)
    pnl_series: pd.Series | None = None
    individual_metrics: list = field(default_factory=list)
    error: str | None = None


class AlphaCombiner:
    def __init__(self, config: dict):
        self.config = config
        self.bt_cfg = config['backtest']
        self.backtester = Backtester(config)
        self.db = Database(get_db_path(config))

    def run(self, method: str = 'equal', min_sharpe: float = 0.5,
            max_alphas: int = 20, frequency: str = '1d') -> CombineResult:
        """Full combination pipeline."""

        # 1. Query good alphas
        raw_alphas = self.db.get_top_alphas(min_sharpe=min_sharpe, limit=max_alphas)
        raw_alphas = [a for a in raw_alphas if a['frequency'] == frequency]

        if len(raw_alphas) < 2:
            return CombineResult(method=method,
                                 error=f'Need at least 2 alphas, found {len(raw_alphas)}')

        console.print(f'Found {len(raw_alphas)} alphas with Sharpe >= {min_sharpe}')

        # 2. Load data once, compute each alpha's signal
        data = self.backtester.load_data(frequency)
        evaluator = AlphaEvaluator(data)

        capital = self.bt_cfg.get('capital', 10000)
        cost_bps = self.bt_cfg.get('transaction_cost_bps', 5)
        lookahead = self.bt_cfg.get('lookahead', 1)

        fwd_returns = compute_forward_returns(data, frequency, lookahead)

        alpha_records = []
        individual_metrics = []

        for a in raw_alphas:
            try:
                signal = evaluator.evaluate(a['expression'])
                record = AlphaRecord(
                    expression=a['expression'],
                    frequency=a['frequency'],
                    sharpe_ratio=a['sharpe_ratio'],
                    signal=signal,
                )

                # Compute individual PnL stats (for inverse-vol weighting + comparison)
                positions = normalize_positions(signal, capital)
                common_idx = positions.index.intersection(fwd_returns.index)
                common_cols = positions.columns.intersection(fwd_returns.columns)
                pos = positions.loc[common_idx, common_cols]
                fwd = fwd_returns.loc[common_idx, common_cols]
                bar_pnl = (pos * fwd).sum(axis=1)
                costs = apply_transaction_costs(pos, cost_bps)
                net_pnl = (bar_pnl - costs) / capital
                net_pnl = net_pnl.dropna()
                record.pnl_std = net_pnl.std()

                indiv_m = compute_all_metrics(net_pnl, frequency)
                individual_metrics.append(indiv_m)

                alpha_records.append(record)
                console.print(f'  Evaluated: {a["expression"][:60]}... Sharpe={a["sharpe_ratio"]:.2f}')

            except (EvaluationError, Exception) as e:
                logger.warning(f'Skipping {a["expression"][:60]}: {e}')

        if len(alpha_records) < 2:
            return CombineResult(method=method,
                                 error=f'Only {len(alpha_records)} alphas evaluated successfully')

        # 3. Compute weights
        weights = self._compute_weights(alpha_records, method)

        # 4. Combine signals
        combined_signal = self._combine_signals(alpha_records, weights)

        # 5. Backtest combined
        bt_result = self._backtest_combined(combined_signal, data, frequency)

        if bt_result.error:
            return CombineResult(method=method, weights=weights, error=bt_result.error)

        # 6. Build result
        result = CombineResult(
            method=method,
            weights={a.expression: weights[a.expression] for a in alpha_records},
            combined_metrics=bt_result.metrics,
            combined_metrics_oos=bt_result.metrics_oos,
            pnl_series=bt_result.pnl_series,
            individual_metrics=individual_metrics,
        )

        # 7. Report
        self._print_report(result, alpha_records)

        return result

    def _compute_weights(self, alphas: list[AlphaRecord], method: str) -> dict:
        n = len(alphas)

        if method == 'equal':
            return {a.expression: 1.0 / n for a in alphas}

        elif method == 'sharpe':
            total = sum(max(a.sharpe_ratio, 0) for a in alphas)
            if total == 0:
                return {a.expression: 1.0 / n for a in alphas}
            return {a.expression: max(a.sharpe_ratio, 0) / total for a in alphas}

        elif method == 'inverse_vol':
            inv_vols = [1.0 / a.pnl_std if a.pnl_std > 0 else 0.0 for a in alphas]
            total = sum(inv_vols)
            if total == 0:
                return {a.expression: 1.0 / n for a in alphas}
            return {a.expression: iv / total for a, iv in zip(alphas, inv_vols)}

        else:
            raise ValueError(f'Unknown weighting method: {method}')

    def _combine_signals(self, alphas: list[AlphaRecord], weights: dict) -> pd.DataFrame:
        """Weighted sum of cross-sectionally z-scored alpha signals."""
        # Find common index/columns
        common_idx = alphas[0].signal.index
        common_cols = alphas[0].signal.columns
        for a in alphas[1:]:
            common_idx = common_idx.intersection(a.signal.index)
            common_cols = common_cols.intersection(a.signal.columns)

        combined = pd.DataFrame(0.0, index=common_idx, columns=common_cols)

        for a in alphas:
            w = weights[a.expression]
            aligned = a.signal.loc[common_idx, common_cols]
            # Cross-sectional z-score so different magnitudes don't dominate
            row_mean = aligned.mean(axis=1)
            row_std = aligned.std(axis=1).replace(0, 1)
            normalized = aligned.sub(row_mean, axis=0).div(row_std, axis=0)
            combined += w * normalized

        return combined

    def _backtest_combined(self, combined_signal: pd.DataFrame,
                           data: dict, frequency: str) -> BacktestResult:
        """Backtest a pre-computed combined signal. Mirrors Backtester.run() logic."""
        capital = self.bt_cfg.get('capital', 10000)
        cost_bps = self.bt_cfg.get('transaction_cost_bps', 5)
        lookahead = self.bt_cfg.get('lookahead', 1)
        sampling = self.bt_cfg.get('sampling', 'quarterly')

        positions = normalize_positions(combined_signal, capital)
        fwd_returns = compute_forward_returns(data, frequency, lookahead)

        common_idx = positions.index.intersection(fwd_returns.index)
        common_cols = positions.columns.intersection(fwd_returns.columns)
        positions = positions.loc[common_idx, common_cols]
        fwd_returns = fwd_returns.loc[common_idx, common_cols]

        bar_pnl = (positions * fwd_returns).sum(axis=1)
        costs = apply_transaction_costs(positions, cost_bps)
        net_pnl = bar_pnl - costs
        pnl_returns = net_pnl / capital
        pnl_returns = pnl_returns.dropna()

        if len(pnl_returns) < 30:
            return BacktestResult(error=f'Too few periods: {len(pnl_returns)}')

        try:
            oos_mask = _oos_mask(pnl_returns.index, sampling)
            pnl_oos = pnl_returns[oos_mask]
            pnl_is = pnl_returns[~oos_mask]
        except Exception:
            pnl_oos = pd.Series(dtype=float)
            pnl_is = pnl_returns

        metrics_is = compute_all_metrics(pnl_is, frequency) if len(pnl_is) > 10 else {}
        metrics_oos = compute_all_metrics(pnl_oos, frequency) if len(pnl_oos) > 10 else {}
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

    def _print_report(self, result: CombineResult, alpha_records: list[AlphaRecord]):
        console.print(f'\n[bold]Alpha Combination Report ({result.method})[/bold]\n')
        console.print(f'Alphas combined: {len(alpha_records)}')
        console.print(f'Weighting method: {result.method}\n')

        # Weights table
        wt = Table(title='Alpha Weights')
        wt.add_column('#', style='dim')
        wt.add_column('Expression', max_width=65)
        wt.add_column('Weight', style='green')
        wt.add_column('Sharpe', style='cyan')
        for i, a in enumerate(alpha_records, 1):
            expr = a.expression[:60] + '...' if len(a.expression) > 60 else a.expression
            wt.add_row(str(i), expr,
                        f'{result.weights[a.expression]:.3f}',
                        f'{a.sharpe_ratio:.2f}')
        console.print(wt)
        console.print()

        # Comparison table
        ct = Table(title='Individual vs Combined Performance')
        ct.add_column('Alpha', style='cyan')
        ct.add_column('Sharpe', justify='right')
        ct.add_column('Return', justify='right')
        ct.add_column('MaxDD', justify='right')
        ct.add_column('Win Rate', justify='right')
        ct.add_column('Sortino', justify='right')

        for i, m in enumerate(result.individual_metrics, 1):
            ct.add_row(
                f'Alpha {i}',
                f'{m.get("sharpe_ratio", 0):.2f}',
                f'{m.get("annualized_return", 0):.1%}',
                f'{m.get("max_drawdown", 0):.1%}',
                f'{m.get("win_rate", 0):.1%}',
                f'{m.get("sortino_ratio", 0):.2f}',
            )

        cm = result.combined_metrics
        ct.add_row(
            f'[bold]COMBINED[/bold]',
            f'[bold]{cm.get("sharpe_ratio", 0):.2f}[/bold]',
            f'[bold]{cm.get("annualized_return", 0):.1%}[/bold]',
            f'[bold]{cm.get("max_drawdown", 0):.1%}[/bold]',
            f'[bold]{cm.get("win_rate", 0):.1%}[/bold]',
            f'[bold]{cm.get("sortino_ratio", 0):.2f}[/bold]',
        )
        console.print(ct)

        # Summary
        avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in result.individual_metrics])
        comb_sharpe = cm.get('sharpe_ratio', 0)
        improvement = ((comb_sharpe / avg_sharpe) - 1) * 100 if avg_sharpe > 0 else 0
        console.print(f'\nAvg individual Sharpe: {avg_sharpe:.2f}')
        console.print(f'Combined Sharpe:      {comb_sharpe:.2f}')
        console.print(f'Diversification gain: {improvement:+.1f}%\n')
