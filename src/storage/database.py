"""SQLite database operations for alpha research."""

import hashlib
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from src.storage.models import TABLES

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist, and run migrations."""
        with self._conn() as conn:
            for name, ddl in TABLES.items():
                conn.execute(ddl)
            # Migration: add OOS columns if missing
            cols = [r[1] for r in conn.execute('PRAGMA table_info(backtest_results)').fetchall()]
            if 'sharpe_oos' not in cols:
                conn.execute('ALTER TABLE backtest_results ADD COLUMN sharpe_oos REAL')
                conn.execute('ALTER TABLE backtest_results ADD COLUMN sharpe_is REAL')
                conn.execute('ALTER TABLE backtest_results ADD COLUMN n_oos INTEGER')
                conn.execute('ALTER TABLE backtest_results ADD COLUMN n_is INTEGER')

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def save_strategy(self, text: str, category: str | None = None,
                      provider: str | None = None, model: str | None = None) -> int:
        """Save a strategy idea. Returns strategy ID."""
        with self._conn() as conn:
            cur = conn.execute(
                'INSERT INTO strategies (strategy_text, category, llm_provider, llm_model) VALUES (?, ?, ?, ?)',
                (text, category, provider, model)
            )
            return cur.lastrowid

    def save_alpha(self, strategy_id: int, expression: str, frequency: str) -> int | None:
        """Save an alpha expression. Returns alpha ID, or None if duplicate."""
        expr_hash = hashlib.sha256(expression.encode()).hexdigest()
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    'INSERT INTO alphas (strategy_id, expression, frequency, expression_hash) VALUES (?, ?, ?, ?)',
                    (strategy_id, expression, frequency, expr_hash)
                )
                return cur.lastrowid
        except sqlite3.IntegrityError:
            logger.debug(f'Duplicate alpha: {expression[:60]}...')
            return None

    def mark_alpha_invalid(self, alpha_id: int):
        """Mark an alpha as invalid (failed evaluation)."""
        with self._conn() as conn:
            conn.execute('UPDATE alphas SET is_valid = 0 WHERE id = ?', (alpha_id,))

    def save_backtest_result(self, alpha_id: int, metrics: dict) -> int:
        """Save backtest results for an alpha (including OOS metrics)."""
        with self._conn() as conn:
            cur = conn.execute(
                '''INSERT INTO backtest_results
                   (alpha_id, sharpe_ratio, sortino_ratio, annualized_return,
                    max_drawdown, calmar_ratio, win_rate, profit_factor,
                    total_return, num_periods, sharpe_oos, sharpe_is, n_oos, n_is)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (alpha_id,
                 metrics.get('sharpe_ratio'),
                 metrics.get('sortino_ratio'),
                 metrics.get('annualized_return'),
                 metrics.get('max_drawdown'),
                 metrics.get('calmar_ratio'),
                 metrics.get('win_rate'),
                 metrics.get('profit_factor'),
                 metrics.get('total_return'),
                 metrics.get('num_periods'),
                 metrics.get('sharpe_oos'),
                 metrics.get('sharpe_is'),
                 metrics.get('n_oos'),
                 metrics.get('n_is'))
            )
            return cur.lastrowid

    def get_top_alphas(self, min_sharpe: float = 0, limit: int = 20) -> list[dict]:
        """Get top-performing alphas by Sharpe ratio (deduplicated per alpha)."""
        with self._conn() as conn:
            rows = conn.execute(
                '''SELECT a.expression, a.frequency, s.strategy_text, s.category,
                          MAX(b.sharpe_ratio) as sharpe_ratio, b.sortino_ratio,
                          b.annualized_return, b.max_drawdown, b.total_return,
                          b.win_rate
                   FROM backtest_results b
                   JOIN alphas a ON b.alpha_id = a.id
                   JOIN strategies s ON a.strategy_id = s.id
                   WHERE b.sharpe_ratio >= ? AND a.is_valid = 1
                   GROUP BY a.id
                   ORDER BY sharpe_ratio DESC
                   LIMIT ?''',
                (min_sharpe, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def save_pipeline_run(self, stats: dict) -> int:
        """Record a pipeline run."""
        with self._conn() as conn:
            cur = conn.execute(
                '''INSERT INTO pipeline_runs
                   (started_at, completed_at, strategies_generated, alphas_generated,
                    alphas_valid, alphas_above_threshold, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (stats.get('started_at'), datetime.now().isoformat(),
                 stats.get('strategies', 0), stats.get('alphas', 0),
                 stats.get('valid', 0), stats.get('good', 0), 'completed')
            )
            return cur.lastrowid

    def get_stats(self) -> dict:
        """Get overall database statistics."""
        with self._conn() as conn:
            stats = {}
            stats['total_strategies'] = conn.execute('SELECT COUNT(*) FROM strategies').fetchone()[0]
            stats['total_alphas'] = conn.execute('SELECT COUNT(*) FROM alphas').fetchone()[0]
            stats['valid_alphas'] = conn.execute('SELECT COUNT(*) FROM alphas WHERE is_valid = 1').fetchone()[0]
            stats['total_backtests'] = conn.execute('SELECT COUNT(*) FROM backtest_results').fetchone()[0]
            row = conn.execute('SELECT MAX(sharpe_ratio) FROM backtest_results').fetchone()
            stats['best_sharpe'] = row[0] if row[0] else 0
            return stats
