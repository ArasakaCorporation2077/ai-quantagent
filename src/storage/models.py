"""SQLite table schemas for alpha research storage."""

TABLES = {
    'strategies': """
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_text TEXT NOT NULL,
            category TEXT,
            llm_provider TEXT,
            llm_model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    'alphas': """
        CREATE TABLE IF NOT EXISTS alphas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER REFERENCES strategies(id),
            expression TEXT NOT NULL,
            frequency TEXT NOT NULL,
            expression_hash TEXT,
            is_valid BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(expression_hash)
        )
    """,
    'backtest_results': """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alpha_id INTEGER REFERENCES alphas(id),
            sharpe_ratio REAL,
            sortino_ratio REAL,
            annualized_return REAL,
            max_drawdown REAL,
            calmar_ratio REAL,
            win_rate REAL,
            profit_factor REAL,
            total_return REAL,
            num_periods INTEGER,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    'pipeline_runs': """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            strategies_generated INTEGER DEFAULT 0,
            alphas_generated INTEGER DEFAULT 0,
            alphas_valid INTEGER DEFAULT 0,
            alphas_above_threshold INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running'
        )
    """,
}
