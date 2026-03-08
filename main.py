"""AI Agent Hedge Fund - CLI Entry Point."""

import asyncio
import logging

import click
from rich.console import Console
from rich.logging import RichHandler

from src.config import load_config, load_secrets

console = Console()


def setup_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=level,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option('--config', default=None, help='Path to config.yaml')
@click.pass_context
def cli(ctx, config):
    """AI Agent Hedge Fund - Automated Alpha Discovery for Crypto Futures."""
    setup_logging()
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['secrets'] = load_secrets()


@cli.command()
@click.option('--symbols', default=None, help='Comma-separated symbols (default: all)')
@click.option('--frequencies', default=None, help='Comma-separated frequencies (default: all)')
@click.pass_context
def download(ctx, symbols, frequencies):
    """Download historical kline data from Binance."""
    from src.data.downloader import BinanceDataDownloader

    config = ctx.obj['config']
    sym_list = symbols.split(',') if symbols else config['symbols']
    freq_list = frequencies.split(',') if frequencies else config['frequencies']

    downloader = BinanceDataDownloader(config)
    asyncio.run(downloader.download_all(sym_list, freq_list))


@cli.command()
@click.option('--symbols', default=None, help='Comma-separated symbols (default: all)')
@click.option('--frequencies', default=None, help='Comma-separated frequencies (default: all)')
@click.pass_context
def process(ctx, symbols, frequencies):
    """Process raw CSVs into enriched parquet files."""
    from src.data.processor import DataProcessor

    config = ctx.obj['config']
    sym_list = symbols.split(',') if symbols else config['symbols']
    freq_list = frequencies.split(',') if frequencies else config['frequencies']

    processor = DataProcessor(config)
    processor.process_all(sym_list, freq_list)


@cli.command()
@click.option('--expr', required=True, help='Alpha expression to evaluate')
@click.option('--freq', required=True, help='Frequency (e.g., 1h)')
@click.pass_context
def backtest(ctx, expr, freq):
    """Backtest a single alpha expression."""
    from src.backtest.engine import Backtester

    config = ctx.obj['config']
    bt = Backtester(config)
    result = bt.run(expr, freq)
    if result.error:
        console.print(f'[red]Error: {result.error}[/red]')
    else:
        console.print(result.metrics)


@cli.command()
@click.option('--iterations', default=10, help='Number of pipeline iterations')
@click.pass_context
def pipeline(ctx, iterations):
    """Run the full alpha discovery pipeline."""
    from src.orchestrator.pipeline import AlphaPipeline

    config = ctx.obj['config']
    secrets = ctx.obj['secrets']
    pipe = AlphaPipeline(config, secrets)
    stats = pipe.run(iterations)
    console.print(stats)


@cli.command()
@click.option('--top', default=20, help='Number of top alphas to show')
@click.pass_context
def report(ctx, top):
    """Show top discovered alphas from the database."""
    from src.storage.database import Database
    from src.config import get_db_path

    config = ctx.obj['config']
    db = Database(get_db_path(config))
    alphas = db.get_top_alphas(limit=top)
    for a in alphas:
        console.print(a)


@cli.command(name='half-life')
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--max-alphas', default=20, help='Max alphas to analyze')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.option('--max-lag', default=10, help='Maximum forward lag to test')
@click.option('--combined', is_flag=True, help='Analyze combined signal instead of individual')
@click.option('--method', default='sharpe', type=click.Choice(['equal', 'sharpe']),
              help='Weighting method for combined analysis')
@click.pass_context
def half_life(ctx, min_sharpe, max_alphas, frequency, max_lag, combined, method):
    """Analyze alpha signal half-life (IC and spread decay)."""
    from src.alpha.analytics import (
        analyze_alpha_halflife, analyze_combined_halflife, print_halflife_report
    )
    from src.backtest.engine import Backtester
    from src.config import get_db_path
    from src.storage.database import Database

    config = ctx.obj['config']
    db = Database(get_db_path(config))
    bt = Backtester(config)

    raw_alphas = db.get_top_alphas(min_sharpe=min_sharpe, limit=max_alphas)
    raw_alphas = [a for a in raw_alphas if a['frequency'] == frequency]
    console.print(f'Loaded {len(raw_alphas)} alphas (Sharpe >= {min_sharpe}, freq={frequency})')

    data = bt.load_data(frequency)

    if combined:
        result = analyze_combined_halflife(raw_alphas, data, max_lag, method)
        print_halflife_report(result)
    else:
        for a in raw_alphas[:5]:
            try:
                result = analyze_alpha_halflife(a['expression'], data, max_lag)
                print_halflife_report(result)
            except Exception as e:
                console.print(f'[red]Error analyzing {a["expression"][:50]}: {e}[/red]')


@cli.command()
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--max-alphas', default=50, help='Max alphas to consider')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.option('--symbols', default='5,10,15', help='Comma-separated symbol counts to test')
@click.option('--rebal', default='1,3,7', help='Comma-separated rebalance bar counts')
@click.option('--methods', default='equal,sharpe', help='Comma-separated methods')
@click.option('--corr-threshold', default=0.85, help='Correlation pruning threshold')
@click.pass_context
def grid(ctx, min_sharpe, max_alphas, frequency, symbols, rebal, methods, corr_threshold):
    """Grid search for optimal portfolio parameters (symbols, rebalancing, method)."""
    from src.backtest.grid import run_grid, print_grid_report
    from src.config import get_db_path
    from src.storage.database import Database

    config = ctx.obj['config']
    db = Database(get_db_path(config))

    raw_alphas = db.get_top_alphas(min_sharpe=min_sharpe, limit=max_alphas)
    raw_alphas = [a for a in raw_alphas if a['frequency'] == frequency]
    console.print(f'Loaded {len(raw_alphas)} alphas (Sharpe >= {min_sharpe}, freq={frequency})')

    sym_counts = [int(x) for x in symbols.split(',')]
    rebal_bars = [int(x) for x in rebal.split(',')]
    method_list = methods.split(',')

    console.print(f'Grid: symbols={sym_counts} x rebal={rebal_bars} x methods={method_list}')
    console.print(f'Total combinations: {len(sym_counts) * len(rebal_bars) * len(method_list)}')

    df = run_grid(config, raw_alphas,
                  symbol_counts=sym_counts,
                  rebalance_bars_list=rebal_bars,
                  methods=method_list,
                  frequency=frequency,
                  corr_threshold=corr_threshold)

    print_grid_report(df, frequency)


@cli.command()
@click.option('--method', default='equal',
              type=click.Choice(['equal', 'sharpe', 'inverse_vol']),
              help='Weighting method for alpha combination')
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--max-alphas', default=20, help='Max number of alphas to combine')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.option('--corr-threshold', default=0.85, help='Correlation pruning threshold (default: 0.85)')
@click.option('--all-methods', is_flag=True, help='Run all three weighting methods')
@click.pass_context
def combine(ctx, method, min_sharpe, max_alphas, frequency, corr_threshold, all_methods):
    """Combine top alphas into a portfolio and compare performance."""
    from src.orchestrator.combiner import AlphaCombiner

    config = ctx.obj['config']
    combiner = AlphaCombiner(config)

    if all_methods:
        for m in ['equal', 'sharpe', 'inverse_vol']:
            console.print(f'\n{"="*60}')
            result = combiner.run(m, min_sharpe, max_alphas, frequency, corr_threshold)
            if result.error:
                console.print(f'[red]Error ({m}): {result.error}[/red]')
    else:
        result = combiner.run(method, min_sharpe, max_alphas, frequency, corr_threshold)
        if result.error:
            console.print(f'[red]Error: {result.error}[/red]')


@cli.command()
@click.option('--method', default='sharpe',
              type=click.Choice(['equal', 'sharpe']),
              help='Weighting method (default: sharpe)')
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--capital', default=None, type=float, help='Capital amount (default: from config)')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.option('--corr-threshold', default=0.85, help='Correlation pruning threshold (default: 0.85)')
@click.pass_context
def signal(ctx, method, min_sharpe, capital, frequency, corr_threshold):
    """Generate live trading signal from combined alphas."""
    from src.orchestrator.signal import LiveSignalGenerator

    config = ctx.obj['config']
    gen = LiveSignalGenerator(config)
    gen.run(method=method, min_sharpe=min_sharpe, frequency=frequency,
            capital=capital, corr_threshold=corr_threshold)


@cli.command()
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--max-alphas', default=100, help='Max alphas to analyze')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.option('--corr-threshold', default=0.85, help='Correlation threshold (default: 0.85)')
@click.pass_context
def prune(ctx, min_sharpe, max_alphas, frequency, corr_threshold):
    """Analyze and prune correlated alphas."""
    from src.alpha.pruner import prune_alphas, print_prune_report
    from src.backtest.engine import Backtester
    from src.config import get_db_path
    from src.storage.database import Database

    config = ctx.obj['config']
    db = Database(get_db_path(config))
    bt = Backtester(config)

    raw_alphas = db.get_top_alphas(min_sharpe=min_sharpe, limit=max_alphas)
    raw_alphas = [a for a in raw_alphas if a['frequency'] == frequency]
    console.print(f'Loaded {len(raw_alphas)} alphas (Sharpe >= {min_sharpe}, freq={frequency})')

    data = bt.load_data(frequency)
    kept, removed = prune_alphas(raw_alphas, data, threshold=corr_threshold)
    print_prune_report(kept, removed, corr_threshold)


@cli.command()
@click.option('--method', default='sharpe',
              type=click.Choice(['equal', 'sharpe']),
              help='Weighting method (default: sharpe)')
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--capital', default=None, type=float, help='Capital amount (default: from config)')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.option('--corr-threshold', default=0.85, help='Correlation pruning threshold (default: 0.85)')
@click.option('--confirm', is_flag=True, help='Actually execute orders (default: dry run)')
@click.pass_context
def execute(ctx, method, min_sharpe, capital, frequency, corr_threshold, confirm):
    """Generate signal and execute on Hyperliquid (dry run by default)."""
    from src.orchestrator.signal import LiveSignalGenerator
    from src.execution.hyperliquid import HyperliquidExecutor

    config = ctx.obj['config']
    secrets = ctx.obj['secrets']

    # 1. Auto capital: balance × multiplier
    executor = HyperliquidExecutor(secrets)
    if capital is None:
        exec_cfg = config.get('execution', {})
        multiplier = exec_cfg.get('capital_multiplier', 0)
        if multiplier > 0:
            balance = executor.get_account_value()
            capital = round(balance * multiplier, 2)
            console.print(f'[dim]Auto capital: ${balance:.0f} x {multiplier} = ${capital:.0f}[/dim]')

    # 2. Generate signal
    gen = LiveSignalGenerator(config)
    result = gen.run(method=method, min_sharpe=min_sharpe,
                     frequency=frequency, capital=capital,
                     corr_threshold=corr_threshold)
    if not result:
        console.print('[red]No signal generated[/red]')
        return

    # 3. Execute on Hyperliquid
    executor.execute_signal(result['positions'], dry_run=not confirm)


@cli.command()
@click.pass_context
def positions(ctx):
    """Show current Hyperliquid positions."""
    from src.execution.hyperliquid import HyperliquidExecutor

    secrets = ctx.obj['secrets']
    executor = HyperliquidExecutor(secrets)
    executor.print_positions()


if __name__ == '__main__':
    cli()
