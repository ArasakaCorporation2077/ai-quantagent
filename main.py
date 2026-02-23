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


@cli.command()
@click.option('--method', default='equal',
              type=click.Choice(['equal', 'sharpe', 'inverse_vol']),
              help='Weighting method for alpha combination')
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--max-alphas', default=20, help='Max number of alphas to combine')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.option('--all-methods', is_flag=True, help='Run all three weighting methods')
@click.pass_context
def combine(ctx, method, min_sharpe, max_alphas, frequency, all_methods):
    """Combine top alphas into a portfolio and compare performance."""
    from src.orchestrator.combiner import AlphaCombiner

    config = ctx.obj['config']
    combiner = AlphaCombiner(config)

    if all_methods:
        for m in ['equal', 'sharpe', 'inverse_vol']:
            console.print(f'\n{"="*60}')
            result = combiner.run(m, min_sharpe, max_alphas, frequency)
            if result.error:
                console.print(f'[red]Error ({m}): {result.error}[/red]')
    else:
        result = combiner.run(method, min_sharpe, max_alphas, frequency)
        if result.error:
            console.print(f'[red]Error: {result.error}[/red]')


@cli.command()
@click.option('--method', default='sharpe',
              type=click.Choice(['equal', 'sharpe']),
              help='Weighting method (default: sharpe)')
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--capital', default=None, type=float, help='Capital amount (default: from config)')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.pass_context
def signal(ctx, method, min_sharpe, capital, frequency):
    """Generate live trading signal from combined alphas."""
    from src.orchestrator.signal import LiveSignalGenerator

    config = ctx.obj['config']
    gen = LiveSignalGenerator(config)
    gen.run(method=method, min_sharpe=min_sharpe, frequency=frequency, capital=capital)


@cli.command()
@click.option('--method', default='sharpe',
              type=click.Choice(['equal', 'sharpe']),
              help='Weighting method (default: sharpe)')
@click.option('--min-sharpe', default=0.5, help='Minimum Sharpe to include')
@click.option('--capital', default=None, type=float, help='Capital amount (default: from config)')
@click.option('--frequency', default='1d', help='Frequency (default: 1d)')
@click.option('--confirm', is_flag=True, help='Actually execute orders (default: dry run)')
@click.pass_context
def execute(ctx, method, min_sharpe, capital, frequency, confirm):
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
                     frequency=frequency, capital=capital)
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
