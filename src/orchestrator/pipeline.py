"""Main automated alpha discovery pipeline."""

import logging
from datetime import datetime

from rich.console import Console
from rich.table import Table

from src.alpha.validator import AlphaValidator, ValidationError
from src.backtest.engine import Backtester
from src.config import get_db_path
from src.llm.client import LLMClient
from src.llm.generator import AlphaGenerator
from src.storage.database import Database

logger = logging.getLogger(__name__)
console = Console()


class AlphaPipeline:
    def __init__(self, config: dict, secrets: dict):
        self.config = config
        self.llm_cfg = config['llm']
        self.pipe_cfg = config.get('pipeline', {})

        # Init LLM client
        provider = self.llm_cfg['provider']
        api_key = secrets.get(f'{provider}_api_key', '')
        if not api_key or api_key.startswith('YOUR_'):
            raise ValueError(
                f'Missing API key for {provider}. '
                f'Update config/secrets.yaml with your {provider} API key.'
            )

        self.llm = LLMClient(provider, api_key, self.llm_cfg['model'])
        self.generator = AlphaGenerator(self.llm, config)
        self.backtester = Backtester(config)
        self.db = Database(get_db_path(config))
        self.validator = AlphaValidator()

    def run(self, n_iterations: int = 10) -> dict:
        """Main loop: generate strategies -> alpha expressions -> backtest -> store."""
        stats = {
            'started_at': datetime.now().isoformat(),
            'strategies': 0,
            'alphas': 0,
            'valid': 0,
            'good': 0,
            'errors': 0,
        }

        min_sharpe = self.config['backtest'].get('min_sharpe', 0.5)
        allowed_freqs = set(self.config.get('frequencies', []))

        console.print(f'\n[bold]Starting alpha discovery pipeline ({n_iterations} iterations)[/bold]\n')

        for i in range(n_iterations):
            console.print(f'[cyan]--- Iteration {i+1}/{n_iterations} ---[/cyan]')

            try:
                batch = self.generator.generate_diverse_batch(
                    n_categories=4,
                    n_per_category=2,
                )
            except Exception as e:
                logger.error(f'Generation failed: {e}')
                stats['errors'] += 1
                continue

            stats['strategies'] += len(batch)

            for strat in batch:
                idea = strat['strategy_idea']
                category = strat.get('category')
                console.print(f'  Strategy: {idea[:80]}...')

                sid = self.db.save_strategy(
                    idea, category,
                    self.llm_cfg['provider'], self.llm_cfg['model']
                )

                for alpha_item in strat['alphas']:
                    expr = alpha_item['alpha']
                    freq = alpha_item['frequency']
                    stats['alphas'] += 1

                    # Skip frequencies not in config (memory safety)
                    if allowed_freqs and freq not in allowed_freqs:
                        logger.debug(f'Skipping freq {freq} not in config')
                        continue

                    # Validate
                    try:
                        self.validator.validate(expr)
                    except ValidationError as e:
                        logger.debug(f'Invalid: {expr[:60]}... - {e}')
                        continue

                    stats['valid'] += 1

                    # Save (dedup via hash)
                    aid = self.db.save_alpha(sid, expr, freq)
                    if aid is None:
                        continue  # duplicate

                    # Backtest
                    result = self.backtester.run(expr, freq)

                    if result.error:
                        logger.debug(f'Backtest error: {result.error}')
                        self.db.mark_alpha_invalid(aid)
                        continue

                    self.db.save_backtest_result(aid, result.metrics)

                    sharpe = result.metrics.get('sharpe_ratio', 0)
                    if sharpe >= min_sharpe:
                        stats['good'] += 1
                        console.print(
                            f'    [green]GOOD[/green] Sharpe={sharpe:.2f} '
                            f'Freq={freq} Expr={expr[:60]}...'
                        )
                    else:
                        console.print(
                            f'    [dim]Sharpe={sharpe:.2f} Freq={freq}[/dim]'
                        )

            # Progress summary
            console.print(
                f'  [yellow]Progress: {stats["strategies"]} strategies, '
                f'{stats["alphas"]} alphas, {stats["valid"]} valid, '
                f'{stats["good"]} good[/yellow]\n'
            )

        # Save run
        self.db.save_pipeline_run(stats)

        # Final report
        self._print_report(stats)

        return stats

    def _print_report(self, stats: dict):
        """Print a summary report."""
        console.print('\n[bold]Pipeline Complete[/bold]\n')

        table = Table(title='Run Summary')
        table.add_column('Metric', style='cyan')
        table.add_column('Value', style='green')
        table.add_row('Strategies Generated', str(stats['strategies']))
        table.add_row('Alphas Generated', str(stats['alphas']))
        table.add_row('Valid Alphas', str(stats['valid']))
        table.add_row('Above Sharpe Threshold', str(stats['good']))
        table.add_row('Errors', str(stats['errors']))
        console.print(table)

        # Show top alphas
        top = self.db.get_top_alphas(limit=10)
        if top:
            console.print('\n[bold]Top 10 Alphas[/bold]')
            top_table = Table()
            top_table.add_column('Sharpe', style='green')
            top_table.add_column('Freq')
            top_table.add_column('Return')
            top_table.add_column('MaxDD')
            top_table.add_column('Expression')
            for a in top:
                top_table.add_row(
                    f'{a["sharpe_ratio"]:.2f}',
                    a['frequency'],
                    f'{a["annualized_return"]:.2%}' if a['annualized_return'] else 'N/A',
                    f'{a["max_drawdown"]:.2%}' if a['max_drawdown'] else 'N/A',
                    a['expression'][:70] + '...' if len(a['expression']) > 70 else a['expression'],
                )
            console.print(top_table)
