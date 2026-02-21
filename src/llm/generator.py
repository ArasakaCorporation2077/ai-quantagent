"""Two-stage alpha generation using LLMs."""

import logging

from src.llm.client import LLMClient
from src.llm.prompts import (
    STAGE1_SYSTEM, STAGE1_USER, STAGE1_USER_WITH_CATEGORY,
    STAGE2_SYSTEM, STAGE2_USER,
    DIVERSITY_SYSTEM, DIVERSITY_USER,
)
from src.llm.response_parser import (
    extract_json_array, validate_alpha_response,
    parse_categories, ResponseParseError,
)

logger = logging.getLogger(__name__)


class AlphaGenerator:
    def __init__(self, llm_client: LLMClient, config: dict):
        self.client = llm_client
        self.llm_cfg = config['llm']
        self.temp_stage1 = self.llm_cfg.get('temperature_stage1', 1.0)
        self.temp_stage2 = self.llm_cfg.get('temperature_stage2', 0.3)
        self.max_retries = self.llm_cfg.get('max_retries', 3)

    def generate_strategy_idea(self, category: str | None = None) -> str:
        """Stage 1: Generate a trading strategy idea in 3 sentences."""
        if category:
            user_prompt = STAGE1_USER_WITH_CATEGORY.format(category=category)
        else:
            user_prompt = STAGE1_USER

        response = self.client.generate(
            system_prompt=STAGE1_SYSTEM,
            user_prompt=user_prompt,
            temperature=self.temp_stage1,
            max_tokens=500,
        )
        return response.strip()

    def generate_alpha_expressions(self, strategy_idea: str) -> list[dict]:
        """Stage 2: Generate alpha expressions for a strategy idea.

        Returns list of {"frequency": "...", "alpha": "..."} dicts.
        """
        user_prompt = STAGE2_USER.format(strategy_idea=strategy_idea)

        for attempt in range(self.max_retries):
            try:
                response = self.client.generate(
                    system_prompt=STAGE2_SYSTEM,
                    user_prompt=user_prompt,
                    temperature=self.temp_stage2,
                    max_tokens=2000,
                )

                parsed = extract_json_array(response)
                valid = validate_alpha_response(parsed)

                if valid:
                    return valid

                logger.warning(f'No valid alphas in response (attempt {attempt+1})')

            except ResponseParseError as e:
                logger.warning(f'Parse error (attempt {attempt+1}): {e}')
            except Exception as e:
                logger.warning(f'LLM error (attempt {attempt+1}): {e}')

        return []

    # Predefined categories (avoids unreliable LLM parsing for simple list)
    STRATEGY_CATEGORIES = [
        'Momentum',
        'Mean-Reversion',
        'Volume / Order-Flow',
        'Volatility',
        'Trend-Following',
        'Market Microstructure',
        'Cross-Asset Correlation',
        'Seasonality / Cyclical',
    ]

    def generate_categories(self, n_categories: int = 8) -> list[str]:
        """Return diverse strategy categories (predefined for reliability)."""
        import random
        cats = random.sample(self.STRATEGY_CATEGORIES, min(n_categories, len(self.STRATEGY_CATEGORIES)))
        logger.info(f'Using {len(cats)} categories: {cats}')
        return cats

    def generate_batch(self, n_strategies: int = 10) -> list[dict]:
        """Generate a batch of strategies with alpha expressions (no diversity)."""
        results = []
        for i in range(n_strategies):
            logger.info(f'Generating strategy {i+1}/{n_strategies}')
            idea = self.generate_strategy_idea()
            alphas = self.generate_alpha_expressions(idea)
            if alphas:
                results.append({
                    'strategy_idea': idea,
                    'alphas': alphas,
                })
            else:
                logger.warning(f'Strategy {i+1} produced no valid alphas')
        return results

    def generate_diverse_batch(
        self, n_categories: int = 8, n_per_category: int = 2
    ) -> list[dict]:
        """Generate strategies with category-based diversity."""
        categories = self.generate_categories(n_categories)
        results = []

        for cat in categories:
            for j in range(n_per_category):
                logger.info(f'Category: {cat} ({j+1}/{n_per_category})')
                try:
                    idea = self.generate_strategy_idea(category=cat)
                    alphas = self.generate_alpha_expressions(idea)
                    if alphas:
                        results.append({
                            'strategy_idea': idea,
                            'category': cat,
                            'alphas': alphas,
                        })
                except Exception as e:
                    logger.error(f'Error generating for {cat}: {e}')

        logger.info(f'Generated {len(results)} strategies across {len(categories)} categories')
        return results
