"""Prompt templates for LLM-based alpha generation."""

from src.data.schema import VALID_FREQUENCIES

_FREQUENCIES_STR = ', '.join(VALID_FREQUENCIES)

# Blog-matching data_types and transforms with comments
_DATA_TYPES_BLOCK = """data_types = [
    'open', # The opening price of the period
    'high', # The highest price of the period
    'low', # The lowest price of the period
    'close', # The closing price of the period
    'vwap', # The volume-weighted average price of the period
    'buy_volume', # The sum of the buy volumes of the period
    'sell_volume', # The sum of the sell volumes of the period
    'total_volume', # The sum of the buy and sell volumes of the period
    'buy_trades_count', # The number of buy trades in the period
    'sell_trades_count', # The number of sell trades in the period
    'total_trades_count', # The number of buy and sell trades in the period
]"""

_TRANSFORMS_BLOCK = """transforms_available = [
    'add',  # 2 inputs: (x1, x2), returns x1 + x2
    'sub',  # 2 inputs: (x1, x2), returns x1 - x2
    'mul',  # 2 inputs: (x1, x2), returns x1 * x2
    'div',  # 2 inputs: (x1, x2), returns x1 / x2 (protected division)
    'sqrt',  # 1 input: (x1), returns sqrt(x1) (protected for negative values)
    'log',  # 1 input: (x1), returns log(x1) (protected for zero/negative values)
    'abs',  # 1 input: (x1), returns absolute value of x1
    'neg',  # 1 input: (x1), returns -x1
    'inv',  # 1 input: (x1), returns 1/x1 (protected inverse)
    'max',  # 2 inputs: (x1, x2), returns element-wise maximum
    'min',  # 2 inputs: (x1, x2), returns element-wise minimum
    'rank',  # 1 input: (x1), returns percentile rank transformation
    'scale',  # 2 inputs: (x1, a), returns x1 scaled by a/sum(abs(x1))
    'signedpower',  # 2 inputs: (x1, a), returns sign(x1) * abs(x1)^a (default a=2)
    'delay',  # 2 inputs: (x1, d), returns x1 shifted by d periods
    'correlation',  # 3 inputs: (x1, x2, d), returns rolling correlation over d periods
    'covariance',  # 3 inputs: (x1, x2, d), returns rolling covariance over d periods
    'delta',  # 2 inputs: (x1, d), returns x1 - x1_shifted_by_d
    'decay_linear',  # 2 inputs: (x1, d), returns linear weighted average over d periods
    'ts_min',  # 2 inputs: (x1, d), returns rolling minimum over d periods
    'ts_max',  # 2 inputs: (x1, d), returns rolling maximum over d periods
    'ts_argmin',  # 2 inputs: (x1, d), returns index of minimum in rolling window of d periods
    'ts_argmax',  # 2 inputs: (x1, d), returns index of maximum in rolling window of d periods
    'ts_rank',  # 2 inputs: (x1, d), returns rank within rolling window of d periods
    'ts_sum',  # 2 inputs: (x1, d), returns rolling sum over d periods
    'ts_mean',  # 2 inputs: (x1, d), returns rolling mean over d periods
    'ts_product',  # 2 inputs: (x1, d), returns rolling product over d periods
    'ts_stddev',  # 2 inputs: (x1, d), returns rolling standard deviation over d periods
    'ts_zscore',  # 2 inputs: (x1, d), returns rolling z-score (x1 - rolling_mean) / rolling_std over d periods
    'ts_sma',  # 3 inputs: (x1, n, m), returns exponential moving average with alpha=m/n
    'ts_wma',  # 2 inputs: (x1, d), returns weighted moving average over d periods
    'sign',  # 1 input: (x1), returns sign of x1 (-1, 0, or 1)
    'power',  # 2 inputs: (x1, x2), returns x1^x2
    'ifcondition_g',  # 4 inputs: (cond1, cond2, x1, x2), returns x1 if cond1 > cond2 else x2
    'ifcondition_e',  # 4 inputs: (cond1, cond2, x1, x2), returns x1 if cond1 == cond2 else x2
    'ifcondition_ge',  # 4 inputs: (cond1, cond2, x1, x2), returns x1 if cond1 >= cond2 else x2
    'ts_sumif',  # 4 inputs: (x1, cond1, cond2, d), returns rolling sum of x1 where cond1 > cond2
    'ts_count',  # 3 inputs: (cond1, cond2, d), returns rolling count where cond1 > cond2
    'ts_highday',  # 2 inputs: (x1, d), returns days since highest value in window of d periods
    'ts_lowday',  # 2 inputs: (x1, d), returns days since lowest value in window of d periods
]"""


# ---------------------------------------------------------------------------
# Stage 1: Strategy Idea Generation
# ---------------------------------------------------------------------------

STAGE1_SYSTEM = """You are a quantitative researcher specializing in crypto futures markets.
You come up with novel, diverse trading strategy ideas rooted in fundamental market theories.
Your strategies must be implementable using mathematical transforms on OHLCV and volume data."""

STAGE1_USER = f"""Come up with a novel trading strategy that can be represented as a feature for an asset.
Do not respond with the feature, simply the trading strategy in only 3 sentences.
It should be rooted in a fundamental theory of how you view the markets and must be suitable for crypto markets.

It must only use these input datas:

{_DATA_TYPES_BLOCK}

and it should be able to be constructed with these transforms:

{_TRANSFORMS_BLOCK}"""

STAGE1_USER_WITH_CATEGORY = STAGE1_USER + """

IMPORTANT: Your strategy must be in the category: {category}
Make sure it is fundamentally different from typical {category} approaches.
Be creative and think from first principles."""


# ---------------------------------------------------------------------------
# Stage 2: Alpha Expression Generation
# ---------------------------------------------------------------------------

STAGE2_SYSTEM = f"""You are a quantitative researcher designing alpha features for crypto futures.

Given a trading strategy idea, construct alpha expressions using the provided transforms and data types.

{_DATA_TYPES_BLOCK}

{_TRANSFORMS_BLOCK}

Available frequencies: {_FREQUENCIES_STR}

RULES:
1. Output ONLY a JSON array of objects, each with "frequency" and "alpha" keys.
2. This is simply an example of how one would be formatted, you should use the input data, parameters, and transform you feel best suit the strategy:
   [{{"frequency": "1h", "alpha": "scale(ts_zscore(div(sub(buy_volume,sell_volume),total_volume),20))"}}]
3. Prefer simplicity wherever possible when designing your features so that we avoid overfitting.
4. DO NOT use any data types not listed above. We do NOT have bid/ask spread data (no open_bid_price, close_ask_price, open_ask_price, close_bid_price, etc.).
5. OPEN, HIGH, LOW, CLOSE are mid-prices. Do not try to calculate midprice from bid/ask.
6. Every alpha expression must be syntactically valid with matching parentheses and valid transform/data names.
7. Generate one alpha per frequency. Not all frequencies need an alpha - pick the ones that best suit the strategy.
8. Use window parameters appropriate to the frequency (larger windows for higher frequencies).
9. Numbers in expressions must be integers (e.g., 20, not 20.0).
10. Return ONLY the JSON array, no commentary or explanation."""

STAGE2_USER = """Strategy idea: {strategy_idea}

Generate alpha expressions for this strategy across multiple frequencies.
Return ONLY valid JSON array."""


# ---------------------------------------------------------------------------
# Diversity: Category Generation
# ---------------------------------------------------------------------------

DIVERSITY_SYSTEM = """You are a quantitative researcher. Generate distinct categories of trading strategies."""

DIVERSITY_USER = """List {n_categories} distinct categories of quantitative trading strategies suitable for crypto futures.
For each category, respond with ONLY the category name on a separate line.
Focus on fundamentally different market theories.
Examples: momentum, mean-reversion, volume-imbalance, volatility, microstructure"""
