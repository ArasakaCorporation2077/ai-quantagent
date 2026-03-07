"""All 42 transform function implementations for alpha expressions."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Arithmetic / Element-wise
# ---------------------------------------------------------------------------

def add(x1: pd.Series, x2: pd.Series) -> pd.Series:
    return x1 + x2


def sub(x1: pd.Series, x2: pd.Series) -> pd.Series:
    return x1 - x2


def mul(x1: pd.Series, x2: pd.Series) -> pd.Series:
    return x1 * x2


def div(x1: pd.Series, x2: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = x1 / x2
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], np.nan)
        return result


def sqrt_(x: pd.Series) -> pd.Series:
    """Protected sqrt: sign(x) * sqrt(|x|) to preserve sign information."""
    with np.errstate(invalid='ignore'):
        return np.sign(x) * np.sqrt(np.abs(x))


def log_(x: pd.Series) -> pd.Series:
    """Protected log: sign(x) * log(|x|) to preserve sign information."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sign(x) * np.log(np.abs(x).clip(lower=1e-10))


def abs_(x: pd.Series) -> pd.Series:
    return x.abs()


def neg(x: pd.Series) -> pd.Series:
    return -x


def inv(x: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 1.0 / x
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], np.nan)
        return result


def sign_(x: pd.Series) -> pd.Series:
    return np.sign(x)


def power_(x1: pd.Series, x2) -> pd.Series:
    if isinstance(x2, pd.Series):
        return x1.abs().clip(lower=1e-10) ** x2 * np.sign(x1)
    return x1.abs().clip(lower=1e-10) ** float(x2) * np.sign(x1)


def signedpower(x: pd.Series, a=2) -> pd.Series:
    a = float(a)
    return np.sign(x) * (x.abs() ** a)


def max_(x1: pd.Series, x2: pd.Series) -> pd.Series:
    return np.maximum(x1, x2)


def min_(x1: pd.Series, x2: pd.Series) -> pd.Series:
    return np.minimum(x1, x2)


# ---------------------------------------------------------------------------
# Cross-sectional (applied across symbols at each timestamp)
# These are marked in the registry; evaluator handles groupby logic.
# When evaluated per-symbol, rank → ts_rank-like behavior, scale → identity.
# ---------------------------------------------------------------------------

def rank_(x: pd.Series) -> pd.Series:
    return x.rank(pct=True)


def scale_(x: pd.Series, a=1) -> pd.Series:
    a = float(a)
    denom = x.abs().sum()
    if denom == 0:
        return x * 0
    return a * x / denom


# ---------------------------------------------------------------------------
# Time-series (rolling window operations)
# ---------------------------------------------------------------------------

def delay(x: pd.Series, d) -> pd.Series:
    d = max(1, int(d))
    return x.shift(d)


def delta(x: pd.Series, d) -> pd.Series:
    d = max(1, int(d))
    return x - x.shift(d)


def ts_sum(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    return x.rolling(d, min_periods=1).sum()


def ts_mean(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    return x.rolling(d, min_periods=1).mean()


def ts_min(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    return x.rolling(d, min_periods=1).min()


def ts_max(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    return x.rolling(d, min_periods=1).max()


def ts_stddev(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    return x.rolling(d, min_periods=2).std()


def ts_product(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    return x.rolling(d, min_periods=1).apply(np.prod, raw=True)


def ts_rank(x: pd.Series, d) -> pd.Series:
    """Rank of last value within window (blog-matching implementation)."""
    d = max(2, int(d))

    def _rank_last(arr):
        if len(arr) < 1:
            return np.nan
        return (arr < arr[-1]).sum() / len(arr) + (arr == arr[-1]).sum() / (2 * len(arr))

    return x.rolling(d, min_periods=1).apply(_rank_last, raw=True)


def ts_argmin(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    return x.rolling(d, min_periods=1).apply(lambda arr: np.argmin(arr), raw=True)


def ts_argmax(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    return x.rolling(d, min_periods=1).apply(lambda arr: np.argmax(arr), raw=True)


def ts_zscore(x: pd.Series, d) -> pd.Series:
    d = max(2, int(d))
    m = x.rolling(d, min_periods=2).mean()
    s = x.rolling(d, min_periods=2).std()
    return (x - m) / s.replace(0, np.nan)


def ts_sma(x: pd.Series, n, m=None) -> pd.Series:
    """Exponential moving average with alpha = m/n. If m is not given, plain SMA."""
    n = max(1, int(n))
    if m is None:
        return ts_mean(x, n)
    m = int(m)
    if m <= 0 or m / n > 1:
        m = 1
    return x.ewm(alpha=m / n).mean()


def ts_wma(x: pd.Series, d) -> pd.Series:
    """Weighted moving average with linearly increasing weights (blog formula)."""
    d = max(2, int(d))
    weights = 2 * np.arange(1, d + 1, dtype=float) / (d * (d + 1))

    def _wma(arr):
        if len(arr) < 1:
            return np.nan
        w = weights[-len(arr):]
        w = w / w.sum()
        return np.dot(arr, w)

    return x.rolling(d, min_periods=1).apply(_wma, raw=True)


def decay_linear(x: pd.Series, d) -> pd.Series:
    """Linear weighted average (alias for ts_wma)."""
    return ts_wma(x, d)


def ts_highday(x: pd.Series, d) -> pd.Series:
    """Days since highest value in window."""
    d = max(2, int(d))

    def _days_since_high(arr):
        if len(arr) < 1:
            return np.nan
        return len(arr) - 1 - np.argmax(arr)

    return x.rolling(d, min_periods=1).apply(_days_since_high, raw=True)


def ts_lowday(x: pd.Series, d) -> pd.Series:
    """Days since lowest value in window."""
    d = max(2, int(d))

    def _days_since_low(arr):
        if len(arr) < 1:
            return np.nan
        return len(arr) - 1 - np.argmin(arr)

    return x.rolling(d, min_periods=1).apply(_days_since_low, raw=True)


# ---------------------------------------------------------------------------
# Multi-input time-series
# ---------------------------------------------------------------------------

def correlation_(x: pd.Series, y: pd.Series, d) -> pd.Series:
    return x.rolling(int(d), min_periods=2).corr(y)


def covariance_(x: pd.Series, y: pd.Series, d) -> pd.Series:
    return x.rolling(int(d), min_periods=2).cov(y)


# ---------------------------------------------------------------------------
# Conditional operations
# ---------------------------------------------------------------------------

def ifcondition_g(cond1: pd.Series, cond2: pd.Series, x1: pd.Series, x2: pd.Series) -> pd.Series:
    return pd.Series(np.where(cond1 > cond2, x1, x2), index=cond1.index)


def ifcondition_e(cond1: pd.Series, cond2: pd.Series, x1: pd.Series, x2: pd.Series) -> pd.Series:
    return pd.Series(np.where(cond1 == cond2, x1, x2), index=cond1.index)


def ifcondition_ge(cond1: pd.Series, cond2: pd.Series, x1: pd.Series, x2: pd.Series) -> pd.Series:
    return pd.Series(np.where(cond1 >= cond2, x1, x2), index=cond1.index)


def ts_sumif(x: pd.Series, cond1: pd.Series, cond2: pd.Series, d) -> pd.Series:
    d = int(d)
    masked = x.where(cond1 > cond2, 0)
    return masked.rolling(d, min_periods=1).sum()


def ts_count(cond1: pd.Series, cond2: pd.Series, d) -> pd.Series:
    d = int(d)
    return (cond1 > cond2).astype(float).rolling(d, min_periods=1).sum()


# ---------------------------------------------------------------------------
# Transform registry: name -> (function, arg_types, category)
#   arg_types: 'series' = pd.Series, 'number' = int/float constant
#   category: 'elementwise', 'timeseries', 'crosssectional'
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY = {
    # Arithmetic
    'add':           (add,            ['series', 'series'],                        'elementwise'),
    'sub':           (sub,            ['series', 'series'],                        'elementwise'),
    'mul':           (mul,            ['series', 'series'],                        'elementwise'),
    'div':           (div,            ['series', 'series'],                        'elementwise'),
    'sqrt':          (sqrt_,          ['series'],                                  'elementwise'),
    'log':           (log_,           ['series'],                                  'elementwise'),
    'abs':           (abs_,           ['series'],                                  'elementwise'),
    'neg':           (neg,            ['series'],                                  'elementwise'),
    'inv':           (inv,            ['series'],                                  'elementwise'),
    'sign':          (sign_,          ['series'],                                  'elementwise'),
    'power':         (power_,         ['series', 'number'],                        'elementwise'),
    'signedpower':   (signedpower,    ['series', 'number'],                        'elementwise'),
    'max':           (max_,           ['series', 'series'],                        'elementwise'),
    'min':           (min_,           ['series', 'series'],                        'elementwise'),
    # Cross-sectional
    'rank':          (rank_,          ['series'],                                  'crosssectional'),
    'scale':         (scale_,         ['series', 'float_opt'],                           'crosssectional'),
    # Time-series
    'delay':         (delay,          ['series', 'number'],                        'timeseries'),
    'delta':         (delta,          ['series', 'number'],                        'timeseries'),
    'ts_sum':        (ts_sum,         ['series', 'number'],                        'timeseries'),
    'ts_mean':       (ts_mean,        ['series', 'number'],                        'timeseries'),
    'ts_min':        (ts_min,         ['series', 'number'],                        'timeseries'),
    'ts_max':        (ts_max,         ['series', 'number'],                        'timeseries'),
    'ts_stddev':     (ts_stddev,      ['series', 'number'],                        'timeseries'),
    'ts_product':    (ts_product,     ['series', 'number'],                        'timeseries'),
    'ts_rank':       (ts_rank,        ['series', 'number'],                        'timeseries'),
    'ts_argmin':     (ts_argmin,      ['series', 'number'],                        'timeseries'),
    'ts_argmax':     (ts_argmax,      ['series', 'number'],                        'timeseries'),
    'ts_zscore':     (ts_zscore,      ['series', 'number'],                        'timeseries'),
    'ts_sma':        (ts_sma,         ['series', 'number', 'number'],              'timeseries'),
    'ts_wma':        (ts_wma,         ['series', 'number'],                        'timeseries'),
    'decay_linear':  (decay_linear,   ['series', 'number'],                        'timeseries'),
    'ts_highday':    (ts_highday,     ['series', 'number'],                        'timeseries'),
    'ts_lowday':     (ts_lowday,      ['series', 'number'],                        'timeseries'),
    'correlation':   (correlation_,   ['series', 'series', 'number'],              'timeseries'),
    'covariance':    (covariance_,    ['series', 'series', 'number'],              'timeseries'),
    # Conditional
    'ifcondition_g': (ifcondition_g,  ['series', 'series', 'series', 'series'],    'elementwise'),
    'ifcondition_e': (ifcondition_e,  ['series', 'series', 'series', 'series'],    'elementwise'),
    'ifcondition_ge':(ifcondition_ge, ['series', 'series', 'series', 'series'],    'elementwise'),
    'ts_sumif':      (ts_sumif,       ['series', 'series', 'series', 'number'],    'timeseries'),
    'ts_count':      (ts_count,       ['series', 'series', 'number'],              'timeseries'),
}
