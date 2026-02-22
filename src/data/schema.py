"""Column definitions and constants for the data pipeline."""

# Raw kline CSV columns from data.binance.vision
KLINE_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

KLINE_DTYPES = {
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    'quote_asset_volume': 'float64',
    'number_of_trades': 'int64',
    'taker_buy_base_asset_volume': 'float64',
    'taker_buy_quote_asset_volume': 'float64',
}

# Data types available to the alpha expression engine
ALPHA_DATA_TYPES = [
    'open',
    'high',
    'low',
    'close',
    'vwap',
    'buy_volume',
    'sell_volume',
    'total_volume',
    'buy_trades_count',
    'sell_trades_count',
    'total_trades_count',
]

# Final processed parquet columns
PROCESSED_COLUMNS = [
    'open', 'high', 'low', 'close',
    'vwap',
    'buy_volume', 'sell_volume', 'total_volume',
    'buy_trades_count', 'sell_trades_count', 'total_trades_count',
]

# Valid frequencies
VALID_FREQUENCIES = ['1m', '5m', '15m', '1h', '4h', '12h', '1d']

# Annualization factors (bars per year, approximate)
ANNUALIZATION_FACTORS = {
    '1m': 525_600,
    '5m': 105_120,
    '15m': 35_040,
    '1h': 8_760,
    '4h': 2_190,
    '12h': 730,
    '1d': 365,
}
