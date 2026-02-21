"""Evaluate parsed alpha expression ASTs against market data."""

import logging

import numpy as np
import pandas as pd

from src.alpha.parser import AlphaParser, FunctionNode, DataRefNode, NumberNode, Node, ParseError
from src.alpha.transforms import TRANSFORM_REGISTRY
from src.data.schema import ALPHA_DATA_TYPES

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    pass


class AlphaEvaluator:
    """Evaluates alpha expressions against a panel of market data.

    Data is stored as a dict of {symbol: DataFrame} where each DataFrame
    has ALPHA_DATA_TYPES as columns and a datetime index.
    """

    def __init__(self, data: dict[str, pd.DataFrame]):
        self.symbol_data = data
        self.symbols = sorted(data.keys())
        self._panel = None  # lazy-built MultiIndex DataFrame

    @property
    def panel(self) -> pd.DataFrame:
        """Build MultiIndex (symbol, timestamp) panel on first access."""
        if self._panel is None:
            frames = []
            for sym in self.symbols:
                df = self.symbol_data[sym].copy()
                df['symbol'] = sym
                df.index.name = 'timestamp'
                frames.append(df)
            combined = pd.concat(frames)
            combined = combined.set_index('symbol', append=True)
            combined = combined.reorder_levels(['symbol', 'timestamp']).sort_index()
            self._panel = combined
        return self._panel

    def evaluate(self, expression: str) -> pd.DataFrame:
        """Parse and evaluate an alpha expression.

        Returns a DataFrame with columns=symbols, index=timestamps,
        containing the alpha signal values.
        """
        parser = AlphaParser()
        ast = parser.parse(expression)
        result = self._eval_node(ast)

        # Convert to pivoted form: timestamps x symbols
        if isinstance(result, pd.Series):
            result = result.unstack(level='symbol')
        elif isinstance(result, (int, float)):
            # Constant expression
            idx = self.panel.index
            result = pd.Series(float(result), index=idx).unstack(level='symbol')

        return result

    def evaluate_single(self, expression: str, symbol: str) -> pd.Series:
        """Evaluate an alpha expression for a single symbol."""
        parser = AlphaParser()
        ast = parser.parse(expression)
        df = self.symbol_data[symbol]
        return self._eval_node_single(ast, df)

    def _eval_node(self, node: Node) -> pd.Series | float:
        """Evaluate an AST node against the full panel."""
        if isinstance(node, NumberNode):
            return node.value

        if isinstance(node, DataRefNode):
            if node.name not in ALPHA_DATA_TYPES:
                raise EvaluationError(f'Unknown data reference: {node.name}')
            return self.panel[node.name]

        if isinstance(node, FunctionNode):
            if node.name not in TRANSFORM_REGISTRY:
                raise EvaluationError(f'Unknown function: {node.name}')

            func, arg_types, category = TRANSFORM_REGISTRY[node.name]

            # Validate argument count
            if len(node.args) != len(arg_types):
                # Allow scale with optional second arg
                if node.name == 'scale' and len(node.args) in (1, 2):
                    pass
                elif node.name == 'ts_sma' and len(node.args) in (2, 3):
                    pass
                else:
                    raise EvaluationError(
                        f'{node.name} expects {len(arg_types)} args, got {len(node.args)}'
                    )

            # Evaluate child nodes
            raw_args = [self._eval_node(child) for child in node.args]

            # Convert number args
            args = []
            for i, (val, expected) in enumerate(zip(raw_args, arg_types)):
                if expected == 'number':
                    if isinstance(val, pd.Series):
                        raise EvaluationError(
                            f'{node.name} arg {i} should be a number, got series'
                        )
                    args.append(float(val) if not isinstance(val, (int, float)) else val)
                else:
                    if isinstance(val, (int, float)):
                        # Broadcast scalar to series
                        args.append(pd.Series(val, index=self.panel.index))
                    else:
                        args.append(val)

            if category == 'crosssectional':
                return self._apply_crosssectional(func, args)
            elif category == 'timeseries':
                return self._apply_timeseries(func, args)
            else:
                return func(*args)

        raise EvaluationError(f'Unknown node type: {type(node)}')

    def _apply_timeseries(self, func, args) -> pd.Series:
        """Apply a time-series function per-symbol to avoid cross-symbol contamination."""
        series_args_idx = [i for i, a in enumerate(args) if isinstance(a, pd.Series)]
        scalar_args = {i: a for i, a in enumerate(args) if not isinstance(a, pd.Series)}

        results = []
        for sym in self.symbols:
            sym_args = []
            for i in range(len(args)):
                if i in scalar_args:
                    sym_args.append(scalar_args[i])
                else:
                    # Extract this symbol's data
                    try:
                        sym_args.append(args[i].loc[sym])
                    except KeyError:
                        sym_args.append(args[i].xs(sym, level='symbol'))

            result = func(*sym_args)
            result = pd.DataFrame({'value': result})
            result['symbol'] = sym
            result.index.name = 'timestamp'
            results.append(result)

        combined = pd.concat(results)
        combined = combined.set_index('symbol', append=True)
        combined = combined.reorder_levels(['symbol', 'timestamp']).sort_index()
        return combined['value']

    def _apply_crosssectional(self, func, args) -> pd.Series:
        """Apply a cross-sectional function across symbols at each timestamp."""
        # Unstack to (timestamp x symbol), apply, restack
        series_arg = args[0]
        extra_args = args[1:]

        unstacked = series_arg.unstack(level='symbol')

        # Apply function row-wise (across symbols)
        applied = unstacked.apply(lambda row: func(row, *extra_args), axis=1)

        # Restack back to MultiIndex
        return applied.stack().reorder_levels(['symbol', 'timestamp']).sort_index()

    def _eval_node_single(self, node: Node, df: pd.DataFrame) -> pd.Series | float:
        """Evaluate an AST node against a single symbol's DataFrame."""
        if isinstance(node, NumberNode):
            return node.value

        if isinstance(node, DataRefNode):
            if node.name not in df.columns:
                raise EvaluationError(f'Column not found: {node.name}')
            return df[node.name]

        if isinstance(node, FunctionNode):
            if node.name not in TRANSFORM_REGISTRY:
                raise EvaluationError(f'Unknown function: {node.name}')

            func, arg_types, category = TRANSFORM_REGISTRY[node.name]

            raw_args = [self._eval_node_single(child, df) for child in node.args]

            args = []
            for i, (val, expected) in enumerate(zip(raw_args, arg_types)):
                if expected == 'number':
                    args.append(float(val) if not isinstance(val, (int, float)) else val)
                else:
                    if isinstance(val, (int, float)):
                        args.append(pd.Series(val, index=df.index))
                    else:
                        args.append(val)

            return func(*args)

        raise EvaluationError(f'Unknown node type: {type(node)}')
