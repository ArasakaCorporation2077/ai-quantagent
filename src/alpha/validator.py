"""Validate alpha expressions before evaluation."""

from src.alpha.parser import AlphaParser, FunctionNode, DataRefNode, NumberNode, Node, ParseError
from src.alpha.transforms import TRANSFORM_REGISTRY
from src.data.schema import ALPHA_DATA_TYPES


class ValidationError(Exception):
    pass


class AlphaValidator:
    """Validates alpha expressions for correctness before evaluation."""

    def validate(self, expression: str) -> dict:
        """Validate an expression string. Returns info dict or raises ValidationError."""
        try:
            parser = AlphaParser()
            ast = parser.parse(expression)
        except ParseError as e:
            raise ValidationError(f'Parse error: {e}')

        info = {
            'max_lookback': 0,
            'data_refs': set(),
            'functions_used': set(),
            'depth': 0,
        }
        self._validate_node(ast, info, depth=0)
        info['data_refs'] = list(info['data_refs'])
        info['functions_used'] = list(info['functions_used'])
        return info

    def _validate_node(self, node: Node, info: dict, depth: int):
        info['depth'] = max(info['depth'], depth)

        if isinstance(node, NumberNode):
            return

        if isinstance(node, DataRefNode):
            if node.name not in ALPHA_DATA_TYPES:
                raise ValidationError(
                    f'Unknown data reference: {node.name}. '
                    f'Valid types: {ALPHA_DATA_TYPES}'
                )
            info['data_refs'].add(node.name)
            return

        if isinstance(node, FunctionNode):
            if node.name not in TRANSFORM_REGISTRY:
                raise ValidationError(
                    f'Unknown function: {node.name}. '
                    f'Valid functions: {list(TRANSFORM_REGISTRY.keys())}'
                )

            info['functions_used'].add(node.name)

            _, arg_types, _ = TRANSFORM_REGISTRY[node.name]

            # Flexible arg count for scale(x) or scale(x, a), ts_sma(x,n) or ts_sma(x,n,m)
            min_args = len(arg_types)
            max_args = len(arg_types)
            if node.name == 'scale':
                min_args = 1
            elif node.name == 'ts_sma':
                min_args = 2

            if not (min_args <= len(node.args) <= max_args):
                raise ValidationError(
                    f'{node.name} expects {min_args}-{max_args} args, got {len(node.args)}'
                )

            # Estimate lookback from number args in time-series functions
            _, _, category = TRANSFORM_REGISTRY[node.name]
            if category == 'timeseries':
                for i, (child, expected) in enumerate(zip(node.args, arg_types)):
                    if expected == 'number' and isinstance(child, NumberNode):
                        info['max_lookback'] += int(child.value)

            # Recurse into children
            for child in node.args:
                self._validate_node(child, info, depth + 1)

            return

        raise ValidationError(f'Unknown node type: {type(node)}')
