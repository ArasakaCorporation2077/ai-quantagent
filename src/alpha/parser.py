"""Recursive descent parser for alpha expressions.

Parses expressions like:
    scale(add(ts_zscore(div(sub(buy_volume,sell_volume),total_volume),20),delta(close,1)))

Into an AST of FunctionNode, DataRefNode, and NumberNode objects.
"""

from dataclasses import dataclass
from enum import Enum, auto

from src.data.schema import ALPHA_DATA_TYPES
from src.alpha.transforms import TRANSFORM_REGISTRY


# ---------------------------------------------------------------------------
# AST Node types
# ---------------------------------------------------------------------------

@dataclass
class FunctionNode:
    name: str
    args: list  # list[Node]


@dataclass
class DataRefNode:
    name: str


@dataclass
class NumberNode:
    value: float


Node = FunctionNode | DataRefNode | NumberNode


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TokenType(Enum):
    IDENT = auto()
    NUMBER = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int


class ParseError(Exception):
    pass


def tokenize(expr: str) -> list[Token]:
    """Split an alpha expression string into tokens."""
    tokens = []
    i = 0
    n = len(expr)

    while i < n:
        c = expr[i]

        # Skip whitespace
        if c in ' \t\n\r':
            i += 1
            continue

        if c == '(':
            tokens.append(Token(TokenType.LPAREN, '(', i))
            i += 1
        elif c == ')':
            tokens.append(Token(TokenType.RPAREN, ')', i))
            i += 1
        elif c == ',':
            tokens.append(Token(TokenType.COMMA, ',', i))
            i += 1
        elif c.isdigit() or (c == '-' and i + 1 < n and expr[i + 1].isdigit()
                              and (not tokens or tokens[-1].type in (TokenType.LPAREN, TokenType.COMMA))):
            # Number (integer or float, possibly negative)
            start = i
            if c == '-':
                i += 1
            while i < n and (expr[i].isdigit() or expr[i] == '.'):
                i += 1
            tokens.append(Token(TokenType.NUMBER, expr[start:i], start))
        elif c.isalpha() or c == '_':
            # Identifier (function name or data reference)
            start = i
            while i < n and (expr[i].isalnum() or expr[i] == '_'):
                i += 1
            tokens.append(Token(TokenType.IDENT, expr[start:i], start))
        else:
            raise ParseError(f'Unexpected character {c!r} at position {i}')

    tokens.append(Token(TokenType.EOF, '', n))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class AlphaParser:
    """Recursive descent parser for alpha expressions."""

    def parse(self, expression: str) -> Node:
        """Parse an expression string and return the AST root."""
        expression = expression.strip()
        if not expression:
            raise ParseError('Empty expression')

        self.tokens = tokenize(expression)
        self.pos = 0
        node = self._parse_expression()

        # Ensure we consumed all tokens
        if self._current().type != TokenType.EOF:
            tok = self._current()
            raise ParseError(f'Unexpected token {tok.value!r} at position {tok.pos}')

        return node

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._advance()
        if tok.type != tt:
            raise ParseError(f'Expected {tt.name}, got {tok.type.name} ({tok.value!r}) at position {tok.pos}')
        return tok

    def _parse_expression(self) -> Node:
        tok = self._current()

        if tok.type == TokenType.NUMBER:
            self._advance()
            return NumberNode(float(tok.value))

        if tok.type == TokenType.IDENT:
            name = tok.value
            self._advance()

            # Check if it's a function call (followed by '(')
            if self._current().type == TokenType.LPAREN:
                self._advance()  # consume '('
                args = self._parse_arg_list()
                self._expect(TokenType.RPAREN)
                return FunctionNode(name, args)
            else:
                # Data reference
                return DataRefNode(name)

        raise ParseError(f'Unexpected token {tok.value!r} at position {tok.pos}')

    def _parse_arg_list(self) -> list[Node]:
        """Parse comma-separated arguments until ')'."""
        args = []

        # Handle empty argument list
        if self._current().type == TokenType.RPAREN:
            return args

        args.append(self._parse_expression())

        while self._current().type == TokenType.COMMA:
            self._advance()  # consume ','
            args.append(self._parse_expression())

        return args
