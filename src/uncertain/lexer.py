import re
from dataclasses import dataclass
from typing import Iterator

@dataclass
class Token:
    type: str
    value: str
    line: int
    col: int

# Token types
LET = "LET"
IDENT = "IDENT"
NUMBER = "NUMBER"
PLUS = "PLUS"
MINUS = "MINUS"
STAR = "STAR"
SLASH = "SLASH"
LPAREN = "LPAREN"
RPAREN = "RPAREN"
LANGLE = "LANGLE"
RANGLE = "RANGLE"
COMMA = "COMMA"
COLON = "COLON"
EQUALS = "EQUALS"
SEMI = "SEMI"
MEASURED = "MEASURED"
NORMAL = "NORMAL"
UNIFORM = "UNIFORM"
EMPIRICAL = "EMPIRICAL"
LOGNORMAL = "LOGNORMAL"
POISSON = "POISSON"
BINOMIAL = "BINOMIAL"
GAMMA = "GAMMA"
BERNOULLI = "BERNOULLI"
NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"
GEOMETRIC = "GEOMETRIC"
EXPONENTIAL = "EXPONENTIAL"
EXACT = "EXACT"
VAR = "VAR"
WHILE = "WHILE"
FOR = "FOR"
IF = "IF"
ELSE = "ELSE"

class LexerError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"{message} at {line}:{col}")
        self.line = line
        self.col = col

def tokenize(source: str) -> Iterator[Token]:
    token_specification = [
        ('NUMBER',   r'\d+(\.\d*)?'),  # Integer or decimal number
        ('IDENT',    r'[A-Za-z_][A-Za-z0-9_]*'), # Identifiers
        ('COMMENT',  r'//.*'),         # Comments
        ('PLUS',     r'\+'),
        ('MINUS',    r'-'),
        ('STAR',     r'\*'),
        ('SLASH',    r'/'),
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        ('LBRACE',   r'\{'),
        ('RBRACE',   r'\}'),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('LANGLE',   r'<'),
        ('RANGLE',   r'>'),
        ('COMMA',    r','),
        ('COLON',    r':'),
        ('EQUALS',   r'='),
        ('SEMI',     r';'),
        ('SKIP',     r'[ \t]+'),       # Skip over spaces and tabs
        ('NEWLINE',  r'\n|\r\n'),      # Line endings
        ('MISMATCH', r'.'),            # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    for mo in re.finditer(tok_regex, source):
        kind = mo.lastgroup
        value = mo.group(kind)
        column = mo.start() - line_start + 1
        if kind == 'NUMBER':
            yield Token(NUMBER, value, line_num, column)
        elif kind == 'IDENT':
            if value == 'let':
                yield Token(LET, value, line_num, column)
            elif value == 'var':
                yield Token(VAR, value, line_num, column)
            elif value == 'while':
                yield Token(WHILE, value, line_num, column)
            elif value == 'for':
                yield Token(FOR, value, line_num, column)
            elif value == 'if':
                yield Token(IF, value, line_num, column)
            elif value == 'else':
                yield Token(ELSE, value, line_num, column)
            elif value == 'Measured':
                yield Token(MEASURED, value, line_num, column)
            elif value == 'Normal':
                yield Token(NORMAL, value, line_num, column)
            elif value == 'Uniform':
                yield Token(UNIFORM, value, line_num, column)
            elif value == 'Empirical':
                yield Token("EMPIRICAL", value, line_num, column)
            elif value == 'LogNormal':
                yield Token("LOGNORMAL", value, line_num, column)
            elif value == 'Poisson':
                yield Token("POISSON", value, line_num, column)
            elif value == 'Binomial':
                yield Token("BINOMIAL", value, line_num, column)
            elif value == 'Gamma':
                yield Token("GAMMA", value, line_num, column)
            elif value == 'Bernoulli':
                yield Token("BERNOULLI", value, line_num, column)
            elif value == 'NegativeBinomial':
                yield Token("NEGATIVE_BINOMIAL", value, line_num, column)
            elif value == 'Geometric':
                yield Token("GEOMETRIC", value, line_num, column)
            elif value == 'Exponential':
                yield Token("EXPONENTIAL", value, line_num, column)
            elif value == 'Exact':
                yield Token(EXACT, value, line_num, column)
            else:
                yield Token(IDENT, value, line_num, column)
        elif kind in ('PLUS', 'MINUS', 'STAR', 'SLASH', 'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET', 'LANGLE', 'RANGLE', 'COMMA', 'COLON', 'EQUALS', 'SEMI'):
            yield Token(kind, value, line_num, column)
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
        elif kind in ('SKIP', 'COMMENT'):
            pass
        elif kind == 'MISMATCH':
            raise LexerError(f"Unexpected character {value!r}", line_num, column)
