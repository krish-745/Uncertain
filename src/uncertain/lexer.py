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
LE = "LE"
GE = "GE"
EQEQ = "EQEQ"
NEQ = "NEQ"
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
FN = "FN"
RETURN = "RETURN"

KEYWORDS = {
    'let': LET,
    'var': VAR,
    'while': WHILE,
    'for': FOR,
    'if': IF,
    'else': ELSE,
    'fn': FN,
    'return': RETURN,
    'Measured': MEASURED,
    'Normal': NORMAL,
    'Uniform': UNIFORM,
    'Empirical': EMPIRICAL,
    'LogNormal': LOGNORMAL,
    'Poisson': POISSON,
    'Binomial': BINOMIAL,
    'Gamma': GAMMA,
    'Bernoulli': BERNOULLI,
    'NegativeBinomial': NEGATIVE_BINOMIAL,
    'Geometric': GEOMETRIC,
    'Exponential': EXPONENTIAL,
    'Exact': EXACT,
}

class LexerError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"{message} at {line}:{col}")
        self.line = line
        self.col = col

token_specification = [
    ('NUMBER',   r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'),  # 1, 1.5, .5, 1e-3
    ('IDENT',    r'[A-Za-z_][A-Za-z0-9_]*'), # Identifiers
    ('COMMENT',  r'//[^\r\n]*'),   # Comments
    ('PLUS',     r'\+'),
    ('ARROW',    r'->'),
    ('MINUS',    r'-'),
    ('STAR',     r'\*'),
    ('SLASH',    r'/'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('LBRACE',   r'\{'),
    ('RBRACE',   r'\}'),
    ('LBRACKET', r'\['),
    ('RBRACKET', r'\]'),
    ('DOT',      r'\.'),
    ('LE',       r'<='),
    ('GE',       r'>='),
    ('EQEQ',     r'=='),
    ('NEQ',      r'!='),
    ('LANGLE',   r'<'),
    ('RANGLE',   r'>'),
    ('COMMA',    r','),
    ('COLON',    r':'),
    ('EQUALS',   r'='),
    ('SEMI',     r';'),
    ('SKIP',     r'[ \t\f]+'),     # Skip over spaces and tabs
    ('NEWLINE',  r'\r\n|\n|\r'),   # Line endings
    ('MISMATCH', r'.'),            # Any other character
]
_tok_regex = re.compile('|'.join('(?P<%s>%s)' % pair for pair in token_specification))

def tokenize(source: str) -> Iterator[Token]:
    line_num = 1
    line_start = 0
    for mo in _tok_regex.finditer(source):
        kind = mo.lastgroup
        value = mo.group(kind)
        column = mo.start() - line_start + 1
        if kind == 'NUMBER':
            yield Token(NUMBER, value, line_num, column)
        elif kind == 'IDENT':
            yield Token(KEYWORDS.get(value, IDENT), value, line_num, column)
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
        elif kind in ('SKIP', 'COMMENT'):
            pass
        elif kind == 'MISMATCH':
            raise LexerError(f"Unexpected character {value!r}", line_num, column)
        else:
            yield Token(kind, value, line_num, column)
