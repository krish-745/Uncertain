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
EXACT = "EXACT"

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
            elif value == 'Measured':
                yield Token(MEASURED, value, line_num, column)
            elif value == 'Normal':
                yield Token(NORMAL, value, line_num, column)
            elif value == 'Exact':
                yield Token(EXACT, value, line_num, column)
            else:
                yield Token(IDENT, value, line_num, column)
        elif kind in ('PLUS', 'MINUS', 'STAR', 'SLASH', 'LPAREN', 'RPAREN', 'LANGLE', 'RANGLE', 'COMMA', 'COLON', 'EQUALS', 'SEMI'):
            yield Token(kind, value, line_num, column)
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
        elif kind in ('SKIP', 'COMMENT'):
            pass
        elif kind == 'MISMATCH':
            raise LexerError(f"Unexpected character {value!r}", line_num, column)
