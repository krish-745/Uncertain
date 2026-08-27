import pytest
from uncertain.lexer import tokenize, Token
from uncertain.parser import parse, ParseError
from uncertain.ast_nodes import *

def test_lexer():
    source = "let a: Measured<Normal(10.0, 2.0)> = sensor_read();"
    tokens = list(tokenize(source))
    
    types = [t.type for t in tokens]
    expected = [
        "LET", "IDENT", "COLON", "MEASURED", "LANGLE", "NORMAL", "LPAREN",
        "NUMBER", "COMMA", "NUMBER", "RPAREN", "RANGLE", "EQUALS",
        "IDENT", "LPAREN", "RPAREN", "SEMI"
    ]
    assert types == expected

def test_parser_basic_let():
    source = "let a = 10.0;"
    stmts, expr = parse(source)
    assert len(stmts) == 1
    assert expr is None
    
    stmt = stmts[0]
    assert stmt.name == "a"
    assert stmt.type_ann is None
    assert isinstance(stmt.value, NumberLit)
    assert stmt.value.value == 10.0

def test_parser_type_ann():
    source = "let a: Measured<Normal(10, 2)> = 5;"
    stmts, expr = parse(source)
    assert len(stmts) == 1
    
    stmt = stmts[0]
    assert isinstance(stmt.type_ann, NormalLit)
    assert stmt.type_ann.mean.value == 10.0
    assert stmt.type_ann.stddev.value == 2.0

def test_parser_expression():
    source = "a + b * 2"
    stmts, expr = parse(source)
    assert len(stmts) == 0
    assert isinstance(expr, BinOp)
    assert expr.op == "+"
    assert isinstance(expr.left, VarRef)
    assert expr.left.name == "a"
    assert isinstance(expr.right, BinOp)
    assert expr.right.op == "*"
    
def test_parser_kwargs():
    source = "correlated(a, b, cov=0.5)"
    stmts, expr = parse(source)
    assert isinstance(expr, Call)
    assert expr.name == "correlated"
    assert len(expr.args) == 2
    assert "cov" in expr.kwargs
    assert expr.kwargs["cov"].value == 0.5

def test_parser_unary():
    source = "-a + 5"
    stmts, expr = parse(source)
    assert isinstance(expr, BinOp)
    assert expr.op == "+"
    assert isinstance(expr.left, BinOp)
    assert expr.left.op == "-"
    assert isinstance(expr.left.right, VarRef)
    assert expr.left.right.name == "a"
