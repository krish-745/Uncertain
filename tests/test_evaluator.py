import pytest
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext
from uncertain.evaluator import evaluate

def test_evaluator():
    source = """
    let a = sensor_read();
    let b = sensor_read();
    let c = a + b;
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    for stmt in stmts:
        check_stmt(stmt, ctx)
        
    env = evaluate(stmts, ctx)
    assert "c" in env
    assert env["c"].mean == 20.0
    assert "a" in env
    assert env["a"].mean == 10.0
