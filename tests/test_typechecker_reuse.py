import pytest
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext

def test_typecheck_reuse_error():
    source = """
    let a = sensor_read();
    let b = a * a;
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    check_stmt(stmts[0], ctx)
    diags = check_stmt(stmts[1], ctx)
    
    assert len(diags) == 1
    assert diags[0].kind == "uncertain-reuse"
    assert "a" in diags[0].overlapping_vars

def test_typecheck_reuse_nested():
    source = """
    let a = sensor_read();
    let c = (a + 1.0) * (a - 2.0);
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    check_stmt(stmts[0], ctx)
    diags = check_stmt(stmts[1], ctx)
    
    assert len(diags) == 1
    assert diags[0].kind == "uncertain-reuse"
    assert "a" in diags[0].overlapping_vars

def test_typecheck_square_no_error():
    source = """
    let a = sensor_read();
    let b = square(a);
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    check_stmt(stmts[0], ctx)
    diags = check_stmt(stmts[1], ctx)
    
    assert len(diags) == 0

def test_typecheck_correlated_no_error():
    source = """
    let a = sensor_read();
    let b = correlated(a, a, cov=0.05);
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    check_stmt(stmts[0], ctx)
    diags = check_stmt(stmts[1], ctx)
    
    assert len(diags) == 0
