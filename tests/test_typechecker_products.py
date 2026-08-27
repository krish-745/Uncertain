import pytest
import math
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext

def test_typecheck_mul():
    source = """
    let a = sensor_read();
    let b = sensor_read();
    let c = a * b;
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    for stmt in stmts:
        diags = check_stmt(stmt, ctx)
        assert len(diags) == 0
        
    typ = ctx.lookup("c")
    assert typ is not None
    assert typ.dist.mean == 100.0
    assert math.isclose(typ.dist.stddev, 14.14213, rel_tol=1e-3)

def test_typecheck_div():
    source = """
    let a = sensor_read();
    let b = sensor_read();
    let c = a / b;
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    for stmt in stmts:
        diags = check_stmt(stmt, ctx)
        assert len(diags) == 0
        
    typ = ctx.lookup("c")
    assert typ is not None
    assert typ.dist.mean == 1.0
