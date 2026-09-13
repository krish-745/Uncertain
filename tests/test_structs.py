import pytest
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext
import math

def test_struct_lit_and_access():
    source = """
    let gps = { x: sensor_read(), y: sensor_read() };
    let sum = gps.x + gps.y;
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    check_stmt(stmts[0], ctx)
    check_stmt(stmts[1], ctx)
    
    sum_t = ctx.lookup("sum")
    assert math.isclose(sum_t.dist.mean, 20.0)
    assert sum_t.dist.stddev > 0
