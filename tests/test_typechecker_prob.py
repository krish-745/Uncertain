import pytest
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext
import math

def test_prob_query():
    source = """
    let temp = sensor_read(); // Mean 10, std 1
    let p_over_10 = prob(temp > 10.0);
    let p_under_10 = prob(temp < 10.0);
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    # temp
    check_stmt(stmts[0], ctx)
    # p_over_10
    check_stmt(stmts[1], ctx)
    # p_under_10
    check_stmt(stmts[2], ctx)
    
    t1 = ctx.lookup("p_over_10")
    t2 = ctx.lookup("p_under_10")
    
    assert math.isclose(t1.dist.mean, 0.5, abs_tol=1e-3)
    assert t1.dist.stddev == 0.0
    assert t1.dist.family == "Exact"
    
    assert math.isclose(t2.dist.mean, 0.5, abs_tol=1e-3)

def test_prob_diff():
    source = """
    let t1 = sensor_read();
    let t2 = sensor_read();
    let p = prob(t1 > t2);
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    check_stmt(stmts[0], ctx)
    check_stmt(stmts[1], ctx)
    check_stmt(stmts[2], ctx)
    
    p = ctx.lookup("p")
    assert math.isclose(p.dist.mean, 0.5, abs_tol=1e-3)

def test_prob_correlated():
    source = """
    let base = sensor_read();
    let a = base + 2.0;
    let b = base + 2.0;
    let p = prob(a > b);
    """
    # They are perfectly correlated, a - b = 0 with 0 variance!
    stmts, _ = parse(source)
    ctx = TypeContext()
    check_stmt(stmts[0], ctx)
    check_stmt(stmts[1], ctx)
    check_stmt(stmts[2], ctx)
    check_stmt(stmts[3], ctx)
    
    p = ctx.lookup("p")
    assert p.dist.mean == 0.0 # because mean(a - b) = 0, mean > 0 is false, p=0.0!

