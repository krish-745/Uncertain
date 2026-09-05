import pytest
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext
import math

def test_distributions():
    source = """
    let u = uniform_read();
    let emp = empirical_read([2.0, 4.0, 6.0]);
    let logn = lognormal_read(0.0, 1.0);
    let pois = poisson_read(5.0);
    let bino = binomial_read(10.0, 0.5);
    let gam = gamma_read(2.0, 2.0);
    let bern = bernoulli_read(0.8);
    let nbin = negbinom_read(5.0, 0.5);
    let geo = geometric_read(0.5);
    let exp = exponential_read(2.0);
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = []
    for stmt in stmts:
        diags.extend(check_stmt(stmt, ctx))
    
    assert len(diags) == 0
    
    assert ctx.lookup("u").dist.family == "Uniform"
    
    emp_dist = ctx.lookup("emp").dist
    assert emp_dist.family == "Empirical"
    assert math.isclose(emp_dist.mean, 4.0)
    
    assert ctx.lookup("logn").dist.family == "LogNormal"
    assert ctx.lookup("pois").dist.family == "Poisson"
    assert ctx.lookup("bino").dist.family == "Binomial"
    assert ctx.lookup("gam").dist.family == "Gamma"
    assert ctx.lookup("bern").dist.family == "Bernoulli"
    assert ctx.lookup("nbin").dist.family == "NegativeBinomial"
    assert ctx.lookup("geo").dist.family == "Geometric"
    assert ctx.lookup("exp").dist.family == "Exponential"

def test_approximation_warning():
    source = """
    let u = uniform_read();
    let n = sensor_read();
    let res = u + n;
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = []
    for stmt in stmts:
        diags.extend(check_stmt(stmt, ctx))
    
    assert any(d.kind == "approximation-warning" for d in diags)
