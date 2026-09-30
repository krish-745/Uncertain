"""prob(): exact probabilities for Normal and single-reading comparisons, warnings otherwise."""
import math
import pytest
from uncertain.parser import parse
from uncertain.typechecker import TypeContext, check_program

def run(source: str):
    stmts, expr = parse(source)
    ctx = TypeContext()
    diags = check_program(stmts, ctx, expr)
    return ctx, diags

def p_of(source: str):
    ctx, diags = run(source)
    assert not [d for d in diags if d.severity == "error"], diags
    return ctx.lookup("p").dist.mean, [d.kind for d in diags]

@pytest.mark.parametrize("source,expected", [
    # one non-Normal reading: exact CDF of that family
    ("let u = uniform_read(); let p = prob(u > 3);", 0.7),
    ("let u = uniform_read(); let p = prob(2 * u + 1 < 5);", 0.2),            # u < 2
    ("let u = uniform_read(); let p = prob(10 - u >= 4);", 0.6),              # u <= 6 (coefficient flips the inequality)
    ("let n = poisson_read(3); let p = prob(n >= 3);", 1 - math.exp(-3) * (1 + 3 + 4.5)),
    ("let n = poisson_read(3); let p = prob(n > 3);", 1 - math.exp(-3) * (1 + 3 + 4.5 + 4.5)),
    ("let n = poisson_read(3); let p = prob(n * 2 + 1 > 7);", 1 - math.exp(-3) * (1 + 3 + 4.5 + 4.5)),
    ("let b = binomial_read(10, 0.5); let p = prob(b <= 5);", 638 / 1024),
    ("let f = bernoulli_read(0.3); let p = prob(f > 0.5);", 0.3),
    ("let g = geometric_read(0.5); let p = prob(g <= 2);", 0.75),
    ("let e = exponential_read(2); let p = prob(e < 1);", 1 - math.exp(-2)),
    ("let d = empirical_read([1, 2, 3, 4]); let p = prob(d >= 2);", 0.75),
    ("let l = lognormal_read(0, 1); let p = prob(l < 1);", 0.5),
    # linear combinations of Normal readings: exact normal CDF
    ("let a = normal_read(0, 1); let b = normal_read(1, 1); let p = prob(a + b > 1);", 0.5),
    ("let a = normal_read(10, 2); let p = prob(a < 12);", 0.5 * (1 + math.erf(1 / math.sqrt(2)))),
])
def test_exact_probabilities(source, expected):
    p, diag_kinds = p_of(source)
    assert math.isclose(p, expected, rel_tol=1e-9, abs_tol=1e-12)
    assert diag_kinds == []   # exact: no approximation warning

@pytest.mark.parametrize("source,reason", [
    ("let u = uniform_read(); let v = uniform_read(); let p = prob(u + v > 10);", "non-Normal inputs (Uniform)"),
    ("let a = sensor_read(); let b = sensor_read(); let p = prob(a * b > 100);", "nonlinear operation"),
    ("let a = sensor_read(); let p = prob(1 / a > 0.1);", "nonlinear operation"),
    ("let a = sensor_read(); let p = prob(sqrt(a) > 3);", "nonlinear operation"),
    ("let n = poisson_read(3); let a = sensor_read(); let p = prob(n + a > 13);", "non-Normal inputs (Poisson)"),
])
def test_approximate_probabilities_warn(source, reason):
    _, diag_kinds = p_of(source)
    ctx, diags = run(source)
    warnings = [d for d in diags if d.kind == "approximation-warning" and "prob()" in d.extra["msg"]]
    assert len(warnings) == 1
    assert reason in warnings[0].extra["msg"]

def test_non_normal_warning_survives_linear_operations():
    # `u + 0` used to lose track of being Uniform
    _, diags = run("let u = uniform_read(); let v = u + 0; let w = v * v;")
    assert [d.kind for d in diags] == ["approximation-warning"]

np = pytest.importorskip("numpy")

@pytest.mark.parametrize("source,sampler", [
    ("let g = gamma_read(2, 1.5); let p = prob(g > 4);", lambda r, n: r.gamma(2, 1.5, n) > 4),
    ("let n = negbinom_read(3, 0.4); let p = prob(n <= 2);",
     # successes (prob 0.4) before the 3rd failure == numpy's negative_binomial(r, 1-p)
     lambda r, n: r.negative_binomial(3, 0.6, n) <= 2),
    ("let x = poisson_read(12); let p = prob(x < 10);", lambda r, n: r.poisson(12, n) < 10),
    ("let b = binomial_read(40, 0.3); let p = prob(b >= 15);", lambda r, n: r.binomial(40, 0.3, n) >= 15),
])
def test_exact_probabilities_match_sampling(source, sampler):
    p, diag_kinds = p_of(source)
    empirical = sampler(np.random.default_rng(7), 2_000_000).mean()
    assert diag_kinds == []
    assert abs(p - empirical) < 1.5e-3
