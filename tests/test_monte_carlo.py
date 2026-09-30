"""Cross-check the analytic formulas (and the correlation tracking) against Monte Carlo sampling."""
import math
import pytest

np = pytest.importorskip("numpy")

from uncertain.parser import parse
from uncertain.typechecker import TypeContext, check_program

N = 1_000_000

def analyse(source: str, name: str):
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = [d for d in check_program(stmts, ctx) if d.severity == "error"]
    assert diags == []
    return ctx.lookup(name).dist

def assert_matches(dist, samples, tol=0.02):
    mean, std = samples.mean(), samples.std()
    scale = max(std, 1e-9)
    # compare the mean relative to the spread (so means near zero are handled), and the stddev relatively
    assert abs(dist.mean - mean) <= tol * max(abs(mean), scale), f"mean {dist.mean} vs sampled {mean}"
    assert abs(dist.stddev - std) <= tol * scale, f"stddev {dist.stddev} vs sampled {std}"

@pytest.fixture
def rng():
    return np.random.default_rng(1234)

@pytest.mark.parametrize("mu,sigma", [(10.0, 1.0), (0.0, 1.0), (1.5707963, 1.0), (-2.0, 3.0)])
def test_unary_functions(rng, mu, sigma):
    x = rng.normal(mu, sigma, N)
    src = f"let x = normal_read({mu}, {sigma});"
    assert_matches(analyse(src + "let y = sin(x);", "y"), np.sin(x))
    assert_matches(analyse(src + "let y = cos(x);", "y"), np.cos(x))
    assert_matches(analyse(src + "let y = abs(x);", "y"), np.abs(x))
    assert_matches(analyse(src + "let y = square(x);", "y"), x**2)
    assert_matches(analyse(src + "let y = pow(x, 3);", "y"), x**3)

def test_exp_is_lognormal(rng):
    x = rng.normal(1.0, 0.5, N)
    assert_matches(analyse("let x = normal_read(1, 0.5); let y = exp(x);", "y"), np.exp(x))

def test_products(rng):
    x = rng.normal(10.0, 2.0, N)
    y = rng.normal(4.0, 0.5, N)
    src = "let x = normal_read(10, 2); let y = normal_read(4, 0.5);"
    assert_matches(analyse(src + "let p = x * y;", "p"), x * y)
    assert_matches(analyse(src + "let p = x * x;", "p"), x * x)
    assert_matches(analyse(src + "let p = x * (x + y);", "p"), x * (x + y))

def test_explicit_covariance(rng):
    cov = 0.5
    s = rng.multivariate_normal([20.0, 4.0], [[9.0, cov], [cov, 0.25]], N)
    src = f"let x = normal_read(20, 3); let y = normal_read(4, 0.5); let p = correlated(x, y, cov={cov});"
    assert_matches(analyse(src, "p"), s[:, 0] * s[:, 1])

@pytest.mark.parametrize("fn", ["sin", "cos", "exp", "abs", "square"])
def test_correlation_survives_nonlinear_functions(rng, fn):
    # f(x) - x only has the right spread if Cov(f(x), x) is tracked correctly
    x = rng.normal(0.5, 0.8, N)
    f = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "abs": np.abs, "square": np.square}[fn]
    src = f"let x = normal_read(0.5, 0.8); let d = {fn}(x) - x;"
    assert_matches(analyse(src, "d"), f(x) - x)
