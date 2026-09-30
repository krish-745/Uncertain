import math
from dataclasses import dataclass

class MathDomainError(Exception):
    pass

@dataclass(frozen=True)
class Dist:
    mean: float
    stddev: float   # 0.0 represents Exact(value)
    family: str = "Normal" # "Normal", "Uniform", "Exact", ...

def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def add(a: Dist, b: Dist) -> Dist:
    return Dist(a.mean + b.mean, math.sqrt(a.stddev**2 + b.stddev**2))

def sub(a: Dist, b: Dist) -> Dist:
    return Dist(a.mean - b.mean, math.sqrt(a.stddev**2 + b.stddev**2))

def product(a: Dist, b: Dist, cov: float = 0.0) -> Dist:
    # Exact moments of XY for jointly normal X, Y (Isserlis' theorem):
    # E[XY]   = μx μy + c
    # Var(XY) = μx²σy² + μy²σx² + 2 μx μy c + σx²σy² + c²
    # With c = 0 this is also exact for any pair of independent variables.
    mean = a.mean * b.mean + cov
    variance = (a.mean**2) * (b.stddev**2) + (b.mean**2) * (a.stddev**2) + 2 * a.mean * b.mean * cov \
        + (a.stddev**2) * (b.stddev**2) + cov**2
    return Dist(mean, math.sqrt(max(0.0, variance)))

def mul_independent(a: Dist, b: Dist) -> Dist:
    return product(a, b, 0.0)

def correlated_product(a: Dist, b: Dist, cov: float) -> Dist:
    return product(a, b, cov)

def div_independent(a: Dist, b: Dist) -> Dist:
    if b.mean == 0:
        raise MathDomainError("cannot divide by a distribution with a zero mean")
    mean = a.mean / b.mean
    variance = (a.stddev**2 / b.mean**2) + ((a.mean**2 * b.stddev**2) / b.mean**4)
    return Dist(mean, math.sqrt(max(0.0, variance)))

def abs_dist(a: Dist) -> Dist:
    if a.stddev == 0:
        return Dist(abs(a.mean), 0.0)
    # Folded normal distribution
    mu, sigma = a.mean, a.stddev
    mean = sigma * math.sqrt(2.0 / math.pi) * math.exp(-mu**2 / (2 * sigma**2)) + mu * (1.0 - 2.0 * _phi(-mu / sigma))
    variance = mu**2 + sigma**2 - mean**2
    return Dist(mean, math.sqrt(max(0.0, variance)))

def log_dist(a: Dist) -> Dist:
    if a.mean <= 0:
        raise MathDomainError(f"cannot compute the log of a distribution with non-positive mean ({a.mean})")
    return Dist(math.log(a.mean), a.stddev / a.mean)

def exp_dist(a: Dist) -> Dist:
    # exp of a normal is log-normal: exact moments
    try:
        mean = math.exp(a.mean + a.stddev**2 / 2.0)
        variance = math.expm1(a.stddev**2) * math.exp(2.0 * a.mean + a.stddev**2)
    except OverflowError:
        raise MathDomainError(f"exp() overflows for a distribution with mean {a.mean} and stddev {a.stddev}")
    return Dist(mean, math.sqrt(max(0.0, variance)))

def sin_dist(a: Dist) -> Dist:
    # Exact moments for a normal input: E[sin X] = sin μ · e^{-σ²/2}
    damp = math.exp(-a.stddev**2 / 2.0)
    mean = math.sin(a.mean) * damp
    second = 0.5 * (1.0 - math.exp(-2.0 * a.stddev**2) * math.cos(2.0 * a.mean))
    return Dist(mean, math.sqrt(max(0.0, second - mean**2)))

def cos_dist(a: Dist) -> Dist:
    # Exact moments for a normal input: E[cos X] = cos μ · e^{-σ²/2}
    damp = math.exp(-a.stddev**2 / 2.0)
    mean = math.cos(a.mean) * damp
    second = 0.5 * (1.0 + math.exp(-2.0 * a.stddev**2) * math.cos(2.0 * a.mean))
    return Dist(mean, math.sqrt(max(0.0, second - mean**2)))

def pow_const(a: Dist, n: int) -> Dist:
    if n == 0:
        return Dist(1.0, 0.0)
    elif n == 1:
        return Dist(a.mean, a.stddev)
    elif n == 2:
        return square(a)
    elif n == 3:
        mean = (a.mean ** 3) + 3 * a.mean * (a.stddev ** 2)
        variance = 9 * (a.mean ** 4) * (a.stddev ** 2) + 36 * (a.mean ** 2) * (a.stddev ** 4) + 15 * (a.stddev ** 6)
        return Dist(mean, math.sqrt(max(0.0, variance)))
    else:
        if n < 0 and a.mean == 0:
            raise MathDomainError(f"cannot raise a distribution with a zero mean to a negative power ({n})")
        mean = a.mean ** n
        variance = (n * (a.mean ** (n - 1)) * a.stddev) ** 2
        return Dist(mean, math.sqrt(max(0.0, variance)))

def square(a: Dist) -> Dist:
    # E[X^2] = μ^2 + σ^2
    # Var(X^2) = 2σ^4 + 4μ^2σ^2 for a normal X
    mean = (a.mean ** 2) + (a.stddev ** 2)
    variance = 2 * (a.stddev ** 4) + 4 * (a.mean ** 2) * (a.stddev ** 2)
    return Dist(mean, math.sqrt(variance))

def sqrt_dist(a: Dist) -> Dist:
    if a.mean < 0:
        raise MathDomainError(f"cannot compute the square root of a distribution with a negative mean ({a.mean})")
    # delta method: stddev' = stddev / (2*sqrt(mean))
    mean = math.sqrt(a.mean)
    stddev = a.stddev / (2 * math.sqrt(a.mean)) if a.mean > 0 else 0.0
    return Dist(mean, stddev)

# --- Named distribution families ------------------------------------------------

# family -> parameter names, in the order they appear in `Measured<Family(...)>` and `*_read(...)`
FAMILY_PARAMS: dict[str, tuple[str, ...]] = {
    "Normal": ("mean", "stddev"),
    "Uniform": ("min", "max"),
    "Exact": ("value",),
    "LogNormal": ("mu", "sigma"),
    "Poisson": ("lambda",),
    "Binomial": ("n", "p"),
    "Gamma": ("k", "theta"),
    "Bernoulli": ("p",),
    "NegativeBinomial": ("r", "p"),
    "Geometric": ("p",),
    "Exponential": ("lambda",),
}

def _require(cond: bool, family: str, msg: str):
    if not cond:
        raise MathDomainError(f"invalid {family} parameters: {msg}")

def moments(family: str, params: list[float]) -> Dist:
    """Mean and standard deviation of a named distribution, validating its parameters."""
    if family not in FAMILY_PARAMS:
        raise MathDomainError(f"unknown distribution family '{family}'")
    expected = len(FAMILY_PARAMS[family])
    if len(params) != expected:
        raise MathDomainError(f"{family} takes {expected} parameter(s), got {len(params)}")

    try:
        if family == "Normal":
            mean, sd = params
            _require(sd >= 0, family, f"stddev must be >= 0 (got {sd})")
            return Dist(mean, sd, "Normal")
        if family == "Exact":
            return Dist(params[0], 0.0, "Exact")
        if family == "Uniform":
            lo, hi = params
            _require(lo <= hi, family, f"min must be <= max (got {lo}, {hi})")
            return Dist((lo + hi) / 2.0, (hi - lo) / math.sqrt(12.0), "Uniform")
        if family == "LogNormal":
            mu, sigma = params
            _require(sigma >= 0, family, f"sigma must be >= 0 (got {sigma})")
            mean = math.exp(mu + sigma**2 / 2.0)
            sd = math.sqrt(math.expm1(sigma**2) * math.exp(2.0 * mu + sigma**2))
            return Dist(mean, sd, "LogNormal")
        if family == "Poisson":
            lam = params[0]
            _require(lam >= 0, family, f"lambda must be >= 0 (got {lam})")
            return Dist(lam, math.sqrt(lam), "Poisson")
        if family == "Binomial":
            n, p = params
            _require(n >= 0 and float(n).is_integer(), family, f"n must be a non-negative integer (got {n})")
            _require(0 <= p <= 1, family, f"p must be in [0, 1] (got {p})")
            return Dist(n * p, math.sqrt(n * p * (1 - p)), "Binomial")
        if family == "Gamma":
            k, theta = params
            _require(k > 0 and theta > 0, family, f"k and theta must be > 0 (got {k}, {theta})")
            return Dist(k * theta, math.sqrt(k) * theta, "Gamma")
        if family == "Bernoulli":
            p = params[0]
            _require(0 <= p <= 1, family, f"p must be in [0, 1] (got {p})")
            return Dist(p, math.sqrt(p * (1 - p)), "Bernoulli")
        if family == "NegativeBinomial":
            # Number of successes before the r-th failure, where p is the success probability.
            r, p = params
            _require(r > 0, family, f"r must be > 0 (got {r})")
            _require(0 <= p < 1, family, f"p must be in [0, 1) (got {p})")
            return Dist(p * r / (1 - p), math.sqrt(p * r) / (1 - p), "NegativeBinomial")
        if family == "Geometric":
            # Number of trials up to and including the first success.
            p = params[0]
            _require(0 < p <= 1, family, f"p must be in (0, 1] (got {p})")
            return Dist(1.0 / p, math.sqrt(1.0 - p) / p, "Geometric")
        if family == "Exponential":
            lam = params[0]
            _require(lam > 0, family, f"lambda must be > 0 (got {lam})")
            return Dist(1.0 / lam, 1.0 / lam, "Exponential")
    except OverflowError:
        raise MathDomainError(f"{family} parameters {params} are too large to represent")
    raise AssertionError(family)

def empirical(values: list[float]) -> Dist:
    if not values:
        raise MathDomainError("an empirical distribution needs at least one value")
    mean = sum(values) / len(values)
    stddev = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return Dist(mean, stddev, "Empirical")
