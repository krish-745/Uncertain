import math
from dataclasses import dataclass

class MathDomainError(Exception):
    pass

@dataclass(frozen=True)
class Dist:
    mean: float
    stddev: float   # 0.0 represents Exact(value)
    family: str = "Normal" # "Normal", "Uniform", "Exact"

def add(a: Dist, b: Dist) -> Dist:
    return Dist(a.mean + b.mean, math.sqrt(a.stddev**2 + b.stddev**2))

def sub(a: Dist, b: Dist) -> Dist:
    return Dist(a.mean - b.mean, math.sqrt(a.stddev**2 + b.stddev**2))

def mul_independent(a: Dist, b: Dist) -> Dist:
    mean = a.mean * b.mean
    variance = (a.mean**2 * b.stddev**2) + (b.mean**2 * a.stddev**2) + (a.stddev**2 * b.stddev**2)
    return Dist(mean, math.sqrt(max(0.0, variance)))

def div_independent(a: Dist, b: Dist) -> Dist:
    if b.mean == 0:
        raise MathDomainError("cannot divide by a distribution with a zero mean")
    mean = a.mean / b.mean
    variance = (a.stddev**2 / b.mean**2) + ((a.mean**2 * b.stddev**2) / b.mean**4)
    return Dist(mean, math.sqrt(max(0.0, variance)))

def abs_dist(a: Dist) -> Dist:
    return Dist(abs(a.mean), a.stddev)

def log_dist(a: Dist) -> Dist:
    if a.mean <= 0:
        raise MathDomainError(f"cannot compute the log of a distribution with non-positive mean ({a.mean})")
    return Dist(math.log(a.mean), a.stddev / a.mean)

def exp_dist(a: Dist) -> Dist:
    mean = math.exp(a.mean)
    return Dist(mean, a.stddev * mean)

def pow_const(a: Dist, n: int) -> Dist:
    if n == 2:
        return square(a)
    elif n == 3:
        mean = (a.mean ** 3) + 3 * a.mean * (a.stddev ** 2)
        variance = 9 * (a.mean ** 4) * (a.stddev ** 2) + 36 * (a.mean ** 2) * (a.stddev ** 4) + 15 * (a.stddev ** 6)
        return Dist(mean, math.sqrt(max(0.0, variance)))
    else:
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

def correlated_product(a: Dist, b: Dist, cov: float) -> Dist:
    # Var(XY) with explicit covariance term
    # Var(XY) ≈ μx^2 σy^2 + μy^2 σx^2 + 2 μx μy Cov(X,Y)
    mean = a.mean * b.mean
    variance = (a.mean ** 2) * (b.stddev ** 2) + (b.mean ** 2) * (a.stddev ** 2) + 2 * a.mean * b.mean * cov
    return Dist(mean, math.sqrt(max(0.0, variance)))
