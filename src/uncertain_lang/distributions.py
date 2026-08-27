import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Dist:
    mean: float
    stddev: float   # 0.0 represents Exact(value)

def add(a: Dist, b: Dist) -> Dist:
    return Dist(a.mean + b.mean, math.sqrt(a.stddev**2 + b.stddev**2))

def sub(a: Dist, b: Dist) -> Dist:
    return Dist(a.mean - b.mean, math.sqrt(a.stddev**2 + b.stddev**2))

def mul_independent(a: Dist, b: Dist) -> Dist:
    mean = a.mean * b.mean
    # delta-method relative-variance formula; guard against mean == 0
    rel = math.sqrt((a.stddev / a.mean) ** 2 + (b.stddev / b.mean) ** 2) if a.mean and b.mean else 0.0
    return Dist(mean, abs(mean) * rel)

def div_independent(a: Dist, b: Dist) -> Dist:
    mean = a.mean / b.mean
    rel = math.sqrt((a.stddev / a.mean) ** 2 + (b.stddev / b.mean) ** 2) if a.mean and b.mean else 0.0
    return Dist(mean, abs(mean) * rel)

def square(a: Dist) -> Dist:
    # Var(X^2) = 2σ^4 + 4μ^2σ^2 for a normal X
    mean = a.mean ** 2
    variance = 2 * (a.stddev ** 4) + 4 * (a.mean ** 2) * (a.stddev ** 2)
    return Dist(mean, math.sqrt(variance))

def sqrt_dist(a: Dist) -> Dist:
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
