import numpy as np
import sys
import os

# Add the src directory to path to allow running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from uncertain.distributions import (
    Dist, add, sub, mul_independent, div_independent, square, sqrt_dist, correlated_product
)

def mc_check(op_name, analytic_dist, sample_fn, n=1_000_000, rtol=5e-2):
    samples = sample_fn(n)
    empirical_mean = samples.mean()
    empirical_std = samples.std()
    
    mean_err = abs(analytic_dist.mean - empirical_mean) / max(abs(empirical_mean), 1e-9)
    std_err = abs(analytic_dist.stddev - empirical_std) / max(empirical_std, 1e-9)
    
    print(f"[{op_name}] Analytic: (mean={analytic_dist.mean:.4f}, stddev={analytic_dist.stddev:.4f})")
    print(f"[{op_name}] Empiric : (mean={empirical_mean:.4f}, stddev={empirical_std:.4f})")
    
    if mean_err > rtol or std_err > rtol:
        print(f"  -> FAIL: mean_err={mean_err:.2%}, std_err={std_err:.2%}")
        return False
    print(f"  -> PASS: mean_err={mean_err:.2%}, std_err={std_err:.2%}")
    return True

def run_report():
    print("Running Monte Carlo Cross-Checks...\n")
    
    rng = np.random.default_rng(42)
    
    # 1. Add
    d1 = Dist(10.0, 2.0)
    d2 = Dist(5.0, 1.0)
    mc_check(
        "add", 
        add(d1, d2), 
        lambda n: rng.normal(d1.mean, d1.stddev, n) + rng.normal(d2.mean, d2.stddev, n)
    )

    # 2. Sub
    mc_check(
        "sub", 
        sub(d1, d2), 
        lambda n: rng.normal(d1.mean, d1.stddev, n) - rng.normal(d2.mean, d2.stddev, n)
    )

    # 3. Mul Independent
    d3 = Dist(20.0, 3.0)
    d4 = Dist(4.0, 0.5)
    mc_check(
        "mul_independent", 
        mul_independent(d3, d4), 
        lambda n: rng.normal(d3.mean, d3.stddev, n) * rng.normal(d4.mean, d4.stddev, n)
    )

    # 4. Div Independent
    mc_check(
        "div_independent", 
        div_independent(d3, d4), 
        lambda n: rng.normal(d3.mean, d3.stddev, n) / rng.normal(d4.mean, d4.stddev, n)
    )

    # 5. Square
    mc_check(
        "square", 
        square(d3), 
        lambda n: rng.normal(d3.mean, d3.stddev, n) ** 2
    )

    # 6. Sqrt
    mc_check(
        "sqrt_dist", 
        sqrt_dist(d3), 
        lambda n: np.sqrt(rng.normal(d3.mean, d3.stddev, n))
    )

    # 7. Correlated product (cov = 0.5)
    cov = 0.5
    mean = [d3.mean, d4.mean]
    cov_matrix = [
        [d3.stddev**2, cov],
        [cov, d4.stddev**2]
    ]
    def sample_corr(n):
        samples = rng.multivariate_normal(mean, cov_matrix, n)
        return samples[:, 0] * samples[:, 1]

    mc_check(
        "correlated_product", 
        correlated_product(d3, d4, cov), 
        sample_corr
    )

if __name__ == "__main__":
    run_report()
