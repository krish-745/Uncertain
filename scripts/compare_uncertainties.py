import math
import uncertainties
from uncertain.distributions import Dist, square

def main():
    print("--- Head-to-Head Comparison: Self-Multiplication (a * a) ---")
    print("We have a sensor reading: a = 10.0 \u00b1 2.0")
    print("What is the variance of a * a?\n")

    # 1. Naive (Human) Approach: Treat as independent
    # Var(X*Y) = mu_x^2 * var_y + mu_y^2 * var_x = 100*4 + 100*4 = 800 -> stddev = sqrt(800) = 28.28
    print("1. Naive Hand Calculation (Assuming Independence):")
    print("   Result: 100.00 \u00b1 28.28")
    print("   (DANGEROUS: Silently understates variance by ignoring correlation)\n")

    # 2. Python's `uncertainties` package
    a_unc = uncertainties.ufloat(10.0, 2.0)
    res_unc = a_unc * a_unc
    print("2. Python's `uncertainties` package (a * a):")
    print(f"   Result: {res_unc.nominal_value:.2f} \u00b1 {res_unc.std_dev:.2f}")
    print("   (BETTER: Detects correlation, but uses linear Taylor approximation, dropping higher-order terms.)\n")

    # 3. Uncertain DSL
    # Uncertain forces you to use `square(a)` instead of `a * a` at compile time.
    # This guarantees we use the EXACT formula for Var(X^2) = 2*sigma^4 + 4*mu^2*sigma^2
    a_dist = Dist(10.0, 2.0)
    res_dist = square(a_dist)
    print("3. Uncertain DSL (forces `square(a)` at compile time):")
    print(f"   Result: {res_dist.mean:.2f} \u00b1 {res_dist.stddev:.2f}")
    print("   (PERFECT: Compiler caught the reuse, forced explicit intent, and used the exact higher-order formula.)\n")

if __name__ == "__main__":
    main()
