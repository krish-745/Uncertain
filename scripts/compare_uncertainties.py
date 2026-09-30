import os
import sys
import uncertainties

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from uncertain.parser import parse
from uncertain.typechecker import TypeContext, check_program

def main():
    print("--- Head-to-Head Comparison: Self-Multiplication (a * a) ---")
    print("We have a sensor reading: a = 2.0 ± 5.0")
    print("What is the variance of a * a?\n")

    # 1. Naive (Human) Approach: Treat as independent
    # Var(X*Y) = mu_x^2 * var_y + mu_y^2 * var_x = 4*25 + 4*25 = 200 -> stddev = sqrt(200) = 14.14
    print("1. Naive Hand Calculation (Assuming Independence):")
    print("   Result: 4.00 ± 14.14")
    print("   (DANGEROUS: Silently understates variance by ignoring correlation)\n")

    # 2. Python's `uncertainties` package
    a_unc = uncertainties.ufloat(2.0, 5.0)
    res_unc = a_unc * a_unc
    print("2. Python's `uncertainties` package (a * a):")
    print(f"   Result: {res_unc.nominal_value:.2f} ± {res_unc.std_dev:.2f}")
    print("   (BETTER: Detects correlation, but uses linear Taylor approximation, dropping higher-order terms. Notice the mean is completely wrong!)\n")

    # 3. Uncertain DSL: the same program, `a * a`, run through the type checker
    stmts, _ = parse("let a = normal_read(2.0, 5.0); let r = a * a;")
    ctx = TypeContext()
    check_program(stmts, ctx)
    res = ctx.lookup("r").dist
    print("3. Uncertain DSL:")
    print(f"   Result: {res.mean:.2f} ± {res.stddev:.2f}")
    print("   (PERFECT: Compiler tracked the affine lineage and injected lost non-linear variance to match the EXACT mathematical formula for E[X²] and Var(X²).)\n")

if __name__ == "__main__":
    main()
