import uncertainties
from uncertain.distributions import Dist, square

# 2.0 +/- 5.0
print("--- 2.0 +/- 5.0 ---")
a_unc = uncertainties.ufloat(2.0, 5.0)
print(f"unc: {a_unc * a_unc}")

a_dist = Dist(2.0, 5.0)
res = square(a_dist)
print(f"exact: {res.mean:.2f} +/- {res.stddev:.2f}")

# 0.0 +/- 10.0
print("\n--- 0.0 +/- 10.0 ---")
a_unc = uncertainties.ufloat(0.0, 10.0)
print(f"unc: {a_unc * a_unc}")

a_dist = Dist(0.0, 10.0)
res = square(a_dist)
print(f"exact: {res.mean:.2f} +/- {res.stddev:.2f}")
