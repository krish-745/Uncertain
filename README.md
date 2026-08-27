# Uncertain

*A statically-typed arithmetic DSL where every value's type encodes its distributional uncertainty — and the compiler proves how that uncertainty compounds.*

## Why Uncertain?

When you write equations for measurements, variables are not exact values—they are distributions. If you reuse a variable without tracking its correlation, your calculated variance will be silently understated. `Uncertain` catches these correlation bugs at compile-time by enforcing dependency tracking in its type system.

## Hero Demo: Catching Correlation Bugs

```calc
let a = sensor_read();
let variance_est = a * a;
```

Running `uncertain run` catches the hidden dependency reuse:

```
error: uncertain reuse without correlation annotation
  --> line 2:20
   |
 2 | let variance_est = a * a;
   |                    ^^^^^ `a` appears twice in this product
   |
   = note: treating repeated occurrences of `a` as independent understates
     the true variance of the result.
   = help: use `square(a)` for the correct self-product variance formula,
     or wrap with `correlated(..., ..., cov = ...)` if you have an explicit
     covariance estimate.
```
