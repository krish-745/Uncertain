# Uncertain Error Catalog

This catalog documents the structured diagnostics emitted by the `uncertain` compiler and typechecker. Every error is designed to pinpoint the exact location of the issue and provide actionable help.

## `uncertain-reuse`
**Description:** The intellectual centerpiece of the language. This error is triggered when a variable (representing a probability distribution) appears in both operands of a product or ratio without an explicit correlation annotation. Treating reused variables as independent understates the true variance of the result.

**Example:**
```
let x = Normal(10.0, 2.0);
let y = x * x;
```

**Diagnostic Output:**
```text
error: uncertain reuse without correlation annotation
  --> line 2:9
   |
 2 | let y = x * x;
   |         ^^^^^ `x` appears twice in this product
   |
   = note: treating repeated occurrences of `x` as independent understates
     the true variance of the result.
   = help: use `square(x)` for the correct self-product variance formula,
     or wrap with `correlated(..., ..., cov = ...)` if you have an explicit
     covariance estimate.
```

---

## `undefined-var`
**Description:** Emitted when a variable is referenced before it has been defined using a `let` binding.

**Example:**
```
let y = x + 1.0;
```

**Diagnostic Output:**
```text
error: undefined variable `x`
  --> line 1:9
   |
 1 | let y = x + 1.0;
   |         ^ not found in scope
   |
   = note: `x` must be declared with `let` before use.
```

---

## `type-mismatch`
**Description:** Emitted when the user provides an explicit type annotation (e.g., `Normal(mean, stddev)`) for a `let` binding, but the typechecker infers a distribution with a different mean or standard deviation.

**Example:**
```
let a: Normal(10.0, 0.0) = 5.0 + 4.0;
```

**Diagnostic Output:**
```text
error: type annotation mismatch
  --> line 1:1
   |
 1 | let a: Normal(10.0, 0.0) = 5.0 + 4.0;
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ inferred type does not match annotation
   |
   = note: the distribution computed by the typechecker differs from the explicit type annotation.
```

---

## `math-domain-error`
**Description:** Emitted during typechecking/evaluation when an invalid mathematical operation is performed on a distribution, such as dividing by a distribution with a zero mean, or attempting to take the square root of a distribution with a negative mean.

**Example:**
```
let a = Normal(-5.0, 1.0);
let b = sqrt(a);
```

**Diagnostic Output:**
```text
error: math domain error
  --> line 2:9
   |
 2 | let b = sqrt(a);
   |         ^^^^^^^ invalid operation
   |
   = note: cannot compute the square root of a distribution with a negative mean (-5.0)
```

---

## `approximation-warning`
**Description:** Emitted as a warning when a mathematical operation combines variables from different distribution families (e.g., adding a Normal distribution and a Poisson distribution). The compiler falls back to a Normal approximation using moment-matching.

**Example:**
```calc
let a = Normal(10.0, 1.0);
let b = Poisson(5);
let c = a + b;
```

**Diagnostic Output:**
```text
warning: approximation-warning
  --> line 3:9
   |
 3 | let c = a + b;
   |         ^^^^^ Moment-matching approximation used for non-Normal distribution
   |
   = note: Moment-matching approximation used for non-Normal distribution
```

---

## `uncertain-branch`
**Description:** Emitted when attempting to use a probability distribution (uncertain variable) as the condition for an `if` statement or loop, or when a loop exceeds the maximum compile-time unrolling limit (1000 iterations). Branching logic must be deterministic.

**Example:**
```calc
let a = Normal(10.0, 1.0);
if (a > 5.0) {
    let b = 1;
}
```

**Diagnostic Output:**
```text
error: uncertain-branch
  --> line 2:5
   |
 2 | if (a > 5.0) {
   |     ^^^^^^^ Cannot branch on a non-deterministic condition
   |
   = note: Cannot branch on a non-deterministic condition
```
