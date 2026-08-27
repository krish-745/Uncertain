# Uncertain

*A statically-typed arithmetic DSL where every value's type encodes its distributional uncertainty — and the compiler proves how that uncertainty compounds.*

## Why Uncertain?

When you write equations for physical measurements, sensor data, or statistical variables, those values are almost never exact—they are probability distributions. 

If you reuse a variable in a normal programming language without explicitly tracking its mathematical correlation (e.g. `a * a`), your calculated variance will be silently understated, leading to overconfidence in faulty data. 

**`Uncertain` catches these correlation bugs at compile-time by enforcing dependency tracking in its type system.**

---

## Quick Start

### Prerequisites
- [uv](https://astral.sh/uv/) (Fast Python package and environment manager)
- Python 3.12+

### Installation

The compiler is available on PyPI. You can install it globally via `pip` or `uv`:
```bash
pip install uncertain
# or
uv tool install uncertain
```

If you want to install from source:
```bash
git clone https://github.com/yourusername/uncertain.git
cd uncertain
pip install .
```

### Running Scripts

Run an example calculation using the newly installed CLI:
```bash
uncertain examples/my_experiment.calc
```

---

## The Hero Demo: Catching Correlation Bugs

Consider this simple program:

```calc
let a = sensor_read();
let variance_est = a * a;
```

Running `uncertain` on this file immediately catches the hidden dependency reuse:

```text
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

### Catching Runtime Math Errors at Compile Time
Beyond type checking, `uncertain` uses the same beautiful diagnostic system to catch mathematical domain errors before your program even evaluates. For example, taking the square root of a distribution with a negative mean:

```calc
let a = sensor_read() - 15.0;
let b = sqrt(a);
```

Yields a pinpointed math-domain error:
```text
error: math domain error
  --> line 2:9
   |
 2 | let b = sqrt(a);
   |         ^^^^^^^ invalid operation
   |
   = note: cannot compute the square root of a distribution with a negative mean (-5.0)
```

> **Note:** For a comprehensive list of all diagnostics emitted by the compiler, check out the [Error Catalog](docs/error-catalog.md).

---

## Head-to-Head: `uncertain` vs. Python's `uncertainties`

Python's popular `uncertainties` package is fantastic, but it operates entirely at *runtime* using linear approximations (the Delta method). 

To see exactly why a compiler-enforced approach is safer and more precise, we've included a script that computes `a * a` where `a = 10.0 ± 2.0`.

Run it yourself:
```bash
uv run python scripts/compare_uncertainties.py
```

**The Output:**
```text
1. Naive Hand Calculation (Assuming Independence):
   Result: 100.00 ± 28.28
   (DANGEROUS: Silently understates variance by ignoring correlation)

2. Python's `uncertainties` package (a * a):
   Result: 100.00 ± 40.00
   (BETTER: Detects correlation, but uses linear Taylor approximation, dropping higher-order terms.)

3. Uncertain DSL (forces `square(a)` at compile time):
   Result: 100.00 ± 40.40
   (PERFECT: Compiler caught the reuse, forced explicit intent, and used the exact higher-order formula.)
```

This single comparison proves the core thesis of the project: `uncertainties` will happily compute a linear approximation of `a * a` without telling you. `uncertain` will throw a **compile-time error**, forcing you to explicitly choose `square(a)`, which in turn applies the *exact* mathematical formula for the variance of a squared distribution.

---

## Language Guide

### Declaring Variables
All inputs are declared using the built-in `sensor_read()` function. The compiler infers the uncertainty, but you can also provide explicit type annotations to ensure your expectations match reality.

```calc
// Inferred type
let width = sensor_read(); 

// Explicitly type-checked distribution
let length: Measured<Normal(10.0, 1.0)> = sensor_read();
```

### Math & Operations
You can safely combine independent measurements using standard arithmetic. The compiler propagates the mean and standard deviation automatically using the Delta-method.

```calc
let w = sensor_read();
let h = sensor_read();

let perimeter = w + w + h + h;
let area = w * h;
let ratio = w / h;
let root = sqrt(w);
```

### Safe Variable Reuse
If you *must* multiply correlated variables, you must use mathematically safe functions provided by the language to bypass the compiler error:

```calc
// For perfect self-correlation (squaring a variable)
let w_squared = square(w);

// If you have a known covariance estimate for two distinct variables
let correlated_area = correlated(w, h, cov=0.5);
```

---

## Testing

The project has a highly robust, fuzzed test suite covering lexical analysis, recursive-descent parsing, bidirectional type checking, and hypothesis-driven property tests for the distribution math.

- **Fuzzing**: The parser and typechecker are subjected to thousands of randomly generated inputs and deep AST structures via `hypothesis` to ensure zero unhandled exceptions.
- **Performance**: The typechecker dependency-union performance is strictly validated against large generated programs.

To run the full test suite:
```bash
# Windows
.\test.bat

# Linux/macOS
uv run pytest tests/
```

To run the Monte Carlo simulation cross-validation (which proves our analytic Delta-method formulas match empirical random sampling):
```bash
uv run python scripts/monte_carlo_report.py
```
