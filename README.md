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

Clone the repository and you're ready to go!
```bash
git clone https://github.com/yourusername/uncertain.git
cd uncertain
```

### Running Scripts
We've included handy wrapper scripts that automatically manage dependencies using `uv`.

Run an example calculation:
```bash
# Windows
.\run.bat examples/my_experiment.calc

# Linux/macOS
uv run python -m uncertain_lang.cli examples/my_experiment.calc
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

The project has a robust test suite covering lexical analysis, recursive-descent parsing, bidirectional type checking, and hypothesis-driven property tests for the distribution math.

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
