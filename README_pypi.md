# Uncertain — The Uncertainty-Aware Programming Language

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/uncertain-lang/) [![PyPI](https://img.shields.io/badge/PyPI-uncertain--lang-F800D7?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/uncertain-lang/) [![Pytest](https://img.shields.io/badge/Pytest-tested-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/krish-745/Uncertain) [![Hypothesis](https://img.shields.io/badge/Hypothesis-fuzz%20tested-6B5B95?style=for-the-badge&logo=python&logoColor=white)](https://github.com/krish-745/Uncertain) [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

> **A statically-typed arithmetic DSL where every value's type encodes its distributional uncertainty — and the compiler proves how that uncertainty compounds.**

**Source & docs on GitHub:** [github.com/krish-745/Uncertain](https://github.com/krish-745/Uncertain)

---

## Table of Contents

- [Why Uncertain?](#why-uncertain)
- [Quick Start](#quick-start)
- [The Hero Demo](#the-hero-demo)
- [Head-to-Head Comparison](#head-to-head-comparison)
- [Language Guide](#language-guide)
- [Testing](#testing)

---

## Why Uncertain?

When you write equations for physical measurements, sensor data, or statistical variables, those values are almost never exact — they are probability distributions.

A silent, common bug arises when you reuse a variable without tracking its mathematical correlation. For example, writing `a * a` in a normal programming language (or using a runtime uncertainty library) treats the two occurrences of `a` as if they were independent measurements. This **silently understates the true variance**, leading to overconfidence in your results.

**`Uncertain` eliminates these correlation bugs entirely using compile-time Affine Arithmetic.** The compiler tracks partial dependencies between variables, allowing it to accurately simulate covariance for reused variables without requiring manual annotations.

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **[uv](https://astral.sh/uv/)** — fast Python package and environment manager (recommended)

### Installation

```bash
# Using pip
pip install uncertain-lang

# Using uv (recommended)
uv tool install uncertain-lang
```

To install from source:

```bash
git clone https://github.com/krish-745/Uncertain.git
cd Uncertain
pip install .
```

### Running a Script

Once installed, the `uncertain` CLI is available on your `PATH`. Pass it any `.calc` file:

```bash
uncertain my_experiment.calc
```

---

## The Hero Demo

### Automatic Covariance Tracking

Consider this simple program:

```calc
let a = sensor_read();
let variance_est = a * a;
let diff = a - a;
```

> **Note:** `sensor_read()` is a built-in that returns a `Normal(10.0, 1.0)` distribution — a sensor reading with a mean of 10 and a standard deviation of 1.

Running `uncertain` on this file perfectly tracks the dependency reuse. Instead of naively treating the two `a`s as independent, it automatically calculates the correct exact variance for `a * a`, and precisely evaluates `a - a` as `0.0` with `0.0` variance!

### Probability Queries

You can explicitly evaluate the probability of events or confidence intervals at compile time using the `prob(...)` built-in:

```calc
let a = sensor_read();
let b = sensor_read();
let is_a_bigger = prob(a > b);
```
The compiler calculates the exact normal CDF distribution to return the exact probability scalar.

### Catching Runtime Math Errors at Compile Time

Beyond correlation tracking, `uncertain` uses its diagnostic system to catch mathematical domain errors **before your program ever evaluates**. For example, taking the square root of a distribution with a negative mean:

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

> **See also:** For the full list of compiler diagnostics, visit the [Error Catalog on GitHub](https://github.com/krish-745/Uncertain/blob/main/docs/error-catalog.md).

---

## Head-to-Head Comparison

### `uncertain` vs. Python's `uncertainties`

Python's popular [`uncertainties`](https://pythonhosted.org/uncertainties/) package is a fantastic tool, but it operates entirely at *runtime* using **linear approximations** (the first-order Taylor / Delta method).

To see the difference, clone the repo and run the included comparison script, which computes `a * a` where `a = 2.0 ± 5.0`:

```bash
uv run python scripts/compare_uncertainties.py
```

**Output:**

```text
--- Head-to-Head Comparison: Self-Multiplication (a * a) ---
We have a sensor reading: a = 2.0 ± 5.0
What is the variance of a * a?

1. Naive Hand Calculation (Assuming Independence):
   Result: 4.00 ± 14.14
   (DANGEROUS: Silently understates variance by ignoring correlation)

2. Python's `uncertainties` package (a * a):
   Result: 4.00 ± 20.00
   (BETTER: Detects correlation, but uses linear Taylor approximation, dropping higher-order terms. Notice the mean is completely wrong!)

3. Uncertain DSL:
   Result: 29.00 ± 40.62
   (PERFECT: Compiler tracked the affine lineage and injected lost non-linear variance to match the EXACT mathematical formula for E[X²] and Var(X²).)
```

`uncertainties` computes a linear approximation of `a * a` resulting in a heavily understated mean and variance. `uncertain`'s affine tracking engine automatically injects the lost non-linear variance bounds to compute the *exact* mathematical formula for the squared Normal distribution:

```
E[X²]   = μ² + σ²
Var(X²) = 2σ⁴ + 4μ²σ²
```

---

## Language Guide

### 11 Statistical Distributions

`Uncertain` ships with a comprehensive built-in type system of 11 named distributions (plus the special `Exact` scalar type). Annotate variables with exactly the distribution that models your data — the type-checker automatically calculates their mathematical mean and variance.

| Distribution | Type Annotation | Read Function | Notes |
|---|---|---|---|
| Normal | `Normal(μ, σ)` | `sensor_read()` | Default for `sensor_read()` |
| LogNormal | `LogNormal(μ, σ)` | `lognormal_read(μ, σ)` | |
| Gamma | `Gamma(α, β)` | `gamma_read(α, β)` | Shape, Scale |
| Uniform | `Uniform(a, b)` | `uniform_read()` | |
| Exponential | `Exponential(λ)` | `exponential_read(λ)` | |
| Poisson | `Poisson(λ)` | `poisson_read(λ)` | Discrete |
| Binomial | `Binomial(n, p)` | `binomial_read(n, p)` | Discrete |
| Bernoulli | `Bernoulli(p)` | `bernoulli_read(p)` | Discrete |
| Geometric | `Geometric(p)` | `geometric_read(p)` | Discrete |
| NegativeBinomial | `NegativeBinomial(r, p)` | `negbinom_read(r, p)` | Discrete |
| Empirical | `Empirical([...])` | `empirical_read([...])` | Discrete, custom data |

```calc
// Continuous distributions
let normal:  Measured<Normal(10.0, 1.0)>  = sensor_read();
let price:   Measured<LogNormal(1, 0.5)>  = lognormal_read(1, 0.5);
let wear:    Measured<Gamma(2, 2.5)>      = gamma_read(2, 2.5);
let timeout: Measured<Uniform(0, 10)>     = uniform_read();
let failure: Measured<Exponential(0.1)>   = exponential_read(0.1);

// Discrete distributions
let clicks:  Measured<Poisson(5)>               = poisson_read(5);
let success: Measured<Binomial(100, 0.9)>       = binomial_read(100, 0.9);
let flag:    Measured<Bernoulli(0.5)>           = bernoulli_read(0.5);
let trials:  Measured<Geometric(0.2)>           = geometric_read(0.2);
let batch:   Measured<NegativeBinomial(5, 0.5)> = negbinom_read(5, 0.5);
let custom:  Measured<Empirical([1, 5, 9])>     = empirical_read([1, 5, 9]);
```

### Compile-Time Control Flow & Mutability

The language supports block scoping, mutable variables (`var`), arrays (`[...]`), `while` and `for` loops, and `if/else` branching. The compiler seamlessly tracks dependency lineages across block reassignments.

> **Important:** `Uncertain` enforces a strict separation between random variables and control-flow. Because loops and branches are **unrolled and evaluated entirely at compile-time**, you **cannot** branch on an uncertain variable (e.g., `stddev > 0`). Branching is restricted to deterministic values such as loop counters. Violating this rule causes the compiler to emit an `uncertain-branch` error.

```calc
let sensors = [normal, price, wear];
var sum = 0;

for (var i = 0; i < 3; i = i + 1) {
    if (i < 2) {
        sum = sum + normal; // Compiler accurately tracks each branch!
    } else {
        sum = sum + price;
    }
}
```

### Math & Operations

Combine independent measurements using standard arithmetic. The compiler propagates means and standard deviations automatically using the Delta method.

```calc
let w = sensor_read();
let h = sensor_read();

let perimeter = w + w + h + h;  // addition: σ² summed in quadrature
let area      = w * h;          // independent multiplication: Delta-method
let ratio     = w / h;          // independent division: Delta-method
let root      = sqrt(w);        // sqrt: Delta-method propagation
```

> **Approximation Warning:** When you combine variables from different distribution families (e.g., `Normal` + `Poisson`), the compiler falls back to a **Normal approximation via moment-matching** and emits an `approximation-warning`, so you are never silently affected by precision trade-offs.

### Structs & Records

Group data logically using struct literals:

```calc
let gps_coords = { x: sensor_read(), y: sensor_read() };
let total_dist = sqrt(gps_coords.x * gps_coords.x + gps_coords.y * gps_coords.y);
```

### Modules & Imports

Build reusable libraries of constants and equations using the `import` statement. Imports are evaluated at compile time and exposed as structs.

```calc
// In physics.calc
let g = 9.81;

// In main.calc
import physics as phys;
let acceleration = phys.g;
```

### Functions & Arrays

`Uncertain` supports custom function definitions and first-class arrays. Functions are evaluated dynamically during type-checking.

```calc
fn compute_risk(base_risk: Measured<Normal(0,1)>, multiplier) -> Measured<Normal> {
    return base_risk * multiplier;
}

let risks = [sensor_read(), sensor_read(), sensor_read()];

// Higher-order array functions evaluate at compile-time:
let mapped = map(risks, compute_risk);
let total_risk = reduce(mapped, add_risks, 0.0);
```

### Language Server (LSP)

The compiler ships with a built-in Language Server to provide real-time diagnostic squiggles and hover information directly in your editor (like VS Code, Neovim, etc.).

To start the LSP server, simply run:
```bash
uncertain --lsp
```

### CLI Reference

```
uncertain <file.calc> [--check-only]
```

| Flag | Description |
|---|---|
| `<file.calc>` | Path to the `.calc` source file to compile and evaluate |
| `--check-only` | Run only the type-checker; do not evaluate. Exits `0` on success, `1` on errors. |

**Type-check only (no evaluation):**
```bash
uncertain my_experiment.calc --check-only
```

**Full evaluation:**
```bash
uncertain my_experiment.calc
```

A successful run prints each named binding and its inferred distribution:

```text
perimeter = (mean=40.0000, stddev=2.8284)
area      = (mean=100.0000, stddev=2.0000)
...
```

---

## Testing

The project has a robust, property-based test suite covering the full compiler pipeline.

| Test Area | Description |
|---|---|
| **Lexer & Parser** | Fuzz-tested with thousands of randomly generated inputs via `hypothesis` to ensure zero unhandled exceptions |
| **Type-checker** | Property tests for bidirectional inference, dependency tracking, and diagnostic correctness |
| **Distribution Math** | Hypothesis-driven property tests validating analytic formulas for all operations |
| **Performance** | Type-checker dependency-union performance is validated against large, generated programs |

Clone the repository and run the full test suite:

```bash
# Windows
.\test.bat

# Linux / macOS
uv run pytest tests/
```

Run the Monte Carlo cross-validation (proves that analytic Delta-method formulas match empirical random sampling):

```bash
uv run python scripts/monte_carlo_report.py
```

---

**Built with ❤️ to keep data precise**
