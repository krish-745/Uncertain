<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyPI-uncertain--lang-F800D7?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI" />
  <img src="https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge&logo=python&logoColor=white" alt="uv" />
  <img src="https://img.shields.io/badge/Pytest-tested-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest" />
  <img src="https://img.shields.io/badge/Hypothesis-fuzz%20tested-6B5B95?style=for-the-badge&logo=python&logoColor=white" alt="Hypothesis" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
</p>

# Uncertain — The Uncertainty-Aware Programming Language

> **A statically-typed arithmetic DSL where every value's type encodes its distributional uncertainty — and the compiler proves how that uncertainty compounds.**

**Available on PyPI:** [pypi.org/project/uncertain-lang/](https://pypi.org/project/uncertain-lang/)

---

## Table of Contents

- [Why Uncertain?](#why-uncertain)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running a Script](#running-a-script)
- [The Hero Demo](#the-hero-demo)
  - [Catching Correlation Bugs](#catching-correlation-bugs)
  - [Catching Math Errors at Compile Time](#catching-runtime-math-errors-at-compile-time)
- [Head-to-Head Comparison](#head-to-head-comparison)
- [Language Guide](#language-guide)
  - [11 Statistical Distributions](#11-statistical-distributions)
  - [Control Flow & Mutability](#compile-time-control-flow--mutability)
  - [Math & Operations](#math--operations)
  - [Safe Variable Reuse](#safe-variable-reuse)
  - [CLI Reference](#cli-reference)
- [Architecture](#architecture)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Why Uncertain?

When you write equations for physical measurements, sensor data, or statistical variables, those values are almost never exact — they are probability distributions.

A silent, common bug arises when you reuse a variable without tracking its mathematical correlation. For example, writing `a * a` in a normal programming language (or using a runtime uncertainty library) treats the two occurrences of `a` as if they were independent measurements. This **silently understates the true variance**, leading to overconfidence in your results.

**`Uncertain` catches these correlation bugs at compile-time by enforcing dependency tracking in its type system.** Instead of a wrong answer, you get a pinpointed compiler error.

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **[uv](https://astral.sh/uv/)** — fast Python package and environment manager (recommended)

### Installation

The compiler is published on PyPI and can be installed globally:

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
uncertain examples/my_experiment.calc
```

---

## The Hero Demo

### Catching Correlation Bugs

Consider this simple program:

```calc
let a = sensor_read();
let variance_est = a * a;
```

> **Note:** `sensor_read()` is a built-in that returns a `Normal(10.0, 1.0)` distribution — a sensor reading with a mean of 10 and a standard deviation of 1.

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

Beyond correlation checking, `uncertain` uses the same diagnostic system to catch mathematical domain errors **before your program ever evaluates**. For example, taking the square root of a distribution with a negative mean:

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

> **See also:** For the full list of all diagnostics emitted by the compiler, see the [Error Catalog](docs/error-catalog.md).

---

## Head-to-Head Comparison

### `uncertain` vs. Python's `uncertainties`

Python's popular [`uncertainties`](https://pythonhosted.org/uncertainties/) package is a fantastic tool, but it operates entirely at *runtime* using **linear approximations** (the first-order Taylor / Delta method).

To see exactly why a compiler-enforced approach is safer and more precise, we include a script that computes `a * a` where `a = 2.0 ± 5.0`:

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

3. Uncertain DSL (forces `square(a)` at compile time):
   Result: 29.00 ± 40.62
   (PERFECT: Compiler caught the reuse, forced explicit intent, and used the exact higher-order formula.)
```

This single comparison proves the core thesis: `uncertainties` will happily compute a linear approximation of `a * a` without warning you. `uncertain` throws a **compile-time error**, forcing you to explicitly choose `square(a)`, which applies the *exact* mathematical formula for the variance of a squared Normal distribution:

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

### Safe Variable Reuse

When you need to multiply a variable by itself or combine two known-correlated variables, use the mathematically safe built-in functions to bypass the `uncertain-reuse` error:

```calc
// For perfect self-correlation (squaring a variable):
// Uses the exact formula E[X²] = μ² + σ²  and  Var(X²) = 2σ⁴ + 4μ²σ²
let w_squared = square(w);

// For two distinct variables with a known covariance:
// Uses the full Var(XY) formula including the covariance term
let correlated_area = correlated(w, h, cov=0.5);
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
uncertain examples/my_experiment.calc --check-only
```

**Full evaluation:**
```bash
uncertain examples/demo.calc
```

A successful run prints each named binding and its inferred distribution:

```text
perimeter = (mean=40.0000, stddev=2.8284)
area      = (mean=100.0000, stddev=2.0000)
...
```

---

## Architecture

The compiler is structured as a classic pipeline. All source lives under `src/uncertain/`.

```
src/uncertain/
├── cli.py            — Entry point; argument parsing and pipeline orchestration
├── lexer.py          — Tokeniser; converts .calc source text into a token stream
├── parser.py         — Recursive-descent parser; produces a typed AST
├── ast_nodes.py      — AST node dataclasses
├── typechecker.py    — Bidirectional type-checker; infers and propagates distributions,
│                       tracks dependency lineages, and emits structured diagnostics
├── dependency.py     — Dependency-set algebra used by the type-checker for correlation tracking
├── distributions.py  — Core distribution math (Delta-method, square, sqrt, correlated product)
├── diagnostics.py    — Structured diagnostic formatting (Rust-style error messages)
└── evaluator.py      — Final evaluation pass; produces runtime values from the typed AST
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

**Run the full test suite:**

```bash
# Windows
.\test.bat

# Linux / macOS
uv run pytest tests/
```

**Run the Monte Carlo cross-validation** (proves that analytic Delta-method formulas match empirical random sampling):

```bash
uv run python scripts/monte_carlo_report.py
```

**Run the head-to-head comparison:**

```bash
uv run python scripts/compare_uncertainties.py
```

---

## Contributing

Contributions are welcome! To get started:

1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/krish-745/Uncertain.git
   cd Uncertain
   uv sync --all-extras
   ```
2. Make your changes and ensure all tests pass:
   ```bash
   uv run pytest tests/
   ```
3. Open a pull request with a clear description of the change and why it's needed.

Please open an issue first for any significant feature additions or breaking changes.

---

<p align="center">
  <b>Built with ❤️ to keep data precise</b>
</p>
