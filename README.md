<h1 align="center">Uncertain</h1>

<p align="center">
  <b>A small language where every number carries its uncertainty,<br>and the compiler works out exactly how that uncertainty propagates.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/uncertain-lang/"><img src="https://img.shields.io/pypi/v/uncertain-lang?color=F800D7" alt="PyPI version"></a>
  <a href="https://pypi.org/project/uncertain-lang/"><img src="https://img.shields.io/pypi/pyversions/uncertain-lang" alt="Python versions"></a>
  <a href="https://github.com/krish-745/Uncertain/actions/workflows/smoke-test.yml"><img src="https://github.com/krish-745/Uncertain/actions/workflows/smoke-test.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/krish-745/Uncertain/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>

```calc
let width  = sensor_read();          // Normal(10, 1): a measurement with noise
let area   = width * width;          // the compiler knows both factors are the same variable
let p_big  = prob(area > 110);       // probability, computed at compile time
```

```text
$ uncertain area.calc
width = (mean=10.0000, stddev=1.0000)
area = (mean=101.0000, stddev=20.0499)
p_big = (mean=0.3268, stddev=0.0000)
```

Values in `Uncertain` are probability distributions, not single numbers. The type checker tracks each value's mean, standard deviation and **correlation with every other value**, so reusing a variable, as in `width * width`, gives the mathematically correct answer instead of silently understating the uncertainty.

---

## Why?

Measurements, sensor readings and estimates are never exact. The classic way to propagate their uncertainty by hand, or with a runtime library, treats every operand as independent. That quietly breaks as soon as a value is used twice:

| Computing `a * a` for `a = 2.0 ± 5.0` | Result | |
|---|---|---|
| Naive propagation (assumes independence) | `4.00 ± 14.14` | variance understated |
| Python's [`uncertainties`](https://pythonhosted.org/uncertainties/) (linear approximation) | `4.00 ± 20.00` | mean and variance wrong |
| **Uncertain** | **`29.00 ± 40.62`** | exact: E[X²] = μ² + σ², Var(X²) = 2σ⁴ + 4μ²σ² |

`Uncertain` tracks dependencies between values with **affine arithmetic** at compile time. It uses exact moment formulas wherever they exist, and it rejects programs whose results would be meaningless (dividing by a distribution centred on zero, branching on an uncertain value, impossible distribution parameters) with precise, Rust-style diagnostics.

*The comparison above comes from [`scripts/compare_uncertainties.py`](https://github.com/krish-745/Uncertain/blob/main/scripts/compare_uncertainties.py).*

---

## Install

Requires Python 3.10 or newer.

```bash
pip install uncertain-lang        # or: uv tool install uncertain-lang
```

Then run any `.calc` file:

```bash
uncertain my_experiment.calc
```

To work on the compiler itself:

```bash
git clone https://github.com/krish-745/Uncertain.git
cd Uncertain
uv sync --all-extras
uv run uncertain examples/my_experiment.calc
```

---

## A quick tour

### Correlations are tracked automatically

```calc
let a = sensor_read();
let b = sensor_read();       // an independent reading

let square_a = a * a;        // (mean=101, stddev=20.05): exact, a is correlated with itself
let diff     = a - a;        // (mean=0,   stddev=0): exactly zero
let product  = a * b;        // (mean=100, stddev=14.18): independent factors
```

Every uncertain value remembers which sources of randomness it depends on, and how strongly. `a * a`, `a - a`, `(a + b) * a` and values passed through functions all come out right without any annotations.

### Probability queries

```calc
let temp      = normal_read(21.5, 0.8);
let threshold = 23.0;
let p_hot     = prob(temp > threshold);   // 0.0304
let p_warm    = prob(temp - 21.5 >= 1);   // 0.1056
```

`prob()` accepts `<`, `>`, `<=` and `>=`. It computes the distribution of the difference between both sides, including any correlation between them, and evaluates the normal CDF.

### Mistakes are caught before anything runs

```calc
let a = sensor_read() - 15.0;
let b = sqrt(a);
```

```text
error: math domain error
  --> line 2:9
   |
 2 | let b = sqrt(a);
   |         ^^^^^^^ invalid operation
   |
   = note: cannot compute the square root of a distribution with a negative mean (-5.0)
```

### Types describe distributions

```calc
let reading: Measured<Normal(10.0, 2.0)> = sensor_read();
```

```text
error: type annotation mismatch
  --> line 1:1
   |
 1 | let reading: Measured<Normal(10.0, 2.0)> = sensor_read();
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ inferred type does not match annotation
   |
   = note: `reading` is annotated as Normal(10, 2) (mean 10, stddev 2), but the inferred distribution has mean 10, stddev 1
```

---

## Language guide

### Values and variables

```calc
let g = 9.81;                    // numbers: 1, 1.5, .5, 1e-3
var count = 0;                   // `var` signals the value will be reassigned
count = count + 1;
let noisy = normal_read(0, 2);   // an uncertain value
// comments start with //
```

A value with standard deviation 0 is *deterministic*. Deterministic values can be used anywhere; uncertain values are rejected where a fixed number is required (loop conditions, array indices, distribution parameters).

### Distributions

Create uncertain values with a `*_read` function, and optionally state the expected distribution with a `Measured<...>` annotation. Annotations are checked against the inferred mean and standard deviation (to 0.1%).

| Family | Annotation | Constructor | Notes |
|---|---|---|---|
| Normal | `Normal(μ, σ)` | `normal_read(μ, σ)` | `sensor_read()` is `Normal(10, 1)` |
| Uniform | `Uniform(a, b)` | `uniform_read()` | `uniform_read()` is `Uniform(0, 10)` |
| LogNormal | `LogNormal(μ, σ)` | `lognormal_read(μ, σ)` | μ, σ of the underlying normal |
| Gamma | `Gamma(k, θ)` | `gamma_read(k, θ)` | shape k, scale θ |
| Exponential | `Exponential(λ)` | `exponential_read(λ)` | rate λ |
| Poisson | `Poisson(λ)` | `poisson_read(λ)` | discrete |
| Binomial | `Binomial(n, p)` | `binomial_read(n, p)` | discrete; integer n |
| Bernoulli | `Bernoulli(p)` | `bernoulli_read(p)` | discrete |
| Geometric | `Geometric(p)` | `geometric_read(p)` | discrete; trials up to and including the first success (mean 1/p) |
| NegativeBinomial | `NegativeBinomial(r, p)` | `negbinom_read(r, p)` | discrete; successes before the r-th failure (mean pr/(1−p)) |
| Empirical | `Empirical([...])` | `empirical_read([...])` | mean and (population) stddev of your data |
| Exact | `Exact(v)` | any deterministic value | stddev 0 |

```calc
let price: Measured<LogNormal(1, 0.5)>          = lognormal_read(1, 0.5);
let wear:  Measured<Gamma(2, 2.5)>              = gamma_read(2, 2.5);
let hits:  Measured<Binomial(100, 0.9)>         = binomial_read(100, 0.9);
let data:  Measured<Empirical([-1.5, 2, 4.5])>  = empirical_read([-1.5, 2, 4.5]);
```

Parameters can be any deterministic expression, including variables and negative numbers. Invalid parameters, such as `poisson_read(-1)` or `binomial_read(10, 1.5)`, are compile errors.

### Arithmetic and math functions

| Operation | Accuracy for Normal inputs |
|---|---|
| `a + b`, `a - b`, scaling by a constant | exact, for every family |
| `a * b` (with any correlation), `square(a)`, `pow(a, 3)` | exact |
| `exp(a)`, `sin(a)`, `cos(a)`, `abs(a)` | exact |
| `a / b`, `sqrt(a)`, `log(a)`, `pow(a, n)` for other n | first-order (delta method) |

The compiler tells you when a result is approximate:

- a `delta-method-warning` when a first-order result is unreliable, because the input's standard deviation is large compared with its mean;
- an `approximation-warning` when a non-Normal value (Uniform, Poisson, ...) is multiplied, divided, passed to a nonlinear function or used in `prob()`. Sums of any family are always exact.

If two *independent* readings are known to be correlated, say so explicitly:

```calc
let x = normal_read(20, 3);
let y = normal_read(4, 0.5);
let xy = correlated(x, y, cov=0.5);   // product of x and y with Cov(x, y) = 0.5
```

### Control flow

Loops and branches are **unrolled at compile time**, so their conditions must be deterministic. Branching on an uncertain value is an `uncertain-branch` error; use `prob()` to reason about it instead.

```calc
let readings = [sensor_read(), sensor_read(), normal_read(12, 2)];
var total = 0;

for (var i = 0; i < 3; i = i + 1) {
    if (i != 1) {
        total = total + readings[i];
    }
}
// total = (mean=22, stddev=2.2361)

var n = 1;
while (n < 100) {
    n = n * 2;
}
```

Conditions support `<`, `>`, `<=`, `>=`, `==` and `!=`. A loop may run at most 1000 iterations by default (`--max-unroll N` changes this). Variables declared inside a block remain visible after it.

### Arrays and structs

```calc
let samples = [sensor_read(), sensor_read(), sensor_read()];
let first   = samples[0];                  // indices must be deterministic integers

let point = { x: normal_read(3, 0.1), y: normal_read(4, 0.1) };
let dist  = sqrt(square(point.x) + square(point.y));
```

### Functions

```calc
fn scale(v: Measured<Normal(10, 1)>) -> Measured<Normal(20, 2)> {
    return v * 2.0;
}

fn add(total, v) {
    return total + v;
}

fn is_large(v) {
    return v > 15;                          // must be deterministic for filter()
}

let readings = [sensor_read(), sensor_read(), sensor_read()];
let scaled   = map(readings, scale);        // 3 x (mean=20, stddev=2)
let total    = reduce(scaled, add, 0.0);    // (mean=60, stddev=3.4641)
let large    = filter([10, 20, 30], is_large);   // [20, 30]
```

Functions are evaluated at every call site during type checking, and argument and return annotations are checked on each call. `map` and `filter` take a one-argument function, `reduce` a two-argument function and an initial value. Recursion works as long as it stops after a bounded number of deterministic steps (at most 64 nested calls).

### Modules

```calc
// physics.calc
let g = 9.81;
let drag = normal_read(0.47, 0.02);
```

```calc
// main.calc
import physics as phys;                     // resolved relative to main.calc
let weight = 70 * phys.g;
```

An import exposes the module's top-level values as a struct. Nested paths such as `import lib.physics as p;` load `lib/physics.calc`.

---

## Diagnostics

| Code | Meaning |
|---|---|
| `syntax-error` | the source could not be parsed |
| `undefined-var` | a variable is used before it is declared |
| `type-mismatch` | a `Measured<...>` annotation does not match the inferred distribution |
| `math-domain-error` | an undefined operation or invalid distribution parameters |
| `uncertain-branch` | a condition or comparison depends on an uncertain value |
| `not-constant` | a parameter, exponent, covariance or index must be deterministic |
| `invalid-operand` | wrong kind of value, e.g. arithmetic on an array or a missing struct field |
| `unknown-function` / `arity-mismatch` | calling a function that doesn't exist, or with the wrong arguments |
| `invalid-return` | a function without `return`, or `return` outside a function |
| `loop-limit` / `recursion-limit` | a loop or recursion that doesn't terminate at compile time |
| `import-error` | a module could not be found, parsed or checked, or imports itself |
| `approximation-warning` / `delta-method-warning` | *(warnings)* the result is an approximation |

Run `uncertain --explain <code>` for details, or see the [error catalog](https://github.com/krish-745/Uncertain/blob/main/docs/error-catalog.md) for an example of each.

---

## Tooling

### Command line

```text
uncertain FILE [--check-only] [--output text|json] [--max-unroll N]
uncertain --explain CODE
uncertain --lsp
uncertain --version
```

| Option | Description |
|---|---|
| `FILE` | the `.calc` program to check; imports are resolved relative to it |
| `--check-only` | report diagnostics without printing values |
| `--output json` | print one JSON document: `{"status", "diagnostics", "values", "result"}` |
| `--max-unroll N` | maximum loop iterations (default 1000) |
| `--explain CODE` | explain a diagnostic code |
| `--lsp` | start the language server on stdio |

The CLI prints every top-level value. If the program ends with an expression without a trailing `;`, that value is printed after `=>`. Exit codes: `0` success (warnings allowed), `1` errors in the program, `2` internal compiler error.

### Editor support

`uncertain --lsp` is a Language Server that provides live diagnostics, plus hover showing the inferred distribution of the variable or struct field under the cursor. Any editor with a generic LSP client can use it. For example, in Neovim:

```lua
vim.api.nvim_create_autocmd({ "BufEnter" }, {
  pattern = "*.calc",
  callback = function()
    vim.lsp.start({ name = "uncertain", cmd = { "uncertain", "--lsp" } })
  end,
})
```

### Browser playground

[`docs/playground.html`](https://github.com/krish-745/Uncertain/blob/main/docs/playground.html) runs the compiler in the browser via Pyodide, with no install. From a clone of the repository:

```bash
python -m http.server 8000
# then open http://localhost:8000/docs/playground.html
```

---

## How it works

Every uncertain value is represented as an affine form over independent standard-normal noise sources:

```
X = μ + c₁·ε₁ + c₂·ε₂ + ... + cₙ·εₙ
```

- Each `*_read()` call introduces a fresh noise source εₖ.
- Linear operations combine the coefficients exactly, so every variance and covariance, and effects like `a - a = 0`, follow directly.
- Nonlinear operations (`*`, `exp`, `sin`, ...) use exact moment formulas for the result's mean and variance. Their linear part uses the slope E[f′(X)], which by Stein's lemma reproduces the exact covariance with the inputs. Any remaining variance becomes a new independent noise source, so later operations see the correct total spread and correlation.

Because everything is computed from the program text, loops and branches are unrolled and functions are expanded at compile time. The checked program is its own answer: there is no separate runtime.

### Limitations

- **Distribution shape:** the compiler tracks means, variances and covariances, not full distribution shapes. After a nonlinear operation, results are summarised by their moments, and `prob()` assumes the compared difference is normal.
- **Control flow:** it must be deterministic. You can't branch on uncertain values (by design).
- **Variables:** `let` and `var` currently behave identically; the distinction documents intent but reassigning a `let` is not yet an error.
- **Function scope:** functions see the caller's variables (dynamic scope), and functions defined in an imported module are not exported, only its values.

---

## Development

```
src/uncertain/
├── cli.py            command-line entry point
├── lexer.py          source text → tokens
├── parser.py         tokens → AST (ast_nodes.py)
├── typechecker.py    inference, affine dependency tracking, control-flow unrolling;
│                     check_program() is the entry point for the CLI, LSP and playground
├── distributions.py  moment formulas for named families, products and nonlinear functions
├── dependency.py     affine-form algebra
├── diagnostics.py    diagnostic kinds and Rust-style rendering
├── output.py         text and JSON rendering of values
└── server.py         language server (diagnostics and hover)
```

```bash
uv run pytest tests/                           # full test suite
uv run python scripts/monte_carlo_report.py    # analytic formulas vs. random sampling
uv run python scripts/gen_error_catalog.py     # regenerate docs/error-catalog.md
```

The test suite includes:

- unit and regression tests for every language feature;
- property-based fuzzing (via Hypothesis) of the lexer, parser and whole programs, checking that the compiler never crashes;
- Monte Carlo cross-checks of every analytic formula and of correlations through nonlinear functions;
- golden tests of the rendered diagnostics;
- tests of the language server.

CI runs the suite on Python 3.10 to 3.13.

Contributions are welcome. Please open an issue first for significant changes, and make sure `uv run pytest tests/` passes. See [CHANGELOG.md](https://github.com/krish-745/Uncertain/blob/main/CHANGELOG.md) for release history.

## License

[MIT](https://github.com/krish-745/Uncertain/blob/main/LICENSE)
