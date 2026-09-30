# Changelog

## 1.0.1 (2026-09-30)

Documentation release; no changes to the compiler.

- Rewrote the README: a quick tour with real output, a complete language guide, a diagnostics reference, editor setup, "How it works" and "Limitations" sections. Every example is verified against the compiler.
- Merged `README_pypi.md` into `README.md`, so GitHub and PyPI show the same, up-to-date page.
- Added project links, license and classifiers to the package metadata.
- Added this changelog.

## 1.0.0 (2026-09-30)

A stability and correctness release.

**Fixed**
- The language server failed to start with pygls 2.x; it now works and supports hover.
- `for` loops with a simple accumulator body gave wrong results (the loop variable never advanced, start values and fractional bounds were ignored).
- The mean of `a * b` ignored the covariance of its operands (`a * a` gave 100 instead of 101 for `Normal(10, 1)`).
- Arithmetic on arrays or structs, top-level `return`, unbounded recursion, `map` with a wrong-arity function and invalid distribution parameters crashed the compiler; they are now diagnostics.
- `empirical_read` silently dropped negative values; annotations with negative parameters were silently not checked.
- Error messages for unknown functions, wrong argument counts, non-arrays, etc. were all rendered as "type annotation mismatch".
- Imports were resolved relative to the working directory instead of the importing file; circular imports recursed forever.
- An `if` on an uncertain condition executed the `else` branch.

**Added**
- Exact moments for `exp`, `sin`, `cos` and `abs`; correlations are tracked through nonlinear functions.
- `normal_read(mean, stddev)`; comparison operators `<=`, `>=`, `==`, `!=`; numbers like `.5` and `1e-3`.
- Distribution parameters may be any deterministic expression, and are validated.
- Function argument and return annotations are checked.
- New diagnostic codes, `delta-method-warning`, "did you mean" hints, and `--version`.
- The program's trailing expression is printed after `=>`; `--output json` prints a single JSON document.

**Changed**
- `approximation-warning` is no longer emitted for sums, which are exact for every family.
- `numpy` is now only a development dependency.
- The playground installs the compiler from PyPI.

## 0.6.0

Automatic covariance tracking, probability queries (`prob`), structs and imports.

## 0.5.2

Web playground, loop iteration limits, math enhancements.

## 0.5.0

Functions, arrays and the language server.
