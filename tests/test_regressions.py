"""Regression tests for bugs found in the v0.6.0 audit. Each test reproduces one reported issue."""
import math
import pytest
from uncertain.parser import parse, ParseError
from uncertain.typechecker import TypeContext, check_program, ERROR_TYPE

def check(source: str, **ctx_kwargs):
    stmts, expr = parse(source)
    ctx = TypeContext(**ctx_kwargs)
    diags = check_program(stmts, ctx, expr)
    return ctx, diags

def kinds(diags):
    return [d.kind for d in diags]

def value(ctx, name):
    return ctx.lookup(name).dist

# --- for loops (the removed "optimised" loop path) ---------------------------------

def test_for_loop_body_sees_loop_variable():
    ctx, diags = check("var s = 0; for (var i = 0; i < 5; i = i + 1) { s = s + i; }")
    assert diags == []
    assert value(ctx, "s").mean == 10.0

def test_for_loop_respects_start_value():
    ctx, _ = check("var s = 0; for (var i = 2; i < 5; i = i + 1) { s = s + 1; }")
    assert value(ctx, "s").mean == 3.0

def test_for_loop_fractional_bound():
    ctx, _ = check("var s = 0; for (var i = 0; i < 2.5; i = i + 1) { s = s + 1; }")
    assert value(ctx, "s").mean == 3.0

def test_loop_variable_usable_after_loop():
    ctx, diags = check("var s = 0; for (var i = 0; i < 3; i = i + 1) { s = s + 1; } let z = i + 1;")
    assert diags == []
    assert value(ctx, "z").mean == 4.0

def test_loop_limit_is_exact():
    ctx, diags = check("var a = 0; while (a < 10) { a = a + 1; }", max_unroll=10)
    assert diags == []
    assert value(ctx, "a").mean == 10.0
    ctx, diags = check("var a = 0; while (a < 11) { a = a + 1; }", max_unroll=10)
    assert kinds(diags) == ["loop-limit"]
    assert value(ctx, "a").mean == 10.0

def test_loop_diagnostics_are_not_repeated_per_iteration():
    _, diags = check("var s = 0; for (var i = 0; i < 5; i = i + 1) { s = s + uniform_read() * uniform_read(); }")
    assert kinds(diags) == ["approximation-warning"]

# --- products and correlation --------------------------------------------------------

def test_self_product_mean_includes_covariance():
    ctx, diags = check("let a = sensor_read(); let p = a * a; let q = square(a);")
    assert diags == []
    p, q = value(ctx, "p"), value(ctx, "q")
    assert math.isclose(p.mean, 101.0)
    assert math.isclose(p.mean, q.mean) and math.isclose(p.stddev, q.stddev)

def test_correlated_matches_tracked_product():
    ctx, diags = check("let a = sensor_read(); let c = correlated(a, a, cov=1.0); let m = a * a;")
    assert diags == []
    c, m = value(ctx, "c"), value(ctx, "m")
    assert math.isclose(c.mean, m.mean) and math.isclose(c.stddev, m.stddev)

def test_correlated_rejects_impossible_covariance():
    _, diags = check("let a = sensor_read(); let b = sensor_read(); let c = correlated(a, b, cov=5);")
    assert kinds(diags) == ["math-domain-error"]

# --- non-scalar operands -------------------------------------------------------------

@pytest.mark.parametrize("source", [
    "let a = [1, 2]; let b = a + 1;",
    "let s = {x: 1}; let b = s * 2;",
    "let a = [1]; let p = prob(a > 1);",
    "let a = [1, 2]; let b = a[a];",
    "let a = [1, 2]; let b = sqrt(a);",
])
def test_arithmetic_on_containers_is_a_diagnostic(source):
    _, diags = check(source)
    assert kinds(diags) == ["invalid-operand"]

def test_array_index_errors():
    assert kinds(check("let a = [1, 2]; let b = a[2];")[1]) == ["invalid-operand"]
    assert kinds(check("let a = [1, 2]; let b = a[0.5];")[1]) == ["invalid-operand"]
    assert kinds(check("let a = [1, 2]; let b = a[sensor_read()];")[1]) == ["not-constant"]

def test_missing_struct_field():
    _, diags = check("let s = {x: 1}; let y = s.z;")
    assert kinds(diags) == ["invalid-operand"]
    assert "fields: x" in diags[0].extra["msg"]

# --- functions -----------------------------------------------------------------------

def test_top_level_return_is_a_diagnostic():
    ctx, diags = check("let a = 1; return a; let b = 2;")
    assert kinds(diags) == ["invalid-return"]
    assert value(ctx, "b").mean == 2.0  # checking continues after the error

def test_unbounded_recursion_is_a_diagnostic():
    _, diags = check("fn f(x) { return f(x); } let y = f(1);")
    assert kinds(diags) == ["recursion-limit"]

def test_bounded_recursion_works():
    ctx, diags = check("fn fact(n) { if (n < 2) { return 1; } return n * fact(n - 1); } let y = fact(10);")
    assert diags == []
    assert value(ctx, "y").mean == 3628800.0

def test_map_checks_function_arity():
    _, diags = check("fn f() { return 1; } let y = map([1, 2], f);")
    assert kinds(diags) == ["arity-mismatch"]

def test_function_without_return():
    _, diags = check("fn f(x) { let y = x; } let z = f(1);")
    assert kinds(diags) == ["invalid-return"]

def test_function_annotations_are_checked():
    _, diags = check("fn f(x: Measured<Normal(0, 1)>) -> Measured<Exact(99)> { return x; } let y = f(5);")
    assert kinds(diags) == ["type-mismatch", "type-mismatch"]
    _, diags = check("fn f(x: Measured<Normal(10, 1)>) -> Measured<Normal(20, 2)> { return x * 2; } let y = f(sensor_read());")
    assert diags == []

def test_diagnostics_before_nested_return_are_kept():
    _, diags = check("fn f(x) { if (x > 0) { let q = nope; return x; } return 0; } let y = f(1);")
    assert kinds(diags) == ["undefined-var"]

def test_builtin_arity_and_unknown_function_messages():
    _, diags = check("let b = sqrt(1, 2);")
    assert kinds(diags) == ["arity-mismatch"]
    _, diags = check("let b = sqr(1);")
    assert kinds(diags) == ["unknown-function"]
    assert "did you mean 'sqrt'" in diags[0].extra["msg"]
    _, diags = check("let b = correlated(1, 2, covariance=0);")
    assert kinds(diags) == ["arity-mismatch"]

# --- constants and distribution parameters -------------------------------------------

@pytest.mark.parametrize("source", [
    "let g = gamma_read(-1, 2);",
    "let g = exp(1000);",
    "let z = 0; let g = pow(z, -1);",
    "let g = poisson_read(1/0);",
    "let g = poisson_read(-3);",
    "let g = binomial_read(10, 1.5);",
    "let g = exponential_read(0);",
    "let g = lognormal_read(1000, 1);",
])
def test_invalid_numeric_input_is_a_math_domain_error(source):
    _, diags = check(source)
    assert kinds(diags) == ["math-domain-error"]

def test_distribution_parameters_can_be_variables_and_negative():
    ctx, diags = check("let lam = 5; let g = poisson_read(lam); let n = normal_read(-3, 2 * 1);")
    assert diags == []
    assert math.isclose(value(ctx, "g").stddev, math.sqrt(5))
    assert value(ctx, "n").mean == -3.0

def test_uncertain_parameter_is_rejected():
    _, diags = check("let g = poisson_read(sensor_read());")
    assert kinds(diags) == ["not-constant"]

def test_empirical_keeps_negative_values():
    ctx, diags = check("let e = empirical_read([-1, 1]);")
    assert diags == []
    assert value(ctx, "e").mean == 0.0
    assert value(ctx, "e").stddev == 1.0

def test_annotation_with_negative_parameters_is_checked():
    _, diags = check("let x: Measured<Normal(-10, 1)> = sensor_read();")
    assert kinds(diags) == ["type-mismatch"]
    assert "mean -10" in diags[0].extra["msg"]
    _, diags = check("let x: Measured<Normal(-10, 1)> = normal_read(-10, 1);")
    assert diags == []

def test_invalid_annotation_parameters():
    _, diags = check("let x: Measured<Poisson(-1)> = 1;")
    assert kinds(diags) == ["math-domain-error"]

# --- control flow --------------------------------------------------------------------

def test_uncertain_if_runs_neither_branch():
    ctx, diags = check("let a = sensor_read(); var r = 0; if (a > 5) { r = 1; } else { r = 2; }")
    assert kinds(diags) == ["uncertain-branch"]
    assert value(ctx, "r").mean == 0.0

def test_errors_do_not_cascade():
    ctx, diags = check("let a = b + 1; let c = a * 2; if (a > 1) { let d = 1; }")
    assert kinds(diags) == ["undefined-var"]
    assert ctx.lookup("c") is ERROR_TYPE

# --- lexer / parser ------------------------------------------------------------------

def test_comparison_operators():
    ctx, diags = check("let a = 2; let b = a == 2; let c = a != 2; let d = a >= 3; let e = a <= 2;")
    assert diags == []
    assert [value(ctx, n).mean for n in "bcde"] == [1.0, 0.0, 0.0, 1.0]

def test_prob_with_non_strict_inequality():
    ctx, diags = check("let a = sensor_read(); let p = prob(a <= 10);")
    assert diags == []
    assert math.isclose(value(ctx, "p").mean, 0.5)

def test_number_formats():
    ctx, diags = check("let a = 1e-3; let b = .5; let c = 2.5E2;")
    assert diags == []
    assert [value(ctx, n).mean for n in "abc"] == [0.001, 0.5, 250.0]

def test_annotation_followed_by_equals_without_space():
    ctx, diags = check("let x: Measured<Normal(10, 1)>= sensor_read();")
    assert diags == []

def test_trailing_expression_is_evaluated():
    ctx, diags = check("let a = sensor_read(); a + 1")
    assert diags == []
    assert ctx.result.dist.mean == 11.0

def test_lone_carriage_return_line_endings():
    ctx, diags = check("let a = 1;\rlet b = a + 1;\r")
    assert diags == []
    assert value(ctx, "b").mean == 2.0

def test_multiline_spans_do_not_go_negative():
    stmts, _ = parse("let a = 1 +\n    2;")
    assert stmts[0].span.length == -1   # "to the end of the first line"

# --- imports -------------------------------------------------------------------------

def test_circular_import(tmp_path):
    (tmp_path / "a.calc").write_text("import b as b;\nlet x = 1;")
    (tmp_path / "b.calc").write_text("import a as a;\nlet y = 2;")
    _, diags = check("import a as a;", base_dir=str(tmp_path))
    assert kinds(diags) == ["import-error"]
    assert "circular import" in diags[0].extra["msg"]

def test_import_errors_are_summarised(tmp_path):
    (tmp_path / "bad.calc").write_text("let x = nope;")
    _, diags = check("import bad as b;", base_dir=str(tmp_path))
    assert kinds(diags) == ["import-error"]
    assert "bad.calc:1:9" in diags[0].extra["msg"]

def test_missing_module(tmp_path):
    ctx, diags = check("import nothere as n; let y = n.x;", base_dir=str(tmp_path))
    assert kinds(diags) == ["import-error"]

def test_imported_values_stay_correlated(tmp_path):
    (tmp_path / "m.calc").write_text("let s = sensor_read();")
    ctx, diags = check("import m as m; let d = m.s - m.s; let fresh = sensor_read(); let e = fresh - m.s;", base_dir=str(tmp_path))
    assert diags == []
    assert value(ctx, "d").stddev == 0.0
    assert math.isclose(value(ctx, "e").stddev, math.sqrt(2))
