import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from uncertain.parser import parse, ParseError
from uncertain.lexer import LexerError
from uncertain.typechecker import check_stmt, check_program, TypeContext

# 1. Fuzzing with completely random text to ensure no unhandled exceptions
@given(st.text(max_size=100))
@settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_parser_random_text(source):
    try:
        parse(source)
    except (ParseError, LexerError):
        pass # Expected


# 2. Fuzzing with structurally valid but random arithmetic expressions
expr_strategy = st.recursive(
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False).map(lambda f: str(round(f, 2))),
    lambda children: st.one_of(
        st.tuples(children, st.sampled_from(["+", "-", "*", "/"]), children).map(lambda t: f"({t[0]} {t[1]} {t[2]})"),
        children.map(lambda c: f"square({c})"),
        children.map(lambda c: f"sqrt({c})")
    ),
    max_leaves=10
)

@st.composite
def prog_strategy(draw):
    expr = draw(expr_strategy)
    return f"let fuzz_var = {expr};"

@given(prog_strategy())
@settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_parser_and_typechecker(source):
    try:
        stmts, _ = parse(source)
    except (ParseError, LexerError):
        return
        
    ctx = TypeContext()
    for stmt in stmts:
        # This should return a list of Diagnostics or empty, but NEVER raise a Python exception
        # Even math domain errors (Option B) are returned as Diagnostics
        _ = check_stmt(stmt, ctx)


# 3. Fuzzing whole programs over the full expression language (arrays, structs, builtins,
#    comparisons, control flow). check_program must never raise and never hit an internal error.
_atoms = st.one_of(
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False).map(lambda f: repr(round(f, 2))),
    st.sampled_from(["x", "y", "arr", "s", "s.a", "arr[0]", "arr[1]", "undefined_v", "sensor_read()",
                     "uniform_read()", "poisson_read(3)", "normal_read(0, 1)", "[]", "[1, x]", "{a: x}"]),
)
_unary = ["square", "sqrt", "abs", "log", "exp", "sin", "cos"]
rich_expr = st.recursive(
    _atoms,
    lambda children: st.one_of(
        st.tuples(children, st.sampled_from(["+", "-", "*", "/", "<", ">", "<=", ">=", "==", "!="]), children)
          .map(lambda t: f"({t[0]} {t[1]} {t[2]})"),
        st.tuples(st.sampled_from(_unary), children).map(lambda t: f"{t[0]}({t[1]})"),
        st.tuples(children, children).map(lambda t: f"pow({t[0]}, {t[1]})"),
        st.tuples(children, children).map(lambda t: f"prob({t[0]} > {t[1]})"),
        st.tuples(children, children).map(lambda t: f"correlated({t[0]}, {t[1]}, cov=0.1)"),
        st.tuples(children, children).map(lambda t: f"[{t[0]}, {t[1]}][{t[1]}]"),
        children.map(lambda c: f"map(arr, twice)"),
        children.map(lambda c: f"f({c})"),
    ),
    max_leaves=8,
)

_PRELUDE = """
let x = sensor_read();
var y = 2;
let arr = [1, x];
let s = {a: x, b: [y]};
fn twice(v) { return v * 2; }
fn f(v) { if (y > 1) { return v + 1; } return f(v); }
"""

@st.composite
def rich_program(draw):
    e1, e2, e3 = draw(rich_expr), draw(rich_expr), draw(rich_expr)
    stmt = draw(st.sampled_from([
        "let r = {0};",
        "y = {0};",
        "if ({0}) {{ y = {1}; }} else {{ y = {2}; }}",
        "while ({0}) {{ y = y + 1; }}",
        "for (var i = 0; i < {0}; i = i + 1) {{ y = y + {1}; }}",
        "return {0};",
        "fn g(q) {{ return {0}; }} let r = g({1});",
        "let r: Measured<Normal({0}, {1})> = {2};",
    ]))
    return _PRELUDE + stmt.format(e1, e2, e3)

@given(rich_program())
@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_rich_programs_never_crash(source):
    try:
        stmts, expr = parse(source)
    except (ParseError, LexerError):
        return
    diags = check_program(stmts, TypeContext(max_unroll=50), expr)
    assert not [d for d in diags if d.kind == "internal-error"], [d.extra for d in diags if d.kind == "internal-error"]
