import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from uncertain.parser import parse, ParseError
from uncertain.lexer import LexerError
from uncertain.typechecker import check_stmt, TypeContext

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
