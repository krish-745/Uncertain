"""`let` immutability, lexical function scope and functions exported from modules."""
from uncertain.parser import parse
from uncertain.typechecker import TypeContext, check_program

def check(source: str, **ctx_kwargs):
    stmts, expr = parse(source)
    ctx = TypeContext(**ctx_kwargs)
    return ctx, check_program(stmts, ctx, expr)

def kinds(diags):
    return [d.kind for d in diags]

def mean(ctx, name):
    return ctx.lookup(name).dist.mean

# --- let / var ---------------------------------------------------------------------

def test_let_cannot_be_reassigned():
    ctx, diags = check("let a = 1; a = 2;")
    assert kinds(diags) == ["immutable-assign"]
    assert "var a" in diags[0].extra["msg"]
    assert mean(ctx, "a") == 1.0   # the assignment is rejected

def test_var_can_be_reassigned():
    ctx, diags = check("var a = 1; a = 2;")
    assert diags == []
    assert mean(ctx, "a") == 2.0

def test_let_loop_counter_is_an_error():
    _, diags = check("var s = 0; for (let i = 0; i < 3; i = i + 1) { s = s + i; }")
    assert kinds(diags) == ["immutable-assign"]

def test_redeclaring_is_allowed():
    # shadowing with a new declaration is not a reassignment (e.g. `let` inside a loop body)
    ctx, diags = check("let a = 1; let a = 2; var t = 0; for (var i = 0; i < 3; i = i + 1) { let sq = i * i; t = t + sq; }")
    assert diags == []
    assert mean(ctx, "a") == 2.0
    assert mean(ctx, "t") == 5.0

def test_var_can_become_let():
    _, diags = check("var a = 1; let a = 2; a = 3;")
    assert kinds(diags) == ["immutable-assign"]

def test_parameters_are_immutable():
    _, diags = check("fn f(x) { x = x + 1; return x; } let y = f(1);")
    assert kinds(diags) == ["immutable-assign"]

def test_imported_module_is_immutable(tmp_path):
    (tmp_path / "m.calc").write_text("let g = 9.81;")
    _, diags = check("import m as m; m = 1;", base_dir=str(tmp_path))
    assert kinds(diags) == ["immutable-assign"]

# --- lexical scope -------------------------------------------------------------------

def test_function_sees_definition_scope_not_caller_scope():
    source = """
    fn outer() {
        let secret = 42;
        return inner();
    }
    fn inner() {
        return secret;
    }
    let r = outer();
    """
    _, diags = check(source)
    assert kinds(diags) == ["undefined-var"]   # `secret` is local to outer(), invisible to inner()

def test_function_sees_globals_defined_before_the_call():
    ctx, diags = check("var y = 2; fn f(v) { return v + y; } y = 3; let r = f(1);")
    assert diags == []
    assert mean(ctx, "r") == 4.0   # closures see the current value of the variable

def test_nested_function_captures_enclosing_parameters():
    source = """
    fn make(k) {
        fn scale(v) { return v * k; }
        return scale(10);
    }
    let r = make(3);
    """
    ctx, diags = check(source)
    assert diags == []
    assert mean(ctx, "r") == 30.0

def test_function_can_update_outer_var():
    ctx, diags = check("var total = 0; fn add(v) { total = total + v; return total; } let a = add(2); let b = add(3);")
    assert diags == []
    assert mean(ctx, "total") == 5.0
    assert (mean(ctx, "a"), mean(ctx, "b")) == (2.0, 5.0)

def test_function_cannot_update_outer_let():
    _, diags = check("let total = 0; fn add(v) { total = total + v; return total; } let a = add(2);")
    assert kinds(diags) == ["immutable-assign"]

def test_recursion_and_mutual_recursion():
    source = """
    fn is_even(n) { if (n == 0) { return 1; } return is_odd(n - 1); }
    fn is_odd(n) { if (n == 0) { return 0; } return is_even(n - 1); }
    let e = is_even(10);
    let o = is_even(7);
    """
    ctx, diags = check(source)
    assert diags == []
    assert (mean(ctx, "e"), mean(ctx, "o")) == (1.0, 0.0)

# --- module functions ----------------------------------------------------------------

def test_calling_module_functions(tmp_path):
    (tmp_path / "geometry.calc").write_text(
        "let pi = 3.14159;\n"
        "fn circle_area(r) { return pi * square(r); }\n"   # uses a module-level value
    )
    ctx, diags = check("import geometry as geo; let a = geo.circle_area(2); let p = geo.pi;", base_dir=str(tmp_path))
    assert diags == []
    assert abs(mean(ctx, "a") - 12.56636) < 1e-9

def test_module_function_does_not_see_importer_scope(tmp_path):
    (tmp_path / "lib.calc").write_text("fn get() { return hidden; }")
    _, diags = check("let hidden = 1; import lib as l; let x = l.get();", base_dir=str(tmp_path))
    assert kinds(diags) == ["undefined-var"]

def test_module_functions_in_higher_order_calls(tmp_path):
    (tmp_path / "ops.calc").write_text("fn double(v) { return v * 2; }\nfn add(a, b) { return a + b; }")
    ctx, diags = check("import ops as o; let xs = map([1, 2, 3], o.double); let s = reduce(xs, o.add, 0);", base_dir=str(tmp_path))
    assert diags == []
    assert mean(ctx, "s") == 12.0

def test_nested_modules(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "units.calc").write_text("fn km(m) { return m / 1000; }")
    ctx, diags = check("import lib.units as u; let d = u.km(2500);", base_dir=str(tmp_path))
    assert diags == []
    assert mean(ctx, "d") == 2.5

def test_unknown_module_function(tmp_path):
    (tmp_path / "ops.calc").write_text("fn double(v) { return v * 2; }")
    _, diags = check("import ops as o; let x = o.triple(1);", base_dir=str(tmp_path))
    assert kinds(diags) == ["unknown-function"]
    assert "double" in diags[0].extra["msg"]

def test_calling_a_function_on_a_non_module():
    _, diags = check("let x = 1; let y = x.f(2);")
    assert kinds(diags) == ["invalid-operand"]
