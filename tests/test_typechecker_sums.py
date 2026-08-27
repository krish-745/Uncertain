import pytest
from uncertain_lang.parser import parse
from uncertain_lang.typechecker import check_stmt, TypeContext, synth

def test_typecheck_sum():
    source = """
    let a = 10.0;
    let b = 5.0;
    let c = a + b;
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    for stmt in stmts:
        diags = check_stmt(stmt, ctx)
        assert len(diags) == 0
        
    typ = ctx.lookup("c")
    assert typ is not None
    assert typ.dist.mean == 15.0
    assert typ.dist.stddev == 0.0

def test_typecheck_annotation_mismatch():
    source = "let a: Measured<Normal(10.0, 1.0)> = 15.0;"
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    diags = check_stmt(stmts[0], ctx)
    assert len(diags) == 1
    assert diags[0].kind == "type-mismatch"

def test_typecheck_undefined_var():
    source = "let a = b + 1.0;"
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    diags = check_stmt(stmts[0], ctx)
    assert len(diags) == 1
    assert diags[0].kind == "undefined-var"
    assert diags[0].extra.get("name") == "b"

def test_typecheck_sensor_read():
    source = "let a: Measured<Normal(10.0, 1.0)> = sensor_read();"
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    diags = check_stmt(stmts[0], ctx)
    assert len(diags) == 0
