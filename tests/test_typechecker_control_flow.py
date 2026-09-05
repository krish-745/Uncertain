import pytest
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext

def test_if_stmt_deterministic():
    source = """
    let a = 10.0;
    let b = 0.0;
    if (a > 5.0) {
        b = 1.0;
    } else {
        b = 2.0;
    }
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = []
    for stmt in stmts:
        diags.extend(check_stmt(stmt, ctx))
    
    assert len(diags) == 0
    assert ctx.lookup("b").dist.mean == 1.0

def test_while_stmt_deterministic():
    source = """
    let a = 0.0;
    while (a < 5.0) {
        a = a + 1.0;
    }
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = []
    for stmt in stmts:
        diags.extend(check_stmt(stmt, ctx))
    
    assert len(diags) == 0
    assert ctx.lookup("a").dist.mean == 5.0

def test_for_stmt_deterministic():
    source = """
    let b = 0.0;
    for (let i = 0.0; i < 5.0; i = i + 1.0) {
        b = b + 2.0;
    }
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = []
    for stmt in stmts:
        diags.extend(check_stmt(stmt, ctx))
    
    assert len(diags) == 0
    assert ctx.lookup("b").dist.mean == 10.0
    assert ctx.lookup("i").dist.mean == 5.0

def test_uncertain_branch_if():
    source = """
    let a = sensor_read();
    let b = 0.0;
    if (a > 5.0) {
        b = 1.0;
    }
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = []
    for stmt in stmts:
        diags.extend(check_stmt(stmt, ctx))
    
    assert any(d.kind == "uncertain-branch" for d in diags)

def test_uncertain_branch_while():
    source = """
    let a = sensor_read();
    while (a < 20.0) {
        a = a + 1.0;
    }
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = []
    for stmt in stmts:
        diags.extend(check_stmt(stmt, ctx))
    
    assert any(d.kind == "uncertain-branch" for d in diags)

def test_loop_iteration_limit():
    source = """
    let a = 0.0;
    while (a < 1005.0) {
        a = a + 1.0;
    }
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    diags = []
    for stmt in stmts:
        diags.extend(check_stmt(stmt, ctx))
    
    assert any(d.kind == "uncertain-branch" and "limit exceeded" in d.extra.get("msg", "") for d in diags)
