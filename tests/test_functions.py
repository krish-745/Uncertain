def test_fn_parsing():
    from uncertain.parser import parse
    from uncertain.ast_nodes import FnDefStmt
    source = """fn propagate(a, b) {
        return a + b;
    }"""
    stmts, expr = parse(source)
    assert len(stmts) == 1
    assert isinstance(stmts[0], FnDefStmt)
    assert stmts[0].name == "propagate"
    assert len(stmts[0].args) == 2
    assert stmts[0].args[0].name == "a"

def test_fn_eval():
    from uncertain.parser import parse
    from uncertain.typechecker import TypeContext, check_stmt
    source = """fn propagate(a, b) {
        return a + b;
    }
    let c = propagate(10.0, 5.0);
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    for stmt in stmts:
        check_stmt(stmt, ctx)
    c = ctx.lookup("c")
    assert c.dist.mean == 15.0