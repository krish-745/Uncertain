def test_arrays():
    from uncertain.parser import parse
    from uncertain.typechecker import TypeContext, check_stmt
    source = """let arr = [10.0, 20.0];
    fn map_func(x) { return x * 2.0; }
    let arr2 = map(arr, map_func);
    let val = arr2[1];
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    for stmt in stmts:
        check_stmt(stmt, ctx)
    val = ctx.lookup("val")
    assert val.dist.mean == 40.0

def test_reduce():
    from uncertain.parser import parse
    from uncertain.typechecker import TypeContext, check_stmt
    source = """let arr = [1.0, 2.0, 3.0];
    fn sum(acc, x) { return acc + x; }
    let val = reduce(arr, sum, 0.0);
    """
    stmts, _ = parse(source)
    ctx = TypeContext()
    for stmt in stmts:
        check_stmt(stmt, ctx)
    val = ctx.lookup("val")
    assert val.dist.mean == 6.0
