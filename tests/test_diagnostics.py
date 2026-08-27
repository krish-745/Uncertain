import os
import pytest
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext
from uncertain.diagnostics import format_diagnostic

def test_golden_reuse_error():
    base_dir = os.path.dirname(__file__)
    calc_file = os.path.join(base_dir, "golden", "reuse_error.calc")
    expected_file = os.path.join(base_dir, "golden", "reuse_error.expected.txt")
    
    with open(calc_file, "r") as f:
        source = f.read()
    with open(expected_file, "r") as f:
        expected = f.read().strip()
        
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    check_stmt(stmts[0], ctx)
    diags = check_stmt(stmts[1], ctx)
    
    assert len(diags) == 1
    actual = format_diagnostic(diags[0], source.splitlines())
    
    assert actual == expected
