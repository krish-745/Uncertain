import time
import pytest
from uncertain.parser import parse
from uncertain.typechecker import TypeContext, check_stmt

def test_typechecker_performance():
    # Programmatically generate 500 statements of chained additions
    # let x_0 = 1.0;
    # let x_1 = x_0 + 1.0;
    # let x_2 = x_1 + 1.0;
    
    statements = ["let x_0: Measured<Normal(1.0, 0.0)> = 1.0;"]
    for i in range(1, 501):
        statements.append(f"let x_{i} = x_{i-1} + 1.0;")
        
    source = "\n".join(statements)
    
    stmts, _ = parse(source)
    ctx = TypeContext()
    
    start_time = time.time()
    all_diags = []
    for stmt in stmts:
        diags = check_stmt(stmt, ctx)
        all_diags.extend(diags)
    end_time = time.time()
    
    assert not all_diags, "Generated program should have no type errors"
    
    duration = end_time - start_time
    assert duration < 1.0, f"Typechecking 500 statements took too long: {duration:.4f}s"
