import os
import pytest
from uncertain.parser import parse
from uncertain.typechecker import check_stmt, TypeContext

def test_imports(tmp_path):
    math_path = tmp_path / "math.calc"
    math_path.write_text("let PI = 3.14159;")
    
    physics_dir = tmp_path / "physics"
    physics_dir.mkdir()
    kinematics_path = physics_dir / "kinematics.calc"
    kinematics_path.write_text("let g = 9.81;")
    
    main_path = tmp_path / "main.calc"
    source = """
    import math as m;
    import physics.kinematics as phys;
    
    let a = m.PI;
    let b = phys.g;
    """
    main_path.write_text(source)
    
    stmts, _ = parse(source)
    ctx = TypeContext(base_dir=str(tmp_path))
    
    for stmt in stmts:
        check_stmt(stmt, ctx)
        
    a = ctx.lookup("a")
    b = ctx.lookup("b")
    
    assert a.dist.mean == 3.14159
    assert b.dist.mean == 9.81
