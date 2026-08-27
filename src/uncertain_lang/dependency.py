from uncertain_lang.ast_nodes import *
from uncertain_lang.diagnostics import Diagnostic

DepSet = frozenset[str]

def deps_of(expr: Expr, env: dict[str, DepSet]) -> DepSet:
    if isinstance(expr, NumberLit):
        return frozenset()
        
    elif isinstance(expr, VarRef):
        return env.get(expr.name, frozenset({expr.name}))
        
    elif isinstance(expr, BinOp):
        return deps_of(expr.left, env) | deps_of(expr.right, env)
        
    elif isinstance(expr, Call):
        if expr.name in ("square", "sqrt") and len(expr.args) == 1:
            return deps_of(expr.args[0], env)
            
        elif expr.name == "correlated" and len(expr.args) >= 2:
            return deps_of(expr.args[0], env) | deps_of(expr.args[1], env)
            
        elif expr.name == "sensor_read":
            return frozenset()
            
        # fallback for unknown calls
        deps = frozenset()
        for arg in expr.args:
            deps |= deps_of(arg, env)
        return deps
        
    return frozenset()

def check_reuse(op: str, left_deps: DepSet, right_deps: DepSet, span: Span) -> Diagnostic | None:
    if op in ("*", "/"):
        overlap = left_deps & right_deps
        if overlap:
            return Diagnostic(
                kind="uncertain-reuse",
                span=span,
                overlapping_vars=overlap,
                extra={"op": op}
            )
    return None
