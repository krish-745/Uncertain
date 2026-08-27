import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
from uncertain_lang.ast_nodes import *
from uncertain_lang.distributions import (
    Dist, add, sub, mul_independent, div_independent, square, sqrt_dist, correlated_product
)
from uncertain_lang.diagnostics import Diagnostic
from uncertain_lang.dependency import check_reuse, DepSet

@dataclass(frozen=True)
class MeasuredType:
    dist: Dist
    deps: DepSet

ERROR_TYPE = MeasuredType(Dist(0.0, 0.0), frozenset())

class TypeContext:
    def __init__(self):
        self.bindings: dict[str, MeasuredType] = {}

    def lookup(self, name: str) -> Optional[MeasuredType]:
        return self.bindings.get(name)

    def bind(self, name: str, typ: MeasuredType):
        self.bindings[name] = typ

def eval_const(expr: Expr) -> float:
    # helper for cov kwargs
    if isinstance(expr, NumberLit):
        return expr.value
    # for simplicity, we only allow constants
    return 0.0

def synth(expr: Expr, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    if isinstance(expr, NumberLit):
        return MeasuredType(Dist(expr.value, 0.0), frozenset()), []
        
    elif isinstance(expr, VarRef):
        typ = ctx.lookup(expr.name)
        if not typ:
            diag = Diagnostic("undefined-var", expr.span, extra={"name": expr.name})
            return ERROR_TYPE, [diag]
        # Direct var ref, its deps are just itself
        # Wait, the design doc: env.get(n, {n}). In synth, typ.deps already has it?
        # When we bind, what are its deps? `let a = 1;` -> deps = {}.
        # `let a = sensor_read();` -> deps = {"a"}.
        # So we should probably just return `MeasuredType(typ.dist, frozenset({expr.name}))` 
        # so that when it is referenced, it introduces its own name as a dependency!
        # Wait, if `a = b + c`, `a`'s deps are `b` and `c`?
        # The design doc says: DepSet = frozenset[str] # set of root variable names contributing uncertainty.
        # If `let a = b + c`, and we use `a`, is the dep `a`, or `{b, c}`?
        # If we just track the bound variables themselves, when we use `a * a`, the dep is `{a}`.
        # Actually, in a straight-line let-binding DSL, `a` is a new variable. 
        # Let's just return typ.deps for now. But wait, if `a` is bound to `sensor_read()`, its deps should be `{"a"}`!
        # The binding logic in `check_stmt` should handle adding `{"a"}`.
        return typ, []
        
    elif isinstance(expr, BinOp):
        lt, ld = synth(expr.left, ctx)
        rt, rd = synth(expr.right, ctx)
        diags = ld + rd
        
        if expr.op in ("+", "-"):
            fn = add if expr.op == "+" else sub
            return MeasuredType(fn(lt.dist, rt.dist), lt.deps | rt.deps), diags
            
        elif expr.op in ("*", "/"):
            diag = check_reuse(expr.op, lt.deps, rt.deps, expr.span)
            if diag:
                diags.append(diag)
                return ERROR_TYPE, diags
            fn = mul_independent if expr.op == "*" else div_independent
            return MeasuredType(fn(lt.dist, rt.dist), lt.deps | rt.deps), diags
            
    elif isinstance(expr, Call):
        if expr.name == "square" and len(expr.args) == 1:
            at, ad = synth(expr.args[0], ctx)
            return MeasuredType(square(at.dist), at.deps), ad
            
        elif expr.name == "sqrt" and len(expr.args) == 1:
            at, ad = synth(expr.args[0], ctx)
            return MeasuredType(sqrt_dist(at.dist), at.deps), ad
            
        elif expr.name == "correlated" and len(expr.args) >= 2:
            at, ad = synth(expr.args[0], ctx)
            bt, bd = synth(expr.args[1], ctx)
            cov_expr = expr.kwargs.get("cov", NumberLit(0.0, expr.span))
            cov = eval_const(cov_expr)
            # Pre-cleared reuse, so we don't check_reuse
            return MeasuredType(correlated_product(at.dist, bt.dist, cov), at.deps | bt.deps), ad + bd
            
        elif expr.name == "sensor_read":
            return MeasuredType(Dist(10.0, 1.0), frozenset()), []
            
        return ERROR_TYPE, []

    return ERROR_TYPE, []

def check_stmt(stmt: LetStmt, ctx: TypeContext) -> list[Diagnostic]:
    inferred, diags = synth(stmt.value, ctx)
    if stmt.type_ann:
        if isinstance(stmt.type_ann, NormalLit):
            if isinstance(stmt.type_ann.mean, NumberLit) and isinstance(stmt.type_ann.stddev, NumberLit):
                exp_mean = stmt.type_ann.mean.value
                exp_std = stmt.type_ann.stddev.value
                
                mean_ok = math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3)
                std_ok = math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)
                
                if not (mean_ok and std_ok):
                    diags.append(Diagnostic("type-mismatch", stmt.span))
    
    # Crucial: when binding a variable, if its uncertainty > 0, it acts as a root variable
    # Wait, the design doc says: "let variance_est = a * a; ... 'a' appears twice".
    # So the deps of `a` must be `{"a"}`!
    # If `let b = a + 1`, deps of `b` should be `{"a"}` (because it's derived from `a`).
    # Wait, if `a` is bound, the variable `a` itself is what we refer to.
    # What if `let c = a + b`? Then `c` has deps `{"a", "b"}` if we just pass `inferred.deps`.
    # Let's just bind `inferred` directly, but if `inferred` has some deps, they propagate.
    # What if `let a = sensor_read();`? `sensor_read` returns `deps = {}`. 
    # But `a` should introduce its own name!
    # So if we bind `name`, we should add `name` to the deps, OR we just let `VarRef` add it.
    # Let's let `VarRef` add it! But wait, `VarRef` currently just returns `typ.deps`.
    # Let's change `VarRef` to return `typ.deps | {stmt.name}`? No, `VarRef` knows the variable name.
    # Let's look at `check_stmt`:
    
    final_deps = inferred.deps | frozenset({stmt.name})
    final_typ = MeasuredType(inferred.dist, final_deps)
    ctx.bind(stmt.name, final_typ)
    return diags
