import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
from uncertain.ast_nodes import *
from uncertain.distributions import (
    Dist, add, sub, mul_independent, div_independent, square, sqrt_dist, correlated_product,
    abs_dist, log_dist, exp_dist, sin_dist, cos_dist, pow_const,
    MathDomainError
)
from uncertain.diagnostics import Diagnostic
from uncertain.dependency import check_reuse, DepSet

@dataclass(frozen=True)
class MeasuredType:
    dist: Dist | tuple["MeasuredType", ...]
    deps: DepSet

ERROR_TYPE = MeasuredType(Dist(0.0, 0.0), frozenset())

class TypeContext:
    def __init__(self, parent: "TypeContext" = None, max_unroll: int = 1000):
        self.bindings: dict[str, MeasuredType] = {}
        self.functions: dict[str, FnDefStmt] = {}
        self.parent = parent
        self.max_unroll = parent.max_unroll if parent else max_unroll

    def lookup(self, name: str) -> Optional[MeasuredType]:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def bind(self, name: str, typ: MeasuredType):
        self.bindings[name] = typ
        
    def lookup_fn(self, name: str) -> Optional[FnDefStmt]:
        if name in self.functions:
            return self.functions[name]
        if self.parent:
            return self.parent.lookup_fn(name)
        return None

class ReturnException(Exception):
    def __init__(self, typ: MeasuredType, diags: list[Diagnostic]):
        self.typ = typ
        self.diags = diags

class NotConstantError(Exception):
    pass

def eval_const(expr: Expr) -> float:
    if isinstance(expr, NumberLit):
        return expr.value
    elif isinstance(expr, BinOp):
        if expr.op == "+":
            return eval_const(expr.left) + eval_const(expr.right)
        elif expr.op == "-":
            return eval_const(expr.left) - eval_const(expr.right)
        elif expr.op == "*":
            return eval_const(expr.left) * eval_const(expr.right)
        elif expr.op == "/":
            return eval_const(expr.left) / eval_const(expr.right)
    raise NotConstantError("Expected a constant numeric expression")

def synth(expr: Expr, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    if isinstance(expr, NumberLit):
        return MeasuredType(Dist(expr.value, 0.0), frozenset()), []
        
    elif isinstance(expr, VarRef):
        typ = ctx.lookup(expr.name)
        if not typ:
            diag = Diagnostic("undefined-var", expr.span, extra={"name": expr.name})
            return ERROR_TYPE, [diag]
        return typ, []
        
    elif isinstance(expr, ArrayLit):
        diags = []
        elements = []
        deps = set()
        for el in expr.elements:
            t, d = synth(el, ctx)
            diags.extend(d)
            elements.append(t)
            deps.update(t.deps)
        return MeasuredType(tuple(elements), frozenset(deps)), diags

    elif isinstance(expr, ArrayAccess):
        arr_t, diags = synth(expr.array, ctx)
        idx_t, idx_diags = synth(expr.index, ctx)
        diags.extend(idx_diags)
        
        if not isinstance(arr_t.dist, tuple):
            diags.append(Diagnostic("type-mismatch", expr.array.span, extra={"msg": "Cannot index into a non-array"}))
            return ERROR_TYPE, diags
            
        try:
            # Check if index is deterministic and integer
            if not math.isclose(idx_t.dist.stddev, 0.0, abs_tol=1e-9):
                raise ValueError("Index must be deterministic")
            if not idx_t.dist.mean.is_integer():
                raise ValueError("Index must be an integer")
                
            idx_val = int(idx_t.dist.mean)
            if idx_val < 0 or idx_val >= len(arr_t.dist):
                raise ValueError("Index out of bounds")
                
            return arr_t.dist[idx_val], diags
        except ValueError as e:
            diags.append(Diagnostic("math-domain-error", expr.index.span, extra={"msg": str(e)}))
            return ERROR_TYPE, diags
            
    elif isinstance(expr, BinOp):
        lt, ld = synth(expr.left, ctx)
        rt, rd = synth(expr.right, ctx)
        diags = ld + rd
        
        non_normal_families = ("Uniform", "Empirical", "LogNormal", "Poisson", "Binomial", "Gamma", "Bernoulli", "NegativeBinomial", "Geometric", "Exponential")
        if lt.dist.family in non_normal_families or rt.dist.family in non_normal_families:
            diags.append(Diagnostic("approximation-warning", expr.span, extra={"msg": "Moment-matching approximation used for non-Normal distribution"}, severity="warning"))

        if expr.op in ("+", "-"):
            fn = add if expr.op == "+" else sub
            return MeasuredType(fn(lt.dist, rt.dist), lt.deps | rt.deps), diags
            
        elif expr.op in ("<", ">"):
            if math.isclose(lt.dist.stddev, 0.0, abs_tol=1e-9) and math.isclose(rt.dist.stddev, 0.0, abs_tol=1e-9):
                if expr.op == "<":
                    res = 1.0 if lt.dist.mean < rt.dist.mean else 0.0
                else:
                    res = 1.0 if lt.dist.mean > rt.dist.mean else 0.0
                return MeasuredType(Dist(res, 0.0), lt.deps | rt.deps), diags
            else:
                diags.append(Diagnostic("uncertain-branch", expr.span, extra={"msg": "Cannot branch on a non-deterministic condition"}))
                return ERROR_TYPE, diags

        elif expr.op in ("*", "/"):
            diag = check_reuse(expr.op, lt.deps, rt.deps, expr.span)
            if diag:
                diags.append(diag)
                return ERROR_TYPE, diags
            fn = mul_independent if expr.op == "*" else div_independent
            try:
                res = fn(lt.dist, rt.dist)
            except MathDomainError as e:
                diags.append(Diagnostic("math-domain-error", expr.span, extra={"msg": str(e)}))
                return ERROR_TYPE, diags
            return MeasuredType(res, lt.deps | rt.deps), diags
            
    elif isinstance(expr, Call):
        diags = []
        try:
            if expr.name == "square" and len(expr.args) == 1:
                at, ad = synth(expr.args[0], ctx)
                return MeasuredType(square(at.dist), at.deps), ad + diags
                
            elif expr.name == "sqrt" and len(expr.args) == 1:
                at, ad = synth(expr.args[0], ctx)
                try:
                    res = sqrt_dist(at.dist)
                except MathDomainError as e:
                    return ERROR_TYPE, ad + diags + [Diagnostic("math-domain-error", expr.span, extra={"msg": str(e)})]
                return MeasuredType(res, at.deps), ad + diags

            elif expr.name == "abs" and len(expr.args) == 1:
                at, ad = synth(expr.args[0], ctx)
                return MeasuredType(abs_dist(at.dist), at.deps), ad + diags

            elif expr.name == "log" and len(expr.args) == 1:
                at, ad = synth(expr.args[0], ctx)
                try:
                    res = log_dist(at.dist)
                except MathDomainError as e:
                    return ERROR_TYPE, ad + diags + [Diagnostic("math-domain-error", expr.span, extra={"msg": str(e)})]
                return MeasuredType(res, at.deps), ad + diags

            elif expr.name == "exp" and len(expr.args) == 1:
                at, ad = synth(expr.args[0], ctx)
                return MeasuredType(exp_dist(at.dist), at.deps), ad + diags

            elif expr.name == "sin" and len(expr.args) == 1:
                at, ad = synth(expr.args[0], ctx)
                return MeasuredType(sin_dist(at.dist), at.deps), ad + diags

            elif expr.name == "cos" and len(expr.args) == 1:
                at, ad = synth(expr.args[0], ctx)
                return MeasuredType(cos_dist(at.dist), at.deps), ad + diags

            elif expr.name == "pow" and len(expr.args) == 2:
                at, ad = synth(expr.args[0], ctx)
                n_val = eval_const(expr.args[1])
                if not n_val.is_integer():
                    raise NotConstantError("Exponent must be an integer constant")
                return MeasuredType(pow_const(at.dist, int(n_val)), at.deps), ad + diags
                
            elif expr.name == "correlated" and len(expr.args) >= 2:
                at, ad = synth(expr.args[0], ctx)
                bt, bd = synth(expr.args[1], ctx)
                cov_expr = expr.kwargs.get("cov", NumberLit(0.0, expr.span))
                cov = eval_const(cov_expr)
                return MeasuredType(correlated_product(at.dist, bt.dist, cov), at.deps | bt.deps), ad + bd + diags
                
            elif expr.name == "sensor_read":
                return MeasuredType(Dist(10.0, 1.0), frozenset()), diags
                
            elif expr.name == "uniform_read":
                return MeasuredType(Dist(5.0, 2.8867, "Uniform"), frozenset()), diags
                
            elif expr.name == "empirical_read" and len(expr.args) == 1:
                if isinstance(expr.args[0], ArrayLit):
                    vals = []
                    for el in expr.args[0].elements:
                        if isinstance(el, NumberLit):
                            vals.append(el.value)
                    if vals:
                        mean = sum(vals) / len(vals)
                        stddev = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
                        return MeasuredType(Dist(mean, stddev, "Empirical"), frozenset()), diags
                return ERROR_TYPE, diags
                
            elif expr.name == "lognormal_read" and len(expr.args) == 2:
                mu, sigma = eval_const(expr.args[0]), eval_const(expr.args[1])
                m = math.exp(mu + (sigma**2) / 2.0)
                s = math.sqrt((math.exp(sigma**2) - 1.0) * math.exp(2.0*mu + sigma**2))
                return MeasuredType(Dist(m, s, "LogNormal"), frozenset()), diags
            elif expr.name == "poisson_read" and len(expr.args) == 1:
                lam = eval_const(expr.args[0])
                return MeasuredType(Dist(lam, math.sqrt(lam) if lam >= 0 else 0.0, "Poisson"), frozenset()), diags
            elif expr.name == "binomial_read" and len(expr.args) == 2:
                n, p = eval_const(expr.args[0]), eval_const(expr.args[1])
                return MeasuredType(Dist(n * p, math.sqrt(n * p * (1 - p)) if n > 0 and 0 <= p <= 1 else 0.0, "Binomial"), frozenset()), diags
            elif expr.name == "gamma_read" and len(expr.args) == 2:
                k, theta = eval_const(expr.args[0]), eval_const(expr.args[1])
                return MeasuredType(Dist(k * theta, math.sqrt(k * (theta**2)), "Gamma"), frozenset()), diags
            elif expr.name == "bernoulli_read" and len(expr.args) == 1:
                p = eval_const(expr.args[0])
                return MeasuredType(Dist(p, math.sqrt(p * (1 - p)) if 0 <= p <= 1 else 0.0, "Bernoulli"), frozenset()), diags
            elif expr.name == "negbinom_read" and len(expr.args) == 2:
                r, p = eval_const(expr.args[0]), eval_const(expr.args[1])
                m = (p * r) / (1 - p) if p > 0 and p < 1 and r > 0 else 0.0
                s = math.sqrt((p * r) / ((1 - p)**2)) if p > 0 and p < 1 and r > 0 else 0.0
                return MeasuredType(Dist(m, s, "NegativeBinomial"), frozenset()), diags
            elif expr.name == "geometric_read" and len(expr.args) == 1:
                p = eval_const(expr.args[0])
                m = 1.0 / p if p > 0 and p <= 1 else 0.0
                s = math.sqrt((1.0 - p) / (p**2)) if p > 0 and p <= 1 else 0.0
                return MeasuredType(Dist(m, s, "Geometric"), frozenset()), diags
            elif expr.name == "exponential_read" and len(expr.args) == 1:
                lam = eval_const(expr.args[0])
                m = 1.0 / lam if lam > 0 else 0.0
                s = math.sqrt(1.0 / (lam**2)) if lam > 0 else 0.0
                return MeasuredType(Dist(m, s, "Exponential"), frozenset()), diags
                
            elif expr.name in ("map", "filter") and len(expr.args) == 2:
                arr_t, d1 = synth(expr.args[0], ctx)
                diags.extend(d1)
                if not isinstance(arr_t.dist, tuple):
                    diags.append(Diagnostic("type-mismatch", expr.args[0].span, extra={"msg": "First argument must be an array"}))
                    return ERROR_TYPE, diags
                    
                fn_name = expr.args[1].name if isinstance(expr.args[1], VarRef) else None
                fn_def = ctx.lookup_fn(fn_name) if fn_name else None
                if not fn_def:
                    diags.append(Diagnostic("type-mismatch", expr.args[1].span, extra={"msg": "Second argument must be a function name"}))
                    return ERROR_TYPE, diags
                    
                result_elements = []
                result_deps = set(arr_t.deps)
                for el in arr_t.dist:
                    call_ctx = TypeContext(parent=ctx)
                    call_ctx.bind(fn_def.args[0].name, el)
                    
                    try:
                        for s in fn_def.body.stmts:
                            diags.extend(check_stmt(s, call_ctx))
                        diags.append(Diagnostic("type-mismatch", expr.span, extra={"msg": "Function did not return a value"}))
                        return ERROR_TYPE, diags
                    except ReturnException as r:
                        diags.extend(r.diags)
                        if expr.name == "map":
                            result_elements.append(r.typ)
                            result_deps.update(r.typ.deps)
                        else: # filter
                            if not math.isclose(r.typ.dist.stddev, 0.0, abs_tol=1e-9):
                                diags.append(Diagnostic("uncertain-branch", expr.span, extra={"msg": "Filter condition must be deterministic"}))
                                return ERROR_TYPE, diags
                            if r.typ.dist.mean != 0.0:
                                result_elements.append(el)
                                
                return MeasuredType(tuple(result_elements), frozenset(result_deps)), diags
                
            elif expr.name == "reduce" and len(expr.args) == 3:
                arr_t, d1 = synth(expr.args[0], ctx)
                diags.extend(d1)
                if not isinstance(arr_t.dist, tuple):
                    diags.append(Diagnostic("type-mismatch", expr.args[0].span, extra={"msg": "First argument must be an array"}))
                    return ERROR_TYPE, diags
                    
                fn_name = expr.args[1].name if isinstance(expr.args[1], VarRef) else None
                fn_def = ctx.lookup_fn(fn_name) if fn_name else None
                if not fn_def or len(fn_def.args) != 2:
                    diags.append(Diagnostic("type-mismatch", expr.args[1].span, extra={"msg": "Second argument must be a 2-argument function name"}))
                    return ERROR_TYPE, diags
                    
                acc_t, d2 = synth(expr.args[2], ctx)
                diags.extend(d2)
                
                for el in arr_t.dist:
                    call_ctx = TypeContext(parent=ctx)
                    call_ctx.bind(fn_def.args[0].name, acc_t)
                    call_ctx.bind(fn_def.args[1].name, el)
                    
                    try:
                        for s in fn_def.body.stmts:
                            diags.extend(check_stmt(s, call_ctx))
                        diags.append(Diagnostic("type-mismatch", expr.span, extra={"msg": "Function did not return a value"}))
                        return ERROR_TYPE, diags
                    except ReturnException as r:
                        diags.extend(r.diags)
                        acc_t = r.typ
                        
                return acc_t, diags
                
            fn_def = ctx.lookup_fn(expr.name)
            if fn_def:
                if len(expr.args) != len(fn_def.args):
                    return ERROR_TYPE, diags + [Diagnostic("type-mismatch", expr.span, extra={"msg": f"Argument count mismatch: expected {len(fn_def.args)}, got {len(expr.args)}"})]
                
                call_ctx = TypeContext(parent=ctx)
                for arg_def, arg_expr in zip(fn_def.args, expr.args):
                    arg_typ, d = synth(arg_expr, ctx)
                    diags.extend(d)
                    call_ctx.bind(arg_def.name, arg_typ)
                    
                try:
                    for s in fn_def.body.stmts:
                        diags.extend(check_stmt(s, call_ctx))
                    return ERROR_TYPE, diags + [Diagnostic("type-mismatch", expr.span, extra={"msg": "Function did not return a value"})]
                except ReturnException as r:
                    diags.extend(r.diags)
                    return r.typ, diags
                    
            return ERROR_TYPE, diags + [Diagnostic("type-mismatch", expr.span, extra={"msg": f"Unknown function: {expr.name}"})]
        except NotConstantError as e:
            return ERROR_TYPE, [Diagnostic("type-mismatch", expr.span, extra={"msg": str(e)})]

    elif isinstance(expr, ArrayLit):
        diags = []
        for el in expr.elements:
            _, d = synth(el, ctx)
            diags.extend(d)
        return ERROR_TYPE, diags

    return ERROR_TYPE, []

def try_optimize_for_loop(stmt: ForStmt, ctx: TypeContext) -> tuple[bool, list[Diagnostic]]:
    # A simple heuristic: if the body only modifies one variable that is added to monotonically, unroll the math analytically instead
    if not isinstance(stmt.init, LetStmt) and not isinstance(stmt.init, VarStmt) and not isinstance(stmt.init, AssignStmt):
        return False, []
        
    loop_var = stmt.init.name
    if not isinstance(stmt.condition, BinOp) or stmt.condition.op != "<":
        return False, []
    if not isinstance(stmt.condition.left, VarRef) or stmt.condition.left.name != loop_var:
        return False, []
    if not isinstance(stmt.condition.right, NumberLit):
        return False, []
        
    max_iters = int(stmt.condition.right.value)
    
    if not isinstance(stmt.increment, AssignStmt) or stmt.increment.name != loop_var:
        return False, []
    if not isinstance(stmt.increment.value, BinOp) or stmt.increment.value.op != "+":
        return False, []
    if not isinstance(stmt.increment.value.left, VarRef) or stmt.increment.value.left.name != loop_var:
        return False, []
    if not isinstance(stmt.increment.value.right, NumberLit) or stmt.increment.value.right.value != 1.0:
        return False, []
        
    if len(stmt.body.stmts) != 1:
        return False, []
    
    body_stmt = stmt.body.stmts[0]
    if not isinstance(body_stmt, AssignStmt):
        return False, []
        
    if not isinstance(body_stmt.value, BinOp) or body_stmt.value.op != "+":
        return False, []
        
    # Check if a = a + expr
    left_is_a = isinstance(body_stmt.value.left, VarRef) and body_stmt.value.left.name == body_stmt.name
    right_is_a = isinstance(body_stmt.value.right, VarRef) and body_stmt.value.right.name == body_stmt.name
    
    if not (left_is_a or right_is_a):
        return False, []
        
    addend = body_stmt.value.right if left_is_a else body_stmt.value.left
    addend_typ, diags = synth(addend, ctx)
    
    # Do init
    diags.extend(check_stmt(stmt.init, ctx))
    
    # Now we want to analytically add `addend` `max_iters` times to `a`
    # However, since `+` tracks correlation, we can just do a simple loop but short-circuited mathematically
    # But since it's just `addend`, if addend is uncorrelated with `a`, we just add them.
    # To keep it perfectly identical to unrolling without the Python overhead:
    # We can just run the loop body manually in python without synthesizing the condition/increment repeatedly.
    target_var = body_stmt.name
    for _ in range(max_iters):
        diags.extend(check_stmt(body_stmt, ctx))
        
    from uncertain.distributions import Dist
    ctx.bind(loop_var, MeasuredType(Dist(float(max_iters), 0.0), frozenset()))
        
    return True, diags

def check_stmt(stmt: Stmt, ctx: TypeContext) -> list[Diagnostic]:
    if isinstance(stmt, (LetStmt, VarStmt)):
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
            elif isinstance(stmt.type_ann, UniformLit):
                if isinstance(stmt.type_ann.min_val, NumberLit) and isinstance(stmt.type_ann.max_val, NumberLit):
                    min_val = stmt.type_ann.min_val.value
                    max_val = stmt.type_ann.max_val.value
                    exp_mean = (min_val + max_val) / 2.0
                    exp_std = abs(max_val - min_val) / math.sqrt(12.0)
                    mean_ok = math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3)
                    std_ok = math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)
                    if not (mean_ok and std_ok):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, EmpiricalLit):
                if isinstance(stmt.type_ann.data, ArrayLit):
                    vals = []
                    for el in stmt.type_ann.data.elements:
                        if isinstance(el, NumberLit):
                            vals.append(el.value)
                    if vals:
                        exp_mean = sum(vals) / len(vals)
                        exp_std = math.sqrt(sum((v - exp_mean)**2 for v in vals) / len(vals))
                        mean_ok = math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3)
                        std_ok = math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)
                        if not (mean_ok and std_ok):
                            diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, LogNormalLit):
                if isinstance(stmt.type_ann.mean, NumberLit) and isinstance(stmt.type_ann.stddev, NumberLit):
                    mu = stmt.type_ann.mean.value
                    sigma = stmt.type_ann.stddev.value
                    exp_mean = math.exp(mu + (sigma**2) / 2.0)
                    exp_std = math.sqrt((math.exp(sigma**2) - 1.0) * math.exp(2.0*mu + sigma**2))
                    if not (math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, PoissonLit):
                if isinstance(stmt.type_ann.lam, NumberLit):
                    lam = stmt.type_ann.lam.value
                    exp_mean = lam
                    exp_std = math.sqrt(lam) if lam >= 0 else 0.0
                    if not (math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, BinomialLit):
                if isinstance(stmt.type_ann.n, NumberLit) and isinstance(stmt.type_ann.p, NumberLit):
                    n = stmt.type_ann.n.value
                    p = stmt.type_ann.p.value
                    exp_mean = n * p
                    exp_std = math.sqrt(n * p * (1 - p)) if n > 0 and 0 <= p <= 1 else 0.0
                    if not (math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, GammaLit):
                if isinstance(stmt.type_ann.k, NumberLit) and isinstance(stmt.type_ann.theta, NumberLit):
                    k = stmt.type_ann.k.value
                    theta = stmt.type_ann.theta.value
                    exp_mean = k * theta
                    exp_std = math.sqrt(k * (theta**2))
                    if not (math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, BernoulliLit):
                if isinstance(stmt.type_ann.p, NumberLit):
                    p = stmt.type_ann.p.value
                    exp_mean = p
                    exp_std = math.sqrt(p * (1 - p)) if 0 <= p <= 1 else 0.0
                    if not (math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, NegativeBinomialLit):
                if isinstance(stmt.type_ann.r, NumberLit) and isinstance(stmt.type_ann.p, NumberLit):
                    r = stmt.type_ann.r.value
                    p = stmt.type_ann.p.value
                    exp_mean = (p * r) / (1 - p) if p > 0 and p < 1 and r > 0 else 0.0
                    exp_std = math.sqrt((p * r) / ((1 - p)**2)) if p > 0 and p < 1 and r > 0 else 0.0
                    if not (math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, GeometricLit):
                if isinstance(stmt.type_ann.p, NumberLit):
                    p = stmt.type_ann.p.value
                    exp_mean = 1.0 / p if p > 0 and p <= 1 else 0.0
                    exp_std = math.sqrt((1.0 - p) / (p**2)) if p > 0 and p <= 1 else 0.0
                    if not (math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, ExponentialLit):
                if isinstance(stmt.type_ann.lam, NumberLit):
                    lam = stmt.type_ann.lam.value
                    exp_mean = 1.0 / lam if lam > 0 else 0.0
                    exp_std = math.sqrt(1.0 / (lam**2)) if lam > 0 else 0.0
                    if not (math.isclose(inferred.dist.mean, exp_mean, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, exp_std, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
            elif isinstance(stmt.type_ann, ExactLit):
                if isinstance(stmt.type_ann.value, NumberLit):
                    exp_val = stmt.type_ann.value.value
                    if not (math.isclose(inferred.dist.mean, exp_val, rel_tol=1e-3, abs_tol=1e-3) and math.isclose(inferred.dist.stddev, 0.0, rel_tol=1e-3, abs_tol=1e-3)):
                        diags.append(Diagnostic("type-mismatch", stmt.span))
        
        if isinstance(inferred.dist, tuple):
            final_deps = inferred.deps
        elif inferred.dist.stddev > 0 and not inferred.deps:
            final_deps = frozenset({stmt.name})
        elif inferred.dist.stddev > 0:
            final_deps = inferred.deps | frozenset({stmt.name})
        else:
            final_deps = inferred.deps
            
        ctx.bind(stmt.name, MeasuredType(inferred.dist, final_deps))
        return diags
        
    elif isinstance(stmt, AssignStmt):
        typ = ctx.lookup(stmt.name)
        if not typ:
            return [Diagnostic("undefined-var", stmt.span, extra={"name": stmt.name})]
        inferred, diags = synth(stmt.value, ctx)
        if isinstance(inferred.dist, tuple):
            final_deps = inferred.deps
        elif inferred.dist.stddev > 0 and not inferred.deps:
            final_deps = frozenset({stmt.name})
        elif inferred.dist.stddev > 0:
            final_deps = inferred.deps | frozenset({stmt.name})
        else:
            final_deps = inferred.deps
            
        ctx.bind(stmt.name, MeasuredType(inferred.dist, final_deps))
        return diags
        
    elif isinstance(stmt, WhileStmt):
        diags = []
        iters = 0
        while True:
            cond_typ, d = synth(stmt.condition, ctx)
            diags.extend(d)
            if cond_typ is ERROR_TYPE or cond_typ.dist.mean == 0.0:
                break
            for s in stmt.body.stmts:
                diags.extend(check_stmt(s, ctx))
            iters += 1
            if iters > ctx.max_unroll:
                diags.append(Diagnostic("uncertain-branch", stmt.span, extra={"msg": f"Loop iteration limit exceeded ({ctx.max_unroll})"}))
                break
        return diags
        
    elif isinstance(stmt, IfStmt):
        cond_typ, diags = synth(stmt.condition, ctx)
        if cond_typ is not ERROR_TYPE and cond_typ.dist.mean != 0.0:
            for s in stmt.true_body.stmts:
                diags.extend(check_stmt(s, ctx))
        elif stmt.false_body:
            for s in stmt.false_body.stmts:
                diags.extend(check_stmt(s, ctx))
        return diags
        
    elif isinstance(stmt, ForStmt):
        opt_success, opt_diags = try_optimize_for_loop(stmt, ctx)
        if opt_success:
            return opt_diags
            
        diags = []
        diags.extend(check_stmt(stmt.init, ctx))
        iters = 0
        while True:
            cond_typ, d = synth(stmt.condition, ctx)
            diags.extend(d)
            if cond_typ is ERROR_TYPE or cond_typ.dist.mean == 0.0:
                break
            for s in stmt.body.stmts:
                diags.extend(check_stmt(s, ctx))
            diags.extend(check_stmt(stmt.increment, ctx))
            iters += 1
            if iters > ctx.max_unroll:
                diags.append(Diagnostic("uncertain-branch", stmt.span, extra={"msg": f"Loop iteration limit exceeded ({ctx.max_unroll})"}))
                break
        return diags
        
    elif isinstance(stmt, FnDefStmt):
        ctx.functions[stmt.name] = stmt
        return []
        
    elif isinstance(stmt, ReturnStmt):
        typ, diags = synth(stmt.value, ctx)
        raise ReturnException(typ, diags)
        
    return []
