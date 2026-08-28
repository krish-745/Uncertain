import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
from uncertain.ast_nodes import *
from uncertain.distributions import (
    Dist, add, sub, mul_independent, div_independent, square, sqrt_dist, correlated_product,
    MathDomainError
)
from uncertain.diagnostics import Diagnostic
from uncertain.dependency import check_reuse, DepSet

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
        
        non_normal_families = ("Uniform", "Empirical", "LogNormal", "Poisson", "Binomial", "Gamma", "Bernoulli", "NegativeBinomial", "Geometric", "Exponential")
        if lt.dist.family in non_normal_families or rt.dist.family in non_normal_families:
            diags.append(Diagnostic("approximation-warning", expr.span, extra={"msg": "Moment-matching approximation used for non-Normal distribution"}, severity="warning"))

        if expr.op in ("+", "-"):
            fn = add if expr.op == "+" else sub
            return MeasuredType(fn(lt.dist, rt.dist), lt.deps | rt.deps), diags
            
        elif expr.op in ("<", ">"):
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
        if expr.name == "square" and len(expr.args) == 1:
            at, ad = synth(expr.args[0], ctx)
            return MeasuredType(square(at.dist), at.deps), ad
            
        elif expr.name == "sqrt" and len(expr.args) == 1:
            at, ad = synth(expr.args[0], ctx)
            try:
                res = sqrt_dist(at.dist)
            except MathDomainError as e:
                return ERROR_TYPE, ad + [Diagnostic("math-domain-error", expr.span, extra={"msg": str(e)})]
            return MeasuredType(res, at.deps), ad
            
        elif expr.name == "correlated" and len(expr.args) >= 2:
            at, ad = synth(expr.args[0], ctx)
            bt, bd = synth(expr.args[1], ctx)
            cov_expr = expr.kwargs.get("cov", NumberLit(0.0, expr.span))
            cov = eval_const(cov_expr)
            # Pre-cleared reuse, so we don't check_reuse
            return MeasuredType(correlated_product(at.dist, bt.dist, cov), at.deps | bt.deps), ad + bd
            
        elif expr.name == "sensor_read":
            return MeasuredType(Dist(10.0, 1.0), frozenset()), []
            
        elif expr.name == "uniform_read":
            return MeasuredType(Dist(5.0, 2.8867, "Uniform"), frozenset()), []
            
        elif expr.name == "empirical_read" and len(expr.args) == 1:
            if isinstance(expr.args[0], ArrayLit):
                vals = []
                for el in expr.args[0].elements:
                    if isinstance(el, NumberLit):
                        vals.append(el.value)
                if vals:
                    mean = sum(vals) / len(vals)
                    stddev = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
                    return MeasuredType(Dist(mean, stddev, "Empirical"), frozenset()), []
            return ERROR_TYPE, []
            
        elif expr.name == "lognormal_read" and len(expr.args) == 2:
            mu, sigma = eval_const(expr.args[0]), eval_const(expr.args[1])
            m = math.exp(mu + (sigma**2) / 2.0)
            s = math.sqrt((math.exp(sigma**2) - 1.0) * math.exp(2.0*mu + sigma**2))
            return MeasuredType(Dist(m, s, "LogNormal"), frozenset()), []
        elif expr.name == "poisson_read" and len(expr.args) == 1:
            lam = eval_const(expr.args[0])
            return MeasuredType(Dist(lam, math.sqrt(lam) if lam >= 0 else 0.0, "Poisson"), frozenset()), []
        elif expr.name == "binomial_read" and len(expr.args) == 2:
            n, p = eval_const(expr.args[0]), eval_const(expr.args[1])
            return MeasuredType(Dist(n * p, math.sqrt(n * p * (1 - p)) if n > 0 and 0 <= p <= 1 else 0.0, "Binomial"), frozenset()), []
        elif expr.name == "gamma_read" and len(expr.args) == 2:
            k, theta = eval_const(expr.args[0]), eval_const(expr.args[1])
            return MeasuredType(Dist(k * theta, math.sqrt(k * (theta**2)), "Gamma"), frozenset()), []
        elif expr.name == "bernoulli_read" and len(expr.args) == 1:
            p = eval_const(expr.args[0])
            return MeasuredType(Dist(p, math.sqrt(p * (1 - p)) if 0 <= p <= 1 else 0.0, "Bernoulli"), frozenset()), []
        elif expr.name == "negbinom_read" and len(expr.args) == 2:
            r, p = eval_const(expr.args[0]), eval_const(expr.args[1])
            m = (p * r) / (1 - p) if p > 0 and p < 1 and r > 0 else 0.0
            s = math.sqrt((p * r) / ((1 - p)**2)) if p > 0 and p < 1 and r > 0 else 0.0
            return MeasuredType(Dist(m, s, "NegativeBinomial"), frozenset()), []
        elif expr.name == "geometric_read" and len(expr.args) == 1:
            p = eval_const(expr.args[0])
            m = 1.0 / p if p > 0 and p <= 1 else 0.0
            s = math.sqrt((1.0 - p) / (p**2)) if p > 0 and p <= 1 else 0.0
            return MeasuredType(Dist(m, s, "Geometric"), frozenset()), []
        elif expr.name == "exponential_read" and len(expr.args) == 1:
            lam = eval_const(expr.args[0])
            m = 1.0 / lam if lam > 0 else 0.0
            s = math.sqrt(1.0 / (lam**2)) if lam > 0 else 0.0
            return MeasuredType(Dist(m, s, "Exponential"), frozenset()), []
            
        return ERROR_TYPE, []

    elif isinstance(expr, ArrayLit):
        diags = []
        for el in expr.elements:
            _, d = synth(el, ctx)
            diags.extend(d)
        return ERROR_TYPE, diags

    return ERROR_TYPE, []

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
        
        final_deps = inferred.deps | frozenset({stmt.name})
        ctx.bind(stmt.name, MeasuredType(inferred.dist, final_deps))
        return diags
        
    elif isinstance(stmt, AssignStmt):
        typ = ctx.lookup(stmt.name)
        if not typ:
            return [Diagnostic("undefined-var", stmt.span, extra={"name": stmt.name})]
        inferred, diags = synth(stmt.value, ctx)
        final_deps = inferred.deps | frozenset({stmt.name})
        ctx.bind(stmt.name, MeasuredType(inferred.dist, final_deps))
        return diags
        
    elif isinstance(stmt, WhileStmt):
        _, diags = synth(stmt.condition, ctx)
        for s in stmt.body.stmts:
            diags.extend(check_stmt(s, ctx))
        return diags
        
    return []
