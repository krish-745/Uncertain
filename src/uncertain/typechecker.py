import difflib
import math
import operator
import os
from dataclasses import dataclass, field, fields
from typing import Callable, Optional
from uncertain.ast_nodes import *
from uncertain.distributions import (
    Dist, MathDomainError, product, square, sqrt_dist, abs_dist, log_dist, exp_dist, sin_dist, cos_dist,
    pow_const, moments, empirical, probability, FAMILY_PARAMS,
)
from uncertain.diagnostics import Diagnostic, diagnostic_message
from uncertain.dependency import AffineForm, get_cov, get_var, linear_comb_affine, scale_affine

@dataclass(frozen=True)
class MeasuredType:
    dist: Dist | tuple["MeasuredType", ...] | dict[str, "MeasuredType"]
    deps: AffineForm
    # True if the value is exactly `mean + Σ coeff·ε` over its noise sources. False after a
    # nonlinear operation (product, division, sqrt, ...), whose result is only summarised by moments.
    linear: bool = field(default=True, compare=False)
    # For module structs: the functions the module defines (name -> Closure)
    functions: dict = field(default_factory=dict, compare=False)

@dataclass(frozen=True)
class Closure:
    """A function together with the scope it was defined in (lexical scoping)."""
    fn: FnDefStmt
    env: "TypeContext"

@dataclass(frozen=True)
class NoiseSource:
    """Where an independent noise source ε came from.

    kind "read":     a `*_read()` call; the reading is X = mean + stddev·ε with the given family/params.
    kind "residual": variance left over by a nonlinear operation (not normally distributed).
    """
    kind: str
    family: Optional[str] = None
    params: Optional[tuple] = None
    values: Optional[tuple] = None   # data of an Empirical reading
    dist: Optional[Dist] = None

RESIDUAL = NoiseSource("residual")

# Returned by `synth` after a diagnostic has been reported. Compared by identity, so that
# operations on an erroneous value propagate the error silently instead of cascading.
ERROR_TYPE = MeasuredType(Dist(0.0, 0.0), {})

DET_TOL = 1e-9            # stddev below which a value is treated as deterministic
MAX_CALL_DEPTH = 64       # maximum nesting of user function calls
DELTA_CV_LIMIT = 0.3      # coefficient of variation above which the delta method is flagged

class _SharedState:
    """State shared by every scope of one program check, including imported modules."""
    def __init__(self, max_unroll: int):
        self.max_unroll = max_unroll
        self.next_id = 1
        self.call_depth = 0
        self.import_stack: list[str] = []
        self.sources: dict[int, NoiseSource] = {}

class TypeContext:
    def __init__(self, parent: "TypeContext" = None, max_unroll: int = 1000, base_dir: str = ".", shared: _SharedState = None):
        self.bindings: dict[str, MeasuredType] = {}
        self.immutable: set[str] = set()              # names declared with `let` (or parameters, imports)
        self.functions: dict[str, Closure] = {}
        self.parent = parent
        self.result: Optional[MeasuredType] = None   # type of the program's trailing expression, if any
        if parent is not None:
            self.shared = parent.shared
            self.base_dir = parent.base_dir
        else:
            self.shared = shared or _SharedState(max_unroll)
            self.base_dir = base_dir

    @property
    def max_unroll(self) -> int:
        return self.shared.max_unroll

    def new_source(self, source: NoiseSource) -> int:
        id = self.shared.next_id
        self.shared.next_id += 1
        self.shared.sources[id] = source
        return id

    def lookup(self, name: str) -> Optional[MeasuredType]:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def bind(self, name: str, typ: MeasuredType, mutable: bool = False):
        """Declare `name` in this scope (shadowing any outer binding)."""
        self.bindings[name] = typ
        if mutable:
            self.immutable.discard(name)
        else:
            self.immutable.add(name)

    def assign(self, name: str, typ: MeasuredType) -> Optional[str]:
        """Update an existing variable in the scope that declared it.

        Returns None on success, "undefined" if there is no such variable, or "immutable" if it
        was declared with `let`.
        """
        scope = self
        while scope is not None:
            if name in scope.bindings:
                if name in scope.immutable:
                    return "immutable"
                scope.bindings[name] = typ
                return None
            scope = scope.parent
        return "undefined"

    def lookup_fn(self, name: str) -> Optional[Closure]:
        if name in self.functions:
            return self.functions[name]
        if self.parent:
            return self.parent.lookup_fn(name)
        return None

    def all_fn_names(self) -> set[str]:
        names = set(self.functions)
        if self.parent:
            names |= self.parent.all_fn_names()
        return names

class ReturnException(Exception):
    def __init__(self, typ: MeasuredType, diags: list[Diagnostic], span: Span):
        self.typ = typ
        self.diags = diags
        self.span = span

# --- helpers --------------------------------------------------------------------

def _diag(kind: str, span: Span, msg: str = None, severity: str = "error", **extra) -> Diagnostic:
    if msg is not None:
        extra["msg"] = msg
    return Diagnostic(kind, span, extra=extra, severity=severity)

def _dedupe(diags: list[Diagnostic]) -> list[Diagnostic]:
    """Drop repeated diagnostics (e.g. the same warning from every unrolled loop iteration)."""
    seen = set()
    out = []
    for d in diags:
        key = (d.kind, d.severity, d.span.line, d.span.col, d.span.length, d.extra.get("msg"), d.extra.get("name"))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def _const(value: float, family: str = "Normal") -> MeasuredType:
    return MeasuredType(Dist(value, 0.0, family), {})

def is_scalar(t: MeasuredType) -> bool:
    return isinstance(t.dist, Dist)

def is_deterministic(t: MeasuredType) -> bool:
    return is_scalar(t) and math.isclose(t.dist.stddev, 0.0, abs_tol=DET_TOL)

def _describe(t: MeasuredType) -> str:
    if isinstance(t.dist, tuple):
        return f"an array of length {len(t.dist)}"
    if isinstance(t.dist, dict):
        return "a struct"
    return "a number"

def _require_scalar(t: MeasuredType, span: Span, diags: list[Diagnostic]) -> bool:
    if is_scalar(t):
        return True
    diags.append(_diag("invalid-operand", span, f"expected a number or distribution, got {_describe(t)}"))
    return False

def _as_const(t: MeasuredType, span: Span, diags: list[Diagnostic], what: str) -> Optional[float]:
    if t is ERROR_TYPE or not _require_scalar(t, span, diags):
        return None
    if not is_deterministic(t):
        diags.append(_diag("not-constant", span, f"{what} must be deterministic, but it has stddev {t.dist.stddev:.4g}"))
        return None
    return t.dist.mean

def _const_value(expr: Expr, ctx: "TypeContext", diags: list[Diagnostic], what: str = "this value") -> Optional[float]:
    t, d = synth(expr, ctx)
    diags.extend(d)
    return _as_const(t, expr.span, diags, what)

def _non_normal_families(t: MeasuredType, ctx: "TypeContext") -> set[str]:
    """Families of the non-Normal readings an uncertain value depends on."""
    if not is_scalar(t) or t.dist.stddev <= DET_TOL:
        return set()
    families = set()
    for k in t.deps:
        src = ctx.shared.sources.get(k)
        if src is not None and src.kind == "read" and src.family != "Normal":
            families.add(src.family)
    return families

def _approx_warning(span: Span, operands: list[MeasuredType], diags: list[Diagnostic], what: str, ctx: "TypeContext"):
    families = sorted(set().union(*(_non_normal_families(t, ctx) for t in operands)))
    if families:
        diags.append(_diag("approximation-warning", span,
                           f"{what} assumes Normal inputs; the result for a {'/'.join(families)} input is a moment-matching approximation",
                           severity="warning"))

def _delta_warning(span: Span, t: MeasuredType, diags: list[Diagnostic], what: str):
    d = t.dist
    if d.stddev > DET_TOL and (d.mean == 0 or d.stddev / abs(d.mean) > DELTA_CV_LIMIT):
        cv = "infinite" if d.mean == 0 else f"{d.stddev / abs(d.mean):.2f}"
        diags.append(_diag("delta-method-warning", span,
                           f"{what} uses a first-order approximation, which is unreliable here (stddev/|mean| = {cv})",
                           severity="warning"))

def _math_msg(e: Exception) -> str:
    if isinstance(e, MathDomainError):
        return str(e)
    if isinstance(e, OverflowError):
        return "numeric overflow: the result is too large to represent"
    if isinstance(e, ZeroDivisionError):
        return "division by zero"
    return f"invalid numeric operation ({e})"

MATH_ERRORS = (MathDomainError, OverflowError, ZeroDivisionError, ValueError)

def create_measured(dist: Dist, affine: AffineForm, ctx: TypeContext, linear: bool = False,
                    source: NoiseSource = RESIDUAL) -> MeasuredType:
    """Pair a distribution with a dependency form whose variance matches it exactly.

    `affine` is the linear part of the result in terms of the independent noise sources. Any
    variance not explained by it becomes a fresh, independent noise term (described by `source`);
    if the linear part overstates the variance it is scaled down so that correlations stay consistent.
    """
    var_exact = dist.stddev ** 2
    var_affine = get_var(affine)
    tol = 1e-12 * max(1.0, var_exact)
    if var_affine > var_exact + tol:
        affine = scale_affine(affine, math.sqrt(var_exact / var_affine)) if var_exact > 0 else {}
    elif var_exact > var_affine + tol:
        affine = dict(affine)
        affine[ctx.new_source(source)] = math.sqrt(var_exact - var_affine)
    return MeasuredType(dist, affine, linear=linear or dist.stddev <= DET_TOL)

def _read(family: str, params: list[float], ctx: TypeContext, values: Optional[list[float]] = None) -> MeasuredType:
    """A new reading from a named distribution: a noise source independent of everything so far."""
    dist = empirical(values) if family == "Empirical" else moments(family, params)
    source = NoiseSource("read", family, tuple(params), tuple(values) if values is not None else None, dist)
    return create_measured(dist, {}, ctx, linear=True, source=source)

def _product(lt: MeasuredType, rt: MeasuredType, cov: float, ctx: TypeContext) -> MeasuredType:
    dist = product(lt.dist, rt.dist, cov)
    affine = linear_comb_affine(lt.deps, rt.dist.mean, rt.deps, lt.dist.mean)
    # scaling by a constant keeps a value linear in its noise sources; a true product does not
    linear = (is_deterministic(lt) and rt.linear) or (is_deterministic(rt) and lt.linear)
    return create_measured(dist, affine, ctx, linear=linear)

def _dist_lit_params(lit: DistLit) -> tuple[str, list[Expr]]:
    return DIST_LIT_FAMILIES[type(lit)], [getattr(lit, f.name) for f in fields(lit)]

def _fmt(x: float) -> str:
    return f"{x:.6g}"

# --- expressions ----------------------------------------------------------------

COMPARISONS = {
    "<": operator.lt, ">": operator.gt, "<=": operator.le,
    ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
}

def synth(expr: Expr, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    if isinstance(expr, NumberLit):
        return _const(expr.value), []

    elif isinstance(expr, VarRef):
        typ = ctx.lookup(expr.name)
        if typ is None:
            return ERROR_TYPE, [_diag("undefined-var", expr.span, name=expr.name)]
        return typ, []

    elif isinstance(expr, ArrayLit):
        diags = []
        elements = []
        for el in expr.elements:
            t, d = synth(el, ctx)
            diags.extend(d)
            elements.append(t)
        return MeasuredType(tuple(elements), {}), diags

    elif isinstance(expr, StructLit):
        diags = []
        struct_fields = {}
        for name, el in expr.fields.items():
            t, d = synth(el, ctx)
            diags.extend(d)
            struct_fields[name] = t
        return MeasuredType(struct_fields, {}), diags

    elif isinstance(expr, ArrayAccess):
        return _synth_index(expr, ctx)

    elif isinstance(expr, FieldAccess):
        obj_t, diags = synth(expr.obj, ctx)
        if obj_t is ERROR_TYPE:
            return ERROR_TYPE, diags
        if not isinstance(obj_t.dist, dict):
            diags.append(_diag("invalid-operand", expr.span, f"cannot access field '{expr.field}' of {_describe(obj_t)}"))
            return ERROR_TYPE, diags
        if expr.field not in obj_t.dist:
            available = ", ".join(sorted(obj_t.dist)) or "none"
            diags.append(_diag("invalid-operand", expr.span, f"struct has no field '{expr.field}' (fields: {available})"))
            return ERROR_TYPE, diags
        return obj_t.dist[expr.field], diags

    elif isinstance(expr, BinOp):
        return _synth_binop(expr, ctx)

    elif isinstance(expr, Call):
        return _synth_call(expr, ctx)

    return ERROR_TYPE, [_diag("internal-error", getattr(expr, "span", Span(1, 1, 1)), f"unsupported expression {type(expr).__name__}")]

def _synth_index(expr: ArrayAccess, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    arr_t, diags = synth(expr.array, ctx)
    idx_t, idx_diags = synth(expr.index, ctx)
    diags.extend(idx_diags)
    if arr_t is ERROR_TYPE or idx_t is ERROR_TYPE:
        return ERROR_TYPE, diags
    if not isinstance(arr_t.dist, tuple):
        diags.append(_diag("invalid-operand", expr.array.span, f"cannot index into {_describe(arr_t)}"))
        return ERROR_TYPE, diags

    idx = _as_const(idx_t, expr.index.span, diags, "an array index")
    if idx is None:
        return ERROR_TYPE, diags
    if not float(idx).is_integer():
        diags.append(_diag("invalid-operand", expr.index.span, f"array index must be an integer, got {_fmt(idx)}"))
        return ERROR_TYPE, diags
    if not 0 <= idx < len(arr_t.dist):
        diags.append(_diag("invalid-operand", expr.index.span, f"index {int(idx)} is out of bounds for an array of length {len(arr_t.dist)}"))
        return ERROR_TYPE, diags
    return arr_t.dist[int(idx)], diags

def _synth_binop(expr: BinOp, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    lt, diags = synth(expr.left, ctx)
    rt, rd = synth(expr.right, ctx)
    diags.extend(rd)
    if lt is ERROR_TYPE or rt is ERROR_TYPE:
        return ERROR_TYPE, diags
    left_ok = _require_scalar(lt, expr.left.span, diags)
    right_ok = _require_scalar(rt, expr.right.span, diags)
    if not (left_ok and right_ok):
        return ERROR_TYPE, diags

    op = expr.op
    if op in COMPARISONS:
        if not (is_deterministic(lt) and is_deterministic(rt)):
            diags.append(_diag("uncertain-branch", expr.span,
                               f"cannot compare uncertain values with '{op}' at compile time; use prob(...) to get the probability instead"))
            return ERROR_TYPE, diags
        return _const(1.0 if COMPARISONS[op](lt.dist.mean, rt.dist.mean) else 0.0), diags

    try:
        if op in ("+", "-"):
            # Sums are exact for any distribution family: means add, variances follow the dependencies.
            scale_r = 1.0 if op == "+" else -1.0
            affine = linear_comb_affine(lt.deps, 1.0, rt.deps, scale_r)
            dist = Dist(lt.dist.mean + scale_r * rt.dist.mean, math.sqrt(get_var(affine)))
            return MeasuredType(dist, affine, linear=lt.linear and rt.linear), diags

        if op == "*":
            if not (is_deterministic(lt) or is_deterministic(rt)):
                _approx_warning(expr.span, [lt, rt], diags, "multiplication", ctx)
            return _product(lt, rt, get_cov(lt.deps, rt.deps), ctx), diags

        if op == "/":
            if rt.dist.mean == 0:
                if is_deterministic(rt):
                    raise MathDomainError("division by zero")
                raise MathDomainError("cannot divide by a distribution with a zero mean")
            if not is_deterministic(rt):
                _approx_warning(expr.span, [lt, rt], diags, "division", ctx)
                _delta_warning(expr.span, rt, diags, "division by an uncertain value")
            mean = lt.dist.mean / rt.dist.mean
            # Delta method (exact when dividing by a constant)
            affine = linear_comb_affine(lt.deps, 1.0 / rt.dist.mean, rt.deps, -lt.dist.mean / (rt.dist.mean**2))
            linear = is_deterministic(rt) and lt.linear
            return create_measured(Dist(mean, math.sqrt(get_var(affine))), affine, ctx, linear=linear), diags
    except MATH_ERRORS as e:
        diags.append(_diag("math-domain-error", expr.span, _math_msg(e)))
        return ERROR_TYPE, diags

    diags.append(_diag("internal-error", expr.span, f"unknown operator '{op}'"))
    return ERROR_TYPE, diags

# name -> (distribution function, slope of the linear dependency on the input).
# The slope is E[f'(X)], which by Stein's lemma reproduces Cov(f(X), X) exactly for normal X.
UNARY_BUILTINS: dict[str, tuple[Callable[[Dist], Dist], Callable[[Dist, Dist], float]]] = {
    "square": (square, lambda d, r: 2.0 * d.mean),
    "sqrt": (sqrt_dist, lambda d, r: 0.5 / math.sqrt(d.mean) if d.mean > 0 else 0.0),
    "abs": (abs_dist, lambda d, r: math.erf(d.mean / (d.stddev * math.sqrt(2.0))) if d.stddev > 0 else math.copysign(1.0, d.mean) if d.mean else 0.0),
    "log": (log_dist, lambda d, r: 1.0 / d.mean),
    "exp": (exp_dist, lambda d, r: r.mean),
    "sin": (sin_dist, lambda d, r: math.cos(d.mean) * math.exp(-d.stddev**2 / 2.0)),
    "cos": (cos_dist, lambda d, r: -math.sin(d.mean) * math.exp(-d.stddev**2 / 2.0)),
}
DELTA_METHOD_BUILTINS = ("sqrt", "log")

# `*_read` builtins that construct a named distribution from constant parameters
READ_BUILTINS = {
    "normal_read": "Normal",
    "lognormal_read": "LogNormal",
    "poisson_read": "Poisson",
    "binomial_read": "Binomial",
    "gamma_read": "Gamma",
    "bernoulli_read": "Bernoulli",
    "negbinom_read": "NegativeBinomial",
    "geometric_read": "Geometric",
    "exponential_read": "Exponential",
}

# name -> number of positional arguments (every builtin, including the ones handled specially)
BUILTIN_ARITY = {
    **{name: 1 for name in UNARY_BUILTINS},
    **{name: len(FAMILY_PARAMS[family]) for name, family in READ_BUILTINS.items()},
    "pow": 2, "correlated": 2, "prob": 1,
    "sensor_read": 0, "uniform_read": 0, "empirical_read": 1,
    "map": 2, "filter": 2, "reduce": 3,
}
BUILTIN_KWARGS = {"correlated": {"cov"}}

def _synth_call(expr: Call, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    name = expr.name
    diags: list[Diagnostic] = []

    if expr.target is not None:
        # `module.function(...)`
        target_t, diags = synth(expr.target, ctx)
        if target_t is ERROR_TYPE:
            return ERROR_TYPE, diags
        if not isinstance(target_t.dist, dict):
            diags.append(_diag("invalid-operand", expr.target.span, f"cannot call '{name}' on {_describe(target_t)}"))
            return ERROR_TYPE, diags
        closure = target_t.functions.get(name)
        if closure is None:
            available = ", ".join(sorted(target_t.functions)) or "none"
            diags.append(_diag("unknown-function", expr.span, f"no function named '{name}' here (functions: {available})"))
            return ERROR_TYPE, diags
    elif name in BUILTIN_ARITY:
        arity = BUILTIN_ARITY[name]
        if len(expr.args) != arity:
            return ERROR_TYPE, [_diag("arity-mismatch", expr.span, f"{name}() takes {arity} positional argument(s), got {len(expr.args)}")]
        unexpected = set(expr.kwargs) - BUILTIN_KWARGS.get(name, set())
        if unexpected:
            return ERROR_TYPE, [_diag("arity-mismatch", expr.span, f"{name}() got unexpected keyword argument(s): {', '.join(sorted(unexpected))}")]
        try:
            return _synth_builtin(expr, ctx)
        except MATH_ERRORS as e:
            return ERROR_TYPE, [_diag("math-domain-error", expr.span, _math_msg(e))]

    else:
        closure = ctx.lookup_fn(name)
        if closure is None:
            candidates = difflib.get_close_matches(name, list(BUILTIN_ARITY) + sorted(ctx.all_fn_names()), n=1)
            hint = f"; did you mean '{candidates[0]}'?" if candidates else ""
            return ERROR_TYPE, [_diag("unknown-function", expr.span, f"no function named '{name}'{hint}")]
    if expr.kwargs:
        return ERROR_TYPE, diags + [_diag("arity-mismatch", expr.span, f"user-defined function '{name}' does not accept keyword arguments")]

    arg_types = []
    for arg_expr in expr.args:
        t, d = synth(arg_expr, ctx)
        diags.extend(d)
        arg_types.append(t)
    result, d = _call_function(closure, arg_types, expr.span, ctx)
    diags.extend(d)
    return result, diags

def _synth_builtin(expr: Call, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    name, args = expr.name, expr.args
    diags: list[Diagnostic] = []

    if name == "prob":
        return _synth_prob(expr, ctx)

    if name in UNARY_BUILTINS:
        at, diags = synth(args[0], ctx)
        if at is ERROR_TYPE or not _require_scalar(at, args[0].span, diags):
            return ERROR_TYPE, diags
        dist_fn, slope_fn = UNARY_BUILTINS[name]
        result = dist_fn(at.dist)
        if not is_deterministic(at):
            _approx_warning(expr.span, [at], diags, f"{name}()", ctx)
            if name in DELTA_METHOD_BUILTINS:
                _delta_warning(expr.span, at, diags, f"{name}()")
        return create_measured(result, scale_affine(at.deps, slope_fn(at.dist, result)), ctx), diags

    if name == "pow":
        at, diags = synth(args[0], ctx)
        n_val = _const_value(args[1], ctx, diags, "the exponent")
        if at is ERROR_TYPE or n_val is None or not _require_scalar(at, args[0].span, diags):
            return ERROR_TYPE, diags
        if not float(n_val).is_integer():
            diags.append(_diag("not-constant", args[1].span, f"the exponent must be an integer constant, got {_fmt(n_val)}"))
            return ERROR_TYPE, diags
        n = int(n_val)
        d = at.dist
        result = pow_const(d, n)
        if not is_deterministic(at) and n not in (0, 1):
            _approx_warning(expr.span, [at], diags, "pow()", ctx)
        if n in (0, 1):
            slope = float(n)
        elif n == 3:
            slope = 3.0 * (d.mean**2 + d.stddev**2)   # E[3X²]
        else:
            slope = n * (d.mean ** (n - 1))
        return create_measured(result, scale_affine(at.deps, slope), ctx, linear=(n == 1 and at.linear)), diags

    if name == "correlated":
        at, diags = synth(args[0], ctx)
        bt, bd = synth(args[1], ctx)
        diags.extend(bd)
        cov = 0.0
        if "cov" in expr.kwargs:
            cov = _const_value(expr.kwargs["cov"], ctx, diags, "the covariance")
            if cov is None:
                return ERROR_TYPE, diags
        if at is ERROR_TYPE or bt is ERROR_TYPE:
            return ERROR_TYPE, diags
        if not (_require_scalar(at, args[0].span, diags) and _require_scalar(bt, args[1].span, diags)):
            return ERROR_TYPE, diags
        bound = at.dist.stddev * bt.dist.stddev
        if abs(cov) > bound * (1 + 1e-9) + 1e-12:
            diags.append(_diag("math-domain-error", expr.span,
                               f"cov={_fmt(cov)} is impossible: |cov| cannot exceed stddev(a) * stddev(b) = {_fmt(bound)}"))
            return ERROR_TYPE, diags
        _approx_warning(expr.span, [at, bt], diags, "correlated()", ctx)
        return _product(at, bt, cov, ctx), diags

    if name == "sensor_read":
        return _read("Normal", [10.0, 1.0], ctx), diags

    if name == "uniform_read":
        return _read("Uniform", [0.0, 10.0], ctx), diags

    if name in READ_BUILTINS:
        family = READ_BUILTINS[name]
        params = [_const_value(a, ctx, diags, f"{family} parameter '{p}'") for a, p in zip(args, FAMILY_PARAMS[family])]
        if any(p is None for p in params):
            return ERROR_TYPE, diags
        return _read(family, params, ctx), diags

    if name == "empirical_read":
        values = _const_array(args[0], ctx, diags)
        if values is None:
            return ERROR_TYPE, diags
        return _read("Empirical", [], ctx, values), diags

    if name in ("map", "filter", "reduce"):
        return _synth_higher_order(expr, ctx)

    return ERROR_TYPE, [_diag("internal-error", expr.span, f"builtin '{name}' is not implemented")]

def _const_array(expr: Expr, ctx: TypeContext, diags: list[Diagnostic]) -> Optional[list[float]]:
    t, d = synth(expr, ctx)
    diags.extend(d)
    if t is ERROR_TYPE:
        return None
    if not isinstance(t.dist, tuple):
        diags.append(_diag("invalid-operand", expr.span, f"expected an array of numbers, got {_describe(t)}"))
        return None
    values = []
    for i, el in enumerate(t.dist):
        v = _as_const(el, expr.span, diags, f"element {i} of the data array")
        if v is None:
            return None
        values.append(v)
    return values

def _synth_prob(expr: Call, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    cmp = expr.args[0]
    if not isinstance(cmp, BinOp) or cmp.op not in ("<", ">", "<=", ">="):
        return ERROR_TYPE, [_diag("invalid-operand", expr.span, "prob() requires an inequality such as prob(X > 5)")]

    lt, diags = synth(cmp.left, ctx)
    rt, rd = synth(cmp.right, ctx)
    diags.extend(rd)
    if lt is ERROR_TYPE or rt is ERROR_TYPE:
        return ERROR_TYPE, diags
    if not (_require_scalar(lt, cmp.left.span, diags) and _require_scalar(rt, cmp.right.span, diags)):
        return ERROR_TYPE, diags

    # D = X - Y, including the covariance between them; prob(X op Y) = P(D op 0)
    mean = lt.dist.mean - rt.dist.mean
    affine = linear_comb_affine(lt.deps, 1.0, rt.deps, -1.0)
    stddev = math.sqrt(get_var(affine))

    if math.isclose(stddev, 0.0, abs_tol=DET_TOL):
        p = 1.0 if COMPARISONS[cmp.op](mean, 0.0) else 0.0
        return _const(p, "Exact"), diags

    sources = [ctx.shared.sources.get(k) for k in affine]
    exact_form = lt.linear and rt.linear and all(s is not None and s.kind == "read" for s in sources)

    if exact_form and len(affine) == 1 and sources[0].family != "Normal":
        # D depends on a single reading X = m + s·ε, so D = mean + (c/s)(X - m): use X's exact CDF.
        (coeff,) = affine.values()
        src = sources[0]
        k = coeff / src.dist.stddev
        threshold = src.dist.mean - mean / k
        op = cmp.op if k > 0 else {"<": ">", ">": "<", "<=": ">=", ">=": "<="}[cmp.op]
        p = probability(src.family, list(src.params), op, threshold, list(src.values) if src.values else None)
        return _const(p, "Exact"), diags

    if not (exact_form and all(s.family == "Normal" for s in sources)):
        families = sorted({s.family for s in sources if s is not None and s.kind == "read" and s.family != "Normal"})
        if families:
            reason = f"it depends on non-Normal inputs ({'/'.join(families)})"
        else:
            reason = "it depends on the result of a nonlinear operation (such as a product, division or function)"
        diags.append(_diag("approximation-warning", expr.span,
                           f"prob() treats the difference of both sides as Normal, which is only an approximation here: {reason}",
                           severity="warning"))

    # Exact for a linear combination of Normal readings; a normal approximation otherwise
    cdf_0 = 0.5 * (1.0 + math.erf((0.0 - mean) / stddev / math.sqrt(2.0)))
    p = cdf_0 if cmp.op in ("<", "<=") else 1.0 - cdf_0
    return _const(p, "Exact"), diags

def _resolve_fn_arg(expr: Expr, caller: str, ctx: TypeContext, diags: list[Diagnostic]) -> Optional[Closure]:
    """Resolve a function passed by name (`f`) or from a module (`m.f`)."""
    if isinstance(expr, VarRef):
        closure = ctx.lookup_fn(expr.name)
        if closure is None:
            diags.append(_diag("unknown-function", expr.span, f"no function named '{expr.name}' (functions passed to {caller}() must be defined with `fn`)"))
        return closure
    if isinstance(expr, FieldAccess):
        obj_t, d = synth(expr.obj, ctx)
        diags.extend(d)
        if obj_t is ERROR_TYPE:
            return None
        closure = obj_t.functions.get(expr.field) if isinstance(obj_t.dist, dict) else None
        if closure is None:
            diags.append(_diag("unknown-function", expr.span, f"no function named '{expr.field}' here"))
        return closure
    diags.append(_diag("invalid-operand", expr.span, f"the second argument of {caller}() must be the name of a function"))
    return None

def _synth_higher_order(expr: Call, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    name = expr.name
    arr_t, diags = synth(expr.args[0], ctx)
    closure = _resolve_fn_arg(expr.args[1], name, ctx, diags)
    if arr_t is ERROR_TYPE or closure is None:
        return ERROR_TYPE, diags
    fn_def = closure.fn
    if not isinstance(arr_t.dist, tuple):
        diags.append(_diag("invalid-operand", expr.args[0].span, f"the first argument of {name}() must be an array, got {_describe(arr_t)}"))
        return ERROR_TYPE, diags

    needed = 2 if name == "reduce" else 1
    if len(fn_def.args) != needed:
        diags.append(_diag("arity-mismatch", expr.args[1].span,
                           f"{name}() needs a function of {needed} argument(s), but '{fn_def.name}' takes {len(fn_def.args)}"))
        return ERROR_TYPE, diags

    if name == "reduce":
        acc_t, d = synth(expr.args[2], ctx)
        diags.extend(d)
        if acc_t is ERROR_TYPE:
            return ERROR_TYPE, diags
        for el in arr_t.dist:
            acc_t, d = _call_function(closure, [acc_t, el], expr.span, ctx)
            diags.extend(d)
            if acc_t is ERROR_TYPE:
                return ERROR_TYPE, _dedupe(diags)
        return acc_t, _dedupe(diags)

    results = []
    for el in arr_t.dist:
        r, d = _call_function(closure, [el], expr.span, ctx)
        diags.extend(d)
        if r is ERROR_TYPE:
            return ERROR_TYPE, _dedupe(diags)
        if name == "map":
            results.append(r)
            continue
        keep = _as_const(r, expr.span, diags, "the filter condition")
        if keep is None:
            return ERROR_TYPE, _dedupe(diags)
        if keep != 0.0:
            results.append(el)
    return MeasuredType(tuple(results), {}), _dedupe(diags)

def _call_function(closure: Closure, args: list[MeasuredType], span: Span, ctx: TypeContext) -> tuple[MeasuredType, list[Diagnostic]]:
    fn_def, env = closure.fn, closure.env
    if len(args) != len(fn_def.args):
        return ERROR_TYPE, [_diag("arity-mismatch", span, f"function '{fn_def.name}' expects {len(fn_def.args)} argument(s), got {len(args)}")]
    if any(a is ERROR_TYPE for a in args):
        return ERROR_TYPE, []

    shared = ctx.shared
    if shared.call_depth >= MAX_CALL_DEPTH:
        return ERROR_TYPE, [_diag("recursion-limit", span,
                                  f"calls to '{fn_def.name}' nested more than {MAX_CALL_DEPTH} deep; recursion must stop after a bounded number of deterministic steps")]

    diags: list[Diagnostic] = []
    # Lexical scoping: the body sees the scope the function was defined in, not the caller's
    call_ctx = TypeContext(parent=env)
    for arg_def, arg_typ in zip(fn_def.args, args):
        if arg_def.type_ann is not None:
            diags.extend(check_annotation(arg_typ, arg_def.type_ann, span, env, f"argument '{arg_def.name}' of '{fn_def.name}'"))
        call_ctx.bind(arg_def.name, arg_typ)   # parameters are immutable, like `let`

    shared.call_depth += 1
    try:
        diags.extend(_check_block(fn_def.body.stmts, call_ctx))
        diags.append(_diag("invalid-return", span, f"function '{fn_def.name}' finished without returning a value"))
        return ERROR_TYPE, diags
    except ReturnException as r:
        diags.extend(r.diags)
        result = r.typ
    finally:
        shared.call_depth -= 1

    if fn_def.return_type is not None and result is not ERROR_TYPE:
        diags.extend(check_annotation(result, fn_def.return_type, span, env, f"the return value of '{fn_def.name}'"))
    return result, diags

def check_annotation(inferred: MeasuredType, type_ann: DistLit, span: Span, ctx: TypeContext, subject: str = "the value") -> list[Diagnostic]:
    """Check that `inferred` has the mean/stddev described by a `Measured<...>` annotation."""
    diags: list[Diagnostic] = []
    if inferred is ERROR_TYPE:
        return diags
    family, param_exprs = _dist_lit_params(type_ann)
    if not is_scalar(inferred):
        diags.append(_diag("type-mismatch", span, f"{subject} is annotated as Measured<{family}(...)> but is {_describe(inferred)}"))
        return diags

    try:
        if family == "Empirical":
            values = _const_array(param_exprs[0], ctx, diags)
            if values is None:
                return diags
            expected = empirical(values)
            label = f"Empirical([{', '.join(_fmt(v) for v in values)}])"
        else:
            params = [_const_value(p, ctx, diags, f"{family} parameter '{name}'") for p, name in zip(param_exprs, FAMILY_PARAMS[family])]
            if any(p is None for p in params):
                return diags
            expected = moments(family, params)
            label = f"{family}({', '.join(_fmt(p) for p in params)})"
    except MATH_ERRORS as e:
        diags.append(_diag("math-domain-error", span, f"invalid annotation: {_math_msg(e)}"))
        return diags

    got = inferred.dist
    mean_ok = math.isclose(got.mean, expected.mean, rel_tol=1e-3, abs_tol=1e-3)
    std_ok = math.isclose(got.stddev, expected.stddev, rel_tol=1e-3, abs_tol=1e-3)
    if not (mean_ok and std_ok):
        diags.append(_diag("type-mismatch", span,
                           f"{subject} is annotated as {label} (mean {_fmt(expected.mean)}, stddev {_fmt(expected.stddev)}), "
                           f"but the inferred distribution has mean {_fmt(got.mean)}, stddev {_fmt(got.stddev)}"))
    return diags

# --- statements -----------------------------------------------------------------

def _check_block(stmts: list[Stmt], ctx: TypeContext) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    try:
        for s in stmts:
            diags.extend(check_stmt(s, ctx))
    except ReturnException as r:
        raise ReturnException(r.typ, diags + r.diags, r.span)
    return diags

def check_stmt(stmt: Stmt, ctx: TypeContext) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    try:
        _check_stmt(stmt, ctx, diags)
    except ReturnException as r:
        # Keep the diagnostics produced before the `return` was reached
        raise ReturnException(r.typ, diags + r.diags, r.span)
    return diags

def _condition(cond: Expr, ctx: TypeContext, out: list[Diagnostic], what: str) -> Optional[bool]:
    """Evaluate a branch/loop condition. Returns None if it is erroneous or uncertain."""
    t, d = synth(cond, ctx)
    out.extend(d)
    if t is ERROR_TYPE or not _require_scalar(t, cond.span, out):
        return None
    if not is_deterministic(t):
        out.append(_diag("uncertain-branch", cond.span, f"the condition of this {what} is uncertain (stddev {t.dist.stddev:.4g})"))
        return None
    return t.dist.mean != 0.0

def _check_loop(stmt: WhileStmt | ForStmt, ctx: TypeContext, out: list[Diagnostic], step: Optional[Stmt]):
    what = "for loop" if isinstance(stmt, ForStmt) else "while loop"
    local: list[Diagnostic] = []
    iters = 0
    try:
        while _condition(stmt.condition, ctx, local, what):
            if iters >= ctx.max_unroll:
                local.append(_diag("loop-limit", stmt.span, f"this {what} did not finish within {ctx.max_unroll} iterations"))
                break
            local.extend(_check_block(stmt.body.stmts, ctx))
            if step is not None:
                step_diags = check_stmt(step, ctx)
                local.extend(step_diags)
                if any(d.severity == "error" for d in step_diags):
                    break   # the loop cannot advance; don't pile a loop-limit error on top
            iters += 1
    finally:
        out.extend(_dedupe(local))

def _check_stmt(stmt: Stmt, ctx: TypeContext, out: list[Diagnostic]):
    if isinstance(stmt, (LetStmt, VarStmt)):
        inferred, d = synth(stmt.value, ctx)
        out.extend(d)
        if stmt.type_ann is not None:
            out.extend(check_annotation(inferred, stmt.type_ann, stmt.span, ctx, f"`{stmt.name}`"))
        ctx.bind(stmt.name, inferred, mutable=isinstance(stmt, VarStmt))

    elif isinstance(stmt, AssignStmt):
        inferred, d = synth(stmt.value, ctx)
        out.extend(d)
        status = ctx.assign(stmt.name, inferred)
        if status == "undefined":
            out.append(_diag("undefined-var", stmt.span, name=stmt.name))
        elif status == "immutable":
            out.append(_diag("immutable-assign", stmt.span,
                             f"cannot assign twice to `{stmt.name}`: it was declared with `let` (or is a function parameter or module); "
                             f"declare it with `var {stmt.name} = ...` to allow reassignment"))

    elif isinstance(stmt, WhileStmt):
        _check_loop(stmt, ctx, out, step=None)

    elif isinstance(stmt, ForStmt):
        out.extend(check_stmt(stmt.init, ctx))
        _check_loop(stmt, ctx, out, step=stmt.increment)

    elif isinstance(stmt, IfStmt):
        taken = _condition(stmt.condition, ctx, out, "if statement")
        if taken is None:
            return  # erroneous or uncertain condition: don't guess a branch
        body = stmt.true_body if taken else stmt.false_body
        if body is not None:
            out.extend(_check_block(body.stmts, ctx))

    elif isinstance(stmt, FnDefStmt):
        ctx.functions[stmt.name] = Closure(stmt, ctx)

    elif isinstance(stmt, ReturnStmt):
        typ, d = synth(stmt.value, ctx)
        out.extend(d)
        raise ReturnException(typ, [], stmt.span)

    elif isinstance(stmt, ImportStmt):
        _check_import(stmt, ctx, out)

    else:
        out.append(_diag("internal-error", getattr(stmt, "span", Span(1, 1, 1)), f"unsupported statement {type(stmt).__name__}"))

def _check_import(stmt: ImportStmt, ctx: TypeContext, out: list[Diagnostic]):
    from uncertain.parser import parse, ParseError
    from uncertain.lexer import LexerError

    module = ".".join(stmt.path)
    filepath = os.path.abspath(os.path.join(ctx.base_dir, *stmt.path) + ".calc")

    def fail(msg: str):
        out.append(_diag("import-error", stmt.span, msg))
        ctx.bind(stmt.alias, ERROR_TYPE)

    if not os.path.exists(filepath):
        return fail(f"module '{module}' not found (looked for {filepath})")
    if filepath in ctx.shared.import_stack:
        return fail(f"circular import: module '{module}' is already being imported")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            mod_stmts, _ = parse(f.read())
    except (OSError, UnicodeDecodeError, ParseError, LexerError) as e:
        return fail(f"could not load module '{module}' ({filepath}): {e}")

    mod_ctx = TypeContext(base_dir=os.path.dirname(filepath), shared=ctx.shared)
    ctx.shared.import_stack.append(filepath)
    try:
        mod_diags = check_program(mod_stmts, mod_ctx)
    finally:
        ctx.shared.import_stack.pop()

    errors = [d for d in mod_diags if d.severity == "error"]
    if errors:
        first = errors[0]
        out.append(_diag("import-error", stmt.span,
                         f"module '{module}' has {len(errors)} error(s); first at {filepath}:{first.span.line}:{first.span.col}: {diagnostic_message(first)}"))

    # Expose the module's top-level values as a struct, and its functions as `alias.fn(...)`
    ctx.bind(stmt.alias, MeasuredType(dict(mod_ctx.bindings), {}, functions=dict(mod_ctx.functions)))

# --- programs -------------------------------------------------------------------

def _guarded(span: Span, fn: Callable[[], list[Diagnostic]]) -> list[Diagnostic]:
    """Run one top-level check, turning escaped control flow and crashes into diagnostics."""
    try:
        return fn()
    except ReturnException as r:
        return r.diags + [_diag("invalid-return", r.span, "`return` can only be used inside a function")]
    except RecursionError:
        return [_diag("recursion-limit", span, "this statement is nested too deeply to analyse")]
    except Exception as e:  # pragma: no cover - last-resort safety net
        return [_diag("internal-error", span, f"{type(e).__name__}: {e}")]

def check_program(stmts: list[Stmt], ctx: TypeContext, expr: Optional[Expr] = None) -> list[Diagnostic]:
    """Type-check a whole program. Never raises; problems are reported as diagnostics.

    If `expr` (the program's trailing expression) is given, its type is stored in `ctx.result`.
    """
    diags: list[Diagnostic] = []
    for stmt in stmts:
        diags.extend(_guarded(stmt.span, lambda: check_stmt(stmt, ctx)))

    if expr is not None:
        def check_expr() -> list[Diagnostic]:
            ctx.result, d = synth(expr, ctx)
            return d
        diags.extend(_guarded(expr.span, check_expr))
    return _dedupe(diags)
