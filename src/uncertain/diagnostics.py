from dataclasses import dataclass, field
from uncertain.ast_nodes import Span

@dataclass
class Diagnostic:
    kind: str                  # a key of KINDS
    span: Span
    extra: dict = field(default_factory=dict)
    severity: str = "error"

@dataclass(frozen=True)
class KindInfo:
    title: str          # headline of the rendered diagnostic
    caret: str          # text printed next to the ^^^ underline
    note: str           # default note, used when the diagnostic carries no `msg`
    explanation: str    # long-form text for `uncertain --explain <kind>`

KINDS: dict[str, KindInfo] = {
    "syntax-error": KindInfo(
        "syntax error",
        "could not parse this",
        "",
        "The source code could not be tokenized or parsed.",
    ),
    "type-mismatch": KindInfo(
        "type annotation mismatch",
        "inferred type does not match annotation",
        "the distribution computed by the typechecker differs from the explicit type annotation.",
        "An explicit `Measured<...>` annotation (on a `let`/`var`, a function argument or a function "
        "return type) does not match the mean and standard deviation computed by the type checker.",
    ),
    "undefined-var": KindInfo(
        "undefined variable",
        "not found in scope",
        "variables must be declared with `let` or `var` before use.",
        "A variable is referenced (or assigned) before it has been declared with `let` or `var`.",
    ),
    "immutable-assign": KindInfo(
        "cannot assign twice to an immutable variable",
        "cannot assign",
        "variables declared with `let` cannot be reassigned; use `var` for values that change.",
        "A variable declared with `let` (or a function parameter, or an imported module) was reassigned. "
        "Declare the variable with `var` if it needs to change, e.g. a loop counter or an accumulator.",
    ),
    "math-domain-error": KindInfo(
        "math domain error",
        "invalid operation",
        "mathematical operation is undefined",
        "An operation is mathematically undefined or cannot be represented, such as dividing by a "
        "distribution with a zero mean, taking the log of a non-positive mean, a numeric overflow, or "
        "distribution parameters outside their valid range (e.g. `poisson_read(-1)`).",
    ),
    "approximation-warning": KindInfo(
        "approximation warning",
        "result is approximate",
        "moment formulas assume Normal inputs; the result for a non-Normal input is approximate.",
        "A non-Normal distribution (Uniform, Poisson, ...) is used in a product, ratio, nonlinear "
        "function or `prob()` query. The formulas used are exact for Normal inputs, so the resulting "
        "mean/variance (or probability) is a moment-matching approximation.",
    ),
    "delta-method-warning": KindInfo(
        "delta method may be inaccurate",
        "large relative uncertainty",
        "the first-order (delta method) approximation is unreliable when stddev is large relative to the mean.",
        "Division, `log` and `sqrt` use a first-order Taylor (delta method) approximation. When the "
        "input's standard deviation is large compared to its mean (coefficient of variation > 0.3), "
        "the true distribution is far from the approximation.",
    ),
    "uncertain-branch": KindInfo(
        "uncertain branch",
        "condition is not deterministic",
        "control flow can only depend on deterministic values; use prob(...) to query uncertain comparisons.",
        "A condition (`if`, `while`, `for`, `filter`) or comparison depends on an uncertain value. "
        "Loops and branches are unrolled at compile time, so their conditions must be deterministic. "
        "Use `prob(x > y)` to compute the probability of an uncertain comparison instead.",
    ),
    "loop-limit": KindInfo(
        "loop iteration limit exceeded",
        "loop did not terminate",
        "loops are unrolled at compile time; raise the limit with --max-unroll if this loop is intended.",
        "A `while` or `for` loop ran for more iterations than the unroll limit (default 1000, "
        "configurable with `--max-unroll`).",
    ),
    "recursion-limit": KindInfo(
        "recursion limit exceeded",
        "call nesting is too deep",
        "function calls are expanded at compile time and must terminate after a bounded number of steps.",
        "Function calls are evaluated during type checking. A call chain exceeded the maximum depth, "
        "usually because of unbounded recursion.",
    ),
    "invalid-operand": KindInfo(
        "invalid operand",
        "invalid value here",
        "this operation cannot be applied to this value.",
        "An operation was applied to a value of the wrong shape: arithmetic on an array or struct, "
        "indexing into a non-array, accessing a missing struct field, or an out-of-bounds index.",
    ),
    "not-constant": KindInfo(
        "expected a constant",
        "value is uncertain",
        "this position requires a deterministic value.",
        "Distribution parameters, exponents, covariances and array indices must be deterministic "
        "values (standard deviation 0), known at compile time.",
    ),
    "unknown-function": KindInfo(
        "unknown function",
        "no such function",
        "the function is neither a built-in nor defined with `fn`.",
        "A call refers to a function that is neither built in nor defined with `fn` earlier in the program.",
    ),
    "arity-mismatch": KindInfo(
        "wrong number of arguments",
        "bad arguments",
        "the call does not match the function's parameters.",
        "A function or built-in was called with the wrong number of arguments or an unexpected keyword argument.",
    ),
    "invalid-return": KindInfo(
        "invalid return",
        "return problem",
        "functions must return a value, and `return` is only allowed inside a function.",
        "A function finished without executing a `return` statement, or `return` was used outside a function.",
    ),
    "import-error": KindInfo(
        "import failed",
        "could not import module",
        "the module could not be loaded.",
        "An `import` statement failed: the module file was not found, could not be parsed, contains "
        "errors, or is part of a circular import.",
    ),
    "internal-error": KindInfo(
        "internal compiler error",
        "the compiler crashed here",
        "this is a bug in the Uncertain compiler; please report it.",
        "The compiler hit an unexpected internal error while checking this statement. Please file a bug report.",
    ),
}

def diagnostic_message(diag: Diagnostic) -> str:
    """One-line human-readable message (used by the LSP and JSON output)."""
    info = KINDS.get(diag.kind)
    title = info.title if info else diag.kind
    if diag.kind == "undefined-var" and "name" in diag.extra:
        title = f"undefined variable `{diag.extra['name']}`"
    msg = diag.extra.get("msg")
    return f"{title}: {msg}" if msg else title

def format_diagnostic(diag: Diagnostic, source_lines: list[str]) -> str:
    info = KINDS.get(diag.kind, KindInfo(diag.kind, "error here", "", ""))
    line_idx = diag.span.line - 1
    line_text = source_lines[line_idx] if 0 <= line_idx < len(source_lines) else "<source unavailable>"

    title = info.title
    if diag.kind == "undefined-var" and "name" in diag.extra:
        title = f"undefined variable `{diag.extra['name']}`"
    note = diag.extra.get("msg") or info.note

    # A negative/zero length (e.g. a span covering several lines) underlines to the end of the line.
    col = max(1, diag.span.col)
    length = diag.span.length
    if length <= 0 or col - 1 + length > len(line_text):
        length = max(1, len(line_text) - (col - 1))

    res = f"{diag.severity}: {title}\n"
    res += f"  --> line {diag.span.line}:{diag.span.col}\n"
    res += f"   |\n"
    res += f"{diag.span.line:2d} | {line_text}\n"
    res += f"   | {' ' * (col - 1)}{'^' * length} {info.caret}\n"
    res += f"   |\n"
    if note:
        res += f"   = note: {note}\n"
    return res.rstrip()
