from dataclasses import dataclass, field
from uncertain_lang.ast_nodes import Span

@dataclass
class Diagnostic:
    kind: str                  # "uncertain-reuse" | "type-mismatch" | "undefined-var" | ...
    span: Span
    overlapping_vars: frozenset[str] | None = None
    extra: dict = field(default_factory=dict)

def format_diagnostic(diag: Diagnostic, source_lines: list[str]) -> str:
    line_idx = diag.span.line - 1
    line_text = source_lines[line_idx] if 0 <= line_idx < len(source_lines) else "<source unavailable>"
    
    caret_indent = " " * (diag.span.col - 1)
    caret = "^" * diag.span.length
    
    if diag.kind == "uncertain-reuse":
        overlap_str = ", ".join(f"`{v}`" for v in diag.overlapping_vars) if diag.overlapping_vars else ""
        title = "uncertain reuse without correlation annotation"
        caret_msg = f"{overlap_str} appears twice in this product" if diag.extra.get("op") == "*" else f"{overlap_str} appears in both operands"
        note = f"treating repeated occurrences of {overlap_str} as independent understates\n     the true variance of the result."
        help_msg = f"use `square({next(iter(diag.overlapping_vars))})` for the correct self-product variance formula,\n     or wrap with `correlated(..., ..., cov = ...)` if you have an explicit\n     covariance estimate."
    elif diag.kind == "undefined-var":
        name = diag.extra.get("name", "?")
        title = f"undefined variable `{name}`"
        caret_msg = f"not found in scope"
        note = f"`{name}` must be declared with `let` before use."
        help_msg = ""
    elif diag.kind == "type-mismatch":
        title = "type annotation mismatch"
        caret_msg = "inferred type does not match annotation"
        note = "the distribution computed by the typechecker differs from the explicit type annotation."
        help_msg = ""
    else:
        title = diag.kind
        caret_msg = "error here"
        note = ""
        help_msg = ""
        
    res = f"error: {title}\n"
    res += f"  --> line {diag.span.line}:{diag.span.col}\n"
    res += f"   |\n"
    res += f"{diag.span.line:2d} | {line_text}\n"
    res += f"   | {caret_indent}{caret} {caret_msg}\n"
    res += f"   |\n"
    
    if note:
        res += f"   = note: {note}\n"
    if help_msg:
        res += f"   = help: {help_msg}\n"
        
    return res.rstrip()
