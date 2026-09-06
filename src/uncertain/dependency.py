from uncertain.ast_nodes import *
from uncertain.diagnostics import Diagnostic

DepSet = frozenset[str]

def check_reuse(op: str, left_deps: DepSet, right_deps: DepSet, span: Span) -> Diagnostic | None:
    if op in ("*", "/", "+", "-"):
        overlap = left_deps & right_deps
        if overlap:
            return Diagnostic(
                kind="uncertain-reuse",
                span=span,
                overlapping_vars=overlap,
                extra={"op": op}
            )
    return None
