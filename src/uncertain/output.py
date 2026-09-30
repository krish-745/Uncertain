"""Rendering of checked values for the CLI, the playground and the language server."""
from typing import Any
from uncertain.typechecker import MeasuredType, ERROR_TYPE

def format_value(typ: MeasuredType) -> str:
    if typ is ERROR_TYPE:
        return "<error>"
    if isinstance(typ.dist, tuple):
        return "[" + ", ".join(format_value(el) for el in typ.dist) + "]"
    if isinstance(typ.dist, dict):
        return "{" + ", ".join(f"{name}: {format_value(el)}" for name, el in typ.dist.items()) + "}"
    return f"(mean={typ.dist.mean:.4f}, stddev={typ.dist.stddev:.4f})"

def to_json(typ: MeasuredType) -> Any:
    if typ is ERROR_TYPE:
        return None
    if isinstance(typ.dist, tuple):
        return [to_json(el) for el in typ.dist]
    if isinstance(typ.dist, dict):
        return {name: to_json(el) for name, el in typ.dist.items()}
    return {"mean": typ.dist.mean, "stddev": typ.dist.stddev, "family": typ.dist.family}
