from dataclasses import dataclass
from typing import Union

@dataclass
class Span:
    line: int
    col: int
    length: int

@dataclass
class NumberLit:
    value: float
    span: Span

@dataclass
class VarRef:
    name: str
    span: Span

@dataclass
class BinOp:
    op: str            # "+" "-" "*" "/"
    left: "Expr"
    right: "Expr"
    span: Span

@dataclass
class Call:
    name: str           # "square", "sqrt", "correlated", "sensor_read"
    args: list["Expr"]
    kwargs: dict[str, "Expr"] # e.g. {"cov": NumberLit(...)}
    span: Span

@dataclass
class NormalLit:
    mean: "Expr"
    stddev: "Expr"

@dataclass
class ExactLit:
    value: "Expr"

DistLit = Union[NormalLit, ExactLit]

@dataclass
class LetStmt:
    name: str
    type_ann: DistLit | None
    value: "Expr"
    span: Span

Expr = Union[NumberLit, VarRef, BinOp, Call]
