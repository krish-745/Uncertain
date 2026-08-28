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
class UniformLit:
    min_val: "Expr"
    max_val: "Expr"

@dataclass
class ExactLit:
    value: "Expr"

@dataclass
class EmpiricalLit:
    data: "Expr"

@dataclass
class LogNormalLit:
    mean: "Expr"
    stddev: "Expr"

@dataclass
class PoissonLit:
    lam: "Expr"

@dataclass
class BinomialLit:
    n: "Expr"
    p: "Expr"

@dataclass
class GammaLit:
    k: "Expr"
    theta: "Expr"

@dataclass
class BernoulliLit:
    p: "Expr"

@dataclass
class NegativeBinomialLit:
    r: "Expr"
    p: "Expr"

@dataclass
class GeometricLit:
    p: "Expr"

@dataclass
class ExponentialLit:
    lam: "Expr"

DistLit = Union[NormalLit, UniformLit, ExactLit, EmpiricalLit, LogNormalLit, PoissonLit, BinomialLit, GammaLit, BernoulliLit, NegativeBinomialLit, GeometricLit, ExponentialLit]

@dataclass
class LetStmt:
    name: str
    type_ann: DistLit | None
    value: "Expr"
    span: Span

@dataclass
class VarStmt:
    name: str
    type_ann: DistLit | None
    value: "Expr"
    span: Span

@dataclass
class AssignStmt:
    name: str
    value: "Expr"
    span: Span

@dataclass
class Block:
    stmts: list["Stmt"]
    span: Span

@dataclass
class WhileStmt:
    condition: "Expr"
    body: Block
    span: Span

@dataclass
class ArrayLit:
    elements: list["Expr"]
    span: Span

Stmt = Union[LetStmt, VarStmt, AssignStmt, WhileStmt]
Expr = Union[NumberLit, VarRef, BinOp, Call, ArrayLit]
