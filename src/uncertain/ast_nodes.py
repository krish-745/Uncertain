from dataclasses import dataclass
from typing import Union

@dataclass
class Span:
    line: int
    col: int
    length: int   # a length <= 0 means "to the end of the line" (e.g. a span covering several lines)

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
    op: str            # "+" "-" "*" "/" "<" ">" "<=" ">=" "==" "!="
    left: "Expr"
    right: "Expr"
    span: Span

@dataclass
class Call:
    name: str           # "square", "sqrt", "correlated", "sensor_read"
    args: list["Expr"]
    kwargs: dict[str, "Expr"] # e.g. {"cov": NumberLit(...)}
    span: Span
    target: "Expr | None" = None   # `m` in `m.f(x)`: a call to a function of an imported module

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

# Distribution literal class -> family name (as used by `distributions.moments`)
DIST_LIT_FAMILIES = {
    NormalLit: "Normal",
    UniformLit: "Uniform",
    ExactLit: "Exact",
    EmpiricalLit: "Empirical",
    LogNormalLit: "LogNormal",
    PoissonLit: "Poisson",
    BinomialLit: "Binomial",
    GammaLit: "Gamma",
    BernoulliLit: "Bernoulli",
    NegativeBinomialLit: "NegativeBinomial",
    GeometricLit: "Geometric",
    ExponentialLit: "Exponential",
}

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
class IfStmt:
    condition: "Expr"
    true_body: Block
    false_body: Block | None
    span: Span

@dataclass
class ForStmt:
    init: "Stmt"
    condition: "Expr"
    increment: "Stmt"
    body: Block
    span: Span

@dataclass
class ArrayLit:
    elements: list["Expr"]
    span: Span

@dataclass
class ArgDef:
    name: str
    type_ann: DistLit | None
    span: Span

@dataclass
class FnDefStmt:
    name: str
    args: list[ArgDef]
    return_type: DistLit | None
    body: Block
    span: Span

@dataclass
class ReturnStmt:
    value: "Expr"
    span: Span

@dataclass
class ArrayAccess:
    array: "Expr"
    index: "Expr"
    span: Span

@dataclass
class StructLit:
    fields: dict[str, "Expr"]
    span: Span

@dataclass
class FieldAccess:
    obj: "Expr"
    field: str
    span: Span

@dataclass
class ImportStmt:
    path: list[str]
    alias: str
    span: Span

Stmt = Union[LetStmt, VarStmt, AssignStmt, WhileStmt, IfStmt, ForStmt, FnDefStmt, ReturnStmt, ImportStmt]
Expr = Union[NumberLit, VarRef, BinOp, Call, ArrayLit, ArrayAccess, StructLit, FieldAccess]
