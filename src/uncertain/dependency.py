from typing import Dict
from uncertain.ast_nodes import Span
from uncertain.diagnostics import Diagnostic

AffineForm = dict[int, float]

def get_cov(a: AffineForm, b: AffineForm) -> float:
    cov = 0.0
    for k, v in a.items():
        if k in b:
            cov += v * b[k]
    return cov

def get_var(a: AffineForm) -> float:
    return sum(v * v for v in a.values())

def add_affine(a: AffineForm, b: AffineForm) -> AffineForm:
    res = dict(a)
    for k, v in b.items():
        res[k] = res.get(k, 0.0) + v
    return {k: v for k, v in res.items() if abs(v) > 1e-12}

def sub_affine(a: AffineForm, b: AffineForm) -> AffineForm:
    res = dict(a)
    for k, v in b.items():
        res[k] = res.get(k, 0.0) - v
    return {k: v for k, v in res.items() if abs(v) > 1e-12}

def scale_affine(a: AffineForm, scale: float) -> AffineForm:
    if scale == 0.0:
        return {}
    return {k: v * scale for k, v in a.items() if abs(v) > 1e-12}

def linear_comb_affine(a: AffineForm, scale_a: float, b: AffineForm, scale_b: float) -> AffineForm:
    return add_affine(scale_affine(a, scale_a), scale_affine(b, scale_b))

def check_reuse(op: str, left_deps: AffineForm, right_deps: AffineForm, span: Span) -> Diagnostic | None:
    return None
