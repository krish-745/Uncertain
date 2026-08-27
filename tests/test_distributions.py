import math
import pytest
from hypothesis import given, strategies as st
from uncertain_lang.distributions import (
    Dist, add, sub, mul_independent, div_independent, square, sqrt_dist, correlated_product
)

st_mean = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
st_stddev = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
st_pos_mean = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)

@given(st_mean, st_stddev, st_mean, st_stddev)
def test_add_properties(m1, s1, m2, s2):
    d1 = Dist(m1, s1)
    d2 = Dist(m2, s2)
    res = add(d1, d2)
    assert math.isclose(res.mean, m1 + m2, rel_tol=1e-5, abs_tol=1e-8)
    assert math.isclose(res.stddev, math.sqrt(s1**2 + s2**2), rel_tol=1e-5, abs_tol=1e-8)

@given(st_mean, st_stddev, st_mean, st_stddev)
def test_sub_properties(m1, s1, m2, s2):
    d1 = Dist(m1, s1)
    d2 = Dist(m2, s2)
    res = sub(d1, d2)
    assert math.isclose(res.mean, m1 - m2, rel_tol=1e-5, abs_tol=1e-8)
    assert math.isclose(res.stddev, math.sqrt(s1**2 + s2**2), rel_tol=1e-5, abs_tol=1e-8)

def test_square_vs_mul_independent():
    # Verify that square(a) is not accidentally aliased to mul_independent(a, a)
    d1 = Dist(10.0, 2.0)
    res_sq = square(d1)
    res_mul = mul_independent(d1, d1)
    
    assert res_sq.mean == res_mul.mean
    assert not math.isclose(res_sq.stddev, res_mul.stddev, rel_tol=1e-5)
    
    # The standard deviation of X^2 is larger than independent X1 * X2
    assert res_sq.stddev > res_mul.stddev

def test_exact():
    # 0.0 stddev represents Exact(value)
    d1 = Dist(5.0, 0.0)
    d2 = Dist(10.0, 0.0)
    
    res = add(d1, d2)
    assert res.mean == 15.0
    assert res.stddev == 0.0
    
    res = mul_independent(d1, d2)
    assert res.mean == 50.0
    assert res.stddev == 0.0

@given(st_pos_mean, st_stddev)
def test_sqrt_properties(m, s):
    d = Dist(m, s)
    res = sqrt_dist(d)
    assert math.isclose(res.mean, math.sqrt(m), rel_tol=1e-5, abs_tol=1e-8)
