import math
import numpy as np

from utils.utils import norm_radian, norm_realative_radian


def test_norm_radian_bounds():
    values = np.linspace(-10 * math.pi, 10 * math.pi, num=100)
    for val in values:
        out = norm_radian(val)
        assert -math.pi <= out <= math.pi


def test_norm_radian_wrap():
    assert math.isclose(norm_radian(3 * math.pi), -math.pi, rel_tol=1e-6)


def test_norm_realative_radian_bounds():
    values = np.linspace(-10 * math.pi, 10 * math.pi, num=100)
    for val in values:
        out = norm_realative_radian(val)
        assert -math.pi <= out <= math.pi


def test_norm_realative_radian_symmetry():
    assert math.isclose(norm_realative_radian(math.pi), math.pi, rel_tol=1e-6)
    assert math.isclose(norm_realative_radian(-math.pi), -math.pi, rel_tol=1e-6)
