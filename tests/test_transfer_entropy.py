"""Nearest-neighbour transfer entropy: agreement with the Gaussian value, nonlinearity, null."""
import numpy as np
import pytest

from phc.transfer_entropy import KSGTE


def _system(n, b, quad, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n), np.zeros(n)
    e = rng.standard_normal((2, n))
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + e[0, t]
        y[t] = 0.4 * y[t - 1] + (b * (x[t - 1] ** 2 - 4 / 3) if quad else b * x[t - 1]) + e[1, t]
    return y[1:], y[:-1, None], x[:-1, None]


def test_gaussian_value_of_linear_system():
    """For a linear Gaussian system TE = 0.5 log(var_restricted / var_full)."""
    y, own, c = _system(20000, 0.5, False)
    A = np.column_stack([np.ones_like(y), own[:, 0]])
    B = np.column_stack([A, c[:, 0]])
    r0 = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r1 = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
    exact = 0.5 * np.log(r0.var() / r1.var())
    assert KSGTE(y, own).stat(c)["delta"] == pytest.approx(exact, abs=0.02)


def test_no_link_gives_zero():
    y, own, c = _system(20000, 0.0, False, seed=1)
    assert abs(KSGTE(y, own).stat(c)["delta"]) < 0.01


def test_quadratic_link_without_correlation_is_found():
    y, own, c = _system(5000, 0.6, True, seed=2)
    assert abs(np.corrcoef(y, c[:, 0])[0, 1]) < 0.05          # linear tests see nothing
    assert KSGTE(y, own).stat(c)["delta"] > 0.1


def test_ties_do_not_break_the_estimate():
    y, own, c = _system(3000, 0.0, False, seed=3)
    out = KSGTE(np.round(y), np.round(own)).stat(np.round(c))["delta"]
    assert np.isfinite(out)
