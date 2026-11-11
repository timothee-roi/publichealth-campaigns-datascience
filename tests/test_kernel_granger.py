"""Kernel Granger causality: reduction to the linear case, surrogates, corrections."""
import numpy as np
import pytest

from phc import granger as G
from phc import kernel_granger as K
from phc import simulate as S


def test_degree_one_kernel_is_linear_granger():
    """With a polynomial kernel of degree 1, delta = 1 - RSS_full / RSS_reduced."""
    df = S.independent(seed=3)
    d = K.make_design(df, "gt", "mc", 3)
    delta = K.KGC(d.y, d.own, "poly", 1).stat(d.cause)["delta"]
    y, X_red, X_full = G.lag_design(df, "gt", "mc", 3, 3, by_season=True)
    assert delta == pytest.approx(1 - G._rss(X_full, y) / G._rss(X_red, y), abs=1e-8)


def test_derangements_have_no_fixed_point():
    for p in K.derangements(11, 50, np.random.default_rng(0)):
        assert sorted(p) == list(range(11))
        assert np.all(p != np.arange(11))


def test_weekday_aligned_swap():
    df = S.common_driver(seed=0, weekday=0.5)
    d = K.make_design(df, "gt", "mc", 2)
    perm = K.derangements(11, 1, np.random.default_rng(1))[0]
    assert np.array_equal(d.swapped_panel(perm), d.cause_panel[perm])
    L = d.cause_panel.shape[1]
    aligned = d.swapped_panel(perm, align_weekday=True)
    for s, donor in enumerate(perm):
        j = (d.first_weekday[s] - d.first_weekday[donor] + 3) % 7 - 3
        assert -3 <= j <= 3
        # day 10 of season s receives a donor day with the same weekday
        assert (d.first_weekday[donor] + 10 + j) % 7 == (d.first_weekday[s] + 10) % 7
        assert np.array_equal(aligned[s, 10:L - 3], d.cause_panel[donor, 10 + j:L - 3 + j])


def test_multiple_testing_corrections():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(K.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(K.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    assert K.surrogate_pvalue(1.0, np.zeros(199)) == pytest.approx(1 / 200)


def test_nonlinear_link_is_detected_and_direction_is_right():
    df = S.nonlinear_mc_to_gt(seed=1)
    fwd = K.run_tests(df, "gt", "mc", m=1, n_surrogates=39, seed=0, methods=("lsngc",))
    back = K.run_tests(df, "mc", "gt", m=1, n_surrogates=39, seed=0, methods=("lsngc",))
    assert fwd["lsngc_p_surr"] == pytest.approx(1 / 40)
    assert fwd["lsngc_delta"] > 5 * back["lsngc_delta"]
