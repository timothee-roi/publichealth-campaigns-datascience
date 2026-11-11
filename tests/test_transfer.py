"""Transfer functions: each building block is checked against a case with a known answer."""
import numpy as np
import pandas as pd
import pytest

from phc import simulate as S
from phc import transfer as T


def test_anomalies_remove_profile_weekday_and_season():
    df = S.transfer_system(seed=0)
    a = T.anomalies(df)
    idx = pd.DatetimeIndex(a.index)
    for col in ("gt", "mc"):
        assert np.allclose(a[col].groupby(idx.strftime("%m-%d")).mean(), 0, atol=1e-9)
        assert np.allclose(a[col].groupby(idx.dayofweek).mean(), 0, atol=1e-9)
        assert np.allclose(a[col].groupby(a["season"]).mean(), 0, atol=1e-9)


def test_remove_weekday_recovers_the_weekly_cycle():
    df = S.common_driver(seed=2, weekday=1.0)
    removed = (df["mc"] - T.remove_weekday(df)["mc"]).groupby(df.index.dayofweek).mean()
    truth = np.array([0.4] * 5 + [-1.0] * 2)
    assert np.allclose(removed - removed.mean(), truth - truth.mean(), atol=0.15)


def test_panel_pads_short_seasons():
    df = pd.DataFrame({"gt": np.arange(7.0), "season": list("aaaabbb")})
    P = T.panel(df, "gt")
    assert P.shape == (2, 4) and np.isnan(P[1, 3]) and P[1, 0] == 4


def test_ccf_sign_convention():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((11, 90))
    B = np.roll(A, 3, axis=1)                    # B_t = A_{t-3}: A leads B by 3 days
    r = T.ccf(A, B, 6)
    assert r.idxmax() == 3 and r[3] > 0.9 and abs(r[-3]) < 0.2


def test_phase_delay_sign_convention():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((11, 96))
    Y = np.roll(X, 2, axis=1)                    # Y lags X by 2 days
    sp = T.cross_spectrum(X, Y)
    low = sp[sp["freq"] < 0.2]
    assert np.allclose(low["delay"], 2, atol=0.4)      # the taper blurs the lowest frequencies
    assert low["delay"].mean() == pytest.approx(2, abs=0.05)
    assert (low["coherence"] > 0.9).all()
    assert T.coherence_threshold(11) == pytest.approx(1 - 0.05 ** 0.1)


def test_swap_keeps_missing_days():
    P = np.arange(12.0).reshape(3, 4)
    P[1, 3] = np.nan
    out = T.swap(P, np.array([1, 2, 0]))
    assert np.isnan(out[1, 3]) and np.isnan(out[0, 3]) and out[2, 0] == 0


def test_rational_model_recovers_a_known_transfer_function():
    # y_t = 0.5 / (1 - 0.6 B) x_{t-2} + AR(1) noise, no calendar
    rng = np.random.default_rng(1)
    X = rng.standard_normal((11, 150))
    Y = np.zeros_like(X)
    for s in range(11):
        u = n = 0.0
        for t in range(150):
            u = 0.6 * u + (0.5 * X[s, t - 2] if t >= 2 else 0.0)
            n = 0.4 * n + 0.3 * rng.standard_normal()
            Y[s, t] = u + n
    m = T.select_tf(X, Y)
    assert m.b == 2
    assert m.omega[0] == pytest.approx(0.5, abs=0.03)
    assert m.delta == pytest.approx(0.6, abs=0.03)
    assert m.gain == pytest.approx(1.25, abs=0.08)
    assert m.half_life == pytest.approx(np.log(0.5) / np.log(0.6), abs=0.15)
    assert np.allclose(m.weights(4), [0, 0, 0.5, 0.3, 0.18], atol=0.03)
    fir = T.fit_tf(X, Y, "fir", b=0, p=m.p, K=6)
    assert np.allclose(fir.omega, [0, 0, 0.5, 0.3, 0.18, 0.108, 0.0648], atol=0.04)
    checks = T.residual_checks(X, Y, m)
    assert checks["p_ljung_box"] > 0.01 and checks["p_cross"] > 0.01


def test_full_procedure_on_a_seasonal_system():
    df = S.transfer_system(seed=1)               # nu_k = 0.5 * 0.6^(k-1), k >= 1
    est = T.tf_summary(df, "gt", "mc")
    assert est["b"] == 1
    assert est["lagged"] == pytest.approx(1.249, abs=0.25)
    assert abs(est["same_day"]) < 0.15
    back = T.tf_summary(df, "mc", "gt")          # no link in this direction
    assert back["r2_lagged"] < est["r2_lagged"] / 5


def test_jackknife_interval():
    se, lo, hi = T.jackknife_interval(1.0, np.arange(11.0))
    assert se == pytest.approx(np.sqrt(10 / 11 * 110))
    assert (lo + hi) / 2 == pytest.approx(1.0)
