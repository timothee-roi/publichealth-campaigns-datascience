"""Simulated seasonal systems with a known causal structure, used to validate the
kernel Granger tests before applying them to real data.

Each system has 11 seasons of 91 days, like the Octobre Rose windows, and returns a
frame with columns gt, mc, season indexed by date (same layout as ``load_daily``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SEASONS, L = 11, 91


def _ar1(rng, n, phi=0.5):
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _frame(gt, mc):
    dates, seasons = [], []
    for s in range(N_SEASONS):
        dates.append(pd.date_range(f"{2015 + s}-09-01", periods=L, freq="D"))
        seasons += [str(2015 + s)] * L
    return pd.DataFrame({"gt": np.concatenate(gt), "mc": np.concatenate(mc), "season": seasons},
                        index=pd.DatetimeIndex(np.concatenate(dates), name="date"))


def nonlinear_mc_to_gt(seed=0, strength=0.6):
    """MC -> GT through a quadratic link, with no linear correlation.

    mc_t = 0.5 mc_{t-1} + e;  gt_t = 0.4 gt_{t-1} + strength * (mc_{t-1}^2 - E[mc^2]) + e.
    MC is symmetric around 0, so cov(gt_t, mc_{t-1}) = 0: a linear test cannot see it.
    """
    rng = np.random.default_rng(seed)
    gt, mc = [], []
    for _ in range(N_SEASONS):
        x = _ar1(rng, L)
        y = np.zeros(L)
        e = rng.standard_normal(L)
        for t in range(1, L):
            y[t] = 0.4 * y[t - 1] + strength * (x[t - 1] ** 2 - 4 / 3) + e[t]
        gt.append(y); mc.append(x)
    return _frame(gt, mc)


def independent(seed=0):
    """Two independent AR(1) series: no causality in either direction."""
    rng = np.random.default_rng(seed)
    return _frame([_ar1(rng, L) for _ in range(N_SEASONS)],
                  [_ar1(rng, L) for _ in range(N_SEASONS)])


def _weekly(s, weekday):
    """Weekly cycle of season s: +0.4 * weekday on working days, -weekday at weekends.
    Seasons start on 1 September of successive years, so the cycle shifts between seasons."""
    wd = (pd.Timestamp(f"{2015 + s}-09-01").dayofweek + np.arange(L)) % 7
    return weekday * np.where(wd >= 5, -1.0, 0.4)


def common_driver(seed=0, jitter=0, amplitude=3.0, weekday=0.0):
    """No direct link: a campaign pulse drives both series.

    The pulse peaks around day 45; MC follows it on the same day, GT two days later.
    jitter = 0: same timing every season (a fixed calendar).
    jitter > 0: the peak day moves by up to +/- jitter days from one season to the next.
    weekday > 0: both series also share a weekly cycle (lower at weekends), of amplitude
    ``weekday`` for MC and 0.6 * ``weekday`` for GT.
    """
    rng = np.random.default_rng(seed)
    day = np.arange(L)
    gt, mc = [], []
    for s in range(N_SEASONS):
        c = 45 + (rng.integers(-jitter, jitter + 1) if jitter else 0)
        pulse = np.exp(-0.5 * ((day - c) / 6) ** 2)
        pulse_late = np.exp(-0.5 * ((day - c - 2) / 6) ** 2)
        w = _weekly(s, weekday)
        mc.append(_ar1(rng, L) + amplitude * pulse + w)
        gt.append(_ar1(rng, L) + amplitude * pulse_late + 0.6 * w)
    return _frame(gt, mc)


def transfer_system(seed=0, omega=0.5, delta=0.6, delay=1, feedback=0.0, weekday=0.5,
                    pulse=3.0, jitter=0):
    """GT -> MC through a known transfer function, on top of a shared calendar.

    Day-to-day fluctuations (the part the transfer function acts on):
        x_t = 0.5 x_{t-1} + feedback * y_{t-1} + e_t                    (GT)
        y_t = u_t + n_t,  u_t = delta u_{t-1} + omega x_{t-delay},      (MC)
              n_t = 0.4 n_{t-1} + e'_t.
    The response of MC to a one-unit pulse of GT is omega * delta^(k - delay) for
    k >= delay: gain omega / (1 - delta), half-life log(0.5) / log(delta).
    Observed series add a campaign pulse (as in ``common_driver``) and a weekly cycle
    (lower at weekends) to both. omega = 0 gives a system with no direct link.
    """
    rng = np.random.default_rng(seed)
    day = np.arange(L)
    gt, mc = [], []
    for s in range(N_SEASONS):
        ex, ey = rng.standard_normal(L), rng.standard_normal(L)
        x, y, u, n = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
        for t in range(1, L):
            x[t] = 0.5 * x[t - 1] + feedback * y[t - 1] + ex[t]
            u[t] = delta * u[t - 1] + (omega * x[t - delay] if t >= delay else 0.0)
            n[t] = 0.4 * n[t - 1] + ey[t]
            y[t] = u[t] + n[t]
        c = 45 + (rng.integers(-jitter, jitter + 1) if jitter else 0)
        w = _weekly(s, weekday)
        mc.append(y + pulse * np.exp(-0.5 * ((day - c) / 6) ** 2) + w)
        gt.append(x + pulse * np.exp(-0.5 * ((day - c - 2) / 6) ** 2) + 0.6 * w)
    return _frame(gt, mc)


SCENARIOS = {
    "Nonlinear MC → GT": nonlinear_mc_to_gt,
    "Independent": independent,
    "Common driver, fixed timing": lambda seed=0: common_driver(seed, jitter=0),
    "Common driver, timing varies": lambda seed=0: common_driver(seed, jitter=15),
}
TRUTH = {  # true direct causal links
    "Nonlinear MC → GT": {"MC → GT": True, "GT → MC": False},
    "Independent": {"MC → GT": False, "GT → MC": False},
    "Common driver, fixed timing": {"MC → GT": False, "GT → MC": False},
    "Common driver, timing varies": {"MC → GT": False, "GT → MC": False},
}
