"""Transfer functions between two daily campaign series.

A transfer function describes how an output series y responds over time to an input
series x (Box, Jenkins & Reinsel, *Time Series Analysis*, ch. 11-12):

    y_t = c + nu(B) x_t + N_t,        nu(B) = nu_0 + nu_1 B + nu_2 B^2 + ...,

where B is the backshift operator, nu_k is the response of y, k days later, to a
one-unit rise of x (the impulse-response weights) and N_t is autocorrelated noise,
here an AR(p) process, phi(B) N_t = e_t. Two forms of nu(B) are fitted:

* an unrestricted distributed lag, nu_0 ... nu_K (``fit_tf(..., kind="fir")``);
* a delay b followed by a geometric decay (``kind="rational"``),
      nu(B) = omega B^b / (1 - delta B),   nu_k = omega delta^(k-b) for k >= b,
  which is summarised by three numbers: the delay b, the total gain omega / (1 - delta)
  and the half-life log(1/2) / log(delta).

The steps, in the order used in notebook 06:

1. ``anomalies``  removes what the calendar explains: the average profile of the
   campaign (one value per calendar date), the day of the week and the level of the
   season. The transfer function is estimated on what remains.
2. ``prewhiten`` + ``ccf``  model-free identification: an AR filter that whitens the
   input is applied to both series; the cross-correlation of the two filtered series is
   then proportional to nu_k. Correlations at negative lags reveal feedback (the output
   leading the input), which the single-input model rules out.
3. ``fit_tf``  conditional least squares for the two models above; ``select_tf`` chooses
   the delay and the noise order by AIC on a common sample.
4. ``season_swap_null``, ``jackknife``  inference that respects the unit of
   replication, the season: the input of each season is replaced by that of another
   season (null of no day-to-day link), and seasons are deleted one at a time
   (standard errors).
5. ``cross_spectrum``  the frequency-domain transfer function (coherence, gain, phase).

All lags are taken within a season. Seasons are stored as rows of a (seasons x days)
array, padded with NaN at the end when their lengths differ (leap years).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

P_MAX = 7       # largest AR order of the noise (one week)
B_MAX = 7       # largest delay
K_FIR = 14      # last lag of the unrestricted distributed lag
DELTA_MAX = 0.98


# ---------------------------------------------------------------------------
# 1. Calendar removal
# ---------------------------------------------------------------------------
def anomalies(df: pd.DataFrame, weekday: bool = True, season: bool = True,
              cols=("gt", "mc"), return_components: bool = False):
    """Remove the calendar from each series by an additive least-squares fit.

        x = date profile (one level per month-day) [+ weekday] [+ season] + anomaly.

    The date profile is the campaign's average shape; it is what season-swap surrogates
    hold fixed. The weekday term removes the weekly cycle shared by searches and
    newspapers. The season term removes level differences between seasons, which here
    come mostly from the normalisation of each season by its own maximum.
    Dates are matched by month and day, so that 29 February does not shift the profile.
    """
    idx = pd.DatetimeIndex(df.index)
    blocks = [pd.get_dummies(idx.strftime("%m-%d")).to_numpy(float)]
    names = ["profile"]
    if weekday:
        blocks.append(pd.get_dummies(idx.dayofweek).to_numpy(float)[:, 1:])
        names.append("weekday")
    if season:
        blocks.append(pd.get_dummies(df["season"].to_numpy()).to_numpy(float)[:, 1:])
        names.append("season")
    X = np.hstack(blocks)
    out = df.copy()
    comps = {}
    for c in cols:
        y = df[c].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[c] = y - X @ beta
        if return_components:
            start = 0
            for name, blk in zip(names, blocks):
                part = blk @ beta[start:start + blk.shape[1]]
                comps[(c, name)] = part - (part.mean() if name != "profile" else 0.0)
                start += blk.shape[1]
    if return_components:
        return out, pd.DataFrame(comps, index=df.index)
    return out


def remove_weekday(df: pd.DataFrame, cols=("gt", "mc")) -> pd.DataFrame:
    """Series with their day-of-week effect removed; the campaign profile and the season
    level are kept. The effect is estimated jointly with them, as in ``anomalies``."""
    _, comps = anomalies(df, cols=cols, return_components=True)
    out = df.copy()
    for c in cols:
        out[c] = df[c].to_numpy(float) - comps[(c, "weekday")].to_numpy()
    return out


def panel(df: pd.DataFrame, col: str) -> np.ndarray:
    """(seasons x days) array of one column, NaN-padded at the end of short seasons."""
    groups = [g[col].to_numpy(float) for _, g in df.groupby("season", sort=False)]
    L = max(len(g) for g in groups)
    out = np.full((len(groups), L), np.nan)
    for i, g in enumerate(groups):
        out[i, :len(g)] = g
    return out


# ---------------------------------------------------------------------------
# 2. Pre-whitening and cross-correlation
# ---------------------------------------------------------------------------
def _lagmat(P: np.ndarray, p: int, start: int):
    """Targets and lag matrix of a panel, rows with NaN dropped."""
    L = P.shape[1]
    y = P[:, start:].ravel()
    X = np.column_stack([P[:, start - k:L - k].ravel() for k in range(1, p + 1)]) if p else \
        np.empty((len(y), 0))
    ok = ~np.isnan(y) & ~np.isnan(X).any(1)
    return y[ok], X[ok]


def ar_fit(P: np.ndarray, p: int, start: int | None = None):
    """AR(p) with constant fitted by OLS on a panel. Returns (const, phi, rss, n)."""
    y, X = _lagmat(P, p, p if start is None else start)
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    r = y - Xc @ beta
    return beta[0], beta[1:], float(r @ r), len(y)


def _aic(rss: float, n: int, k: int) -> float:
    return n * np.log(rss / n) + 2 * k


def ar_order(P: np.ndarray, max_p: int = 14) -> int:
    """AR order minimising AIC, all orders compared on the same observations."""
    aics = []
    for p in range(1, max_p + 1):
        _, _, rss, n = ar_fit(P, p, start=max_p)
        aics.append(_aic(rss, n, p + 1))
    return int(np.argmin(aics)) + 1


def ar_filter(P: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """phi(B) applied within seasons; the first len(phi) days of each season are NaN."""
    p, L = len(phi), P.shape[1]
    out = np.full_like(P, np.nan)
    out[:, p:] = P[:, p:] - sum(phi[k - 1] * P[:, p - k:L - k] for k in range(1, p + 1))
    return out


def prewhiten(X: np.ndarray, Y: np.ndarray, p: int | None = None, max_p: int = 14):
    """Filter input and output panels with the AR model of the input.

    Returns (alpha, beta, p): alpha is the whitened input, beta the output passed through
    the same filter. Then corr(alpha_t, beta_{t+k}) is proportional to nu_k.
    """
    p = ar_order(X, max_p) if p is None else p
    _, phi, _, _ = ar_fit(X, p)
    return ar_filter(X, phi), ar_filter(Y, phi), p


def ccf(A: np.ndarray, B: np.ndarray, max_lag: int = 14) -> pd.Series:
    """Cross-correlation r_k = corr(A_t, B_{t+k}) pooled within seasons, k = -K..K.

    k > 0: A leads B by k days. Both panels are centred and scaled over the whole sample;
    every lag is divided by the same number of observations (that of lag 0).
    """
    a = (A - np.nanmean(A)) / np.nanstd(A)
    b = (B - np.nanmean(B)) / np.nanstd(B)
    L = A.shape[1]
    n0 = np.sum(~np.isnan(a * b))
    out = {}
    for k in range(-max_lag, max_lag + 1):
        prod = a[:, :L - k] * b[:, k:] if k >= 0 else a[:, -k:] * b[:, :L + k]
        out[k] = np.nansum(prod) / n0
    return pd.Series(out, name="ccf")


def derangements(n: int, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Random permutations of 0..n-1 with no fixed point."""
    out = []
    while len(out) < size:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            out.append(p)
    return out


def swap(P: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Season s receives the series of season perm[s], day for day.

    Where the donor season is shorter (NaN padding), the recipient keeps a NaN; where it
    is longer, the extra days are dropped.
    """
    out = P[perm].copy()
    out[np.isnan(P)] = np.nan
    return out


def ccf_null(A: np.ndarray, B: np.ndarray, max_lag: int = 14, n: int = 999, seed: int = 0):
    """Season-swap null distribution of the cross-correlation, shape (n, 2K+1)."""
    rng = np.random.default_rng(seed)
    return np.array([ccf(swap(A, p), B, max_lag).to_numpy()
                     for p in derangements(A.shape[0], n, rng)])


# ---------------------------------------------------------------------------
# 3. Transfer-function models
# ---------------------------------------------------------------------------
@dataclass
class TF:
    kind: str                 # "rational" or "fir"
    b: int                    # delay (rational) or first lag (fir)
    p: int                    # AR order of the noise
    const: float
    omega: np.ndarray         # rational: [omega]; fir: nu_b ... nu_K
    delta: float              # decay (rational), 0 for fir
    phi: np.ndarray
    rss: float
    n: int
    k_params: int
    weights_: np.ndarray = field(default=None, repr=False)

    @property
    def aic(self) -> float:
        return _aic(self.rss, self.n, self.k_params)

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.rss / (self.n - self.k_params)))

    def weights(self, horizon: int = 30) -> np.ndarray:
        """Impulse-response weights nu_0 ... nu_horizon."""
        nu = np.zeros(horizon + 1)
        if self.kind == "rational":
            k = np.arange(self.b, horizon + 1)
            nu[self.b:] = self.omega[0] * self.delta ** (k - self.b)
        else:
            m = min(len(self.omega), horizon + 1 - self.b)
            nu[self.b:self.b + m] = self.omega[:m]
        return nu

    @property
    def gain(self) -> float:
        """Total response of the output to a sustained one-unit rise of the input."""
        if self.kind == "rational":
            return float(self.omega[0] / (1 - self.delta))
        return float(self.omega.sum())

    def cumulative(self, horizon: int = 14) -> float:
        """Response accumulated over days 0..horizon after a one-day unit pulse."""
        return float(self.weights(horizon).sum())

    @property
    def half_life(self) -> float:
        """Days for the response to halve after its start (rational model)."""
        if self.kind != "rational" or not 0 < self.delta < 1:
            return 0.0 if self.kind == "rational" and self.delta <= 0 else np.nan
        return float(np.log(0.5) / np.log(self.delta))

    @property
    def mean_lag(self) -> float:
        """Average delay of the response, sum(k nu_k) / sum(nu_k) (rational model)."""
        return float(self.b + self.delta / (1 - self.delta)) if self.kind == "rational" else np.nan


def _shift(X: np.ndarray, k: int) -> np.ndarray:
    """x_{t-k} within seasons; days before the window are set to 0, the mean of an
    anomaly series."""
    if k == 0:
        return X
    out = np.zeros_like(X)
    out[:, k:] = X[:, :-k]
    return out


def _ar_resid(Nt: np.ndarray, phi: np.ndarray, start: int) -> np.ndarray:
    L = Nt.shape[1]
    e = Nt[:, start:].copy()
    for k in range(1, len(phi) + 1):
        e -= phi[k - 1] * Nt[:, start - k:L - k]
    return e


def fit_tf(X: np.ndarray, Y: np.ndarray, kind: str = "rational", b: int = 0, p: int = 1,
           K: int = K_FIR, start: int = P_MAX) -> TF:
    """Conditional least squares for y_t = c + nu(B) x_t + N_t, phi(B) N_t = e_t.

    X, Y  : (seasons x days) panels of the input and the output.
    kind  : "rational" (omega B^b / (1 - delta B)) or "fir" (nu_b ... nu_K).
    start : first day of each season used in the sum of squares (>= p). The same value
            for all candidate models makes their AIC comparable.
    The pre-window input is set to zero, so the first days carry a small start-up error.
    """
    if start < p:
        raise ValueError("start must be >= p")
    Xz = np.where(np.isnan(X), 0.0, X)
    mask = ~np.isnan(Y[:, start:])
    for k in range(1, p + 1):
        mask &= ~np.isnan(Y[:, start - k:Y.shape[1] - k])
    n = int(mask.sum())

    if kind == "fir":
        lags = np.arange(b, K + 1)
        Xs = np.stack([_shift(Xz, k) for k in lags])           # (lags, S, L)

        def mean_part(theta):
            return theta[0] + np.tensordot(theta[1:1 + len(lags)], Xs, axes=1)
        n_mean = 1 + len(lags)
        starts = [np.zeros(n_mean)]
    elif kind == "rational":
        Xb = _shift(Xz, b)

        def mean_part(theta):
            u = signal.lfilter([1.0], [1.0, -theta[2]], Xb, axis=1)
            return theta[0] + theta[1] * u
        n_mean = 3
        starts = [np.array([0.0, 0.0, d]) for d in (0.3, 0.8)]
    else:
        raise ValueError(kind)

    # starting value of the noise model: AR(p) of the output
    _, phi0, _, _ = ar_fit(Y, p) if p else (0, np.empty(0), 0, 0)

    def resid(theta):
        e = _ar_resid(Y - mean_part(theta), theta[n_mean:], start)
        return e[mask]

    lo = np.full(n_mean + p, -np.inf)
    hi = np.full(n_mean + p, np.inf)
    if kind == "rational":
        lo[2], hi[2] = 0.0, DELTA_MAX
    best = None
    for s in starts:
        theta0 = np.concatenate([s, phi0])
        if kind == "rational":                      # start omega at its LS value given delta
            u = signal.lfilter([1.0], [1.0, -s[2]], Xb, axis=1)
            e0 = _ar_resid(Y, phi0, start)[mask]
            u0 = _ar_resid(u, phi0, start)[mask]
            theta0[1] = float(u0 @ e0 / (u0 @ u0))
        sol = optimize.least_squares(resid, theta0, bounds=(lo, hi), method="trf",
                                     x_scale="jac", xtol=1e-10, ftol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
    th = best.x
    rss = float(2 * best.cost)
    if kind == "rational":
        return TF(kind, b, p, th[0], th[1:2].copy(), float(th[2]), th[3:].copy(), rss, n, 3 + p)
    return TF(kind, b, p, th[0], th[1:n_mean].copy(), 0.0, th[n_mean:].copy(), rss, n, n_mean + p)


def select_tf(X: np.ndarray, Y: np.ndarray, delays=range(0, B_MAX + 1),
              p: int | None = None) -> TF:
    """Rational model whose delay b minimises AIC.

    The noise order p is chosen first on the model with no input (an AR model of the
    output, orders 1..P_MAX), so that it cannot depend on the input; all fits use the same
    observations.
    """
    if p is None:
        aics = [_aic(*ar_fit(Y, q, start=P_MAX)[2:], q + 1) for q in range(1, P_MAX + 1)]
        p = int(np.argmin(aics)) + 1
    return min((fit_tf(X, Y, "rational", b=b, p=p) for b in delays), key=lambda f: f.aic)


def residual_checks(X: np.ndarray, Y: np.ndarray, model: TF, max_lag: int = 14) -> dict:
    """Box-Jenkins diagnostic checks of a fitted model.

    * Ljung-Box statistic of the residual autocorrelations (noise model adequate?);
    * portmanteau statistic of the cross-correlations between the pre-whitened input and
      the residuals at lags 0..max_lag (transfer function adequate?).
    P-values use chi-square laws that treat days as independent draws; with seasons as
    the unit of replication they are indicative only.
    """
    Xz = np.where(np.isnan(X), 0.0, X)
    nu = model.weights(X.shape[1])
    mean = model.const + sum(nu[k] * _shift(Xz, k) for k in np.flatnonzero(nu))
    E = np.full_like(Y, np.nan)
    E[:, P_MAX:] = _ar_resid(Y - mean, model.phi, P_MAX)
    n = int(np.sum(~np.isnan(E)))
    acf = ccf(E, E, max_lag).loc[1:].to_numpy()
    q_acf = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, max_lag + 1)))
    A, _, _ = prewhiten(X, Y)
    r = ccf(A, E, max_lag).loc[0:].to_numpy()
    q_ccf = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(0, max_lag + 1)))
    k_tf = 2 if model.kind == "rational" else len(model.omega)
    return dict(n=n, resid_acf=acf, resid_ccf=r,
                ljung_box=q_acf, p_ljung_box=float(stats.chi2.sf(q_acf, max(max_lag - model.p, 1))),
                q_cross=q_ccf, p_cross=float(stats.chi2.sf(q_ccf, max(max_lag + 1 - k_tf, 1))))


# ---------------------------------------------------------------------------
# 4. Inference with seasons as the unit of replication
# ---------------------------------------------------------------------------
def _fits(X: np.ndarray, Y: np.ndarray, p: int, horizon: int, delays) -> dict:
    """Both models and the summaries derived from them, for fixed noise order p."""
    rat = min((fit_tf(X, Y, "rational", b=b, p=p) for b in delays), key=lambda f: f.aic)
    fir = fit_tf(X, Y, "fir", b=0, p=p, K=horizon)
    fir0 = fit_tf(X, Y, "fir", b=0, p=p, K=0)             # same-day term only
    rss_ar = ar_fit(Y, p, start=P_MAX)[2]                  # no input
    w = fir.omega
    out = dict(b=rat.b, p=p, omega=float(rat.omega[0]), delta=rat.delta, gain=rat.gain,
               cum=rat.cumulative(horizon), half_life=rat.half_life, mean_lag=rat.mean_lag,
               same_day=float(w[0]), lagged=float(w[1:].sum()), total=float(w.sum()),
               r2_total=1 - fir.rss / rss_ar, r2_lagged=1 - fir.rss / fir0.rss,
               r2_rational=1 - rat.rss / rss_ar, sigma=rat.sigma, n=rat.n)
    out.update({f"nu_{k}": float(w[k]) for k in range(len(w))})
    return out


def tf_summary(df: pd.DataFrame, x: str, y: str, weekday: bool = True, season: bool = True,
               horizon: int = K_FIR, b: int | None = None, p: int | None = None) -> dict:
    """Full procedure on one data set: calendar removal, then both models.

    Returned quantities (input x, output y, both in the units of the series):
      b, omega, delta       rational model omega B^b / (1 - delta B), b chosen by AIC;
      gain, cum             its total gain and its response cumulated over 0..horizon days;
      half_life, mean_lag   days;
      nu_0 .. nu_horizon    unrestricted weights (distributed-lag model);
      same_day              nu_0, an association whose direction cannot be told apart;
      lagged                nu_1 + ... + nu_horizon;   total = same_day + lagged;
      r2_total              share of the output's unpredicted variance (after its own AR
                            model) explained by the input at lags 0..horizon;
      r2_lagged             share explained by lags 1..horizon beyond the same-day term;
      r2_rational           as r2_total, for the rational model.
    ``b`` and ``p`` fix the delay and the noise order instead of selecting them.
    """
    a = anomalies(df, weekday=weekday, season=season)
    X, Y = panel(a, x), panel(a, y)
    if p is None:
        aics = [_aic(*ar_fit(Y, q, start=P_MAX)[2:], q + 1) for q in range(1, P_MAX + 1)]
        p = int(np.argmin(aics)) + 1
    return _fits(X, Y, p, horizon, range(0, B_MAX + 1) if b is None else [b])


def jackknife(df: pd.DataFrame, func, **kwargs) -> pd.DataFrame:
    """Delete-one-season replicates of ``func(df, **kwargs)`` (a dict of numbers).

    The calendar profile is re-estimated without the deleted season, since ``func``
    receives the raw series.
    """
    seasons = list(dict.fromkeys(df["season"]))
    return pd.DataFrame([func(df[df["season"] != s], **kwargs) for s in seasons], index=seasons)


def jackknife_interval(estimate: float, replicates, level: float = 0.95):
    """Jackknife standard error and Student-t interval (seasons - 1 degrees of freedom)."""
    v = np.asarray(replicates, float)
    n = len(v)
    se = float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return se, estimate - q * se, estimate + q * se


def season_swap_null(df: pd.DataFrame, x: str, y: str, n: int = 199, seed: int = 0,
                     weekday: bool = True, season: bool = True, horizon: int = K_FIR,
                     p: int | None = None) -> pd.DataFrame:
    """Null distribution of the transfer-function summaries when the input of each season
    is replaced by the input of another season (same dates, another year).

    The anomalies are computed once, so the surrogate input has no calendar component left
    and only its day-to-day correspondence with the output is broken. The delay is
    re-selected for every surrogate, so the null accounts for that selection.
    """
    a = anomalies(df, weekday=weekday, season=season)
    X, Y = panel(a, x), panel(a, y)
    if p is None:
        p = tf_summary(df, x, y, weekday, season, horizon)["p"]
    rng = np.random.default_rng(seed)
    return pd.DataFrame([_fits(swap(X, perm), Y, p, horizon, range(0, B_MAX + 1))
                         for perm in derangements(X.shape[0], n, rng)])


def upper_p(obs: float, null) -> float:
    """One-sided surrogate p-value: share of surrogates at least as large as observed."""
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def two_sided_p(obs: float, null) -> float:
    """Two-sided surrogate p-value on |statistic - null mean|."""
    null = np.asarray(null, float)
    d = np.abs(null - null.mean())
    return float((1 + np.sum(d >= abs(obs - null.mean()))) / (1 + len(null)))


# ---------------------------------------------------------------------------
# 5. Frequency domain
# ---------------------------------------------------------------------------
def cross_spectrum(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """Cross-spectral estimates with one Hann-tapered segment per season.

    Returns, for each frequency (cycles per day): squared coherence, gain |S_xy| / S_xx of
    the frequency-response function from x to y, phase (radians) and the corresponding
    delay in days (positive: y lags x). Seasons are truncated to the shortest one.
    """
    L = int(np.min(np.sum(~np.isnan(X), 1)))
    x = X[:, :L] - X[:, :L].mean(1, keepdims=True)
    y = Y[:, :L] - Y[:, :L].mean(1, keepdims=True)
    w = signal.windows.hann(L, sym=False)
    Fx, Fy = np.fft.rfft(x * w, axis=1), np.fft.rfft(y * w, axis=1)
    f = np.fft.rfftfreq(L, d=1.0)
    Sxx, Syy = np.mean(np.abs(Fx) ** 2, 0), np.mean(np.abs(Fy) ** 2, 0)
    Sxy = np.mean(np.conj(Fx) * Fy, 0)
    phase = np.angle(Sxy)
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = -phase / (2 * np.pi * f)
    out = pd.DataFrame({"freq": f, "period": np.r_[np.inf, 1 / f[1:]],
                        "coherence": np.abs(Sxy) ** 2 / (Sxx * Syy),
                        "gain": np.abs(Sxy) / Sxx, "phase": phase, "delay": delay})
    return out.iloc[1:].reset_index(drop=True)


def coherence_threshold(n_segments: int, alpha: float = 0.05) -> float:
    """Level exceeded with probability alpha by the squared coherence of two independent
    series estimated from n_segments independent segments: 1 - alpha^(1 / (n - 1))."""
    return 1 - alpha ** (1 / (n_segments - 1))
