"""Nonlinear (kernel) Granger causality between two daily series.

Notebook 04 uses KGC and lsNGC (``run_tests``). KPCR and GPSIC are also implemented
and tested here, but not used in the notebook.

lsNGC  Large-scale nonlinear Granger causality (Wismuller et al. 2021, Sci. Rep. 11,
       8516), as described by Murphy & Benavoli (2026), Sec. 2.1.3: normalised Gaussian
       RBF features around k-means centres (c_f = 25 for the effect's past, c_g = 5 for
       the cause's past), additive regression, F-test; the false discovery rate is
       controlled across tests.

Kernel methods compared by Murphy & Benavoli (2026),
"Constraint- and Score-Based Nonlinear Granger Causality Discovery with Kernels",
arXiv:2601.09579, for a bivariate system (n_t = 2):

* KGC   Kernel Granger causality (Marinazzo, Pellicoro & Stramaglia 2008, PRL 100,
        144103). The effect y is projected on the range P of the restricted kernel K(X);
        the residual e = (I - P) y is correlated with the eigenvectors w_i of
        K~ = (I - P) K(Z) (I - P). r_i = corr(e, w_i). Filtered index
        delta_f = sum of r_i^2 over the r_i significant after Bonferroni correction
        (alpha / (n_r * n_t (n_t - 1) / 2)). Unfiltered delta = sum of all r_i^2.
* KPCR  Kernel principal component regression (the paper's unified method): projectors
        P, P' on the leading eigenvectors of K(X) and K(Z) (eigenvalue > mu * max),
        degrees of freedom p_r = rank(P), p_u = rank(P'), F-test
        F = [(SSR_r - SSR_u)/(p_u - p_r)] / [SSR_u / (n - p_u - 1)] (one more degree
        of freedom than the paper's n - p_u, for the mean removed by centring), Bonferroni
        alpha' = alpha / (n_t (n_t - 1)).
* GPSIC Gaussian process regression of y on all lagged inputs with an ARD
        squared-exponential kernel (one lengthscale per input, kernel variance fixed to
        1), hyperparameters maximising log marginal likelihood - log(n)/2 * SIC, where
        SIC = sum_d phi_omega(1/l_d), phi_omega(x) = x^2 / (x^2 + omega^2) is a smooth L0
        norm; omega decreases from 100 to 1e-5 over 50 steps. A lag is relevant if its
        lengthscale is < 50; the cause Granger-causes the effect if any of its lags is.

Kernels: squared exponential k(x, x') = exp(-|x - x'|^2 / (2 l^2)) with the paper's
heuristic l = C * d (d = number of input columns of Z; C = 2 as in the paper's KGC runs),
or the inhomogeneous polynomial (1 + x.x')^p. With the polynomial kernel of degree 1,
KGC and KPCR reduce exactly to linear Granger causality.

Additions for campaign data (not in the paper)
----------------------------------------------
* Lags are taken within a season, never across two seasons.
* Optional calendar conditioning: day in the campaign window and day of week are added
  to the restricted and unrestricted inputs.
* Season-swap surrogates. The paper's tests assume no latent confounder; here the
  campaign calendar drives both series. Under the null, the cause series of each season
  is replaced by that of another season (a random derangement), aligned by day in the
  window. The surrogate keeps the cause's own dynamics and its calendar shape, but loses
  its day-to-day link with the effect. p = (1 + #{null >= observed}) / (1 + n_surrogates).
  This removes spurious causality from a common driver with the same timing every year,
  not from a common driver whose timing changes from one year to the next.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

MU = 1e-6            # eigenvalue filter (paper)
ALPHA = 0.05
N_T = 2              # number of series in the system


# ---------------------------------------------------------------------------
# Design: lagged blocks within seasons
# ---------------------------------------------------------------------------
@dataclass
class Design:
    y: np.ndarray            # effect at t (standardised), (n,)
    own: np.ndarray          # past of the effect, (n, m)
    cause: np.ndarray        # past of the cause, (n, m)
    calendar: np.ndarray     # day in window (standardised), sin/cos day of week, (n, 3)
    season: np.ndarray       # season index of each row, (n,)
    cause_panel: np.ndarray  # standardised cause, (n_seasons, L), used for surrogates
    m: int
    start: int

    def X(self, conditional: bool) -> np.ndarray:
        """Restricted inputs: past of the effect (+ calendar)."""
        return np.hstack([self.own, self.calendar]) if conditional else self.own

    def cause_from_panel(self, panel: np.ndarray) -> np.ndarray:
        n_s, L = panel.shape
        idx = np.arange(self.start, L)
        return np.vstack([np.column_stack([panel[s, idx - k] for k in range(1, self.m + 1)])
                          for s in range(n_s)])


def make_design(df: pd.DataFrame, effect: str, cause: str, m: int,
                start: int | None = None) -> Design:
    """Lagged blocks from a daily frame with columns gt, mc, season (date index).

    Seasons are truncated to the shortest season so they can be swapped day by day.
    Each series is standardised over the whole sample. ``start`` (>= m) is the first
    day of each season used as a target: use the same value for every m to compare
    lag orders on the same observations.
    """
    start = m if start is None else start
    if start < m:
        raise ValueError("start must be >= m")
    seasons = list(dict.fromkeys(df["season"]))
    groups = [df[df["season"] == s] for s in seasons]
    L = min(len(g) for g in groups)
    groups = [g.iloc[:L] for g in groups]

    def panel(col):
        v = np.vstack([g[col].to_numpy(float) for g in groups])
        return (v - v.mean()) / v.std()

    e_panel, c_panel = panel(effect), panel(cause)
    idx = np.arange(start, L)
    y = np.concatenate([e_panel[s, idx] for s in range(len(groups))])
    own = np.vstack([np.column_stack([e_panel[s, idx - k] for k in range(1, m + 1)])
                     for s in range(len(groups))])
    cal = []
    for g in groups:
        dow = pd.DatetimeIndex(g.index)[idx].dayofweek.to_numpy()
        cal.append(np.column_stack([idx.astype(float), np.sin(2 * np.pi * dow / 7),
                                    np.cos(2 * np.pi * dow / 7)]))
    cal = np.vstack(cal)
    cal[:, 0] = (cal[:, 0] - cal[:, 0].mean()) / cal[:, 0].std()
    des = Design(y=y, own=own, cause=None, calendar=cal,
                 season=np.repeat(np.arange(len(groups)), len(idx)),
                 cause_panel=c_panel, m=m, start=start)
    des.cause = des.cause_from_panel(c_panel)
    return des


def derangements(n_seasons: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """n random permutations of the seasons with no fixed point."""
    out = []
    while len(out) < n:
        p = rng.permutation(n_seasons)
        if np.all(p != np.arange(n_seasons)):
            out.append(p)
    return out


def surrogate_pvalue(obs: float, null: np.ndarray) -> float:
    return float((1 + np.sum(np.asarray(null) >= obs)) / (1 + len(null)))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _sqdist(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    Y = X if Y is None else Y
    d = np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X @ Y.T
    return np.maximum(d, 0.0)


def gram(X: np.ndarray, kernel: str, param: float) -> np.ndarray:
    """Gram matrix: kernel 'se' (param = lengthscale) or 'poly' (param = degree)."""
    if kernel == "se":
        return np.exp(-_sqdist(X) / (2 * param ** 2))
    if kernel == "poly":
        return (1.0 + X @ X.T) ** int(param)
    raise ValueError(kernel)


def _center(K: np.ndarray) -> np.ndarray:
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def _range_basis(K: np.ndarray, mu: float = MU) -> np.ndarray:
    """Orthonormal eigenvectors of a symmetric PSD matrix with eigenvalue > mu * max."""
    w, V = linalg.eigh((K + K.T) / 2)
    return V[:, w > mu * w.max()]


def se_lengthscale(n_inputs: int, C: float = 2.0) -> float:
    """Paper heuristic l = C * n_t * m, generalised to l = C * (number of inputs)."""
    return C * n_inputs


# ---------------------------------------------------------------------------
# KGC (Marinazzo et al. 2008)
# ---------------------------------------------------------------------------
class KGC:
    """The restricted projection is computed once; ``stat`` accepts any cause block."""

    def __init__(self, y, X, kernel="se", param=None, alpha=ALPHA):
        self.kernel, self.alpha, self.X = kernel, alpha, X
        self.param = param
        self.U = _range_basis(_center(gram(X, kernel, self._param(X.shape[1]))))
        yc = (y - y.mean()) / np.linalg.norm(y - y.mean())
        self.e = yc - self.U @ (self.U.T @ yc)          # e = (I - P) y
        self.n = len(y)

    def _param(self, n_inputs):
        if self.param is not None:
            return self.param
        return se_lengthscale(n_inputs) if self.kernel == "se" else 2

    def stat(self, B) -> dict:
        Z = np.hstack([self.X, B])
        # the restricted and unrestricted kernels use the same lengthscale (that of Z),
        # so that H is contained in H' for the polynomial kernel and nearly so for SE
        Kz = _center(gram(Z, self.kernel, self._param(Z.shape[1])))
        U = self.U
        KU = Kz @ U
        Kt = Kz - U @ KU.T - KU @ U.T + U @ (U.T @ KU) @ U.T   # (I-P) K' (I-P)
        W = _range_basis(Kt)
        r = (W.T @ self.e) / np.linalg.norm(self.e)
        nr = len(r)
        t = r * np.sqrt((self.n - 2) / np.clip(1 - r ** 2, 1e-12, None))
        pv = 2 * stats.t.sf(np.abs(t), self.n - 2)
        thr = self.alpha / (nr * N_T * (N_T - 1) / 2) if nr else 0
        sig = pv < thr
        return {"delta_f": float(np.sum(r[sig] ** 2)), "delta": float(np.sum(r ** 2)),
                "n_r": nr, "n_f": int(sig.sum()),
                "p_min_bonf": float(min(1.0, pv.min() * nr * N_T * (N_T - 1) / 2)) if nr else 1.0}


# ---------------------------------------------------------------------------
# lsNGC (Wismuller et al. 2021), as described in Murphy & Benavoli (2026), Sec. 2.1.3
# ---------------------------------------------------------------------------
def grbf_features(X: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Normalised Gaussian RBF activations around k-means centres.

    f_j(x) = exp(-|x - u_j|^2 / l^2) / sum_k exp(-|x - u_k|^2 / l^2), with l the mean
    distance between the cluster centres.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
    U = km.cluster_centers_
    dc = np.sqrt(_sqdist(U))
    l = dc[np.triu_indices_from(dc, 1)].mean()
    A = -_sqdist(X, U) / l ** 2
    A = np.exp(A - A.max(1, keepdims=True))          # stable softmax
    return A / A.sum(1, keepdims=True)


class LSNGC:
    """Large-scale nonlinear Granger causality (bivariate case).

    Restricted model: y ~ f(X) (c_f = 25 clusters on the effect's past [+ calendar]);
    unrestricted: y ~ [f(X), g(X_a)] (c_g = 5 clusters on the cause's past), fitted by
    least squares; F-test on the added block. The activations of each block sum to 1,
    so the intercept is implicit and the unrestricted design has rank c_f + c_g - 1;
    the F-test uses these actual ranks.
    """

    def __init__(self, y, X, c_f=25, c_g=5, seed=0):
        self.y, self.c_g, self.seed = y, c_g, seed
        self.F_X = grbf_features(X, c_f, seed)
        self.n = len(y)
        self.p_r = np.linalg.matrix_rank(self.F_X)
        b, *_ = np.linalg.lstsq(self.F_X, y, rcond=None)
        r = y - self.F_X @ b
        self.ssr_r = float(r @ r)

    def stat(self, B) -> dict:
        Zf = np.hstack([self.F_X, grbf_features(B, self.c_g, self.seed)])
        p_u = np.linalg.matrix_rank(Zf)
        b, *_ = np.linalg.lstsq(Zf, self.y, rcond=None)
        r = self.y - Zf @ b
        ssr_u = float(r @ r)
        F = ((self.ssr_r - ssr_u) / (p_u - self.p_r)) / (ssr_u / (self.n - p_u))
        return {"delta": 1 - ssr_u / self.ssr_r, "F": F, "p_r": int(self.p_r), "p_u": int(p_u),
                "p_value": float(stats.f.sf(F, p_u - self.p_r, self.n - p_u))}


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * n / (rank + 1))
        adj[i] = running
    return adj


# ---------------------------------------------------------------------------
# KPCR (Murphy & Benavoli 2026) -- implemented and tested, not used in notebook 04
# ---------------------------------------------------------------------------
class KPCR:
    def __init__(self, y, X, kernel="se", param=None, alpha=ALPHA):
        self.kernel, self.param, self.alpha, self.X = kernel, param, alpha, X
        self.y = y - y.mean()
        self.n = len(y)
        self._cache = {}

    def _restricted(self, param):
        if param not in self._cache:
            U = _range_basis(_center(gram(self.X, self.kernel, param)))
            r = self.y - U @ (U.T @ self.y)
            self._cache[param] = (U.shape[1], float(r @ r))
        return self._cache[param]

    def stat(self, B) -> dict:
        Z = np.hstack([self.X, B])
        param = self.param if self.param is not None else (
            se_lengthscale(Z.shape[1]) if self.kernel == "se" else 2)
        p_r, ssr_r = self._restricted(param)
        U = _range_basis(_center(gram(Z, self.kernel, param)))
        p_u = U.shape[1]
        res = self.y - U @ (U.T @ self.y)
        ssr_u = float(res @ res)
        out = {"p_r": p_r, "p_u": p_u, "delta": 1 - ssr_u / ssr_r}
        if p_u > p_r and self.n > p_u + 1:
            F = ((ssr_r - ssr_u) / (p_u - p_r)) / (ssr_u / (self.n - p_u - 1))
            pv = float(stats.f.sf(F, p_u - p_r, self.n - p_u - 1))
            out.update(F=F, p_value=pv, p_bonf=min(1.0, pv * N_T * (N_T - 1)))
        else:
            out.update(F=np.nan, p_value=np.nan, p_bonf=np.nan)
        return out


# ---------------------------------------------------------------------------
# GPSIC (Murphy & Benavoli 2026) -- implemented and tested, not used in notebook 04
# ---------------------------------------------------------------------------
class GPSIC:
    """GP regression with ARD squared-exponential kernel and SIC-penalised evidence.

    Parameters are optimised in log space: log l_d (one per input) and log sigma^2;
    the kernel variance is fixed to 1 because inputs and target are standardised.
    """

    def __init__(self, threshold=50.0, omegas=None, maxiter=40):
        self.threshold = threshold
        self.omegas = np.logspace(2, -5, 50) if omegas is None else omegas
        self.maxiter = maxiter

    def _objective(self, theta, y, D, omega, n):
        d = D.shape[0]
        log_l, log_s2 = theta[:d], theta[d]
        l2 = np.exp(2 * log_l)
        K = np.exp(-0.5 * np.tensordot(1 / l2, D, axes=1))
        s2 = np.exp(log_s2)
        Ky = K + (s2 + 1e-8) * np.eye(n)
        try:
            c = linalg.cho_factor(Ky, lower=True)
        except linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        a = linalg.cho_solve(c, y)
        logdet = 2 * np.sum(np.log(np.diag(c[0])))
        lml = -0.5 * y @ a - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
        Kinv = linalg.cho_solve(c, np.eye(n))
        Wm = np.outer(a, a) - Kinv
        WK = Wm * K
        g_l = 0.5 * np.tensordot(D, WK, axes=([1, 2], [0, 1])) / l2   # d lml / d log l
        g_s = 0.5 * s2 * np.trace(Wm)
        # SIC penalty on inverse lengthscales
        x2 = 1 / l2
        phi = x2 / (x2 + omega ** 2)
        dphi = -2 * x2 * omega ** 2 / (x2 + omega ** 2) ** 2               # d phi / d log l
        pen = 0.5 * np.log(n)
        f = -(lml - pen * phi.sum())
        g = -np.concatenate([g_l - pen * dphi, [g_s]])
        return f, g

    def fit(self, Z, y):
        y = (y - y.mean()) / y.std()
        n, d = Z.shape
        D = np.stack([(Z[:, j, None] - Z[None, :, j]) ** 2 for j in range(d)])
        theta = np.concatenate([np.zeros(d), [np.log(0.5)]])
        bounds = [(np.log(0.05), np.log(1e6))] * d + [(np.log(1e-4), np.log(10.0))]
        for omega in self.omegas:
            res = optimize.minimize(self._objective, theta, args=(y, D, omega, n), jac=True,
                                    method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": self.maxiter})
            theta = res.x
        self.lengthscales_ = np.exp(theta[:d])
        self.noise_ = float(np.exp(theta[d]))
        f, _ = self._objective(theta, y, D, self.omegas[-1], n)
        self.objective_ = -f
        return self


# ---------------------------------------------------------------------------
# Lag selection and full tests for one pair
# ---------------------------------------------------------------------------
def select_lag(df, effect, cause, lags=(1, 2, 3, 5, 7), conditional=False,
               alphas=(0.1, 1.0, 10.0)) -> pd.Series:
    """Leave-one-season-out CV error of a kernel ridge model of the effect on its own
    past (+ calendar), for each lag order. The lag is chosen without the cause.
    All orders are compared on the same observations (start = max(lags))."""
    out = {}
    for m in lags:
        d = make_design(df, effect, cause, m, start=max(lags))
        X = d.X(conditional)
        K = gram(X, "se", se_lengthscale(X.shape[1]) / 2)
        sse = np.zeros(len(alphas))
        for s in np.unique(d.season):
            tr, te = d.season != s, d.season == s
            mu = d.y[tr].mean()
            for j, a in enumerate(alphas):
                c = linalg.solve(K[np.ix_(tr, tr)] + a * np.eye(tr.sum()), d.y[tr] - mu,
                                 assume_a="pos")
                sse[j] += np.sum((d.y[te] - mu - K[np.ix_(te, tr)] @ c) ** 2)
        out[m] = sse.min() / len(d.y)
    return pd.Series(out, name="cv_mse")


def run_tests(df, effect, cause, m, conditional=False, start=None, n_surrogates=199,
              seed=0, methods=("kgc_lin", "kgc_se", "lsngc")) -> dict:
    """KGC (linear and SE kernels) and lsNGC for the direction cause -> effect, with the
    paper's parametric tests and season-swap surrogate p-values."""
    d = make_design(df, effect, cause, m, start=start)
    X = d.X(conditional)
    perms = derangements(d.cause_panel.shape[0], n_surrogates, np.random.default_rng(seed))
    surr = [d.cause_from_panel(d.cause_panel[p]) for p in perms]
    out = {"m": m, "n": len(d.y), "conditional": conditional, "nulls": {}}
    for key in methods:
        if key.startswith("kgc"):
            if key == "kgc_lin":
                test = KGC(d.y, X, "poly", 1)
            else:
                test = KGC(d.y, X, "se", se_lengthscale(X.shape[1] + d.cause.shape[1]))
            o = test.stat(d.cause)
            out.update({f"{key}_delta_f": o["delta_f"], f"{key}_n_r": o["n_r"],
                        f"{key}_n_f": o["n_f"], f"{key}_p_param": o["p_min_bonf"]})
        else:
            test = LSNGC(d.y, X, seed=seed)
            o = test.stat(d.cause)
            out.update({f"{key}_F": o["F"], f"{key}_p_r": o["p_r"], f"{key}_p_u": o["p_u"],
                        f"{key}_p_param": o["p_value"]})
        null = np.array([test.stat(B)["delta"] for B in surr])
        out.update({f"{key}_delta": o["delta"], f"{key}_null_mean": float(null.mean()),
                    f"{key}_p_surr": surrogate_pvalue(o["delta"], null)})
        out["nulls"][key] = (o["delta"], null)
    return out


# ---------------------------------------------------------------------------
# Season bootstrap of the test statistic
# ---------------------------------------------------------------------------
def resample_seasons(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw seasons with replacement; each draw gets a unique season label."""
    seasons = list(dict.fromkeys(df["season"]))
    pick = rng.choice(len(seasons), size=len(seasons), replace=True)
    return pd.concat([df[df["season"] == seasons[i]].assign(season=f"b{j}_{seasons[i]}")
                      for j, i in enumerate(pick)])


def delta_stat(df, effect, cause, m, method, conditional=False, start=None, seed=0) -> float:
    """Statistic delta of one method on one dataset (no surrogates)."""
    d = make_design(df, effect, cause, m, start=start)
    X = d.X(conditional)
    if method == "kgc_lin":
        return KGC(d.y, X, "poly", 1).stat(d.cause)["delta"]
    if method == "kgc_se":
        return KGC(d.y, X, "se", se_lengthscale(X.shape[1] + d.cause.shape[1])).stat(d.cause)["delta"]
    if method == "lsngc":
        return LSNGC(d.y, X, seed=seed).stat(d.cause)["delta"]
    raise ValueError(method)


def bootstrap_delta(df, effect, cause, m, method, n_boot=499, seed=0, conditional=False,
                    start=None) -> np.ndarray:
    """Season-bootstrap distribution of delta (seasons resampled with replacement).

    Not used in the notebooks: duplicated seasons inflate in-sample statistics such as
    delta, so the bootstrap distribution is shifted upwards. Use ``jackknife_delta``.
    """
    rng = np.random.default_rng(seed)
    return np.array([delta_stat(resample_seasons(df, rng), effect, cause, m, method,
                                conditional, start) for _ in range(n_boot)])


def jackknife_delta(df, effect, cause, m, method, seed=0, conditional=False,
                    start=None) -> np.ndarray:
    """Delete-one-season jackknife replicates of delta.

    Unlike the bootstrap, no season is duplicated, which matters here: duplicated seasons
    inflate in-sample fit statistics such as delta.
    """
    seasons = list(dict.fromkeys(df["season"]))
    return np.array([delta_stat(df[df["season"] != s], effect, cause, m, method,
                                conditional, start, seed) for s in seasons])


def jackknife_ci(estimate: float, replicates, level: float = 0.95) -> tuple[float, float, float]:
    """Jackknife standard error and normal-approximation interval around the estimate."""
    v = np.asarray(replicates, float)
    n = len(v)
    se = float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))
    z = stats.norm.ppf(0.5 + level / 2)
    return se, estimate - z * se, estimate + z * se
