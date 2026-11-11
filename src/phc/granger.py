"""Linear Granger causality between Google Trends (GT) and MediaCloud (MC).

Model for the direction cause -> effect, with asymmetric lags (p1, p2):

    full    : effect_t = c + sum_{k=1..p1} a_k effect_{t-k} + sum_{k=1..p2} b_k cause_{t-k} + e_t
    reduced : effect_t = c + sum_{k=1..p1} a_k effect_{t-k}                                 + e_t

H0: b_1 = ... = b_p2 = 0, tested with
    F = [(RSS_red - RSS_full) / p2] / [RSS_full / (n - p1 - p2 - 1)]  ~  F(p2, n - p1 - p2 - 1).

Effect size: gain in adjusted R^2, R2_adj = 1 - (1 - R2)(n - 1)/(n - k - 1),
with k = p1 (reduced) and k = p1 + p2 (full).

Default options reproduce the paper exactly:
  * the daily seasonal windows are concatenated and lags are taken by row
    position, so a few lags cross the boundary between two seasons;
  * each (p1, p2) cell of the AIC grid is estimated on its own sample,
    starting at max(p1, p2).
Two options address these limitations (see notebook 03):
  * ``by_season=True``  : lags never cross a season boundary;
  * ``common_sample=True``: every AIC cell uses the same observations,
    so AIC values are strictly comparable across the grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CAMPAIGNS, DAILY_DIR

DIRECTIONS = [("mc", "gt", "GT → MC"), ("gt", "mc", "MC → GT")]  # (effect, cause, label)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_daily(code: str, data_dir=DAILY_DIR) -> pd.DataFrame:
    """Daily GT and MC series of one campaign, indexed by date.

    Columns: gt (0-100 within season), mc (0-100 within season), season.
    """
    gt = pd.read_csv(data_dir / f"GT_{code}_Granger.csv", parse_dates=["date"],
                     encoding="utf-8-sig")
    mc = pd.read_csv(data_dir / f"MC_{code}_Granger_normalized.csv", parse_dates=["date"])
    df = (gt.rename(columns={"campaign_index": "gt"})[["date", "gt", "season"]]
            .merge(mc.rename(columns={"n_articles": "mc"})[["date", "mc"]],
                   on="date", how="inner")
            .sort_values("date").set_index("date"))
    df["season"] = df["season"].astype(str)
    return df[["gt", "mc", "season"]]


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------
def lag_design(df: pd.DataFrame, effect: str, cause: str, p1: int, p2: int,
               start: int | None = None, by_season: bool = False):
    """Return (y, X_reduced, X_full) without constant.

    start     : index of the first usable observation in each segment
                (default max(p1, p2)). Use the same ``start`` for every model
                to estimate them on a common sample.
    by_season : if True, lags are computed within each season only.
    """
    start = max(p1, p2) if start is None else start
    if start < max(p1, p2):
        raise ValueError("start must be >= max(p1, p2)")
    groups = [g for _, g in df.groupby("season", sort=False)] if by_season else [df]
    ys, xr, xc = [], [], []
    for g in groups:
        e = g[effect].to_numpy(float)
        c = g[cause].to_numpy(float)
        T = len(g)
        if T <= start:
            continue
        idx = np.arange(start, T)
        ys.append(e[idx])
        xr.append(np.column_stack([e[idx - k] for k in range(1, p1 + 1)]))
        xc.append(np.column_stack([c[idx - k] for k in range(1, p2 + 1)]))
    y = np.concatenate(ys)
    X_red = np.vstack(xr)
    X_full = np.hstack([X_red, np.vstack(xc)])
    return y, X_red, X_full


def _rss(X: np.ndarray, y: np.ndarray) -> float:
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    r = y - Xc @ beta
    return float(r @ r)


def gaussian_aic(rss: float, n: int, n_params: int) -> float:
    """AIC of an OLS model with Gaussian errors (same value as statsmodels)."""
    llf = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    return -2 * llf + 2 * n_params


# ---------------------------------------------------------------------------
# Lag selection
# ---------------------------------------------------------------------------
def aic_grid(df: pd.DataFrame, effect: str, cause: str, max_lag: int = 20,
             common_sample: bool = False, by_season: bool = False) -> pd.DataFrame:
    """AIC of the full model for every (p1, p2) in [1, max_lag]^2.

    Rows: p1 (lags of the effect); columns: p2 (lags of the cause).
    """
    grid = pd.DataFrame(np.nan, index=pd.RangeIndex(1, max_lag + 1, name="p1"),
                        columns=pd.RangeIndex(1, max_lag + 1, name="p2"))
    for p1 in grid.index:
        for p2 in grid.columns:
            start = max_lag if common_sample else None
            y, _, X_full = lag_design(df, effect, cause, p1, p2, start, by_season)
            if len(y) < (p1 + p2) * 3 + 5:      # same guard as the original code
                continue
            grid.loc[p1, p2] = gaussian_aic(_rss(X_full, y), len(y), p1 + p2 + 1)
    return grid


def best_lags(grid: pd.DataFrame) -> tuple[int, int]:
    p1, p2 = grid.stack().idxmin()
    return int(p1), int(p2)


# ---------------------------------------------------------------------------
# Granger F-test
# ---------------------------------------------------------------------------
def granger_test(df: pd.DataFrame, effect: str, cause: str, p1: int, p2: int,
                 by_season: bool = False) -> dict:
    y, X_red, X_full = lag_design(df, effect, cause, p1, p2, by_season=by_season)
    n = len(y)
    tss = float(((y - y.mean()) ** 2).sum())
    rss_red, rss_full = _rss(X_red, y), _rss(X_full, y)
    df_num, df_den = p2, n - p1 - p2 - 1
    F = ((rss_red - rss_full) / df_num) / (rss_full / df_den)
    pval = float(stats.f.sf(F, df_num, df_den))
    r2_red, r2_full = 1 - rss_red / tss, 1 - rss_full / tss
    adj = lambda r2, k: 1 - (1 - r2) * (n - 1) / (n - k - 1)
    r2a_red, r2a_full = adj(r2_red, p1), adj(r2_full, p1 + p2)
    return dict(p1=p1, p2=p2, n=n, F=F, pval=pval,
                r2_red=r2_red, r2_full=r2_full,
                r2adj_red=r2a_red, r2adj_full=r2a_full,
                delta_r2adj=r2a_full - r2a_red)


def granger_test_fe(df: pd.DataFrame, effect: str, cause: str, p1: int, p2: int) -> dict:
    """Granger F-test with within-season lags and one intercept per season.

    As ``granger_test(..., by_season=True)``, but both models also contain season fixed
    effects, so that differences in the level of a series between seasons (the size of the
    campaign in a given year) cannot make the past of the cause "predict" the effect. The
    degrees of freedom count the S - 1 extra parameters.
    """
    start = max(p1, p2)
    y, X_red, X_full = lag_design(df, effect, cause, p1, p2, by_season=True)
    counts = [len(g) - start for _, g in df.groupby("season", sort=False) if len(g) > start]
    D = pd.get_dummies(np.repeat(np.arange(len(counts)), counts)).to_numpy(float)[:, 1:]
    n, S = len(y), len(counts)
    tss = float(((y - y.mean()) ** 2).sum())
    rss_red, rss_full = _rss(np.hstack([X_red, D]), y), _rss(np.hstack([X_full, D]), y)
    k_red, k_full = p1 + S - 1, p1 + p2 + S - 1
    F = ((rss_red - rss_full) / p2) / (rss_full / (n - k_full - 1))
    pval = float(stats.f.sf(F, p2, n - k_full - 1))
    adj = lambda rss, k: 1 - (rss / (n - k - 1)) / (tss / (n - 1))
    return dict(p1=p1, p2=p2, n=n, F=F, pval=pval,
                r2adj_red=adj(rss_red, k_red), r2adj_full=adj(rss_full, k_full),
                delta_r2adj=adj(rss_full, k_full) - adj(rss_red, k_red))


def loyo(df: pd.DataFrame, effect: str, cause: str, p1: int, p2: int,
         by_season: bool = False, alpha: float = 0.05) -> pd.DataFrame:
    """Leave-one-season-out: re-run the test with each season removed.

    The lags (p1, p2) are those selected on the full sample, as in the paper.
    """
    rows = []
    for held_out in df["season"].unique():
        train = df[df["season"] != held_out]
        res = granger_test(train, effect, cause, p1, p2, by_season=by_season)
        rows.append({"held_out": held_out, **res, "significant": res["pval"] < alpha})
    return pd.DataFrame(rows)


def coefficients(df: pd.DataFrame, effect: str, cause: str, p1: int, p2: int,
                 by_season: bool = False, level: float = 0.95) -> pd.DataFrame:
    """OLS coefficients of the full model with confidence intervals."""
    import statsmodels.api as sm
    y, _, X_full = lag_design(df, effect, cause, p1, p2, by_season=by_season)
    res = sm.OLS(y, sm.add_constant(X_full)).fit()
    ci = res.conf_int(1 - level)
    names = (["const"] + [f"alpha_{k}" for k in range(1, p1 + 1)]
             + [f"beta_{k}" for k in range(1, p2 + 1)])
    out = pd.DataFrame({"term": names, "coef": res.params,
                        "ci_low": ci[:, 0], "ci_high": ci[:, 1], "pval": res.pvalues})
    out["kind"] = out["term"].str.split("_").str[0]
    out["lag"] = pd.to_numeric(out["term"].str.split("_").str[1], errors="coerce")
    return out


def stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) tests on gt and mc."""
    import warnings
    from statsmodels.tsa.stattools import adfuller, kpss
    rows = []
    for col in ("gt", "mc"):
        x = df[col].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")       # KPSS p-values are bounded
            kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
        rows.append({"series": col, "adf_stat": adf_stat, "adf_pval": adf_p,
                     "kpss_stat": kpss_stat, "kpss_pval": kpss_p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Season bootstrap and impulse responses
# ---------------------------------------------------------------------------
def resample_seasons(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw seasons with replacement; each draw gets a unique season label."""
    seasons = list(dict.fromkeys(df["season"]))
    pick = rng.choice(len(seasons), size=len(seasons), replace=True)
    return pd.concat([df[df["season"] == seasons[i]].assign(season=f"b{j}_{seasons[i]}")
                      for j, i in enumerate(pick)])


def impulse_response(df: pd.DataFrame, effect: str, cause: str, p1: int, p2: int,
                     horizon: int = 30, shock: float = 10.0, by_season: bool = True) -> np.ndarray:
    """Response of the effect to a one-off pulse of ``shock`` points in the cause at day 0.

    Dynamic multipliers of the fitted full model
        effect_t = c + sum_k a_k effect_{t-k} + sum_k b_k cause_{t-k},
    computed recursively: r_0 = 0, r_t = sum_k a_k r_{t-k} + b_t * shock (b_t = 0 for t > p2).
    Units are those of the series (0-100 within each season).
    """
    y, _, X = lag_design(df, effect, cause, p1, p2, by_season=by_season)
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    a, b = beta[1:1 + p1], beta[1 + p1:]
    r = np.zeros(horizon + 1)
    for t in range(1, horizon + 1):
        r[t] = sum(a[k - 1] * r[t - k] for k in range(1, min(t, p1) + 1))
        if t <= p2:
            r[t] += b[t - 1] * shock
    return r


def bootstrap_linear(df: pd.DataFrame, effect: str, cause: str, p1: int, p2: int,
                     n_boot: int = 999, seed: int = 0, horizon: int = 30, shock: float = 10.0,
                     by_season: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Season bootstrap (lags fixed) of the adjusted-R2 gain and of the impulse response.

    Returns (delta_r2adj of shape (n_boot,), impulse responses of shape (n_boot, horizon + 1)).
    """
    rng = np.random.default_rng(seed)
    d_r2, irfs = np.empty(n_boot), np.empty((n_boot, horizon + 1))
    for i in range(n_boot):
        b = resample_seasons(df, rng)
        d_r2[i] = granger_test(b, effect, cause, p1, p2, by_season=by_season)["delta_r2adj"]
        irfs[i] = impulse_response(b, effect, cause, p1, p2, horizon, shock, by_season)
    return d_r2, irfs
