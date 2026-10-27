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
