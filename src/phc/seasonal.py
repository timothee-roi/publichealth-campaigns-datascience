"""Calendar components of the daily campaign series.

Both series follow the calendar: the average shape of the campaign (one value per
calendar date), the day of the week and, because each season is normalised by its own
maximum, a level that differs between seasons. ``anomalies`` removes them by an additive
least-squares fit; what remains is the day-to-day departure of a series from its usual
campaign shape. Notebook 04 repeats the surrogate tests after removing some or all of
these components, and notebook 06 shows what each carries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def variance_shares(df: pd.DataFrame, cols=("gt", "mc")) -> pd.DataFrame:
    """Share of the variance of each series carried by the campaign profile, the day of
    the week and the season level (sequential sums of squares, in that order)."""
    rows = {}
    for col in cols:
        tot = df[col].var(ddof=0)
        v0 = anomalies(df, weekday=False, season=False)[col].var(ddof=0)
        v1 = anomalies(df, weekday=True, season=False)[col].var(ddof=0)
        v2 = anomalies(df, weekday=True, season=True)[col].var(ddof=0)
        rows[col.upper()] = {"campaign profile": 1 - v0 / tot, "weekday": (v0 - v1) / tot,
                             "season level": (v1 - v2) / tot, "anomaly": v2 / tot,
                             "anomaly SD (points)": np.sqrt(v2)}
    return pd.DataFrame(rows).T
