"""Calendar components: each is checked against a case with a known answer."""
import numpy as np
import pandas as pd
import pytest

from phc import simulate as S
from phc import seasonal as T


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



def test_variance_shares_sum_to_one():
    df = S.common_driver(seed=1, weekday=0.5)
    sh = T.variance_shares(df)[["campaign profile", "weekday", "season level", "anomaly"]]
    assert np.allclose(sh.sum(axis=1), 1.0)
