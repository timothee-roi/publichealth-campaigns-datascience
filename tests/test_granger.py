"""Linear Granger causality: algebra and regression tests on the data in the repository."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from phc import granger as G
from phc.config import TAB_DIR


@pytest.fixture(scope="module")
def movember():
    return G.load_daily("Mov")


def test_daily_data_layout():
    for code, days in {"OR": {91}, "Mov": {92}, "MB": {88, 89}, "DJ": {43}}.items():
        df = G.load_daily(code)
        sizes = df.groupby("season").size()
        assert len(sizes) == 11
        assert set(sizes) == days
        assert not df[["gt", "mc"]].isna().any().any()
        # each series is normalised to 100 within each season
        assert np.allclose(df.groupby("season")[["gt", "mc"]].max().to_numpy(), 100)


def test_lags_stay_within_seasons(movember):
    y, X_red, X_full = G.lag_design(movember, "mc", "gt", 2, 3, by_season=True)
    assert len(y) == 11 * (92 - 3)
    first = movember[movember["season"] == movember["season"].iloc[0]]
    assert y[0] == first["mc"].iloc[3]
    assert np.allclose(X_full[0], [first["mc"].iloc[2], first["mc"].iloc[1],
                                   first["gt"].iloc[2], first["gt"].iloc[1], first["gt"].iloc[0]])


def test_f_test_matches_statsmodels(movember):
    res = G.granger_test(movember, "gt", "mc", 8, 2, by_season=True)
    y, X_red, X_full = G.lag_design(movember, "gt", "mc", 8, 2, by_season=True)
    full = sm.OLS(y, sm.add_constant(X_full)).fit()
    red = sm.OLS(y, sm.add_constant(X_red)).fit()
    F, p, _ = full.compare_f_test(red)
    assert res["F"] == pytest.approx(F)
    assert res["pval"] == pytest.approx(p)
    assert res["r2adj_full"] == pytest.approx(full.rsquared_adj)
    assert G.gaussian_aic(full.ssr, len(y), 11) == pytest.approx(full.aic)


def test_table2_is_reproduced():
    """The lags and statistics of the paper's Table 2 follow from the data."""
    table = pd.read_csv(TAB_DIR / "table2_granger_full_sample.csv")
    row = table[(table["campaign"] == "Movember") & (table["direction"] == "MC → GT")].iloc[0]
    df = G.load_daily("Mov")
    grid = G.aic_grid(df, "gt", "mc", max_lag=20, common_sample=True, by_season=True)
    p1, p2 = G.best_lags(grid)
    assert f"({p1}, {p2})" == row["(p1*, p2*)"]
    res = G.granger_test(df, "gt", "mc", p1, p2, by_season=True)
    assert res["n"] == row["n"]
    assert res["F"] == pytest.approx(row["F"], rel=1e-6)
    assert res["delta_r2adj"] == pytest.approx(row["delta_r2adj"], rel=1e-6)


def test_impulse_response_of_known_model():
    # y_t = 0.5 y_{t-1} + 0.3 x_{t-1}: response to a pulse of 10 is 3 * 0.5^(t-1)
    rng = np.random.default_rng(0)
    rows = []
    for s in range(11):
        x = rng.standard_normal(200)
        y = np.zeros(200)
        for t in range(1, 200):
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 1] + 0.01 * rng.standard_normal()
        rows.append(pd.DataFrame({"gt": x, "mc": y, "season": str(s)}))
    df = pd.concat(rows, ignore_index=True)
    r = G.impulse_response(df, "mc", "gt", 1, 1, horizon=5)
    assert np.allclose(r, [0, 3, 1.5, 0.75, 0.375, 0.1875], atol=0.02)


def test_season_fixed_effects_remove_a_level_confound():
    """Seasons differ in level and the two series share these levels, with no day-to-day
    link. The plain test finds a link, the fixed-effects test does not."""
    import numpy as np
    import pandas as pd
    from phc.granger import granger_test, granger_test_fe

    rng = np.random.default_rng(3)
    S, L = 11, 80
    rows = []
    for s in range(S):
        level = rng.normal(0, 3)
        x = level + np.zeros(L)
        y = level + np.zeros(L)
        ex, ey = np.zeros(L), np.zeros(L)
        for t in range(1, L):
            ex[t] = 0.5 * ex[t - 1] + rng.standard_normal()
            ey[t] = 0.5 * ey[t - 1] + rng.standard_normal()
        rows.append(pd.DataFrame({"gt": x + ex, "mc": y + ey, "season": s},
                                 index=pd.date_range(f"{2010 + s}-01-01", periods=L)))
    df = pd.concat(rows)
    plain = granger_test(df, "mc", "gt", 2, 2, by_season=True)
    fe = granger_test_fe(df, "mc", "gt", 2, 2)
    assert plain["pval"] < 1e-3
    assert fe["pval"] > 0.01
    assert fe["n"] == plain["n"]
