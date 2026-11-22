"""History and feature tests on the fixture matches (2008-2023 finals and openers)."""

import numpy as np
import pandas as pd
import pytest

from ipl_late_overs import config, features, history, overs, schemas

from .conftest import load_fixture


@pytest.fixture(scope="module")
def built(fixture_tables):
    t = fixture_tables
    kept, chase, _ = overs.select_chases(t["matches"], t["deliveries"])
    late, loads = overs.build_late_overs(kept, chase)
    hist = history.build_history(t["matches"], t["deliveries"], late)
    hist["late_over_bowler_loads"] = loads
    X = features.build_features(late, hist)
    return late, hist, X


def ids(match_id):
    return load_fixture(match_id)["info"]["registry"]["people"]


def row(df, match_id, over):
    r = df[(df["match_id"] == match_id) & (df["over"] == over)]
    assert len(r) == 1
    return r.iloc[0]


def test_matrix_satisfies_schema_and_has_no_nan(built):
    late, _, X = built
    schemas.FEATURES.validate(schemas.FEATURES.coerce(X))
    assert len(X) == len(late)
    assert np.isfinite(X[features.feature_columns()].astype(float).to_numpy()).all()


def test_bowler_history_by_hand(built):
    """Malinga before the 2019 final: among the fixtures, his only earlier late chase overs are
    overs 15 and 18 of the 2017 final (6 and 7 runs, read from the raw JSON)."""
    x = row(built[2], "1181768", 19)
    assert x["bowler_id"] == ids("1181768")["SL Malinga"]
    assert x["bowler_late_overs"] == 2
    assert x["bowler_late_mean"] == pytest.approx(6.5)


def test_current_match_is_excluded(built):
    """Johnson bowled 3 overs earlier in the 2017 final; none of them may appear in his history.
    His only earlier late chase overs are overs 15 and 19 of the 2014 final (9 and 10 runs)."""
    x = row(built[2], "1082650", 19)
    assert x["bowler_late_overs"] == 2
    assert x["bowler_late_mean"] == pytest.approx(9.5)
    assert x["bowler_overs_before"] == 3  # in-match load comes from the loads table instead


def test_shrinkage_formula(built):
    cfg = config.load_config()["history"]
    x = row(built[2], "1181768", 19)
    k = cfg["shrinkage_k"]["bowler_late_overs"]
    expected = (2 * 6.5 + k * x["league_late_mean"]) / (2 + k)
    assert x["bowler_late_mean_shrunk"] == pytest.approx(expected)


def test_first_match_uses_league_prior(built):
    """The 2008 final is the earliest fixture: no earlier league data, so priors apply."""
    cfg = config.load_config()["history"]
    x = row(built[2], "336040", 19)
    assert x["league_late_mean"] == cfg["league_prior"]["late_over_runs"]
    assert x["bowler_late_overs"] == 0 and x["bowler_late_mean"] == x["league_late_mean"]
    assert x["bowler_low_history"] and x["is_low_history"]


def test_match_order_breaks_same_date_ties_by_match_id():
    m = pd.DataFrame({"match_id": ["20", "3", "100"], "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-01"])})
    o = history.match_order(m).set_index("match_id")["match_order"]
    assert o["100"] < o["3"] < o["20"]  # numeric, not string, order on the same date


def test_asof_is_strictly_earlier():
    hist = pd.DataFrame({"player_id": ["a", "a"], "match_order": [1, 3], "x": [5, 9]})
    keys = pd.DataFrame({"bowler_id": ["a", "a", "a", "b"], "match_order": [1, 3, 4, 4]})
    got = features.asof(keys, hist, "bowler_id", ("x",))["x"].tolist()
    assert got == [0, 5, 9, 0]


def test_future_matches_do_not_change_history(fixture_tables, built):
    """Dropping every match after the 2019 final must not change any feature of that final."""
    t = fixture_tables
    order = history.match_order(t["matches"]).set_index("match_id")["match_order"]
    keep = order[order <= order["1181768"]].index
    sub = {k: v[v["match_id"].isin(keep)] for k, v in t.items() if "match_id" in v.columns}
    kept, chase, _ = overs.select_chases(sub["matches"], sub["deliveries"])
    late, loads = overs.build_late_overs(kept, chase)
    hist = history.build_history(sub["matches"], sub["deliveries"], late)
    hist["late_over_bowler_loads"] = loads
    X_sub = features.build_features(late[late["match_id"] == "1181768"], hist).reset_index(drop=True)
    X_all = built[2][built[2]["match_id"] == "1181768"].reset_index(drop=True)
    pd.testing.assert_frame_equal(X_sub[features.feature_columns()], X_all[features.feature_columns()])


def test_bowler_override_matches_actual_bowler_features(built):
    """Scoring Bumrah for the 2019 final's last over must give his real history and load
    (4 overs, including the 19th), with no label."""
    late, hist, X = built
    r = late[(late["match_id"] == "1181768") & (late["over"] == 19)]
    bumrah = ids("1181768")["JJ Bumrah"]
    xo = features.build_features(r, hist, bowler_override=bumrah).iloc[0]
    assert "label_runs_in_over" not in xo.index
    assert xo["bowler_id"] == bumrah and xo["bowler_overs_before"] == 4
    actual18 = row(X, "1181768", 18)  # Bumrah really bowled over 19 of the same match
    for col in ("bowler_late_overs", "bowler_late_mean", "bowler_late_mean_shrunk", "bowler_overs_all", "bowler_economy_shrunk"):
        assert xo[col] == pytest.approx(actual18[col]), col
    # Non-bowler features are unchanged by the override.
    x = row(X, "1181768", 19)
    for col in ("runs_needed", "striker_sr_shrunk", "league_late_mean"):
        assert xo[col] == x[col]


def test_override_with_unseen_bowler_gets_league_values(built):
    late, hist, _ = built
    r = late[(late["match_id"] == "1181768") & (late["over"] == 19)]
    xo = features.build_features(r, hist, bowler_override="not-a-real-id").iloc[0]
    assert xo["bowler_late_overs"] == 0 and xo["bowler_overs_before"] == 0
    assert xo["bowler_late_mean"] == xo["league_late_mean"] == xo["bowler_late_mean_shrunk"]
    assert xo["bowler_low_history"]


def test_override_per_row(built):
    late, hist, X = built
    alt = features.build_features(late, hist, bowler_override=late["bowler_id"].to_numpy())
    pd.testing.assert_frame_equal(alt[features.feature_columns()], X[features.feature_columns()])


@pytest.mark.parametrize("enabled", [False, True])
def test_id_encoding_on_and_off(built, enabled):
    late, hist, X = built
    enc = None
    if enabled:
        train = late[late["season"] <= 2017]
        enc = features.IdEncoder(["bowler_id", "striker_id"], min_rows=1).fit(train)
    out = features.build_features(late, hist, id_encoder=enc)
    base = features.feature_columns()
    pd.testing.assert_frame_equal(out[base], X[base])
    if not enabled:
        assert list(out.columns) == list(X.columns)
        return
    assert set(enc.columns()) == set(out.columns) - set(X.columns)
    for role in enc.roles:
        cols = [c for c in out.columns if c.startswith(f"{role}=")]
        assert (out[cols].sum(axis=1) == 1).all()
    # A bowler first seen after 2017 is "unknown": the encoder never saw later rows.
    later = late[(late["season"] > 2017) & ~late["bowler_id"].isin(enc.vocab["bowler_id"])]
    assert len(later) and (out.loc[later.index, "bowler_id=unknown"] == 1).all()
