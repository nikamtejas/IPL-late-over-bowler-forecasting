"""Splits, the test-set lock and the baselines, on the fixture matches.

The fixtures cover seasons 2008, 2014, 2017, 2019, 2022 and 2023, so a temporary config
holds 2023 out as the test season and validates on 2019 and 2022.
"""

import numpy as np
import pytest
import yaml

from ipl_late_overs import baselines, config, evaluate, features, history, overs, schemas, splits


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = yaml.safe_load(config.config_path().read_text())
    c["splits"] = {"train_seasons": list(range(2008, 2019)), "validation_seasons": [2019, 2022], "test_seasons": [2023]}
    c["paths"]["splits"] = str(tmp_path / "splits.json")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(c))
    monkeypatch.setenv(config.CONFIG_ENV_VAR, str(p))
    return c


@pytest.fixture
def data(fixture_tables, cfg):
    t = fixture_tables
    kept, chase, _ = overs.select_chases(t["matches"], t["deliveries"])
    late, loads = overs.build_late_overs(kept, chase)
    hist = history.build_history(t["matches"], t["deliveries"], late)
    hist["late_over_bowler_loads"] = loads
    X = features.build_features(late, hist)
    sp = splits.build_splits(t["matches"], late)
    schemas.write_json(sp, schemas.SPLITS_KEYS, config.path("splits"))
    return t["matches"], late, X, sp


def test_splits_are_disjoint_whole_match_and_chronological(data):
    matches, late, _, sp = data
    tr, va, te = (set(sp[k]) for k in ("train_match_ids", "validation_match_ids", "test_match_ids"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(matches["match_id"])
    d = matches.set_index("match_id")["date"]
    assert d[list(te)].min() > d[list(tr | va)].max()
    # every over of a match is in its match's split
    assert sum(v["late_over_rows"] for v in sp["counts"].values()) == len(late)


def test_unassigned_season_fails_loudly(fixture_tables, cfg, tmp_path, monkeypatch):
    cfg["splits"]["train_seasons"] = [2008, 2009]  # 2014 and 2017 are now in no split
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv(config.CONFIG_ENV_VAR, str(p))
    with pytest.raises(AssertionError, match="in no split"):
        splits.build_splits(fixture_tables["matches"], fixture_tables["deliveries"].head(0))


def test_development_rows_and_folds_never_contain_test(data):
    _, _, X, sp = data
    test_ids = set(sp["test_match_ids"])
    assert test_ids and not splits.development_rows(X)["match_id"].isin(test_ids).any()
    seen_train = []
    for f, tr, va in splits.fold_rows(X):
        assert not tr["match_id"].isin(test_ids).any() and not va["match_id"].isin(test_ids).any()
        assert tr["season"].max() < va["season"].min()
        assert set(va["season"]) == {f.validation_season}
        seen_train.append(len(tr))
    assert seen_train == sorted(seen_train)  # expanding window


def test_test_set_is_locked(data):
    _, _, X, _ = data
    with pytest.raises(splits.TestSetLocked):
        splits.test_rows(X, key=None)
    with pytest.raises(splits.TestSetLocked):
        splits.FinalEvaluationKey()


def test_evaluate_refuses_without_final_and_does_not_open_test_this_week(data):
    with pytest.raises(SystemExit, match="--final"):
        evaluate.main([])
    with pytest.raises(SystemExit, match="week 6"):
        evaluate.main(["--final"])
    with pytest.raises(splits.TestSetLocked):  # the --final stub must not have left a key armed
        splits.FinalEvaluationKey()


def test_final_key_opens_test_once(data):
    _, _, X, sp = data
    key = evaluate._open_test_set()
    assert set(splits.test_rows(X, key)["match_id"]) <= set(sp["test_match_ids"])
    with pytest.raises(splits.TestSetLocked):  # one key per arming
        splits.FinalEvaluationKey()


def test_baseline_a_falls_back_to_league_mean(data):
    _, _, X, _ = data
    min_overs = config.load_config()["history"]["min_history"]["bowler_late_overs"]
    pred = baselines.baseline_a(X)
    low = X["bowler_late_overs"] < min_overs
    assert low.any()
    assert np.allclose(pred[low.to_numpy()], X.loc[low, "league_late_mean"])
    assert np.allclose(pred[~low.to_numpy()], X.loc[~low, "bowler_late_mean"])


def test_baseline_b_scaler_fitted_on_fold_train_only(data):
    _, _, X, _ = data
    cols = baselines.baseline_b_features()
    f, tr, va = next(iter(splits.fold_rows(X)))
    model = baselines.baseline_b().fit(tr[cols].astype(float), tr["label_runs_in_over"].astype(float))
    assert np.allclose(model.named_steps["scale"].mean_, tr[cols].astype(float).mean().to_numpy())


def test_validate_reports_each_fold(data):
    _, _, X, _ = data
    res = baselines.validate(X)
    assert [r["validation_season"] for r in res["folds"]] == [2019, 2022]
    r = res["folds"][0]
    va = X[X["season"] == 2019]
    assert r["mae_baseline_a"] == pytest.approx(np.abs(va["label_runs_in_over"] - baselines.baseline_a(va)).mean())
    assert res["summary"]["mae_to_beat"] == min(res["summary"]["mean_fold_mae_baseline_a"], res["summary"]["mean_fold_mae_baseline_b"])
