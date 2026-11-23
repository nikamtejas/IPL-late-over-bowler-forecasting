"""Stage: baselines. The two baselines the model must beat, scored on validation folds only.

A (historical): the bowler's mean runs per late chase over from earlier matches
  (`bowler_late_mean`); the league mean from earlier matches (`league_late_mean`) when the
  bowler has fewer than history.min_history.bowler_late_overs such overs.
B (scoreboard): Poisson regression on the scoreboard only (config baselines.baseline_b.features),
  standardised inside a Pipeline so scaling is fitted on each fold's training rows only.

Both read the feature matrix made by features.build_features; neither adds a feature path.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, schemas, splits

log = logging.getLogger(__name__)


def baseline_a(X: pd.DataFrame) -> np.ndarray:
    min_overs = config.load_config()["history"]["min_history"]["bowler_late_overs"]
    enough = X["bowler_late_overs"] >= min_overs
    return np.where(enough, X["bowler_late_mean"], X["league_late_mean"]).astype(float)


def baseline_b() -> Pipeline:
    c = config.load_config()["baselines"]["baseline_b"]
    return Pipeline([
        ("scale", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=c["alpha"], max_iter=c["max_iter"])),
    ])


def baseline_b_features() -> list[str]:
    return list(config.load_config()["baselines"]["baseline_b"]["features"])


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))))


def validate(X: pd.DataFrame) -> dict:
    cols = baseline_b_features()
    rows = []
    for f, tr, va in splits.fold_rows(X):
        assert len(tr) and len(va), f"empty fold {f}"
        assert tr["season"].max() < va["season"].min(), "fold trains on the future"
        y = va["label_runs_in_over"].to_numpy(dtype=float)
        b = baseline_b().fit(tr[cols].astype(float), tr["label_runs_in_over"].astype(float))
        pa, pb = baseline_a(va), b.predict(va[cols].astype(float))
        rows.append({
            "fold": f.fold, "validation_season": f.validation_season,
            "train_seasons": [min(f.train_seasons), max(f.train_seasons)],
            "train_rows": len(tr), "validation_rows": len(va),
            "mae_baseline_a": mae(y, pa), "mae_baseline_b": mae(y, pb),
            "mean_label": float(y.mean()), "mean_pred_a": float(pa.mean()), "mean_pred_b": float(pb.mean()),
            "share_a_used_league_mean": float((va["bowler_late_overs"] < config.load_config()["history"]["min_history"]["bowler_late_overs"]).mean()),
        })
    df = pd.DataFrame(rows)
    w = df["validation_rows"] / df["validation_rows"].sum()
    summary = {
        "mean_fold_mae_baseline_a": float(df["mae_baseline_a"].mean()),
        "mean_fold_mae_baseline_b": float(df["mae_baseline_b"].mean()),
        "pooled_mae_baseline_a": float((df["mae_baseline_a"] * w).sum()),
        "pooled_mae_baseline_b": float((df["mae_baseline_b"] * w).sum()),
        "best_baseline": "A" if df["mae_baseline_a"].mean() < df["mae_baseline_b"].mean() else "B",
    }
    summary["mae_to_beat"] = min(summary["mean_fold_mae_baseline_a"], summary["mean_fold_mae_baseline_b"])
    return {
        "folds": rows,
        "summary": summary,
        "baseline_definitions": {
            "A": "bowler's mean runs per late chase over from earlier matches; league mean if the bowler "
                 f"has < {config.load_config()['history']['min_history']['bowler_late_overs']} such overs",
            "B": f"Poisson regression (standardised) on {', '.join(cols)}",
        },
    }


def main() -> None:
    X = schemas.read_table(config.path("features"), schemas.FEATURES)
    res = validate(X)
    schemas.write_json(res, schemas.VALIDATION_BASELINES_KEYS, config.path("validation_baselines"))
    for r in res["folds"]:
        log.info("fold %d (%d): MAE A %.3f  B %.3f", r["fold"], r["validation_season"], r["mae_baseline_a"], r["mae_baseline_b"])
    log.info("mean over folds: A %.3f  B %.3f", res["summary"]["mean_fold_mae_baseline_a"], res["summary"]["mean_fold_mae_baseline_b"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
