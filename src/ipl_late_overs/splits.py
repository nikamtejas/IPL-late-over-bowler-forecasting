"""Stage: splits. Chronological, whole-match splits by season, and the only way to get rows by split.

- train: config splits.train_seasons
- validation folds (expanding window): fold k trains on train seasons + validation seasons
  before k, and validates on validation season k
- test: config splits.test_seasons. Locked: `test_rows` needs a FinalEvaluationKey, which only
  `evaluate.py --final` creates. Every other loader here drops test matches. This guards against
  accidental use of the test set; it is not meant to stop someone determined to bypass it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from . import config, schemas

log = logging.getLogger(__name__)


class TestSetLocked(PermissionError):
    pass


class FinalEvaluationKey:
    """Proof that the caller is the final evaluation. Create only via `evaluate.py --final`."""

    _armed = False

    def __init__(self) -> None:
        if not FinalEvaluationKey._armed:
            raise TestSetLocked("the test set can only be opened by `evaluate.py --final`")
        FinalEvaluationKey._armed = False  # one key per arming


@dataclass(frozen=True)
class Fold:
    fold: int
    train_seasons: tuple[int, ...]
    validation_season: int


def folds() -> list[Fold]:
    s = config.load_config()["splits"]
    out, train = [], list(s["train_seasons"])
    for i, season in enumerate(s["validation_seasons"], start=1):
        out.append(Fold(i, tuple(train), season))
        train.append(season)
    return out


def build_splits(matches: pd.DataFrame, late_overs: pd.DataFrame) -> dict:
    s = config.load_config()["splits"]
    groups = {
        "train": matches[matches["season"].isin(s["train_seasons"])],
        "validation": matches[matches["season"].isin(s["validation_seasons"])],
        "test": matches[matches["season"].isin(s["test_seasons"])],
    }
    ids = {k: set(g["match_id"]) for k, g in groups.items()}
    # Whole matches, no overlap, every match assigned.
    assert not (ids["train"] & ids["validation"]) and not (ids["train"] & ids["test"]) and not (ids["validation"] & ids["test"]), \
        "a match_id is in more than one split"
    unassigned = set(matches["match_id"]) - set().union(*ids.values())
    assert not unassigned, f"{len(unassigned)} matches in no split (seasons {sorted(matches.loc[matches['match_id'].isin(unassigned), 'season'].unique())})"
    # Chronology: every test match after every train/validation match; validation after train.
    assert groups["test"]["date"].min() > pd.concat([groups["train"], groups["validation"]])["date"].max(), \
        "a test match is not later than every train/validation match"
    assert groups["validation"]["date"].min() > groups["train"]["date"].max(), "validation overlaps train in time"
    # Late-over rows inherit their match's split.
    rows_by = late_overs.assign(split=late_overs["match_id"].map({m: k for k, v in ids.items() for m in v}))
    assert rows_by["split"].notna().all(), "late-over rows from unassigned matches"

    counts = {
        k: {"seasons": sorted(int(x) for x in g["season"].unique()), "matches": len(g),
            "late_over_rows": int((rows_by["split"] == k).sum()),
            "first_date": str(g["date"].min().date()), "last_date": str(g["date"].max().date())}
        for k, g in groups.items()
    }
    fold_list = []
    for f in folds():
        tr = late_overs["season"].isin(f.train_seasons)
        va = late_overs["season"] == f.validation_season
        fold_list.append({"fold": f.fold, "train_seasons": list(f.train_seasons), "validation_season": f.validation_season,
                          "train_rows": int(tr.sum()), "validation_rows": int(va.sum())})
    return {
        "seasons": {"train": s["train_seasons"], "validation": s["validation_seasons"], "test": s["test_seasons"]},
        "train_match_ids": sorted(ids["train"]),
        "validation_match_ids": sorted(ids["validation"]),
        "test_match_ids": sorted(ids["test"]),
        "folds": fold_list,
        "counts": counts,
    }


def load_splits() -> dict:
    return schemas.read_json(config.path("splits"), schemas.SPLITS_KEYS)


def _test_ids() -> set[str]:
    return set(load_splits()["test_match_ids"])


def development_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Train + validation rows of any table with a match_id column. Test matches are dropped."""
    out = df[~df["match_id"].isin(_test_ids())]
    sp = load_splits()
    dev = set(sp["train_match_ids"]) | set(sp["validation_match_ids"])
    assert out["match_id"].isin(dev).all(), "rows from matches outside the splits"
    return out


def fold_rows(df: pd.DataFrame) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """(fold, train rows, validation rows) for each expanding validation fold. Never test rows."""
    dev = development_rows(df)
    for f in folds():
        yield f, dev[dev["season"].isin(f.train_seasons)], dev[dev["season"] == f.validation_season]


def test_rows(df: pd.DataFrame, key: FinalEvaluationKey) -> pd.DataFrame:
    if not isinstance(key, FinalEvaluationKey):
        raise TestSetLocked("the test set can only be opened by `evaluate.py --final`")
    return df[df["match_id"].isin(_test_ids())]


def write_report(sp: dict) -> None:
    from .audit import md_table

    c = sp["counts"]
    lines = ["# Splits\n", "_Generated by `make splits`. Split by whole match and season; see config.yaml._\n",
             md_table(pd.DataFrame([{"split": k, "seasons": f"{v['seasons'][0]}-{v['seasons'][-1]}", "matches": v["matches"],
                                     "late-over rows": v["late_over_rows"], "dates": f"{v['first_date']} to {v['last_date']}"}
                                    for k, v in ((k, c[k]) for k in ("train", "validation", "test"))])),
             "\n## Validation folds (expanding window)\n",
             md_table(pd.DataFrame([{"fold": f["fold"], "train on": f"{f['train_seasons'][0]}-{f['train_seasons'][-1]}",
                                     "validate on": f["validation_season"], "train rows": f["train_rows"],
                                     "validation rows": f["validation_rows"]} for f in sp["folds"]])),
             "\n## Season timeline\n", "```"]
    role = {s: "train" for s in sp["seasons"]["train"]} | {s: "VAL" for s in sp["seasons"]["validation"]} | \
           {s: "TEST" for s in sp["seasons"]["test"]}
    lines.append("season  " + " ".join(f"{s % 100:02d}" for s in sorted(role)))
    lines.append("split   " + " ".join({"train": "tr", "VAL": "VA", "TEST": "TE"}[role[s]] for s in sorted(role)))
    for f in sp["folds"]:
        lines.append(f"fold {f['fold']}  " + " ".join(
            "##" if s in f["train_seasons"] else ("VV" if s == f["validation_season"] else "..") for s in sorted(role)))
    lines += ["```", "", "`##` = fit on, `VV` = validate on, `..` = unused in that fold. The test seasons (TE) are used "
              "once, in week 6, through `evaluate.py --final`.", ""]
    config.path("splits_report").write_text("\n".join(lines))


def main() -> None:
    matches = schemas.read_table(config.path("matches"), schemas.MATCHES)
    late = schemas.read_table(config.path("late_overs"), schemas.LATE_OVERS)
    sp = build_splits(matches, late)
    schemas.write_json(sp, schemas.SPLITS_KEYS, config.path("splits"))
    write_report(sp)
    log.info("splits: %s", {k: (v["matches"], v["late_over_rows"]) for k, v in ((k, sp["counts"][k]) for k in ("train", "validation", "test"))})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
