"""Stage: overs. Clean the parsed data and build the late-over table.

One row per STARTED late over (overs 15-20) in the second innings of a standard,
unrevised IPL chase. State columns are computed only from deliveries before the
over's first ball. Every exclusion is counted and written to overs_exclusions.json,
which the audit stage reports.

Rules (see also README):
- Wickets fallen count dismissals only. A "retired hurt" entry is not a dismissal
  (the batter may return); "retired out" is.
- balls_remaining follows the over schedule (6 per remaining over). It differs from
  120 - legal balls only in innings where an umpire miscounted an earlier over.
- The over's bowler is whoever bowled its first ball; the previous over's bowler is
  whoever bowled that over's last ball (the one barred from bowling this over).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from . import config, schemas

log = logging.getLogger(__name__)


def select_chases(matches: pd.DataFrame, deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply match- and innings-level exclusions. Returns kept matches, chase deliveries and counts."""
    cfg = config.load_config()
    n_overs, bpo = cfg["match"]["overs"], cfg["match"]["balls_per_over"]

    # Match-level rules, applied in order; each match is counted under the first rule it fails.
    rules = [
        ("no result", matches["result"].fillna("") == "no result"),
        ("fewer than two regular innings", matches["n_innings"] < 2),
        ("non-standard format (overs or balls per over)", (matches["overs"] != n_overs) | (matches["balls_per_over"] != bpo)),
        ("outcome method present (D/L)", matches["method"].notna()),
        (f"revised target: target overs < {n_overs}", matches["target_overs"].fillna(0) < n_overs),
    ]
    remaining = pd.Series(True, index=matches.index)
    match_rules = []
    for name, fails in rules:
        hit = remaining & fails.astype(bool)
        match_rules.append({"rule": name, "matches_removed": int(hit.sum()), "match_ids": sorted(matches.loc[hit, "match_id"])})
        remaining &= ~hit
    kept = matches[remaining].copy()

    d = deliveries[deliveries["match_id"].isin(kept["match_id"])]
    removed_matches_deliveries = len(deliveries) - len(d)
    innings_rules = [
        {"rule": "deliveries of excluded matches", "deliveries_removed": removed_matches_deliveries},
        {"rule": "super-over innings", "deliveries_removed": int(d["is_super_over"].sum())},
        {"rule": "first innings (not a chase)", "deliveries_removed": int(((d["innings"] == 1) & ~d["is_super_over"]).sum())},
    ]
    chase = d[(d["innings"] == 2) & ~d["is_super_over"]].sort_values(["match_id", "ball_seq"]).reset_index(drop=True)
    assert len(chase) + sum(r["deliveries_removed"] for r in innings_rules) == len(deliveries), "deliveries unaccounted for"

    # Every kept match must have a chase, and its target must equal first-innings total + 1.
    missing = set(kept["match_id"]) - set(chase["match_id"])
    assert not missing, f"kept matches without a second innings: {sorted(missing)[:5]}"
    first = d[(d["innings"] == 1) & ~d["is_super_over"]].groupby("match_id")["runs_total"].sum()
    bad = kept[kept["target_runs"].values != first.reindex(kept["match_id"]).values + 1]
    assert bad.empty, f"target != first-innings total + 1 for {list(bad['match_id'])[:5]}"
    assert (chase["batting_team"] != chase["bowling_team"]).all()

    return kept, chase, {"match_rules": match_rules, "innings_rules": innings_rules}


def check_chase_deliveries(chase: pd.DataFrame) -> dict[str, int]:
    """Data checks on the chase deliveries. Hard failures assert; tolerated anomalies are counted."""
    cfg = config.load_config()
    bpo = cfg["match"]["balls_per_over"]
    key = ["match_id", "ball_seq"]
    assert not chase.duplicated(key).any(), "duplicate deliveries"
    required = [c for c in chase.columns if c not in ("wicket_kind", "player_out", "player_out_id", "actual_delivery")]
    nulls = chase[required].isna().sum()
    assert nulls.sum() == 0, f"missing values: {nulls[nulls > 0].to_dict()}"
    assert (chase["n_wickets"].eq(0) == chase["wicket_kind"].isna()).all(), "wicket count/kind mismatch"

    per_over = chase.groupby(["match_id", "over"]).agg(legal=("is_legal", "sum"))
    last = per_over.reset_index().groupby("match_id")["over"].transform("max").values
    is_last = per_over.reset_index()["over"].values == last
    gaps = per_over.reset_index().groupby("match_id")["over"].agg(lambda s: int(s.max() + 1 - s.nunique()))
    assert (gaps == 0).all(), f"missing over numbers in chases: {gaps[gaps > 0].index[:5].tolist()}"
    legal = per_over["legal"].values
    return {
        "chase_deliveries": len(chase),
        "chase_overs": len(per_over),
        "incomplete_overs_mid_innings": int(((legal < bpo) & ~is_last).sum()),
        "overs_with_extra_legal_balls": int((legal > bpo).sum()),
        "final_overs_ended_early": int(((legal < bpo) & is_last).sum()),
    }


def build_late_overs(kept: pd.DataFrame, chase: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config.load_config()
    lo = cfg["late_overs"]
    n_overs, bpo = cfg["match"]["overs"], cfg["match"]["balls_per_over"]

    c = chase.copy()
    c["dismissal"] = (c["n_wickets"] > 0) & ~c["wicket_kind"].fillna("").isin(lo["non_dismissal_wicket_kinds"])
    g = c.groupby("match_id", sort=False)
    # State BEFORE each delivery = cumulative sum minus the delivery itself.
    c["runs_before"] = g["runs_total"].cumsum() - c["runs_total"]
    c["legal_before"] = g["is_legal"].cumsum().astype(int) - c["is_legal"].astype(int)
    c["dismissals_before"] = g["dismissal"].cumsum().astype(int) - c["dismissal"].astype(int)
    c["last_over_of_chase"] = g["over"].transform("max")

    ov = c.groupby(["match_id", "over"], sort=False)
    overs = ov.agg(
        runs_scored=("runs_before", "first"),
        legal_balls_bowled=("legal_before", "first"),
        wickets_fallen=("dismissals_before", "first"),
        striker_id=("batter_id", "first"),
        non_striker_id=("non_striker_id", "first"),
        bowler_id=("bowler_id", "first"),
        last_ball_bowler_id=("bowler_id", "last"),
        runs_in_over=("runs_total", "sum"),
        legal_balls_in_over=("is_legal", "sum"),
        deliveries_in_over=("ball_seq", "size"),
        n_bowlers_in_over=("bowler_id", "nunique"),
        last_over_of_chase=("last_over_of_chase", "first"),
        batting_team=("batting_team", "first"),
        bowling_team=("bowling_team", "first"),
    ).reset_index()

    # Previous over's last-ball bowler (needed for every over, so look up before filtering).
    prev = overs[["match_id", "over", "last_ball_bowler_id"]].assign(over=lambda x: x["over"] + 1)
    overs = overs.merge(prev.rename(columns={"last_ball_bowler_id": "prev_bowler_id"}), on=["match_id", "over"], how="left")

    late = overs[overs["over"].between(lo["first_over_index"], lo["last_over_index"])].copy()
    assert late["prev_bowler_id"].notna().all(), "a late over has no previous over"

    late = late.merge(kept[["match_id", "date", "season", "target_runs"]], on="match_id", validate="many_to_one")
    late["target"] = late["target_runs"]
    late["over_number"] = late["over"] + 1
    late["runs_needed"] = late["target"] - late["runs_scored"]
    late["balls_remaining"] = bpo * (n_overs - late["over"])
    late["wickets_in_hand"] = lo["wickets_per_innings"] - late["wickets_fallen"]
    late["required_rate"] = bpo * late["runs_needed"] / late["balls_remaining"]
    late["over_completed"] = (late["legal_balls_in_over"] >= bpo) | (late["over"] < late["last_over_of_chase"])
    late["chase_ended_in_over"] = late["over"] == late["last_over_of_chase"]

    loads = bowler_loads(c, late[["match_id", "over"]])
    own = loads.rename(columns={"overs_bowled": "bowler_overs_before", "legal_balls_bowled": "bowler_legal_balls_before"})
    late = late.merge(own[["match_id", "over", "bowler_id", "bowler_overs_before", "bowler_legal_balls_before"]],
                      on=["match_id", "over", "bowler_id"], how="left")
    late[["bowler_overs_before", "bowler_legal_balls_before"]] = late[["bowler_overs_before", "bowler_legal_balls_before"]].fillna(0)

    late = late.drop(columns=["target_runs", "last_ball_bowler_id", "last_over_of_chase"])
    late = late.sort_values(["date", "match_id", "over"]).reset_index(drop=True)
    return late, loads


def bowler_loads(c: pd.DataFrame, late_keys: pd.DataFrame) -> pd.DataFrame:
    """For each late over: overs and legal balls bowled earlier in the chase by each fielding bowler."""
    per = c.groupby(["match_id", "over", "bowler_id"], as_index=False).agg(legal=("is_legal", "sum"))
    last_ball = c.groupby(["match_id", "over"], as_index=False)["bowler_id"].last()
    x = late_keys.rename(columns={"over": "at_over"}).merge(per, on="match_id")
    x = x[x["over"] < x["at_over"]]
    loads = x.groupby(["match_id", "at_over", "bowler_id"], as_index=False).agg(
        overs_bowled=("over", "nunique"), legal_balls_bowled=("legal", "sum")
    )
    prev = last_ball.assign(at_over=last_ball["over"] + 1, bowled_previous_over=True)[["match_id", "at_over", "bowler_id", "bowled_previous_over"]]
    loads = loads.merge(prev, on=["match_id", "at_over", "bowler_id"], how="left")
    loads["bowled_previous_over"] = loads["bowled_previous_over"].astype("boolean").fillna(False)
    return loads.rename(columns={"at_over": "over"})


def check_late_overs(late: pd.DataFrame, loads: pd.DataFrame, chase: pd.DataFrame) -> dict[str, int]:
    """Hard checks on the finished table. Everything here must hold, or the stage fails."""
    cfg = config.load_config()
    lo, bpo, n_overs = cfg["late_overs"], cfg["match"]["balls_per_over"], cfg["match"]["overs"]
    assert late["runs_in_over"].notna().all(), "null labels"
    assert (late["runs_needed"] > 0).all(), "an over starts after the target was reached"
    assert late["balls_remaining"].between(bpo * (n_overs - lo["last_over_index"]), bpo * (n_overs - lo["first_over_index"])).all()
    assert (late["wickets_in_hand"] >= 1).all(), "an over starts with no wickets in hand"
    assert not late.duplicated(["match_id", "over"]).any(), "duplicate (match_id, over)"

    # Row counts and runs must reconcile with the deliveries they came from.
    src = chase[chase["over"].between(lo["first_over_index"], lo["last_over_index"])]
    n_src_overs = src[["match_id", "over"]].drop_duplicates().shape[0]
    assert len(late) == n_src_overs, f"{len(late)} rows but {n_src_overs} late overs in deliveries"
    assert late["runs_in_over"].sum() == src["runs_total"].sum(), "label total != delivery runs"
    assert late["deliveries_in_over"].sum() == len(src)
    # State must chain: next over's runs_scored = this over's runs_scored + runs_in_over.
    nxt = late.groupby("match_id")["runs_scored"].shift(-1)
    chained = nxt.isna() | (nxt == late["runs_scored"] + late["runs_in_over"])
    assert chained.all(), "runs_scored does not chain across overs"
    # The over's bowler cannot be the previous over's bowler (quota breaches are only counted).
    same = late["bowler_id"] == late["prev_bowler_id"]
    assert int(same.sum()) == 0, f"{int(same.sum())} overs bowled by the previous over's bowler"
    over_ends = late["chase_ended_in_over"] & ~late["over_completed"]
    return {
        "late_over_rows": len(late),
        "chases_with_late_overs": int(late["match_id"].nunique()),
        "bowler_load_rows": len(loads),
        "overs_not_completed_chase_ended": int(over_ends.sum()),
        "overs_with_more_than_one_bowler": int((late["n_bowlers_in_over"] > 1).sum()),
        "rows_where_120_minus_legal_differs_from_schedule": int(
            ((n_overs * bpo - late["legal_balls_bowled"]) != late["balls_remaining"]).sum()
        ),
        "bowler_over_quota_reached_before_over": int((late["bowler_overs_before"] >= cfg["match"]["max_overs_per_bowler"]).sum()),
    }


def main() -> None:
    matches = schemas.read_table(config.path("matches"), schemas.MATCHES)
    deliveries = schemas.read_table(config.path("deliveries"), schemas.DELIVERIES)

    kept, chase, excl = select_chases(matches, deliveries)
    delivery_checks = check_chase_deliveries(chase)
    late, loads = build_late_overs(kept, chase)
    row_checks = check_late_overs(late, loads, chase)

    no_late = sorted(set(kept["match_id"]) - set(late["match_id"]))
    excl["chase_rules"] = [
        {"rule": "chase ended before over 15 (adds no rows)", "chases": len(no_late)},
    ]
    excl["kept"] = {"matches": len(kept), "chases_with_late_overs": int(late["match_id"].nunique()), "late_over_rows": len(late)}
    excl["row_checks"] = {**delivery_checks, **row_checks}

    late = schemas.write_table(late, schemas.LATE_OVERS, config.path("late_overs"))
    schemas.write_table(loads, schemas.LATE_OVER_BOWLER_LOADS, config.path("late_over_bowler_loads"))
    schemas.write_json(excl, schemas.OVERS_EXCLUSIONS_KEYS, config.path("overs_exclusions"))
    log.info("kept %d matches; %d late-over rows from %d chases", len(kept), len(late), late["match_id"].nunique())
    log.info("checks: %s", excl["row_checks"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
