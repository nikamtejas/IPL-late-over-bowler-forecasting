"""Stage: features. Build history tables and the model matrix.

`build_features` is the ONLY way to turn late-over rows into model inputs. Training,
evaluation and the dashboard all call it; `bowler_override` scores a different bowler
for the same situation (week 7).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from . import config, history, schemas

log = logging.getLogger(__name__)

HISTORY_TABLES = {
    "history_bowler": schemas.BOWLER_HISTORY,
    "history_batter": schemas.BATTER_HISTORY,
    "history_league": schemas.LEAGUE_HISTORY,
}

# name -> (meaning, source). Every feature column must be documented here (asserted).
FEATURE_DOCS: dict[str, tuple[str, str]] = {
    "over_number": ("Over being bowled, 15-20", "late_overs"),
    "target": ("Runs the chasing side needs to win", "late_overs (matches.target_runs)"),
    "runs_scored": ("Chase score before the over", "late_overs"),
    "runs_needed": ("target - runs_scored", "late_overs"),
    "balls_remaining": ("6 x overs left, including this one", "late_overs"),
    "wickets_in_hand": ("10 - dismissals so far (retired hurt not counted)", "late_overs"),
    "required_rate": ("Runs per over needed: 6 x runs_needed / balls_remaining", "late_overs"),
    "bowler_overs_before": ("Overs this bowler has already bowled in this chase", "late_over_bowler_loads"),
    "bowler_late_overs": ("Bowler's earlier late overs (15-20) in IPL chases", "history_bowler"),
    "bowler_late_mean": ("Bowler's mean runs per late chase over (league mean if none)", "history_bowler"),
    "bowler_late_mean_shrunk": ("bowler_late_mean shrunk toward the league mean (k in config)", "history_bowler + history_league"),
    "bowler_overs_all": ("Bowler's earlier overs in all phases (legal balls / 6)", "history_bowler"),
    "bowler_economy_shrunk": ("Bowler's runs conceded per over, all phases, shrunk", "history_bowler + history_league"),
    "bowler_inn1_late_economy_shrunk": ("Bowler's economy in overs 15-20 of first innings, shrunk", "history_bowler + history_league"),
    "striker_balls": ("Striker's earlier balls faced (wides excluded)", "history_batter"),
    "striker_sr_shrunk": ("Striker's strike rate, all phases, shrunk", "history_batter + history_league"),
    "striker_late_sr_shrunk": ("Striker's strike rate in overs 15-20, shrunk", "history_batter + history_league"),
    "striker_dismissal_rate_shrunk": ("Striker's dismissals per ball faced, shrunk", "history_batter + history_league"),
    "non_striker_balls": ("Non-striker's earlier balls faced", "history_batter"),
    "non_striker_sr_shrunk": ("Non-striker's strike rate, shrunk", "history_batter + history_league"),
    "non_striker_late_sr_shrunk": ("Non-striker's strike rate in overs 15-20, shrunk", "history_batter + history_league"),
    "non_striker_dismissal_rate_shrunk": ("Non-striker's dismissals per ball, shrunk", "history_batter + history_league"),
    "league_late_mean": ("League mean runs per late chase over, earlier matches", "history_league"),
    "bowler_low_history": ("bowler_late_overs < min_history.bowler_late_overs", "derived"),
    "striker_low_history": ("striker_balls < min_history.batter_balls", "derived"),
    "is_low_history": ("bowler_low_history or striker_low_history", "derived"),
}


def feature_columns() -> list[str]:
    return [c.name for c in schemas.FEATURES.columns if c.name not in schemas.FEATURE_ID_COLUMNS]


assert set(FEATURE_DOCS) == set(feature_columns()), "FEATURE_DOCS and schemas.FEATURES disagree"


def load_history_tables() -> dict[str, pd.DataFrame]:
    tables = {name: schemas.read_table(config.path(name), schema) for name, schema in HISTORY_TABLES.items()}
    tables["late_over_bowler_loads"] = schemas.read_table(
        config.path("late_over_bowler_loads"), schemas.LATE_OVER_BOWLER_LOADS
    )
    return tables


def asof(keys: pd.DataFrame, hist: pd.DataFrame, player_col: str, sums: tuple[str, ...]) -> pd.DataFrame:
    """For each (player, match_order) in keys: the player's totals from strictly earlier matches.

    Returns one row per input row, in input order; players with no earlier match get 0.
    """
    left = keys[[player_col, "match_order"]].rename(columns={player_col: "player_id"})
    left = left.assign(_row=np.arange(len(left))).sort_values("match_order")
    right = hist[["player_id", "match_order", *sums]].sort_values("match_order")
    out = pd.merge_asof(
        left.astype({"match_order": "int64"}), right.astype({"match_order": "int64"}),
        on="match_order", by="player_id", allow_exact_matches=False, direction="backward",
    )
    return out.sort_values("_row")[list(sums)].fillna(0).reset_index(drop=True).astype(float)


def league_asof(match_orders: pd.Series, league: pd.DataFrame) -> pd.DataFrame:
    """League totals from strictly earlier matches, one row per input match_order."""
    sums = history.BOWLER_SUMS + history.BATTER_SUMS
    left = pd.DataFrame({"match_order": match_orders.astype("int64").to_numpy(), "_row": np.arange(len(match_orders))})
    right = league[["match_order", *sums]].astype({"match_order": "int64"})
    out = pd.merge_asof(left.sort_values("match_order"), right, on="match_order",
                        allow_exact_matches=False, direction="backward")
    return out.sort_values("_row")[list(sums)].fillna(0).reset_index(drop=True).astype(float)


def shrink(total: pd.Series, n: pd.Series, league_rate: pd.Series, k: float) -> pd.Series:
    """(n * mean + k * league_rate) / (n + k), written with totals so n = 0 is safe."""
    return (total + k * league_rate) / (n + k)


def _rate(num: pd.Series, den: pd.Series, prior: float) -> pd.Series:
    return pd.Series(np.where(den > 0, num / den.where(den > 0, 1), prior), index=num.index)


def build_features(
    over_rows: pd.DataFrame,
    history_tables: dict[str, pd.DataFrame],
    bowler_override: Any = None,
    id_encoder: "IdEncoder | None" = None,
) -> pd.DataFrame:
    """Model matrix for late-over rows.

    over_rows: rows of the late_overs table (any subset).
    bowler_override: None (actual bowler), one bowler_id for every row, or a sequence aligned
        with over_rows. Bowler features are then those of the override bowler. The label is
        only returned without an override: an alternative bowler's runs are never observed.
    id_encoder: optional fitted IdEncoder; appends one-hot ID columns.
    """
    cfg = config.load_config()
    k, mins, prior = cfg["history"]["shrinkage_k"], cfg["history"]["min_history"], cfg["history"]["league_prior"]
    rows = over_rows.reset_index(drop=True).copy()

    order = history_tables["history_league"][["match_id", "match_order"]]
    rows = rows.merge(order, on="match_id", how="left", validate="many_to_one")
    assert rows["match_order"].notna().all(), "over rows from matches missing in history_league"

    if bowler_override is not None:
        override = pd.Series(bowler_override if not isinstance(bowler_override, str) else [bowler_override] * len(rows))
        assert len(override) == len(rows), "bowler_override must be one id or one per row"
        rows["bowler_id"] = pd.array(override.to_numpy(), dtype="string")
    assert rows["bowler_id"].notna().all()

    # In-match load of THIS bowler before the over (0 if they haven't bowled yet).
    loads = history_tables["late_over_bowler_loads"][["match_id", "over", "bowler_id", "overs_bowled"]]
    rows = rows.drop(columns=["bowler_overs_before"], errors="ignore").merge(
        loads.rename(columns={"overs_bowled": "bowler_overs_before"}), on=["match_id", "over", "bowler_id"], how="left"
    )
    rows["bowler_overs_before"] = rows["bowler_overs_before"].fillna(0).astype(int)

    lg = league_asof(rows["match_order"], history_tables["history_league"])
    league_late = _rate(lg["late_chase_runs"], lg["late_chase_overs"], prior["late_over_runs"])
    league_econ = _rate(lg["runs_conceded"], lg["legal_balls"] / 6, prior["economy"])
    league_inn1 = _rate(lg["inn1_late_runs_conceded"], lg["inn1_late_legal_balls"] / 6, prior["inn1_late_economy"])
    league_sr = _rate(lg["runs"], lg["balls_faced"], prior["strike_rate"] / 100)
    league_late_sr = _rate(lg["late_runs"], lg["late_balls_faced"], prior["late_strike_rate"] / 100)
    league_out = _rate(lg["dismissals"], lg["balls_faced"], prior["dismissals_per_ball"])

    b = asof(rows, history_tables["history_bowler"], "bowler_id", history.BOWLER_SUMS)
    X = pd.DataFrame(index=rows.index)
    for col in ("over_number", "target", "runs_scored", "runs_needed", "balls_remaining",
                "wickets_in_hand", "required_rate", "bowler_overs_before"):
        X[col] = rows[col]
    X["bowler_late_overs"] = b["late_chase_overs"].astype(int)
    X["bowler_late_mean"] = _rate(b["late_chase_runs"], b["late_chase_overs"], 0).where(b["late_chase_overs"] > 0, league_late)
    X["bowler_late_mean_shrunk"] = shrink(b["late_chase_runs"], b["late_chase_overs"], league_late, k["bowler_late_overs"])
    X["bowler_overs_all"] = b["legal_balls"] / 6
    X["bowler_economy_shrunk"] = 6 * shrink(b["runs_conceded"], b["legal_balls"], league_econ / 6, k["bowler_economy_balls"])
    X["bowler_inn1_late_economy_shrunk"] = 6 * shrink(
        b["inn1_late_runs_conceded"], b["inn1_late_legal_balls"], league_inn1 / 6, k["bowler_inn1_late_balls"])

    for role, col in (("striker", "striker_id"), ("non_striker", "non_striker_id")):
        t = asof(rows, history_tables["history_batter"], col, history.BATTER_SUMS)
        X[f"{role}_balls"] = t["balls_faced"].astype(int)
        X[f"{role}_sr_shrunk"] = 100 * shrink(t["runs"], t["balls_faced"], league_sr, k["batter_balls"])
        X[f"{role}_late_sr_shrunk"] = 100 * shrink(t["late_runs"], t["late_balls_faced"], league_late_sr, k["batter_late_balls"])
        X[f"{role}_dismissal_rate_shrunk"] = shrink(t["dismissals"], t["balls_faced"], league_out, k["batter_dismissal_balls"])

    X["league_late_mean"] = league_late
    X["bowler_low_history"] = X["bowler_late_overs"] < mins["bowler_late_overs"]
    X["striker_low_history"] = X["striker_balls"] < mins["batter_balls"]
    X["is_low_history"] = X["bowler_low_history"] | X["striker_low_history"]

    feats = feature_columns()
    nulls = X[feats].isna().sum()
    assert nulls.sum() == 0, f"NaN features: {nulls[nulls > 0].to_dict()}"
    assert np.isfinite(X[feats].astype(float).to_numpy()).all(), "non-finite features"

    ids = rows[["match_id", "over", "season", "bowler_id"]]
    out = pd.concat([ids, X[feats]], axis=1)
    if bowler_override is None:
        out.insert(4, "label_runs_in_over", rows["runs_in_over"].to_numpy())
    if id_encoder is not None:
        out = pd.concat([out, id_encoder.transform(rows)], axis=1)
    return out


class IdEncoder:
    """One-hot of player IDs seen at least `min_rows` times in TRAINING rows; others -> unknown."""

    def __init__(self, roles: list[str], min_rows: int) -> None:
        self.roles, self.min_rows = list(roles), min_rows
        self.vocab: dict[str, list[str]] = {}

    @classmethod
    def from_config(cls) -> "IdEncoder":
        c = config.load_config()["features"]["id_encoding"]
        return cls(c["roles"], c["min_rows"])

    def fit(self, train_rows: pd.DataFrame) -> "IdEncoder":
        for role in self.roles:
            counts = train_rows[role].value_counts()
            self.vocab[role] = sorted(counts[counts >= self.min_rows].index)
        return self

    def columns(self) -> list[str]:
        return [f"{r}={v}" for r in self.roles for v in [*self.vocab[r], "unknown"]]

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        assert self.vocab, "IdEncoder used before fit"
        parts = []
        for role in self.roles:
            known = rows[role].where(rows[role].isin(self.vocab[role]), "unknown")
            cats = pd.Categorical(known, categories=[*self.vocab[role], "unknown"])
            parts.append(pd.get_dummies(cats, prefix=role, prefix_sep="=").astype(int).set_axis(rows.index))
        return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

def leakage_check(late: pd.DataFrame, X: pd.DataFrame, deliveries: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    """Recompute bowler and striker history for random rows straight from the source tables,
    filtered to earlier matches, and assert it equals the features. Independent of history.py."""
    cfg = config.load_config()
    n = min(cfg["features"]["leakage_check_rows"], len(late))
    idx = late.sample(n=n, random_state=cfg["seed"]).index
    pos = order.set_index("match_id")["match_order"]
    d = deliveries[~deliveries["is_super_over"]].assign(match_order=deliveries["match_id"].map(pos))
    lo = late.assign(match_order=late["match_id"].map(pos))
    res = []
    for i in idx:
        r = late.loc[i]
        cur = pos[r["match_id"]]
        prev_late = lo[(lo["bowler_id"] == r["bowler_id"]) & (lo["match_order"] < cur)]["runs_in_over"]
        db = d[(d["bowler_id"] == r["bowler_id"]) & (d["match_order"] < cur)]
        ds = d[(d["batter_id"] == r["striker_id"]) & (d["match_order"] < cur)]
        x = X.loc[i]
        checks = {
            "bowler_late_overs": (len(prev_late), x["bowler_late_overs"]),
            "bowler_late_mean": (prev_late.mean() if len(prev_late) else x["league_late_mean"], x["bowler_late_mean"]),
            "bowler_overs_all": (db["is_legal"].sum() / 6, x["bowler_overs_all"]),
            "striker_balls": (int((ds["extras_wides"] == 0).sum()), x["striker_balls"]),
        }
        for name, (want, got) in checks.items():
            assert np.isclose(want, got), f"leakage check failed: row {r['match_id']}/{r['over']} {name}: raw {want} != feature {got}"
        res.append({"match_id": r["match_id"], "over": r["over"], **{k: v[1] for k, v in checks.items()}})
    return pd.DataFrame(res)


def label_correlations(X: pd.DataFrame) -> pd.DataFrame:
    """|corr(feature, label)| on TRAINING seasons only; flags anything above the threshold."""
    cfg = config.load_config()
    train = X[X["season"].isin(cfg["splits"]["train_seasons"])]
    feats = feature_columns()
    corr = train[feats].astype(float).corrwith(train["label_runs_in_over"].astype(float))
    out = corr.rename_axis("feature").reset_index(name="corr_with_label")
    out["flagged"] = out["corr_with_label"].abs() > cfg["features"]["label_correlation_threshold"]
    return out.sort_values("corr_with_label", key=abs, ascending=False)


def write_report(X, corr, leak, hist, late, people) -> None:
    cfg = config.load_config()
    from .audit import md_table

    out = ["# Features\n", "_Generated by `make features` (src/ipl_late_overs/features.py). Do not edit by hand._\n"]
    w = out.append
    w("## How history is computed\n")
    w("- **Earlier matches only.** Matches are ordered by (date, numeric match_id). A row's history uses only "
      "matches with a smaller position in that order: earlier dates, and on the same date (double-headers) "
      "only matches with a smaller match_id. Nothing from the current match is used.")
    w("- **Shrinkage.** `shrunk = (n x mean + k x league) / (n + k)`, with the league value also from earlier matches. "
      f"k: {cfg['history']['shrinkage_k']}.")
    w(f"- **Low history.** Bowler with < {cfg['history']['min_history']['bowler_late_overs']} earlier late chase overs, or "
      f"striker with < {cfg['history']['min_history']['batter_balls']} earlier balls faced.")
    w("- **No NaN.** A player with no history gets the league value; the league value before the very first "
      f"match uses the priors in config ({cfg['history']['league_prior']}).")
    w("- **Sources.** Late-chase bowling history uses the late_overs table (same label definition). Economy, "
      "strike rate and dismissals use every parsed delivery except super overs, including D/L and no-result matches.")
    w("- **Runs conceded** = batter runs + wides + no-balls (byes and leg byes are not charged to the bowler). "
      "**Balls faced** excludes wides. **Dismissals** exclude retired hurt.\n")

    w(f"## Feature list ({len(feature_columns())} features)\n")
    w(md_table(pd.DataFrame([{"feature": f, "meaning": m, "source": s} for f, (m, s) in FEATURE_DOCS.items()])))
    w("\nOptional player-ID encoding (`features.id_encoding` in config, currently "
      f"{'on' if cfg['features']['id_encoding']['enabled'] else 'off'}): one-hot of "
      f"{', '.join(cfg['features']['id_encoding']['roles'])} for IDs with >= {cfg['features']['id_encoding']['min_rows']} "
      "training rows, plus an `unknown` column. Fitted on training rows only.\n")

    w("## Leakage checks\n")
    w(f"1. **Recompute from raw.** For {len(leak)} random rows (seed {cfg['seed']}), the bowler's late-chase overs and mean, "
      "overs in all phases, and the striker's balls faced were recomputed directly from late_overs / deliveries filtered "
      "to earlier matches, independently of history.py. **All match** (the stage fails otherwise).")
    flagged = corr[corr["flagged"]]
    w(f"2. **Correlation with the label** (training seasons only). Threshold |r| > "
      f"{cfg['features']['label_correlation_threshold']}: "
      f"**{'none flagged' if flagged.empty else ', '.join(flagged['feature'])}**. Largest |r|:\n")
    w(md_table(corr.head(8).assign(corr_with_label=lambda d: d["corr_with_label"].round(3))))
    w("")

    ex = cfg["late_overs"]["verification_overs"][3]
    row = late[(late["match_id"] == ex["match_id"]) & (late["over"] == ex["over"])].iloc[0]
    bid, mo = row["bowler_id"], hist["history_league"].set_index("match_id").loc[ex["match_id"], "match_order"]
    hb = hist["history_bowler"][hist["history_bowler"]["player_id"] == bid].sort_values("match_order")
    before = hb[hb["match_order"] < mo].tail(1)
    after = hb[hb["match_order"] <= mo].tail(1)
    name = people.set_index("player_id").loc[bid, "display_name"]
    w(f"## Example: {name}'s history before vs after match {ex['match_id']} ({row['date'].date()})\n")
    w("`before` is what any over in that match sees; `after` is what the next match sees.\n")
    tbl = pd.concat([before.assign(when="before"), after.assign(when="after")])
    tbl["late_mean"] = (tbl["late_chase_runs"] / tbl["late_chase_overs"]).round(2)
    tbl["economy"] = (6 * tbl["runs_conceded"] / tbl["legal_balls"]).round(2)
    w(md_table(tbl[["when", "match_id", "late_chase_overs", "late_chase_runs", "late_mean", "legal_balls", "runs_conceded", "economy"]]))
    xr = X[(X["match_id"] == ex["match_id"]) & (X["over"] == ex["over"])].iloc[0]
    w(f"\nIn that match's over {ex['over'] + 1} the features show {int(xr['bowler_late_overs'])} earlier late chase overs, "
      f"mean {xr['bowler_late_mean']:.2f}, shrunk {xr['bowler_late_mean_shrunk']:.2f}: the `before` row, as required.\n")
    p = config.path("features_report")
    p.write_text("\n".join(out) + "\n")


def main() -> None:
    matches = schemas.read_table(config.path("matches"), schemas.MATCHES)
    deliveries = schemas.read_table(config.path("deliveries"), schemas.DELIVERIES)
    late = schemas.read_table(config.path("late_overs"), schemas.LATE_OVERS)
    people = schemas.read_table(config.path("people"), schemas.PEOPLE)

    hist = history.build_history(matches, deliveries, late)
    for name, schema in HISTORY_TABLES.items():
        hist[name] = schemas.write_table(hist[name], schema, config.path(name))
    tables = load_history_tables()

    X = build_features(late, tables)
    assert len(X) == len(late) and (X["label_runs_in_over"].to_numpy() == late["runs_in_over"].to_numpy()).all()
    assert (X["match_id"].to_numpy() == late["match_id"].to_numpy()).all() and (X["over"].to_numpy() == late["over"].to_numpy()).all()
    assert (X["bowler_overs_before"].to_numpy() == late["bowler_overs_before"].to_numpy()).all()
    X = schemas.write_table(X, schemas.FEATURES, config.path("features"))

    leak = leakage_check(late, X, deliveries, hist["history_league"][["match_id", "match_order"]])
    corr = label_correlations(X)
    write_report(X, corr, leak, hist, late, people)
    log.info("features: %d rows x %d features; leakage check on %d rows passed; flagged: %s",
             len(X), len(feature_columns()), len(leak), list(corr.loc[corr["flagged"], "feature"]) or "none")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
