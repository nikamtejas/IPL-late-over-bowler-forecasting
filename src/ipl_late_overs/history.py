"""Player and league history tables (built by the features stage).

Ordering rule: matches are ordered by (date, numeric match_id) -> `match_order`.
A match's history uses only matches with a smaller match_order, i.e. earlier dates,
and on the same date (IPL double-headers) only matches with a smaller match_id.
Nothing from the current match is ever included.

Each table stores cumulative totals per player AFTER each match they took part in.
Looking up history for match m (for any player, including a bowler who did not
play in m) = the player's last row with match_order < m's (see features.asof).

Sources:
- late-chase bowling: the late_overs table (same label definition, kept chases only);
- everything else: all parsed deliveries except super overs, from every match
  (including D/L and no-result matches, whose balls were still bowled).
"""

from __future__ import annotations

import pandas as pd

from . import config

BOWLER_SUMS = ("late_chase_overs", "late_chase_runs", "legal_balls", "runs_conceded",
               "inn1_late_legal_balls", "inn1_late_runs_conceded")
BATTER_SUMS = ("balls_faced", "runs", "late_balls_faced", "late_runs", "dismissals")


def match_order(matches: pd.DataFrame) -> pd.DataFrame:
    """match_id -> match_order (1-based), ordered by date then numeric match_id."""
    m = matches[["match_id", "date"]].copy()
    m["_id_num"] = m["match_id"].astype(int)
    m = m.sort_values(["date", "_id_num"]).reset_index(drop=True)
    m["match_order"] = range(1, len(m) + 1)
    return m[["match_id", "date", "match_order"]]


def _late_mask(d: pd.DataFrame) -> pd.Series:
    lo = config.load_config()["late_overs"]
    return d["over"].between(lo["first_over_index"], lo["last_over_index"])


def bowler_match_stats(deliveries: pd.DataFrame, late_overs: pd.DataFrame) -> pd.DataFrame:
    """Per (match, bowler) totals for that match alone."""
    d = deliveries[~deliveries["is_super_over"]].copy()
    d["conceded"] = d["runs_batter"] + d["extras_wides"] + d["extras_noballs"]
    d["legal"] = d["is_legal"].astype(int)
    inn1_late = (d["innings"] == 1) & _late_mask(d)
    d["inn1_late_legal"] = d["legal"].where(inn1_late, 0)
    d["inn1_late_conceded"] = d["conceded"].where(inn1_late, 0)
    bowl = d.groupby(["match_id", "bowler_id"], as_index=False).agg(
        legal_balls=("legal", "sum"),
        runs_conceded=("conceded", "sum"),
        inn1_late_legal_balls=("inn1_late_legal", "sum"),
        inn1_late_runs_conceded=("inn1_late_conceded", "sum"),
    )
    late = late_overs.groupby(["match_id", "bowler_id"], as_index=False).agg(
        late_chase_overs=("runs_in_over", "size"), late_chase_runs=("runs_in_over", "sum")
    )
    out = bowl.merge(late, on=["match_id", "bowler_id"], how="outer")
    return out.rename(columns={"bowler_id": "player_id"}).fillna(0)


def batter_match_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per (match, batter) totals for that match alone."""
    non_dismissal = config.load_config()["late_overs"]["non_dismissal_wicket_kinds"]
    d = deliveries[~deliveries["is_super_over"]].copy()
    d["faced"] = (d["extras_wides"] == 0).astype(int)
    late = _late_mask(d)
    d["late_faced"] = d["faced"].where(late, 0)
    d["late_runs_bat"] = d["runs_batter"].where(late, 0)
    bat = d.groupby(["match_id", "batter_id"], as_index=False).agg(
        balls_faced=("faced", "sum"), runs=("runs_batter", "sum"),
        late_balls_faced=("late_faced", "sum"), late_runs=("late_runs_bat", "sum"),
    ).rename(columns={"batter_id": "player_id"})
    out_mask = d["player_out_id"].notna() & ~d["wicket_kind"].fillna("").isin(non_dismissal)
    outs = (d[out_mask].groupby(["match_id", "player_out_id"]).size()
            .rename("dismissals").reset_index().rename(columns={"player_out_id": "player_id"}))
    return bat.merge(outs, on=["match_id", "player_id"], how="outer").fillna(0)


def cumulative(per_match: pd.DataFrame, order: pd.DataFrame, sums: tuple[str, ...]) -> pd.DataFrame:
    """Running totals per player, in match_order, after each match the player appears in."""
    x = per_match.merge(order[["match_id", "match_order"]], on="match_id", validate="many_to_one")
    x = x.sort_values(["player_id", "match_order"]).reset_index(drop=True)
    x[list(sums)] = x.groupby("player_id")[list(sums)].cumsum()
    return x[["player_id", "match_id", "match_order", *sums]]


def league_cumulative(bowl: pd.DataFrame, bat: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    """League totals after each match (every match has a row, even with no deliveries)."""
    b = bowl.groupby("match_id")[list(BOWLER_SUMS)].sum()
    t = bat.groupby("match_id")[list(BATTER_SUMS)].sum()
    x = order.set_index("match_id").join(b).join(t).fillna(0).sort_values("match_order")
    sums = list(BOWLER_SUMS + BATTER_SUMS)
    x[sums] = x[sums].cumsum()
    return x.reset_index()[["match_id", "match_order", "date", *sums]]


def build_history(matches: pd.DataFrame, deliveries: pd.DataFrame, late_overs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    order = match_order(matches)
    bowl = bowler_match_stats(deliveries, late_overs)
    bat = batter_match_stats(deliveries)
    # Fail loudly if any total was lost on the way.
    chase_late = late_overs["runs_in_over"]
    assert bowl["late_chase_runs"].sum() == chase_late.sum() and bowl["late_chase_overs"].sum() == len(chase_late)
    assert bowl["legal_balls"].sum() == deliveries.loc[~deliveries["is_super_over"], "is_legal"].sum()
    hist = {
        "history_bowler": cumulative(bowl, order, BOWLER_SUMS),
        "history_batter": cumulative(bat, order, BATTER_SUMS),
        "history_league": league_cumulative(bowl, bat, order),
    }
    lg = hist["history_league"].iloc[-1]
    assert lg["late_chase_overs"] == len(late_overs) and lg["balls_faced"] == bat["balls_faced"].sum()
    return hist
