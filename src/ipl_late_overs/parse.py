"""Stage: parse. Turn every match JSON in the raw zip into tidy Parquet tables.

Reads directly from the zip so data/raw is never extracted or modified. Produces
matches, match_players, people, deliveries and replacements (see schemas.py).
Nothing is filtered here: exclusions happen in later stages and are logged in the audit.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from . import config, schemas

log = logging.getLogger(__name__)

EXTRAS_KINDS = ("wides", "noballs", "byes", "legbyes", "penalty")


class ParseError(ValueError):
    pass


def parse_match(match_id: str, j: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Parse one Cricsheet match JSON into row dicts for each table."""
    info = j["info"]
    people = info["registry"]["people"]

    def pid(name: str) -> str:
        if name not in people:
            raise ParseError(f"{match_id}: '{name}' has no registry ID")
        return people[name]

    teams = info["teams"]
    if len(teams) != 2:
        raise ParseError(f"{match_id}: expected 2 teams, got {teams}")
    outcome = info["outcome"]
    by = outcome.get("by", {})
    innings = j.get("innings", [])
    regular = [inn for inn in innings if not inn.get("super_over")]
    target = regular[1].get("target", {}) if len(regular) >= 2 else {}

    match_row = {
        "match_id": match_id,
        "date": pd.Timestamp(info["dates"][0]),
        "n_dates": len(info["dates"]),
        "season": pd.Timestamp(info["dates"][0]).year,
        "season_label": str(info["season"]),
        "event_name": info["event"]["name"],
        "event_match_number": info["event"].get("match_number"),
        "event_stage": info["event"].get("stage"),
        "team1": teams[0],
        "team2": teams[1],
        "venue": info["venue"],
        "city": info.get("city"),
        "toss_winner": info["toss"]["winner"],
        "toss_decision": info["toss"]["decision"],
        "winner": outcome.get("winner"),
        "result": outcome.get("result"),
        "win_by_runs": by.get("runs"),
        "win_by_wickets": by.get("wickets"),
        "method": outcome.get("method"),
        "eliminator": outcome.get("eliminator"),
        "player_of_match": ";".join(info["player_of_match"]) if info.get("player_of_match") else None,
        "overs": info["overs"],
        "balls_per_over": info["balls_per_over"],
        "n_innings": len(innings),
        "has_super_over": any(inn.get("super_over") for inn in innings),
        "first_innings_team": regular[0]["team"] if regular else None,
        "target_runs": target.get("runs"),
        "target_overs": target.get("overs"),
        "has_miscounted_overs": any("miscounted_overs" in inn for inn in innings),
        "n_absent_hurt": sum(len(inn.get("absent_hurt", [])) for inn in innings),
        "data_version": str(j["meta"]["data_version"]),
    }

    player_rows = []
    for team, names in info["players"].items():
        if team not in teams:
            raise ParseError(f"{match_id}: players listed for unknown team '{team}'")
        for order, name in enumerate(names):
            player_rows.append(
                {"match_id": match_id, "team": team, "player_id": pid(name), "player_name": name, "squad_order": order}
            )

    delivery_rows, replacement_rows = [], []
    for inn_no, inn in enumerate(innings, start=1):
        batting = inn["team"]
        if batting not in teams:
            raise ParseError(f"{match_id} innings {inn_no}: unknown batting team '{batting}'")
        bowling = teams[1] if batting == teams[0] else teams[0]
        seq = 0
        for ov in inn.get("overs", []):
            for ball_in_over, d in enumerate(ov["deliveries"], start=1):
                seq += 1
                runs = d["runs"]
                extras = {k: int(d.get("extras", {}).get(k, 0)) for k in EXTRAS_KINDS}
                unknown = set(d.get("extras", {})) - set(EXTRAS_KINDS)
                if unknown:
                    raise ParseError(f"{match_id}: unknown extras type {unknown}")
                if runs["batter"] + runs["extras"] != runs["total"] or sum(extras.values()) != runs["extras"]:
                    raise ParseError(f"{match_id} inn {inn_no} over {ov['over']} ball {ball_in_over}: runs don't add up {runs} {extras}")
                wickets = d.get("wickets", [])
                w = wickets[0] if wickets else {}
                delivery_rows.append(
                    {
                        "match_id": match_id,
                        "innings": inn_no,
                        "is_super_over": bool(inn.get("super_over", False)),
                        "batting_team": batting,
                        "bowling_team": bowling,
                        "over": ov["over"],
                        "ball_in_over": ball_in_over,
                        "ball_seq": seq,
                        "actual_delivery": d.get("actual_delivery"),
                        "batter": d["batter"],
                        "batter_id": pid(d["batter"]),
                        "non_striker": d["non_striker"],
                        "non_striker_id": pid(d["non_striker"]),
                        "bowler": d["bowler"],
                        "bowler_id": pid(d["bowler"]),
                        "runs_batter": runs["batter"],
                        "runs_extras": runs["extras"],
                        "runs_total": runs["total"],
                        **{f"extras_{k}": v for k, v in extras.items()},
                        "is_legal": extras["wides"] == 0 and extras["noballs"] == 0,
                        "n_wickets": len(wickets),
                        "wicket_kind": w.get("kind"),
                        "player_out": w.get("player_out"),
                        "player_out_id": pid(w["player_out"]) if w else None,
                    }
                )
                for kind, reps in d.get("replacements", {}).items():
                    for r in reps:
                        replacement_rows.append(
                            {
                                "match_id": match_id,
                                "innings": inn_no,
                                "ball_seq": seq,
                                "over": ov["over"],
                                "kind": kind,
                                "team": r.get("team"),
                                "player_in": r["in"],
                                "player_in_id": pid(r["in"]),
                                "player_out": r.get("out"),
                                "player_out_id": pid(r["out"]) if r.get("out") else None,
                                "role": r.get("role"),
                                "reason": r.get("reason"),
                            }
                        )
    return {
        "matches": [match_row],
        "match_players": player_rows,
        "deliveries": delivery_rows,
        "replacements": replacement_rows,
    }


def build_people(matches: pd.DataFrame, match_players: pd.DataFrame) -> pd.DataFrame:
    """Display name = name used in the player's most recent match (ties broken by match_id)."""
    mp = match_players.merge(matches[["match_id", "date"]], on="match_id", validate="many_to_one")
    mp = mp.sort_values(["date", "match_id"])
    g = mp.groupby("player_id", sort=True)
    return pd.DataFrame(
        {
            "display_name": g["player_name"].last(),
            "all_names": g["player_name"].agg(lambda s: ";".join(sorted(set(s)))),
            "n_matches": g["match_id"].nunique(),
        }
    ).reset_index()


def tables_from_matches(parsed: list[dict[str, list[dict[str, Any]]]]) -> dict[str, pd.DataFrame]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in parsed:
        for table, r in p.items():
            rows[table].extend(r)
    out = {
        "matches": schemas.MATCHES.coerce(pd.DataFrame(rows["matches"])),
        "match_players": schemas.MATCH_PLAYERS.coerce(pd.DataFrame(rows["match_players"])),
        "deliveries": schemas.DELIVERIES.coerce(pd.DataFrame(rows["deliveries"])),
        "replacements": schemas.REPLACEMENTS.coerce(
            pd.DataFrame(rows["replacements"], columns=schemas.REPLACEMENTS.names)
        ),
    }
    out["people"] = schemas.PEOPLE.coerce(build_people(out["matches"], out["match_players"]))
    _check_consistency(out)
    return out


def _check_consistency(t: dict[str, pd.DataFrame]) -> None:
    m, d = t["matches"], t["deliveries"]
    # Each Cricsheet season label must map to exactly one calendar year.
    years = m.groupby("season_label")["season"].nunique()
    if (years > 1).any():
        raise ParseError(f"season labels spanning several years: {years[years > 1].to_dict()}")
    orphan = set(d["match_id"]) - set(m["match_id"])
    if orphan:
        raise ParseError(f"deliveries for unknown matches: {sorted(orphan)[:5]}")
    # Batters must belong to the batting squad and bowlers to the fielding squad.
    squad = set(zip(t["match_players"]["match_id"], t["match_players"]["team"], t["match_players"]["player_id"]))
    for role, team_col in (("batter_id", "batting_team"), ("non_striker_id", "batting_team"), ("bowler_id", "bowling_team")):
        bad = [k for k in zip(d["match_id"], d[team_col], d[role]) if k not in squad]
        if bad:
            raise ParseError(f"{len(bad)} deliveries where {role} is not in the {team_col} squad, e.g. {bad[:3]}")


def parse_zip(zip_path: Path) -> dict[str, pd.DataFrame]:
    parsed = []
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(n for n in zf.namelist() if n.endswith(".json"))
        for name in members:
            match_id = Path(name).stem
            parsed.append(parse_match(match_id, json.loads(zf.read(name))))
        expected = _expected_match_count(zf)
    if expected is not None and expected != len(parsed):
        raise ParseError(f"README says {expected} matches but zip holds {len(parsed)} JSON files")
    log.info("parsed %d matches", len(parsed))
    return tables_from_matches(parsed)


def _expected_match_count(zf: zipfile.ZipFile) -> int | None:
    if "README.txt" not in zf.namelist():
        return None
    m = re.search(r"contains (\d+)\s+.*?matches", zf.read("README.txt").decode("utf-8", "replace"), re.S)
    return int(m.group(1)) if m else None


OUTPUTS = {
    "matches": schemas.MATCHES,
    "match_players": schemas.MATCH_PLAYERS,
    "people": schemas.PEOPLE,
    "deliveries": schemas.DELIVERIES,
    "replacements": schemas.REPLACEMENTS,
}


def main() -> None:
    tables = parse_zip(config.path("raw_zip"))
    for name, schema in OUTPUTS.items():
        df = schemas.write_table(tables[name], schema, config.path(name))
        log.info("wrote %s: %d rows", config.path(name).relative_to(config.REPO_ROOT), len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
