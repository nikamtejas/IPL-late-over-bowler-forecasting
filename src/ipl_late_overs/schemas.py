"""Table schemas, defined once. Every Parquet file is validated on write and on read."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Logical dtype -> (pandas dtype used on write, checker)
_DTYPES = {
    "str": ("string", pd.api.types.is_string_dtype),
    "int": ("Int64", pd.api.types.is_integer_dtype),
    "float": ("Float64", pd.api.types.is_float_dtype),
    "bool": ("boolean", pd.api.types.is_bool_dtype),
    "date": ("datetime64[ns]", pd.api.types.is_datetime64_any_dtype),
}


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Col:
    name: str
    dtype: str
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    allowed: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class Schema:
    name: str
    columns: tuple[Col, ...]
    key: tuple[str, ...]
    description: str = ""
    _by_name: dict[str, Col] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {c.name: c for c in self.columns})
        assert len(self._by_name) == len(self.columns), f"{self.name}: duplicate column names"
        assert all(k in self._by_name for k in self.key), f"{self.name}: key column missing"
        assert all(c.dtype in _DTYPES for c in self.columns), f"{self.name}: unknown dtype"

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order columns and cast to the schema's pandas dtypes (fails on missing columns)."""
        self._check_columns(df)
        return df[self.names].astype({c.name: _DTYPES[c.dtype][0] for c in self.columns})

    def validate(self, df: pd.DataFrame) -> None:
        self._check_columns(df)
        for c in self.columns:
            s = df[c.name]
            if not _DTYPES[c.dtype][1](s.dtype):
                raise SchemaError(f"{self.name}.{c.name}: expected {c.dtype}, got {s.dtype}")
            n_null = int(s.isna().sum())
            if n_null and not c.nullable:
                raise SchemaError(f"{self.name}.{c.name}: {n_null} nulls in non-nullable column")
            vals = s.dropna()
            if c.min is not None and (vals < c.min).any():
                raise SchemaError(f"{self.name}.{c.name}: values below {c.min}: {vals[vals < c.min].unique()[:5]}")
            if c.max is not None and (vals > c.max).any():
                raise SchemaError(f"{self.name}.{c.name}: values above {c.max}: {vals[vals > c.max].unique()[:5]}")
            if c.allowed is not None:
                bad = set(vals.unique()) - set(c.allowed)
                if bad:
                    raise SchemaError(f"{self.name}.{c.name}: unexpected values {sorted(bad)[:5]}")
        dup = df.duplicated(list(self.key))
        if dup.any():
            raise SchemaError(f"{self.name}: {int(dup.sum())} duplicate rows on key {self.key}")

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [n for n in self.names if n not in df.columns]
        extra = [n for n in df.columns if n not in self._by_name]
        if missing or extra:
            raise SchemaError(f"{self.name}: missing columns {missing}, unexpected columns {extra}")


def write_table(df: pd.DataFrame, schema: Schema, path: Path) -> pd.DataFrame:
    df = schema.coerce(df).reset_index(drop=True)
    schema.validate(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def read_table(path: Path, schema: Schema) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the stage that produces '{schema.name}'")
    df = schema.coerce(pd.read_parquet(path))
    schema.validate(df)
    return df


def write_json(obj: dict[str, Any], required_keys: tuple[str, ...], path: Path) -> None:
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise SchemaError(f"{path.name}: missing keys {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: Path, required_keys: tuple[str, ...]) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise SchemaError(f"{path.name}: missing keys {missing}")
    return obj


# --------------------------------------------------------------------------- #
# JSON documents
# --------------------------------------------------------------------------- #

DOWNLOAD_MANIFEST_KEYS = ("url", "file", "sha256", "size_bytes", "downloaded_at_utc", "note")


# --------------------------------------------------------------------------- #
# Parsed tables (stage: parse)
# --------------------------------------------------------------------------- #

MATCHES = Schema(
    name="matches",
    description="One row per match (info section of the Cricsheet JSON).",
    key=("match_id",),
    columns=(
        Col("match_id", "str"),
        Col("date", "date"),  # first day of play
        Col("n_dates", "int", min=1),
        Col("season", "int", min=2008),  # calendar year of `date`
        Col("season_label", "str"),  # Cricsheet label, e.g. "2007/08"
        Col("event_name", "str"),
        Col("event_match_number", "int", nullable=True, min=1),
        Col("event_stage", "str", nullable=True),
        Col("team1", "str"),
        Col("team2", "str"),
        Col("venue", "str"),
        Col("city", "str", nullable=True),
        Col("toss_winner", "str"),
        Col("toss_decision", "str", allowed=("bat", "field")),
        Col("winner", "str", nullable=True),
        Col("result", "str", nullable=True, allowed=("tie", "no result", "draw")),
        Col("win_by_runs", "int", nullable=True, min=0),
        Col("win_by_wickets", "int", nullable=True, min=0, max=10),
        Col("method", "str", nullable=True),  # e.g. "D/L"
        Col("eliminator", "str", nullable=True),  # super-over / tie-break winner
        Col("player_of_match", "str", nullable=True),  # ';'-joined names
        Col("overs", "int", min=1),
        Col("balls_per_over", "int", min=1),
        Col("n_innings", "int", min=0),
        Col("has_super_over", "bool"),
        Col("first_innings_team", "str", nullable=True),
        Col("target_runs", "int", nullable=True, min=1),  # 2nd-innings target as recorded
        Col("target_overs", "float", nullable=True, min=0),
        Col("has_miscounted_overs", "bool"),
        Col("n_absent_hurt", "int", min=0),
        Col("data_version", "str"),
    ),
)

MATCH_PLAYERS = Schema(
    name="match_players",
    description="Squad per team per match (info.players), keyed by Cricsheet registry ID. "
    "From 2023 the list includes the impact player (12 per team).",
    key=("match_id", "team", "player_id"),
    columns=(
        Col("match_id", "str"),
        Col("team", "str"),
        Col("player_id", "str"),
        Col("player_name", "str"),
        Col("squad_order", "int", min=0),
    ),
)

PEOPLE = Schema(
    name="people",
    description="Display names per registry ID (players only, not officials).",
    key=("player_id",),
    columns=(
        Col("player_id", "str"),
        Col("display_name", "str"),  # name used in the player's most recent match
        Col("all_names", "str"),  # ';'-joined distinct names seen
        Col("n_matches", "int", min=1),
    ),
)

DELIVERIES = Schema(
    name="deliveries",
    description="One row per delivery as recorded (wides and no-balls are their own rows).",
    key=("match_id", "innings", "ball_seq"),
    columns=(
        Col("match_id", "str"),
        Col("innings", "int", min=1),  # 1-based order in the file; super overs are 3+
        Col("is_super_over", "bool"),
        Col("batting_team", "str"),
        Col("bowling_team", "str"),
        Col("over", "int", min=0),  # 0-indexed
        Col("ball_in_over", "int", min=1),  # 1-based order within the over, incl. extras
        Col("ball_seq", "int", min=1),  # 1-based order within the innings
        Col("actual_delivery", "str", nullable=True),  # Cricsheet's "over.ball" label
        Col("batter", "str"),
        Col("batter_id", "str"),
        Col("non_striker", "str"),
        Col("non_striker_id", "str"),
        Col("bowler", "str"),
        Col("bowler_id", "str"),
        Col("runs_batter", "int", min=0),
        Col("runs_extras", "int", min=0),
        Col("runs_total", "int", min=0),
        Col("extras_wides", "int", min=0),
        Col("extras_noballs", "int", min=0),
        Col("extras_byes", "int", min=0),
        Col("extras_legbyes", "int", min=0),
        Col("extras_penalty", "int", min=0),
        Col("is_legal", "bool"),  # not a wide and not a no-ball
        Col("n_wickets", "int", min=0, max=2),
        Col("wicket_kind", "str", nullable=True),
        Col("player_out", "str", nullable=True),
        Col("player_out_id", "str", nullable=True),
    ),
)

REPLACEMENTS = Schema(
    name="replacements",
    description="Substitutions recorded on deliveries: 'match' (impact player / concussion sub, "
    "from 2023) and 'role' (e.g. a bowler replaced mid-over through injury).",
    key=("match_id", "innings", "ball_seq", "kind", "player_in_id"),
    columns=(
        Col("match_id", "str"),
        Col("innings", "int", min=1),
        Col("ball_seq", "int", min=1),
        Col("over", "int", min=0),
        Col("kind", "str", allowed=("match", "role")),
        Col("team", "str", nullable=True),
        Col("player_in", "str"),
        Col("player_in_id", "str"),
        Col("player_out", "str", nullable=True),
        Col("player_out_id", "str", nullable=True),
        Col("role", "str", nullable=True),
        Col("reason", "str", nullable=True),
    ),
)


# --------------------------------------------------------------------------- #
# Late-over table (stage: overs)
# --------------------------------------------------------------------------- #

OVERS_EXCLUSIONS_KEYS = ("match_rules", "innings_rules", "chase_rules", "kept", "row_checks")

LATE_OVERS = Schema(
    name="late_overs",
    description="One row per started late over (overs 15-20) of an unrevised IPL chase. "
    "State columns describe the scoreboard before the over's first ball; runs_in_over is the label.",
    key=("match_id", "over"),
    columns=(
        Col("match_id", "str"),
        Col("date", "date"),
        Col("season", "int", min=2008),
        Col("batting_team", "str"),
        Col("bowling_team", "str"),
        Col("over", "int", min=0),  # 0-indexed Cricsheet over
        Col("over_number", "int", min=1),  # cricket numbering (over + 1)
        # --- pre-over state (deliveries before the over's first ball only)
        Col("target", "int", min=1),
        Col("runs_scored", "int", min=0),
        Col("runs_needed", "int", min=1),
        Col("legal_balls_bowled", "int", min=0),
        Col("balls_remaining", "int", min=1),  # by the over schedule: 6 * (overs - over)
        Col("wickets_fallen", "int", min=0),
        Col("wickets_in_hand", "int", min=1),
        Col("required_rate", "float", min=0),  # runs per over: 6 * runs_needed / balls_remaining
        Col("striker_id", "str"),
        Col("non_striker_id", "str"),
        Col("bowler_id", "str"),  # bowler of the over's first ball
        Col("prev_bowler_id", "str"),  # bowler of the previous over's last ball
        Col("bowler_overs_before", "int", min=0),  # overs (incl. partial) this bowler bowled earlier
        Col("bowler_legal_balls_before", "int", min=0),
        # --- the over itself (outcome; never a feature)
        Col("runs_in_over", "int", min=0),  # label: sum of runs.total, all extras included
        Col("legal_balls_in_over", "int", min=0),
        Col("deliveries_in_over", "int", min=1),
        Col("n_bowlers_in_over", "int", min=1),  # >1 when a bowler was replaced mid-over
        Col("over_completed", "bool"),
        Col("chase_ended_in_over", "bool"),
    ),
)

LATE_OVER_BOWLER_LOADS = Schema(
    name="late_over_bowler_loads",
    description="For each late-over row: every fielding bowler who bowled earlier in the chase, "
    "with overs and legal balls bowled before that over.",
    key=("match_id", "over", "bowler_id"),
    columns=(
        Col("match_id", "str"),
        Col("over", "int", min=0),
        Col("bowler_id", "str"),
        Col("overs_bowled", "int", min=1),
        Col("legal_balls_bowled", "int", min=0),
        Col("bowled_previous_over", "bool"),
    ),
)


# --------------------------------------------------------------------------- #
# History (stage: features). Cumulative totals per player AFTER each match they
# played in; a lookup for match m uses the last row with match_order < m's.
# --------------------------------------------------------------------------- #

_HISTORY_KEY_COLS = (
    Col("player_id", "str"),
    Col("match_id", "str"),
    Col("match_order", "int", min=1),
)

BOWLER_HISTORY = Schema(
    name="history_bowler",
    description="Cumulative bowling totals per bowler after each match they bowled in.",
    key=("player_id", "match_order"),
    columns=_HISTORY_KEY_COLS + (
        Col("late_chase_overs", "int", min=0),  # late-over rows (same label definition as late_overs)
        Col("late_chase_runs", "int", min=0),  # sum of runs_in_over in those rows
        Col("legal_balls", "int", min=0),  # all phases, both innings
        Col("runs_conceded", "int", min=0),  # batter runs + wides + no-balls
        Col("inn1_late_legal_balls", "int", min=0),  # overs 15-20 of the first innings
        Col("inn1_late_runs_conceded", "int", min=0),
    ),
)

BATTER_HISTORY = Schema(
    name="history_batter",
    description="Cumulative batting totals per batter after each match they batted in.",
    key=("player_id", "match_order"),
    columns=_HISTORY_KEY_COLS + (
        Col("balls_faced", "int", min=0),  # deliveries faced excluding wides
        Col("runs", "int", min=0),  # runs off the bat
        Col("late_balls_faced", "int", min=0),  # overs 15-20, either innings
        Col("late_runs", "int", min=0),
        Col("dismissals", "int", min=0),  # as player_out; retired hurt excluded
    ),
)

LEAGUE_HISTORY = Schema(
    name="history_league",
    description="League-wide cumulative totals after each match (all matches, in match_order).",
    key=("match_order",),
    columns=(
        Col("match_id", "str"),
        Col("match_order", "int", min=1),
        Col("date", "date"),
        Col("late_chase_overs", "int", min=0),
        Col("late_chase_runs", "int", min=0),
        Col("legal_balls", "int", min=0),
        Col("runs_conceded", "int", min=0),
        Col("inn1_late_legal_balls", "int", min=0),
        Col("inn1_late_runs_conceded", "int", min=0),
        Col("balls_faced", "int", min=0),
        Col("runs", "int", min=0),
        Col("late_balls_faced", "int", min=0),
        Col("late_runs", "int", min=0),
        Col("dismissals", "int", min=0),
    ),
)

# --------------------------------------------------------------------------- #
# Feature matrix (stage: features). Built only by features.build_features.
# --------------------------------------------------------------------------- #

FEATURE_ID_COLUMNS = ("match_id", "over", "season", "bowler_id", "label_runs_in_over")

FEATURES = Schema(
    name="features",
    description="Model matrix: one row per late-over row, actual bowler. Every feature is non-null.",
    key=("match_id", "over"),
    columns=(
        Col("match_id", "str"),
        Col("over", "int", min=0),
        Col("season", "int", min=2008),
        Col("bowler_id", "str"),
        Col("label_runs_in_over", "int", min=0),
        # --- match state before the over
        Col("over_number", "int", min=1),
        Col("target", "int", min=1),
        Col("runs_scored", "int", min=0),
        Col("runs_needed", "int", min=1),
        Col("balls_remaining", "int", min=1),
        Col("wickets_in_hand", "int", min=1),
        Col("required_rate", "float", min=0),
        Col("bowler_overs_before", "int", min=0),
        # --- bowler history
        Col("bowler_late_overs", "int", min=0),
        Col("bowler_late_mean", "float", min=0),
        Col("bowler_late_mean_shrunk", "float", min=0),
        Col("bowler_overs_all", "float", min=0),
        Col("bowler_economy_shrunk", "float", min=0),
        Col("bowler_inn1_late_economy_shrunk", "float", min=0),
        # --- batter history (striker and non-striker)
        Col("striker_balls", "int", min=0),
        Col("striker_sr_shrunk", "float", min=0),
        Col("striker_late_sr_shrunk", "float", min=0),
        Col("striker_dismissal_rate_shrunk", "float", min=0, max=1),
        Col("non_striker_balls", "int", min=0),
        Col("non_striker_sr_shrunk", "float", min=0),
        Col("non_striker_late_sr_shrunk", "float", min=0),
        Col("non_striker_dismissal_rate_shrunk", "float", min=0, max=1),
        # --- league
        Col("league_late_mean", "float", min=0),
        # --- flags
        Col("bowler_low_history", "bool"),
        Col("striker_low_history", "bool"),
        Col("is_low_history", "bool"),
    ),
)


# --------------------------------------------------------------------------- #
# Splits and validation results (stages: splits, baselines)
# --------------------------------------------------------------------------- #

SPLITS_KEYS = ("seasons", "train_match_ids", "validation_match_ids", "test_match_ids", "folds", "counts")
VALIDATION_BASELINES_KEYS = ("folds", "summary", "baseline_definitions")
