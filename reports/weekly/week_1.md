# Week 1: project setup and data check

## What was done

- Project skeleton: `config.yaml`, `Makefile`, `README.md`, `requirements.txt` (pinned, Python 3.11),
  project rules (kept locally), package `src/ipl_late_overs/`, data/model/report folders.
- `config.py`: loads and validates `config.yaml` (splits must be disjoint and chronological).
- `schemas.py`: one definition per table (columns, dtypes, nullability, ranges, allowed values,
  primary key); `write_table` / `read_table` validate on every write and read.
- `download.py` (`make download`): fetches the Cricsheet IPL zip, skips if present, records the
  date and SHA-256 in `data/raw/download_manifest.json`, and fails if the file later changes.
- `parse.py` (`make parse`): reads every match JSON straight from the zip (raw never extracted
  or modified) into five Parquet tables in `data/interim/`:
  `matches` (1,243), `match_players` (27,909), `people` (809), `deliveries` (295,732),
  `replacements` (624).
- `audit.py` (`make audit`): generates `reports/data_audit.md`.
- Tests (30, all passing): innings totals for 3 fixture matches vs. ESPNcricinfo scorecards
  (incl. a super-over match and an impact-player match), metadata, registry IDs, extras,
  schema checks, fail-loudly behaviour, download skip/hash checks, config validation.

## Results

- 1,243 matches, 19 seasons (2008-2026), Cricsheet format 1.3.0 throughout.
- Data quality is high: 0 duplicate matches or deliveries; for all 1,203 unrevised matches
  the target equals first-innings total + 1, and every result margin reconciles with the totals.
- Candidates for week-2 exclusion: 9 no-results, 34 revised-target matches (23 D/L, 11 shortened
  without a method), 16 super-over matches (34 super-over innings).
- 8 overs (from Cricsheet's `miscounted_overs` flag) have 5 or 7 legal balls; one is a late
  over in a chase (336015, over index 14). 29 late overs have more than one bowler (mid-over injury).
- Format change: from 2023, squads list 12 players (impact player) and substitutions are recorded.
- Rough count (preliminary, the real table is built in week 2): ~4,270 late chase overs in
  train, ~1,090 in validation and ~670 in test.

## Decisions to review

1. **Season split**: train 2008-2021, validation folds 2022 / 2023 / 2024 (expanding window),
   test 2025-2026. All of validation except 2022, and all of test, are impact-player seasons with
   higher scoring, so the model is tuned on the conditions it will be tested on. The trade-off is a
   training set that is mostly pre-impact-player.
2. **Season numbering** uses the calendar year (Cricsheet's "2020/21" label = the 2020 UAE season).
3. **Parsed table layout**: players per team and registry IDs are stored in long tables
   (`match_players`, `people`) rather than as nested list columns in `matches`, so they can be
   schema-validated and joined.

## Problems

- The plan's "players per team" field in `matches` was split into a separate table (see above).
- Some team names changed over time (e.g. Delhi Daredevils became Delhi Capitals). They are not
  merged, because team is not a model key.
- One name (Harmeet Singh) belongs to two different registry IDs, which confirms that players
  must be keyed by ID.

## Next steps (week 2)

Clean and build the late-over table: second innings of unrevised 20-over matches, one row per
started over 15-20, pre-over scoreboard state, `runs_in_over` label, 5 hand-verified overs.
Decide how to treat the 29 multi-bowler late overs and the miscounted over 336015/14.
