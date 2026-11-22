# Week 2: data cleaning and the late-over table

## What was done

- `overs.py` (`make overs`): applies the exclusions, checks the chase deliveries, and builds
  - `data/processed/late_overs.parquet`: one row per started late over (overs 15-20) of a kept chase,
    with the pre-over scoreboard, striker / non-striker, this over's bowler, the previous over's bowler,
    the label `runs_in_over`, `over_completed` and `legal_balls_in_over`;
  - `data/processed/late_over_bowler_loads.parquet`: overs and legal balls already bowled by every
    fielding bowler before each late over (needed for eligibility in week 7);
  - `data/processed/overs_exclusions.json`: every exclusion rule with its count, read by the audit.
- Schemas for both tables added to `schemas.py` and validated on write and read.
- `figures.py` (`make figures`): distribution of `runs_in_over` by over number.
- `audit.py`: new sections 11 (exclusion log) and 12 (late-over table, checks, 5 verified overs).
- Tests: 43 pass (13 new). They include 5 late overs rebuilt by hand from the raw JSON, the D/L
  exclusion, super-over handling, the retired-hurt / retired-out rule, a leakage check (changing a
  ball inside an over must not change that over's pre-over state), and bowler loads.

## Results

- 1,243 matches -> **1,201 kept**: 9 no-results, 23 D/L, 10 other shortened matches removed.
  175 super-over deliveries removed. 54 kept chases ended before over 15.
- **6,029 late-over rows** from 1,147 chases (overs 15 to 20: 1,147 / 1,122 / 1,083 / 1,025 / 930 / 722).
- Mean runs per over rises from 8.5 (over 15) to 9.7 (over 18), then falls to 8.7 (over 20).
- All hard checks pass: no null labels, runs needed > 0, balls remaining 6-36, one row per
  (match_id, over), row and run totals reconcile with the deliveries, state chains across overs,
  no bowler bowls consecutive overs, no bowler is past 4 overs.
- 5 hand-verified overs (IPL finals 2008, 2014, 2017, 2019, 2022) match the table and the real results.

## Decisions to review

1. **Truncated labels.** In 595 rows (10%) the chase ended mid-over, so `runs_in_over` is capped by
   the runs needed (e.g. 6 off 1 ball in the 2022 final). This is why over 20 averages less than
   over 18. They are kept as the plan says; the model must see `runs_needed` to learn this, and week 6
   reports their error separately.
2. **Balls remaining** follows the over schedule (6 x overs left) rather than 120 - legal balls. The two
   differ only in 12 rows, after an umpire miscounted an earlier over; the schedule is what a captain sees.
3. **Mid-over bowler changes** (18 rows): the over is credited to the bowler of its first ball.
   They are kept and flagged with `n_bowlers_in_over`.
4. **Shortened-but-no-D/L matches** (10) are excluded along with D/L, as the plan requires target overs = 20.

## Problems

- Nullable pandas columns turn comparisons with nulls into NA, which a filter treats as False. This
  was caught in week 1 and the exclusion rules use explicit `fillna`.
- 6 matches have an `absent_hurt` player, which means the chase ends at 9 wickets. `wickets_in_hand`
  ignores this (at most 6 chases are affected).
- Added `matplotlib` (pinned) for static report figures; plotly stays for the app.

## Next steps (week 3)

Player history features (bowler, batter, league), computed only from earlier matches, and the
single `build_features` function with a `bowler_override` argument.
