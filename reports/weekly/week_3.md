# Week 3: player history features

## What was done

- `history.py`: per-player cumulative totals after every match (bowlers and batters) and league
  totals after every match. Matches are ordered by (date, numeric match_id); history for a match
  uses only matches strictly before it in that order (same-date double-headers are ordered by match_id).
- `features.py` (`make features`):
  - `build_features(over_rows, history_tables, bowler_override=None, id_encoder=None)`, the one
    function that builds model inputs. History is looked up "as of" each match, so it works for
    any bowler, which is how `bowler_override` scores an alternative bowler for the same situation.
    With an override no label is returned, because that bowler's runs are never observed.
  - 26 features: 8 for the match state, 6 for the bowler's history, 4 each for the striker and
    non-striker, 1 league average and 3 low-history flags. Everything is shrunk toward the league
    average with `k` from config. Players with no history get the league value, so nothing is NaN.
  - Optional one-hot encoding of player IDs (`IdEncoder`), fitted on training rows only, with an
    `unknown` bucket. It is switched off in config; tests cover it both on and off.
  - Writes `history_{bowler,batter,league}.parquet`, `features.parquet` and `reports/features.md`
    (feature table, leakage checks, before/after example).
- Schemas for the three history tables and the feature matrix in `schemas.py`.
- Tests: 56 pass (13 new).

## Results

- Feature matrix: 6,029 rows x 26 features, no nulls, schema-validated.
- **Leakage check 1 (every `make features` run):** for 300 random rows, the bowler's earlier late
  overs and their mean, the bowler's overs in all phases, and the striker's balls faced were
  recomputed straight from the source tables filtered to earlier matches. They match exactly, and
  the stage fails if they don't.
- **Leakage check 2:** no feature has |correlation| > 0.9 with the label (training seasons only).
  The largest is 0.14 (target), so the signal per over is weak, which is expected for a noisy
  6-ball outcome.
- **Leakage tests:** removing every match after the 2019 final leaves that final's features unchanged;
  the current match never counts (Johnson's 3 earlier overs in the 2017 final are excluded); values
  worked out by hand (Malinga 2 earlier late chase overs, mean 6.5, among the fixtures) match.
- Example in `reports/features.md`: Malinga before the 2019 final had 107 late chase overs at
  7.14 per over. After it he had 109 at 7.26, since the final added 20 and 7.
- Low-history rows: 43% in train, 34% in validation, 37% in test.

## Decisions to review

1. **League average uses all earlier seasons.** It drifts from 8.3 (2011) to 8.9 (2025), lagging the
   recent rise in scoring after the impact-player rule. A recent-seasons league average could be
   added later as a separate feature if validation shows the drift matters.
2. **Shrinkage strengths (`k`) and low-history thresholds** are reasonable defaults, not tuned.
   They could be added to the week-5 grid search.
3. **Sources.** Bowler late-chase history uses only kept chases (same label definition); economy,
   strike rate and dismissals use all matches, including D/L ones, because those balls were bowled.
4. **ID encoding off by default.** The history features already capture who the player is, and
   one-hot IDs add many sparse columns.

## Problems

- An override bowler ID passed as a plain Python string didn't join with the string-typed tables.
  Fixed by casting overrides to the same dtype, and a test covers it.

## Next steps (week 4)

Chronological splits (`splits.py`, `splits.json`, test set locked behind `evaluate.py --final`) and the
two baselines: A = the bowler's historical late-over mean (`bowler_late_mean`); B = a regression on
the scoreboard only.
