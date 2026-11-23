# Week 4: train/test split and baselines

## What was done

- `splits.py` (`make splits`): splits whole matches by season, as set in `config.yaml`.
  - train 2008-2021 (876 matches, 4,268 late-over rows); validation 2022-2024 (219 matches, 1,093 rows);
    test 2025-2026 (148 matches, 668 rows).
  - Three expanding validation folds: fit on 2008-2021 and validate on 2022; fit on 2008-2022 and
    validate on 2023; fit on 2008-2023 and validate on 2024.
  - Asserted: no match_id in more than one split, every match assigned, every test match later than every
    train/validation match, validation later than train.
  - Writes `data/processed/splits.json` and `reports/splits.md` (tables and a season timeline).
- **Test-set lock.** `splits.development_rows` and `splits.fold_rows` always drop test matches.
  `splits.test_rows` needs a `FinalEvaluationKey`, which only `evaluate.py --final` can create (one key
  per run). `evaluate.py` is a stub this week: even `--final` refuses to run until week 6. The test set
  was not opened.
- `baselines.py` (`make baselines`): both baselines are scored on the validation folds only and
  written to `reports/validation_baselines.json`.
  - **A (historical):** the bowler's mean runs per late chase over from earlier matches; the league mean
    from earlier matches when the bowler has fewer than 5 such overs.
  - **B (scoreboard):** Poisson regression on over number, balls remaining, runs needed, wickets in hand and
    required rate, standardised inside a scikit-learn Pipeline so scaling is fitted on each fold's training rows.
- Chart `reports/figures/validation_baselines.png`; audit section 13 now reads `splits.json`.
- Tests: 65 pass (9 new) covering the split rules, the test lock, `evaluate.py` refusing to run, and both baselines.

## Results: validation MAE (runs per over)

| fold | validate on | Baseline A | Baseline B |
|---|---|---|---|
| 1 | 2022 | 3.93 | 3.91 |
| 2 | 2023 | 4.23 | 4.09 |
| 3 | 2024 | 4.53 | 4.31 |
| **mean of folds** | | **4.23** | **4.10** |

**The number to beat is 4.10 (Baseline B).** The scoreboard alone beats the bowler's history in every fold.

## Observations

- Error grows each season (3.9 to 4.5) as scoring rises: the average late over went from 9.5 runs
  (2022) to 9.9 (2024).
- Both baselines **under-predict** by about 0.7 to 1.2 runs per over (predictions average about 8.7-9.0
  against actual 9.5-9.9). Baseline A uses averages from all earlier seasons, and B is fitted mostly on
  pre-2022 overs, so both lag the higher scoring of recent seasons.
- Baseline A falls back to the league mean for 16-29% of validation rows (bowlers with < 5 earlier late
  chase overs).

## Decisions to review

1. **Baseline B is Poisson regression** (label = a count of runs). `over_number` and `balls_remaining`
   carry the same information (`balls_remaining = 6 x (21 - over_number)`); the small L2 penalty
   handles that and both are kept as the plan lists them.
2. **Scoring drift.** Week 5 could add a recent-seasons league average or season-weighted training if the
   random forest also under-predicts. That decision will be made on validation folds only.

## Next steps (week 5)

Random forest in a scikit-learn Pipeline using `build_features`, a grid search over max_depth,
n_estimators and min_samples_leaf by mean validation MAE across these folds, refit on train + validation,
save with joblib, and compare with both baselines overall and by over.
