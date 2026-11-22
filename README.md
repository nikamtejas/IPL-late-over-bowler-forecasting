# IPL late-over bowler forecasting

Forecast the total runs scored in a late over (overs 15-20) of an IPL chase, to help a
captain choose the next bowler.

**Research question:** does match context improve forecasts of the runs scored in a late
over, compared with a bowler's historical late-over average?

Data: ball-by-ball IPL data from [Cricsheet](https://cricsheet.org/) (ODC-BY licence).

## Setup

```bash
make setup        # creates .venv with Python 3.11 and installs requirements.txt
make all          # runs the whole pipeline
make test         # runs the test suite
```

## Pipeline stages

| target | module | output |
|---|---|---|
| `make download` | `download.py` | `data/raw/ipl_json.zip` + `download_manifest.json` (date, SHA-256); skipped if present |
| `make parse` | `parse.py` | `data/interim/{matches,match_players,people,deliveries,replacements}.parquet` |
| `make audit` | `audit.py` | `reports/data_audit.md` |

All settings live in `config.yaml`. Weekly progress logs are
in `reports/weekly/`, and each week is tagged in git (`week-1`, `week-2`, ...).

## Definitions (so far)

- **Season**: calendar year of the match's first day. Cricsheet's own label is kept as `season_label`.
- **Over index**: Cricsheet's 0-indexed over number; a late over is index 14-19 (overs 15-20).
- **Legal delivery**: not a wide and not a no-ball.
- **Player key**: Cricsheet registry ID.
