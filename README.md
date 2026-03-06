# sp500-nasdaq-under20

A lightweight data pipeline that identifies S&P 500 and NASDAQ-100 stocks
trading below a configurable price ceiling and maintains a consolidated daily
OHLCV history for them in a single CSV file, updated automatically every
weekday morning via GitHub Actions.

---

## Contents

```
.
├── scripts/
│   ├── under20_stocks.py            # Fetch tickers, filter by price, save per-ticker CSVs
│   ├── update_under20_master_csv.py # Incremental update of the consolidated CSV
│   ├── log_config.py                # Shared logging configuration
│   └── schemas.py                   # Pandera schema definitions and validators
├── tests/
│   ├── test_under20.py              # Unit tests for date logic, filtering, merging
│   └── test_schemas.py              # Unit tests for schema validation
├── data/
│   ├── under20_history.csv          # Master OHLCV history (auto-updated daily)
│   ├── nasdaq_tickers.csv           # Cached NASDAQ-100 ticker list
│   └── README.md                    # Data dictionary and schema reference
├── .github/
│   ├── workflows/
│   │   ├── update-under20-stocks.yml  # Daily update workflow
│   │   └── tests.yml                  # CI test workflow
│   └── dependabot.yml
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt
```

### Download one CSV per ticker

Writes individual files to `data_under_20/`:

```bash
python scripts/under20_stocks.py --max-price 30
```

### Update the consolidated CSV

Appends new rows to `data/under20_history.csv`:

```bash
python scripts/update_under20_master_csv.py --max-price 30
```

Both scripts must be run from the **repository root**.

---

## CLI reference

All arguments are optional. Defaults match the values used by the scheduled
GitHub Actions workflow.

| Argument | Default | Description |
|---|---|---|
| `--max-price` | `30.0` | Close price ceiling (USD). Tickers with `close < max-price` on the last trading day are included. |
| `--history-period` | `1y` | yfinance period string for history downloads (`1y`, `6mo`, `ytd`, …). |
| `--output-dir` | `data_under_20/` | Directory for per-ticker CSVs (`under20_stocks.py` only). |
| `--batch-size` | `150` | Number of tickers per yfinance batch request. |

Override the log level without changing code:

```bash
LOG_LEVEL=DEBUG python scripts/update_under20_master_csv.py
```

---

## Automated updates

The workflow in `.github/workflows/update-under20-stocks.yml` runs every
weekday at 01:00 UTC and commits any changes to `data/under20_history.csv`
and `data/nasdaq_tickers.csv`.

You can also trigger it manually from the **Actions** tab with a custom
`--max-price` value.

The pipeline only fetches data for the **last trading day** — weekends and
the fixed NYSE holidays (New Year's Day, Independence Day, Christmas) are
automatically skipped. If the script runs on one of those days, it rolls back
to the most recent valid trading day.

---

## Running the tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --tb=short
```

The full suite runs offline — no network calls, no yfinance, no file I/O
(except the two tests that use `tmp_path`). Coverage is reported to the
terminal and written to `coverage.xml`.

---

## Data

See [`data/README.md`](data/README.md) for the full schema reference,
column descriptions, and notes on known data quirks.

---

## Design notes

**Price filter is strictly less than** — a ticker at exactly `max-price` is
excluded. This matches the original intent of the "under N dollars" framing.

**Incremental updates, not full rewrites** — `update_under20_master_csv.py`
only appends rows that are not already present (deduplication key:
`Date + Ticker`). Re-running the script on the same day is safe.

**Name resolution is Wikipedia-first** — company names are sourced from the
same Wikipedia tables used to retrieve the constituent lists. Only tickers
not found in those tables fall back to a batched yfinance lookup, which
keeps the run time short even for large universes.

**Schema validation protects the CSV** — [pandera](https://pandera.readthedocs.io)
schemas are enforced at three points: after the price fetch, before reading
the existing CSV, and before writing the updated one. If validation fails, the
write is aborted so a good existing CSV is never overwritten with bad data.

---

## License

MIT — see [LICENSE](LICENSE).
