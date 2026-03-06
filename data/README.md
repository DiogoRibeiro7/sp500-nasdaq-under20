# Data directory

This directory contains the output files produced and maintained by the
sp500-nasdaq-under20 pipeline.

---

## Files

| File | Updated by | Description |
|---|---|---|
| `under20_history.csv` | Daily (GitHub Actions) | Consolidated OHLCV history for all tickers that recently traded below the price ceiling. |
| `nasdaq_tickers.csv` | On NASDAQ-100 refresh | Cached list of NASDAQ-100 constituent symbols and names, used as a fallback if Wikipedia is unreachable. |

---

## `under20_history.csv` — schema reference

One row per `(Date, Ticker)` pair. Rows are sorted by `Date` ascending,
then `Ticker` alphabetically.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `Date` | `datetime64[ns]` | No | Trading date (YYYY-MM-DD). Always a weekday; never a fixed NYSE holiday. |
| `Ticker` | `str` | No | Exchange ticker symbol (e.g. `SOFI`, `F`, `SIRI`). Upper-case, no whitespace. |
| `Name` | `str` | Yes | Human-readable company name (e.g. `SoFi Technologies Inc`). Sourced from Wikipedia on first appearance; may be `NaN` for tickers added before name resolution was introduced. |
| `Open` | `float64` | Yes | Opening price (USD). |
| `High` | `float64` | Yes | Intraday high price (USD). |
| `Low` | `float64` | Yes | Intraday low price (USD). |
| `Close` | `float64` | Yes | Closing price (USD). This is the raw close, **not** split/dividend adjusted. |
| `Adj Close` | `float64` | Yes | Split- and dividend-adjusted closing price (USD), as calculated by Yahoo Finance. Use this column for return calculations. |
| `Volume` | `float64` | Yes | Number of shares traded. Stored as `float64` because yfinance can return `NaN` for illiquid tickers on some days. Cast to `int64` after dropping nulls if needed. |

### Notes on nullable columns

OHLCV columns are nullable because yfinance occasionally returns `NaN` for
low-liquidity tickers on specific dates (trading halts, data gaps at Yahoo).
These rows are retained rather than dropped so that the date coverage remains
continuous and downstream consumers can decide how to handle the gaps.

### Price threshold

The CSV only contains tickers whose **Close** on the most recent trading day
was **strictly below** the configured price ceiling (default: `30.00 USD`).
The ceiling can be changed via `--max-price`; rows already in the file from
previous runs under a different ceiling are not removed.

### Adjusted vs unadjusted prices

`Close` is the raw closing price as reported by Yahoo Finance.
`Adj Close` is adjusted for splits and dividends. For any multi-period return
or momentum calculation, always use `Adj Close`. The two columns will diverge
after a stock split or special dividend.

### Duplicate handling

The pipeline deduplicates by `(Date, Ticker)` on every run. If a row already
exists for a given date and ticker, the incoming row is discarded unless the
existing row has no `Name` and the incoming one does, in which case the richer
row is kept.

---

## `nasdaq_tickers.csv` — schema reference

| Column | Type | Description |
|---|---|---|
| `Symbol` | `str` | NASDAQ-100 ticker symbol. Upper-case. |
| `Security Name` | `str` | Company name as listed in the Wikipedia NASDAQ-100 table. |

This file is refreshed whenever the main pipeline successfully fetches the
NASDAQ-100 constituent list from Wikipedia. It is used as a fallback cache on
subsequent runs if the Wikipedia request times out or returns unexpected HTML.
If this file is deleted it will be recreated on the next successful run.

---

## Loading the data

### Python / pandas

```python
import pandas as pd

df = pd.read_csv(
    "data/under20_history.csv",
    parse_dates=["Date"],
)

# Latest close per ticker
latest = (
    df.sort_values("Date")
    .groupby("Ticker")
    .last()
    .reset_index()[["Ticker", "Name", "Date", "Close", "Adj Close"]]
)
```

### Filter to a single ticker

```python
sofi = df[df["Ticker"] == "SOFI"].copy()
sofi = sofi.set_index("Date").sort_index()
```

### Compute simple daily returns

```python
df = df.sort_values(["Ticker", "Date"])
df["return"] = df.groupby("Ticker")["Adj Close"].pct_change()
```

### Drop rows with missing OHLCV

```python
ohlcv_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
df_clean = df.dropna(subset=ohlcv_cols)
```
