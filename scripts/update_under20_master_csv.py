"""
Incrementally update a single CSV with daily history of all
S&P500 + NASDAQ (when available) stocks that were under a configurable
price cap on the last trading day.

- Uses helper functions from scripts.under20_stocks
- Maintains a single CSV file at data/under20_history.csv
- On each run:
    1. Compute the last trading day (UTC).
    2. Find all tickers with last close < MAX_PRICE on or before that day.
    3. Download 1 year of daily history for those tickers.
    4. Resolve human-readable company names (Wikipedia first, yfinance fallback).
    5. Validate the merged DataFrame against the master CSV schema.
    6. Append missing rows (with names) to the master CSV (no duplicates).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pandera.errors
import yfinance as yf

# ---------------------------------------------------------------------------
# Path fix: ensure this script can import from its own directory regardless
# of the working directory from which it is invoked.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from log_config import get_logger  # noqa: E402
from schemas import validate_latest_closes, validate_master_csv  # noqa: E402
from under20_stocks import (  # noqa: E402
    BATCH_SIZE,
    HISTORY_PERIOD,
    MAX_PRICE,
    download_history_for_tickers,
    get_index_tickers_and_names,
    get_latest_closes_for_universe,
    get_yesterday_date,
    select_tickers_below_price,
)

log = get_logger(__name__)

MASTER_CSV_PATH: Path = Path("data") / "under20_history.csv"
_NAME_BATCH_SIZE: int = 50


# ---------------------------------------------------------------------------
# Helpers for building / loading the master CSV
# ---------------------------------------------------------------------------


def _build_new_rows_dataframe(
    history: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Convert a dict[ticker -> DataFrame] into a flat long-format DataFrame.

    Columns: Date, Ticker, Open, High, Low, Close, Adj Close, Volume
    The Name column is NOT added here; it is merged later.
    """
    records: List[pd.DataFrame] = []

    for ticker, df in history.items():
        if df.empty:
            continue

        df_local = df.copy()
        df_local.index = pd.to_datetime(df_local.index)
        df_local.reset_index(inplace=True)
        df_local.rename(columns={"index": "Date"}, inplace=True)

        rename_map = {
            "Adj_Close": "Adj Close",
            "adjclose": "Adj Close",
        }
        for old, new in rename_map.items():
            if old in df_local.columns and new not in df_local.columns:
                df_local.rename(columns={old: new}, inplace=True)

        required_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        cols_to_keep = [c for c in required_cols if c in df_local.columns]
        df_local = df_local[cols_to_keep]
        df_local["Ticker"] = ticker
        records.append(df_local)

    if not records:
        return pd.DataFrame(
            columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        )

    return pd.concat(records, ignore_index=True)


def _load_existing_master() -> pd.DataFrame:
    """Load existing master CSV, or return an empty DataFrame with the correct schema."""
    if not MASTER_CSV_PATH.exists():
        return pd.DataFrame(
            columns=["Date", "Ticker", "Name", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        )

    df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["Date"])

    if "Name" not in df.columns:
        log.debug("Existing master CSV has no Name column — adding for backwards compatibility.")
        df["Name"] = pd.NA

    # Validate the file we just loaded so we catch corruption or manual edits
    # that broke the schema before we start appending new data to it.
    try:
        df = validate_master_csv(df)
    except pandera.errors.SchemaErrors as exc:
        log.error(
            "Existing master CSV failed schema validation — it may be corrupt "
            "or was edited manually.\n%s",
            exc.failure_cases.to_string(index=False),
        )
        raise

    return df


def _merge_and_deduplicate(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge existing and new rows, deduplicating by (Date, Ticker).
    Rows with a company name are preferred over those without.
    """
    if new_rows.empty:
        return existing

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])

    has_name = combined["Name"].astype(str).str.strip().ne("")
    combined = combined.assign(_has_name=has_name)
    combined = combined.sort_values(by=["Date", "Ticker", "_has_name"])
    combined = combined.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    combined = combined.drop(columns="_has_name")
    combined = combined.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


def _fetch_names_from_yfinance(
    tickers: List[str],
    batch_size: int = _NAME_BATCH_SIZE,
) -> Dict[str, str]:
    """Fetch company names from yfinance using batched parallel requests."""
    names: Dict[str, str] = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        ticker_str = " ".join(batch)

        try:
            tickers_obj = yf.Tickers(ticker_str)
        except Exception as exc:  # noqa: BLE001
            log.warning("yf.Tickers() failed for batch at index %d: %s", i, exc)
            for t in batch:
                names.setdefault(t, t)
            continue

        for symbol in batch:
            try:
                info: dict = tickers_obj.tickers[symbol].info
                name = info.get("shortName") or info.get("longName")
                names[symbol] = name.strip() if isinstance(name, str) and name.strip() else symbol
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not fetch name for %s: %s", symbol, exc)
                names[symbol] = symbol

    return names


def resolve_ticker_names(
    tickers: List[str],
    known_names: Dict[str, str],
    batch_size: int = _NAME_BATCH_SIZE,
) -> Dict[str, str]:
    """
    Build a complete ticker -> name mapping.

    Priority:
    1. Wikipedia names from *known_names* (no network call).
    2. Batched yfinance lookup for any ticker not covered by Wikipedia.
    3. Ticker symbol itself as a last resort.
    """
    result: Dict[str, str] = {}
    needs_lookup: List[str] = []

    for t in tickers:
        wiki_name = known_names.get(t, "").strip()
        if wiki_name and wiki_name != t:
            result[t] = wiki_name
        else:
            needs_lookup.append(t)

    if needs_lookup:
        log.info(
            "Fetching names from yfinance for %d tickers not covered by Wikipedia.",
            len(needs_lookup),
        )
        yf_names = _fetch_names_from_yfinance(needs_lookup, batch_size=batch_size)
        result.update(yf_names)
    else:
        log.info("All ticker names resolved from Wikipedia — skipping yfinance name fetch.")

    return result


def _attach_names_to_rows(
    rows: pd.DataFrame,
    ticker_names: Dict[str, str],
) -> pd.DataFrame:
    """Add a Name column to *rows* using *ticker_names*."""
    if rows.empty:
        rows = rows.copy()
        rows["Name"] = pd.Series(dtype="string")
        return rows

    rows = rows.copy()
    rows["Name"] = rows["Ticker"].map(ticker_names).fillna(rows["Ticker"])
    return rows


# ---------------------------------------------------------------------------
# Main incremental update pipeline
# ---------------------------------------------------------------------------


def main(
    max_price: float = MAX_PRICE,
    history_period: str = HISTORY_PERIOD,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Main incremental update pipeline.

    Steps
    -----
    1. Compute the last trading day (UTC).
    2. Gather S&P 500 and NASDAQ tickers + Wikipedia names.
    3. Retrieve and validate latest close prices.
    4. Select tickers with close < max_price.
    5. Download 1 year of daily history for selected tickers.
    6. Resolve company names (Wikipedia first, batched yfinance fallback).
    7. Merge into master CSV, validate, and write.
    """
    if max_price <= 0:
        raise ValueError("max_price must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    as_of_date: dt.date = get_yesterday_date()
    log.info("Using last trading day: %s", as_of_date.isoformat())
    log.info("History period: %s  |  max price: %.2f USD", history_period, max_price)

    tickers_by_index, ticker_name_map = get_index_tickers_and_names()
    all_universe = tickers_by_index["sp500"].union(tickers_by_index["nasdaq"])
    log.info("Total unique tickers in universe: %d", len(all_universe))

    latest_closes = get_latest_closes_for_universe(
        all_tickers=all_universe,
        target_date=as_of_date,
        batch_size=batch_size,
    )

    # Validate closes before filtering — catches yfinance dtype regressions early.
    try:
        latest_closes = validate_latest_closes(latest_closes)
    except pandera.errors.SchemaErrors as exc:
        log.error(
            "Latest closes failed schema validation:\n%s",
            exc.failure_cases.to_string(index=False),
        )
        raise

    log.info("Got latest closes for %d tickers.", latest_closes["ticker"].nunique())

    selected_tickers = select_tickers_below_price(
        latest_closes=latest_closes,
        max_price=max_price,
    )
    log.info(
        "Tickers with close < %.2f USD on or before %s: %d",
        max_price,
        as_of_date,
        len(selected_tickers),
    )

    if not selected_tickers:
        log.warning("No tickers met the price criterion. Nothing to update.")
        return

    history = download_history_for_tickers(
        tickers=selected_tickers,
        period=history_period,
        batch_size=batch_size,
    )

    if not history:
        log.warning("No historical data downloaded. Nothing to update.")
        return

    new_rows = _build_new_rows_dataframe(history)
    log.info("New rows collected: %d", len(new_rows))

    ticker_names = resolve_ticker_names(
        tickers=selected_tickers,
        known_names=ticker_name_map,
        batch_size=_NAME_BATCH_SIZE,
    )
    new_rows = _attach_names_to_rows(new_rows, ticker_names)

    existing = _load_existing_master()
    log.info("Existing master rows: %d", len(existing))

    updated = _merge_and_deduplicate(existing, new_rows)
    log.info("Updated master rows after dedupe: %d", len(updated))

    MASTER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    desired_order = ["Date", "Ticker", "Name", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols_in_df = [c for c in desired_order if c in updated.columns]
    updated = updated[cols_in_df]

    # Final validation before writing — the last line of defence against
    # writing a malformed CSV that would corrupt future incremental runs.
    try:
        updated = validate_master_csv(updated)
    except pandera.errors.SchemaErrors as exc:
        log.error(
            "Updated master DataFrame failed schema validation — aborting write "
            "to protect the existing CSV.\n%s",
            exc.failure_cases.to_string(index=False),
        )
        raise

    updated.to_csv(MASTER_CSV_PATH, index=False)
    log.info("Master CSV updated at: %s", MASTER_CSV_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally update the consolidated CSV of stocks that traded "
            "below a target price on the last trading day."
        )
    )
    parser.add_argument("--max-price", type=float, default=MAX_PRICE)
    parser.add_argument("--history-period", type=str, default=HISTORY_PERIOD)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        max_price=args.max_price,
        history_period=args.history_period,
        batch_size=args.batch_size,
    )
