"""
Incrementally update a single CSV with daily history of all
S&P500 + NASDAQ stocks that were < 20 USD on "yesterday".

- Uses helper functions from scripts.under20_stocks
- Maintains a single CSV file at data/under20_history.csv
- On each run:
    1. Compute "yesterday" (UTC).
    2. Find all tickers with last close < 20 USD on or before yesterday.
    3. Download 1 year of daily history for those tickers.
    4. Append missing rows to the master CSV (no duplicates).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List

import pandas as pd

from under20_stocks import (
    HISTORY_PERIOD,
    MAX_PRICE,
    download_history_for_tickers,
    get_index_tickers,
    get_latest_closes_for_universe,
    get_yesterday_date,
    select_tickers_below_price,
)

MASTER_CSV_PATH: Path = Path("data") / "under20_history.csv"


def _build_new_rows_dataframe(
    history: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Convert a dict[ticker -> DataFrame] into a flat DataFrame suitable for CSV.

    Each row has:
        Date, Ticker, Open, High, Low, Close, Adj Close, Volume
    """
    records: List[pd.DataFrame] = []

    for ticker, df in history.items():
        if df.empty:
            continue

        df_local = df.copy()
        df_local.index = pd.to_datetime(df_local.index)
        df_local.reset_index(inplace=True)
        df_local.rename(columns={"index": "Date"}, inplace=True)

        # Normalize column names (yfinance naming)
        rename_map = {
            "Adj Close": "Adj Close",
            "Adj_Close": "Adj Close",
            "adjclose": "Adj Close",
        }
        # This is defensive: we only rename if the column exists
        for old, new in rename_map.items():
            if old in df_local.columns and new not in df_local.columns:
                df_local.rename(columns={old: new}, inplace=True)

        required_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        # Keep only required columns that exist
        cols_to_keep = [c for c in required_cols if c in df_local.columns]
        df_local = df_local[cols_to_keep]

        df_local["Ticker"] = ticker
        records.append(df_local)

    if not records:
        return pd.DataFrame(
            columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]  # type: ignore[list-item]
        )

    return pd.concat(records, ignore_index=True)


def _load_existing_master() -> pd.DataFrame:
    """
    Load existing master CSV if it exists, otherwise return an empty DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least columns ["Date", "Ticker"].
    """
    if not MASTER_CSV_PATH.exists():
        return pd.DataFrame(
            columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]  # type: ignore[list-item]
        )

    df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["Date"])
    return df


def _merge_and_deduplicate(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge existing and new rows, dropping duplicates by (Date, Ticker).

    Parameters
    ----------
    existing : pd.DataFrame
        Existing master dataset.
    new_rows : pd.DataFrame
        New rows to be added.

    Returns
    -------
    pd.DataFrame
        Combined and deduplicated dataset.
    """
    if new_rows.empty:
        return existing

    combined = pd.concat([existing, new_rows], ignore_index=True)

    # Ensure Date is datetime
    combined["Date"] = pd.to_datetime(combined["Date"])

    # Drop duplicate rows based on (Date, Ticker)
    combined = combined.drop_duplicates(subset=["Date", "Ticker"])

    # Sort for readability
    combined = combined.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)
    return combined


def main() -> None:
    """
    Main incremental update pipeline.

    Steps
    -----
    1. Compute "yesterday" (UTC).
    2. Gather S&P 500 and NASDAQ tickers.
    3. Retrieve latest close price up to yesterday for all tickers.
    4. Select tickers with close < MAX_PRICE.
    5. Download 1 year of daily history for selected tickers.
    6. Convert to flat DataFrame and merge into master CSV, deduplicating rows.
    """
    as_of_date: dt.date = get_yesterday_date()
    print(f"[INFO] Using target date (yesterday): {as_of_date.isoformat()}")
    print(f"[INFO] History period: {HISTORY_PERIOD}, max price: {MAX_PRICE} USD")

    index_tickers = get_index_tickers()
    sp500 = index_tickers["sp500"]
    nasdaq = index_tickers["nasdaq"]

    all_universe = sp500.union(nasdaq)
    print(f"[INFO] Total unique tickers in universe: {len(all_universe)}")

    latest_closes = get_latest_closes_for_universe(
        all_tickers=all_universe,
        target_date=as_of_date,
    )
    print(f"[INFO] Got latest closes for {latest_closes['ticker'].nunique()} tickers.")

    selected_tickers = select_tickers_below_price(
        latest_closes=latest_closes,
        max_price=MAX_PRICE,
    )
    print(
        f"[INFO] Tickers with close < {MAX_PRICE} USD on or before {as_of_date}: "
        f"{len(selected_tickers)}"
    )

    if not selected_tickers:
        print("[WARN] No tickers met the price criterion. Nothing to update.")
        return

    history = download_history_for_tickers(
        tickers=selected_tickers,
        period=HISTORY_PERIOD,
    )

    if not history:
        print("[WARN] No historical data downloaded. Nothing to update.")
        return

    new_rows = _build_new_rows_dataframe(history)
    print(f"[INFO] New rows collected: {len(new_rows)}")

    existing = _load_existing_master()
    print(f"[INFO] Existing master rows: {len(existing)}")

    updated = _merge_and_deduplicate(existing, new_rows)
    print(f"[INFO] Updated master rows after dedupe: {len(updated)}")

    MASTER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(MASTER_CSV_PATH, index=False)
    print(f"[INFO] Master CSV updated at: {MASTER_CSV_PATH}")


if __name__ == "__main__":
    main()
