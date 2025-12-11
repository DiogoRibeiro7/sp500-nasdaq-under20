"""
Incrementally update a single CSV with daily history of all
S&P500 + NASDAQ (when available) stocks that were < 20 USD on "yesterday".

- Uses helper functions from scripts.under20_stocks
- Maintains a single CSV file at data/under20_history.csv
- On each run:
    1. Compute "yesterday" (UTC).
    2. Find all tickers with last close < 20 USD on or before yesterday.
    3. Download 1 year of daily history for those tickers.
    4. Fetch human-readable company names from yfinance.
    5. Append missing rows (with names) to the master CSV (no duplicates).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

from under20_stocks import (
    HISTORY_PERIOD,
    MAX_PRICE,
    download_history_for_tickers,
    get_index_tickers_and_names,
    get_latest_closes_for_universe,
    get_yesterday_date,
    select_tickers_below_price,
)

MASTER_CSV_PATH: Path = Path("data") / "under20_history.csv"


# ---------------------------------------------------------------------------
# Helpers for building / loading the master CSV
# ---------------------------------------------------------------------------


def _build_new_rows_dataframe(
    history: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Convert a dict[ticker -> DataFrame] into a flat DataFrame suitable for CSV.

    Each row has (when available):
        Date, Ticker, Open, High, Low, Close, Adj Close, Volume

    Parameters
    ----------
    history : Dict[str, pd.DataFrame]
        Mapping from ticker symbol to its OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per (Date, Ticker).
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

        # Normalize column names (yfinance naming)
        rename_map = {
            "Adj Close": "Adj Close",
            "Adj_Close": "Adj Close",
            "adjclose": "Adj Close",
        }
        for old, new in rename_map.items():
            if old in df_local.columns and new not in df_local.columns:
                df_local.rename(columns={old: new}, inplace=True)

        required_cols = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
        cols_to_keep = [c for c in required_cols if c in df_local.columns]
        df_local = df_local[cols_to_keep]

        df_local["Ticker"] = ticker
        records.append(df_local)

    if not records:
        # Return an empty frame with the expected columns
        return pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]
        )

    return pd.concat(records, ignore_index=True)


def _load_existing_master() -> pd.DataFrame:
    """
    Load existing master CSV if it exists, otherwise return an empty DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least columns ["Date", "Ticker", "Name"].
    """
    if not MASTER_CSV_PATH.exists():
        return pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "Name",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]
        )

    df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["Date"])

    # Backwards-compatibility: if an old CSV has no Name column, add it
    if "Name" not in df.columns:
        df["Name"] = pd.NA

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
        New rows to be added. Must already contain a Name column.

    Returns
    -------
    pd.DataFrame
        Combined and deduplicated dataset.
    """
    if new_rows.empty:
        return existing

    combined = pd.concat([existing, new_rows], ignore_index=True)

    combined["Date"] = pd.to_datetime(combined["Date"])

    # Drop duplicates based on (Date, Ticker) only.
    # We keep the first occurrence (typically the existing row).
    combined = combined.drop_duplicates(subset=["Date", "Ticker"])

    # Sort for readability
    combined = combined.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Helpers for fetching company names from yfinance
# ---------------------------------------------------------------------------


def _fetch_ticker_names(tickers: List[str]) -> Dict[str, str]:
    """
    Fetch human-readable company names for each ticker using yfinance.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols.

    Returns
    -------
    Dict[str, str]
        Mapping ticker -> company name (or ticker itself if unknown).
    """
    names: Dict[str, str] = {}

    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            info: Dict[str, object] = ticker_obj.info  # may trigger a network call

            name = info.get("shortName") or info.get("longName")
            if not isinstance(name, str) or not name.strip():
                name = ticker  # fallback: use ticker as name

            names[ticker] = name.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not fetch name for {ticker}: {exc}")
            names[ticker] = ticker

    return names


def _attach_names_to_rows(
    rows: pd.DataFrame,
    ticker_names: Dict[str, str],
) -> pd.DataFrame:
    """
    Add a Name column to the rows DataFrame based on a ticker->name mapping.

    Parameters
    ----------
    rows : pd.DataFrame
        DataFrame with at least a 'Ticker' column.
    ticker_names : Dict[str, str]
        Mapping from ticker to human-readable company name.

    Returns
    -------
    pd.DataFrame
        Copy of rows with an extra 'Name' column.
    """
    if rows.empty:
        rows["Name"] = pd.Series(dtype="string")
        return rows

    rows = rows.copy()
    rows["Name"] = rows["Ticker"].map(ticker_names).fillna(rows["Ticker"])
    return rows


# ---------------------------------------------------------------------------
# Main incremental update pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Main incremental update pipeline.

    Steps
    -----
    1. Compute "yesterday" (UTC).
    2. Gather S&P 500 and NASDAQ tickers (when NASDAQ is reachable).
    3. Retrieve latest close price up to yesterday for all tickers.
    4. Select tickers with close < MAX_PRICE.
    5. Download 1 year of daily history for selected tickers.
    6. Fetch ticker names from yfinance.
    7. Convert to flat DataFrame, attach names, and merge into master CSV.
    """
    as_of_date: dt.date = get_yesterday_date()
    print(f"[INFO] Using target date (yesterday): {as_of_date.isoformat()}")
    print(f"[INFO] History period: {HISTORY_PERIOD}, max price: {MAX_PRICE} USD")

    tickers_by_index, ticker_name_map = get_index_tickers_and_names()
    sp500 = tickers_by_index["sp500"]
    nasdaq = tickers_by_index["nasdaq"]  # may be empty if NASDAQ fetch failed

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

    # Download historical OHLCV data
    history = download_history_for_tickers(
        tickers=selected_tickers,
        period=HISTORY_PERIOD,
    )

    if not history:
        print("[WARN] No historical data downloaded. Nothing to update.")
        return

    # Convert dict[ticker -> DataFrame] into a long DataFrame
    new_rows = _build_new_rows_dataframe(history)
    print(f"[INFO] New rows collected: {len(new_rows)}")

    # Build name mapping only for the selected tickers
    ticker_names_for_selected = {
        t: ticker_name_map.get(t, t) for t in selected_tickers
    }
    new_rows = _attach_names_to_rows(new_rows, ticker_names_for_selected)

    existing = _load_existing_master()
    print(f"[INFO] Existing master rows: {len(existing)}")

    updated = _merge_and_deduplicate(existing, new_rows)
    print(f"[INFO] Updated master rows after dedupe: {len(updated)}")

    MASTER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Reorder columns for nicer CSV
    desired_order = [
        "Date",
        "Ticker",
        "Name",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    cols_in_df = [c for c in desired_order if c in updated.columns]
    updated = updated[cols_in_df]

    updated.to_csv(MASTER_CSV_PATH, index=False)
    print(f"[INFO] Master CSV updated at: {MASTER_CSV_PATH}")


if __name__ == "__main__":
    main()
