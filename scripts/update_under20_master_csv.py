"""
Incrementally update a single CSV with daily history of all
S&P500 + NASDAQ (when available) stocks that were under a configurable
price cap on "yesterday".

- Uses helper functions from scripts.under20_stocks
- Maintains a single CSV file at data/under20_history.csv
- On each run:
    1. Compute the last trading day (UTC).
    2. Find all tickers with last close < MAX_PRICE on or before that day.
    3. Download 1 year of daily history for those tickers.
    4. Resolve human-readable company names (Wikipedia first, yfinance fallback).
    5. Append missing rows (with names) to the master CSV (no duplicates).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Path fix: ensure this script can import from its own directory regardless
# of the working directory from which it is invoked.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

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

MASTER_CSV_PATH: Path = Path("data") / "under20_history.csv"

# Maximum tickers to pass to yf.Tickers in a single call.
# yfinance spawns one thread per ticker inside Tickers.download(), so keeping
# this below ~50 avoids hammering the Yahoo Finance API and triggering
# rate-limit responses.
_NAME_BATCH_SIZE: int = 50


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

        # Normalize column names (yfinance naming can vary across versions)
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
    """
    Load existing master CSV if it exists, otherwise return an empty DataFrame.
    """
    if not MASTER_CSV_PATH.exists():
        return pd.DataFrame(
            columns=["Date", "Ticker", "Name", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        )

    df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["Date"])

    if "Name" not in df.columns:
        df["Name"] = pd.NA

    return df


def _merge_and_deduplicate(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge existing and new rows, dropping duplicates by (Date, Ticker).
    Rows that carry a company name are preferred over those that don't.
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
    """
    Fetch company names from yfinance for a list of tickers.

    Uses ``yf.Tickers`` so that yfinance can parallelise the underlying HTTP
    requests internally, instead of issuing one blocking call per ticker.
    Falls back gracefully: if an individual ticker's ``.info`` dict is missing
    or raises, the ticker symbol itself is used as the name.

    Parameters
    ----------
    tickers : List[str]
        Ticker symbols to look up.
    batch_size : int
        Maximum tickers per ``yf.Tickers`` call.  Keeping this below ~50
        avoids Yahoo Finance rate-limiting.

    Returns
    -------
    Dict[str, str]
        Mapping ticker -> company name.
    """
    names: Dict[str, str] = {}

    # Chunk the list so we never hand yfinance an unbounded number of tickers
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        ticker_str = " ".join(batch)

        try:
            tickers_obj = yf.Tickers(ticker_str)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] yf.Tickers() failed for batch starting at index {i}: {exc}")
            for t in batch:
                names.setdefault(t, t)
            continue

        for symbol in batch:
            try:
                info: dict = tickers_obj.tickers[symbol].info
                name = info.get("shortName") or info.get("longName")
                names[symbol] = name.strip() if isinstance(name, str) and name.strip() else symbol
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Could not fetch name for {symbol}: {exc}")
                names[symbol] = symbol

    return names


def resolve_ticker_names(
    tickers: List[str],
    known_names: Dict[str, str],
    batch_size: int = _NAME_BATCH_SIZE,
) -> Dict[str, str]:
    """
    Build a complete ticker -> name mapping for *tickers*.

    Strategy (in priority order):
    1. Use the name already present in *known_names* (sourced from Wikipedia
       during the index-ticker fetch).  This covers the vast majority of
       S&P 500 and NASDAQ-100 names with zero extra network calls.
    2. For any ticker whose Wikipedia name is missing or is just the symbol
       itself, fall back to a batched yfinance lookup via ``_fetch_names_from_yfinance``.
    3. If yfinance also fails, use the ticker symbol as the name.

    Parameters
    ----------
    tickers : List[str]
        All tickers that need a name.
    known_names : Dict[str, str]
        Name map already populated from Wikipedia (ticker -> name).
    batch_size : int
        Passed through to ``_fetch_names_from_yfinance``.

    Returns
    -------
    Dict[str, str]
        Complete mapping ticker -> name for every ticker in *tickers*.
    """
    result: Dict[str, str] = {}
    needs_lookup: List[str] = []

    for t in tickers:
        wiki_name = known_names.get(t, "").strip()
        if wiki_name and wiki_name != t:
            # Wikipedia already gave us a real company name — use it.
            result[t] = wiki_name
        else:
            needs_lookup.append(t)

    if needs_lookup:
        print(
            f"[INFO] Fetching names from yfinance for {len(needs_lookup)} tickers "
            f"not covered by Wikipedia index data."
        )
        yf_names = _fetch_names_from_yfinance(needs_lookup, batch_size=batch_size)
        result.update(yf_names)
    else:
        print("[INFO] All ticker names resolved from Wikipedia index data — skipping yfinance name fetch.")

    return result


def _attach_names_to_rows(
    rows: pd.DataFrame,
    ticker_names: Dict[str, str],
) -> pd.DataFrame:
    """
    Add a Name column to *rows* based on *ticker_names*.
    """
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
    3. Retrieve latest close price up to that day for all tickers.
    4. Select tickers with close < max_price.
    5. Download 1 year of daily history for selected tickers.
    6. Resolve company names (Wikipedia first, batched yfinance fallback).
    7. Convert to flat DataFrame, attach names, and merge into master CSV.
    """
    if max_price <= 0:
        raise ValueError("max_price must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    as_of_date: dt.date = get_yesterday_date()
    print(f"[INFO] Using last trading day: {as_of_date.isoformat()}")
    print(f"[INFO] History period: {history_period}, max price: {max_price} USD")

    tickers_by_index, ticker_name_map = get_index_tickers_and_names()
    sp500 = tickers_by_index["sp500"]
    nasdaq = tickers_by_index["nasdaq"]

    all_universe = sp500.union(nasdaq)
    print(f"[INFO] Total unique tickers in universe: {len(all_universe)}")

    latest_closes = get_latest_closes_for_universe(
        all_tickers=all_universe,
        target_date=as_of_date,
        batch_size=batch_size,
    )
    print(f"[INFO] Got latest closes for {latest_closes['ticker'].nunique()} tickers.")

    selected_tickers = select_tickers_below_price(
        latest_closes=latest_closes,
        max_price=max_price,
    )
    print(
        f"[INFO] Tickers with close < {max_price} USD on or before {as_of_date}: "
        f"{len(selected_tickers)}"
    )

    if not selected_tickers:
        print("[WARN] No tickers met the price criterion. Nothing to update.")
        return

    history = download_history_for_tickers(
        tickers=selected_tickers,
        period=history_period,
        batch_size=batch_size,
    )

    if not history:
        print("[WARN] No historical data downloaded. Nothing to update.")
        return

    new_rows = _build_new_rows_dataframe(history)
    print(f"[INFO] New rows collected: {len(new_rows)}")

    # Resolve names: Wikipedia covers most; yfinance fills the gaps in batches.
    ticker_names = resolve_ticker_names(
        tickers=selected_tickers,
        known_names=ticker_name_map,
        batch_size=_NAME_BATCH_SIZE,
    )
    new_rows = _attach_names_to_rows(new_rows, ticker_names)

    existing = _load_existing_master()
    print(f"[INFO] Existing master rows: {len(existing)}")

    updated = _merge_and_deduplicate(existing, new_rows)
    print(f"[INFO] Updated master rows after dedupe: {len(updated)}")

    MASTER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    desired_order = ["Date", "Ticker", "Name", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols_in_df = [c for c in desired_order if c in updated.columns]
    updated = updated[cols_in_df]

    updated.to_csv(MASTER_CSV_PATH, index=False)
    print(f"[INFO] Master CSV updated at: {MASTER_CSV_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally update the consolidated CSV of stocks that traded "
            "below a target price on the last trading day."
        )
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=MAX_PRICE,
        help="Close price ceiling (USD). Default: %(default)s.",
    )
    parser.add_argument(
        "--history-period",
        type=str,
        default=HISTORY_PERIOD,
        help="yfinance period string for historical downloads. Default: %(default)s.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Tickers per yfinance batch request. Default: %(default)s.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        max_price=args.max_price,
        history_period=args.history_period,
        batch_size=args.batch_size,
    )

