"""
Download NASDAQ-100 + S&P 500 stocks that were below a configurable price
threshold on the last trading day ("yesterday") and save their last year of
history.

Requirements:
    pip install yfinance pandas

Notes:
- Uses yfinance built-in helpers for S&P500 and NASDAQ tickers.
- Filters tickers by close price < MAX_PRICE on the latest available date
  up to target_date (usually yesterday). The default max price can be
  overridden via CLI arguments.
- Saves one CSV per ticker with 1 year of daily OHLCV data.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from io import StringIO
from typing import Dict, Iterable, List, Optional, Set, Tuple
import requests
from requests.exceptions import RequestException, Timeout

import pandas as pd
import yfinance as yf

NASDAQ_CACHE_PATH = Path("data") / "nasdaq_tickers.csv"

# ----------------------------- Configuration ---------------------------------


MAX_PRICE: float = 30.0
# Number of tickers per batch when calling yf.download
BATCH_SIZE: int = 150
# Directory where we will save all CSV files
OUTPUT_DIR: Path = Path("data_under_20")
# How many days of history to download (approx. 1 year)
HISTORY_PERIOD: str = "1y"


# ----------------------------- Helper functions ------------------------------


def get_yesterday_date() -> dt.date:
    """
    Return "yesterday" as a calendar date in UTC.

    This is a simple calendar-based definition. The code later finds the
    latest available trading date <= this date for each ticker.
    """
    today_utc: dt.date = dt.datetime.utcnow().date()
    return today_utc - dt.timedelta(days=1)


def _fetch_sp500_from_wikipedia() -> Tuple[List[str], Dict[str, str]]:
    """
    Fetch S&P 500 tickers and company names from Wikipedia.

    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        (list of tickers, mapping ticker -> company name)
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError("Could not read S&P 500 table from Wikipedia HTML.")

    df = tables[0]
    if "Symbol" not in df.columns:
        raise RuntimeError("Wikipedia S&P 500 table has no 'Symbol' column.")
    # Column for name is usually 'Security'
    name_col = "Security" if "Security" in df.columns else None
    if name_col is None:
        raise RuntimeError("Wikipedia S&P 500 table has no 'Security' column for names.")

    symbols = (
        df["Symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )
    names = (
        df[name_col]
        .astype(str)
        .str.strip()
        .tolist()
    )

    ticker_to_name = {sym: nm for sym, nm in zip(symbols, names)}
    return symbols, ticker_to_name


def _load_cached_nasdaq_tickers() -> Tuple[List[str], Dict[str, str]]:
    """
    Load cached NASDAQ tickers (and optional names) from NASDAQ_CACHE_PATH.

    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        (list of ticker symbols, mapping ticker -> company name) or empty values
        if the cache does not exist.
    """
    if not NASDAQ_CACHE_PATH.exists():
        return [], {}

    df = pd.read_csv(NASDAQ_CACHE_PATH)
    if "Symbol" not in df.columns:
        raise RuntimeError(
            f"Cached NASDAQ tickers file {NASDAQ_CACHE_PATH} has no 'Symbol' column."
        )

    symbols = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )

    if "Security Name" in df.columns:
        names = (
            df["Security Name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )
    else:
        names = symbols

    ticker_to_name = {sym: nm for sym, nm in zip(symbols, names)}
    return symbols, ticker_to_name


def _save_cached_nasdaq_tickers(symbols: List[str], ticker_to_name: Dict[str, str]) -> None:
    """
    Persist NASDAQ ticker symbols and names to NASDAQ_CACHE_PATH.
    """
    if not symbols:
        return

    NASDAQ_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "Symbol": symbols,
            "Security Name": [ticker_to_name.get(sym, "") for sym in symbols],
        }
    )
    df.to_csv(NASDAQ_CACHE_PATH, index=False)


def _fetch_nasdaq100_from_wikipedia() -> Tuple[List[str], Dict[str, str]]:
    """
    Fetch NASDAQ-100 tickers and company names from Wikipedia.

    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        (list of tickers, mapping ticker -> company name)
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Timeout as exc:
        raise RuntimeError("Timeout fetching NASDAQ-100 tickers from Wikipedia") from exc
    except RequestException as exc:
        raise RuntimeError(f"Error fetching NASDAQ-100 tickers from Wikipedia: {exc}") from exc

    tables = pd.read_html(StringIO(resp.text))
    candidate_df: Optional[pd.DataFrame] = None
    for table in tables:
        if "Ticker" in table.columns:
            candidate_df = table.copy()
            break
    if candidate_df is None:
        raise RuntimeError("Could not locate NASDAQ-100 table on Wikipedia page.")

    name_column: Optional[str] = None
    for col in ["Company", "Company Name", "Security", "Name"]:
        if col in candidate_df.columns:
            name_column = col
            break
    if name_column is None:
        raise RuntimeError("NASDAQ-100 table missing company name column.")

    candidate_df = candidate_df[candidate_df["Ticker"].notna()].copy()
    candidate_df["Ticker"] = candidate_df["Ticker"].astype(str).str.upper().str.strip()
    candidate_df[name_column] = (
        candidate_df[name_column].fillna("").astype(str).str.strip()
    )

    symbols = candidate_df["Ticker"].tolist()
    names = candidate_df[name_column].tolist()
    ticker_to_name = {sym: nm for sym, nm in zip(symbols, names)}
    return symbols, ticker_to_name


def get_index_tickers_and_names() -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Fetch tickers and company names for S&P 500 and NASDAQ.

    Returns
    -------
    Tuple[Dict[str, Set[str]], Dict[str, str]]
        (
            {
              "sp500": {ticker, ...},
              "nasdaq": {ticker, ...}
            },
            {ticker: name, ...}
        )
    """
    # ---- S&P 500 ----
    print("[INFO] Fetching S&P 500 tickers from Wikipedia...")
    sp500_list, sp500_name_map = _fetch_sp500_from_wikipedia()
    print(f"[INFO] Retrieved {len(sp500_list)} S&P 500 tickers.")

    # ---- NASDAQ ----
    cached_nasdaq, cached_nasdaq_names = _load_cached_nasdaq_tickers()
    print("[INFO] Fetching NASDAQ-100 tickers from Wikipedia...")
    try:
        nasdaq_list, nasdaq_name_map = _fetch_nasdaq100_from_wikipedia()
        print(f"[INFO] Retrieved {len(nasdaq_list)} NASDAQ-100 tickers.")
        _save_cached_nasdaq_tickers(nasdaq_list, nasdaq_name_map)
    except RuntimeError as exc:
        print(f"[WARN] Could not fetch NASDAQ-100 tickers: {exc}")
        if cached_nasdaq:
            print(
                f"[INFO] Using {len(cached_nasdaq)} cached NASDAQ tickers from "
                f"{NASDAQ_CACHE_PATH}"
            )
            nasdaq_list, nasdaq_name_map = cached_nasdaq, cached_nasdaq_names
        else:
            print("[WARN] Continuing with S&P 500 universe only.")
            nasdaq_list, nasdaq_name_map = [], {}

    tickers_by_index: Dict[str, Set[str]] = {
        "sp500": {t for t in sp500_list if t},
        "nasdaq": {t for t in nasdaq_list if t},
    }

    ticker_to_name: Dict[str, str] = {}
    ticker_to_name.update(sp500_name_map)
    ticker_to_name.update(nasdaq_name_map)

    if not tickers_by_index["sp500"]:
        raise RuntimeError("Could not retrieve any S&P 500 tickers.")

    return tickers_by_index, ticker_to_name


def get_index_tickers() -> Dict[str, Set[str]]:
    """
    Backwards-compatible wrapper that returns only ticker sets per index.
    """
    tickers_by_index, _ = get_index_tickers_and_names()
    return tickers_by_index


def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    """
    Yield successive chunks of the given iterable.

    Parameters
    ----------
    iterable : Iterable[str]
        Iterable of tickers.
    size : int
        Maximum chunk size.

    Yields
    ------
    List[str]
        List of tickers in each chunk.
    """
    chunk: List[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def get_latest_close_for_batch(
    tickers: List[str],
    target_date: dt.date,
) -> pd.DataFrame:
    """
    Download up to a few days of data and extract the latest close for each ticker
    up to target_date.

    Parameters
    ----------
    tickers : List[str]
        List of tickers to query (batch).
    target_date : dt.date
        Date representing "yesterday" (calendar). For each ticker we take
        the last available row with date <= target_date.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["ticker", "date", "close"], one row per ticker
        that has at least one observation up to target_date.
    """
    if not tickers:
        raise ValueError("Ticker list for batch is empty.")

    # We download a small window: last 7 calendar days should cover weekends/holidays
    # and still include the last trading day before or at target_date.
    start_date: dt.date = target_date - dt.timedelta(days=7)
    # Use interval=1d; group_by="ticker" yields a column structure per ticker
    data: pd.DataFrame = yf.download(
        tickers=tickers,
        start=start_date.isoformat(),
        end=(target_date + dt.timedelta(days=1)).isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    # Handle the case where we requested a single ticker (yfinance shape is different)
    records: List[Dict[str, object]] = []

    def _extract_for_single_ticker(
        df_single: pd.DataFrame, ticker: str
    ) -> Optional[Tuple[pd.Timestamp, float]]:
        """
        For a single-ticker DataFrame, find the last row with index <= target_date.
        Returns (date, close) or None if nothing is found.
        """
        if df_single.empty:
            return None

        # Ensure DateTimeIndex
        df_single = df_single.copy()
        df_single.index = pd.to_datetime(df_single.index)

        # Filter by date <= target_date
        mask = df_single.index.date <= target_date
        df_filtered = df_single.loc[mask]

        if df_filtered.empty:
            return None

        last_row = df_filtered.iloc[-1]
        last_date = df_filtered.index[-1]
        close_price = float(last_row["Close"])
        return last_date, close_price

    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker mode
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                # yfinance may fail silently for some symbols
                continue
            df_single = data[ticker].dropna(how="all")
            result = _extract_for_single_ticker(df_single, ticker)
            if result is not None:
                last_date, close_price = result
                records.append(
                    {
                        "ticker": ticker,
                        "date": last_date,
                        "close": close_price,
                    }
                )
    else:
        # Single-ticker mode
        if len(tickers) != 1:
            # Defensive check: this should not happen, but better to fail loudly.
            raise RuntimeError(
                "Unexpected single-ticker data shape for multiple tickers."
            )
        ticker = tickers[0]
        df_single = data.dropna(how="all")
        result = _extract_for_single_ticker(df_single, ticker)
        if result is not None:
            last_date, close_price = result
            records.append(
                {
                    "ticker": ticker,
                    "date": last_date,
                    "close": close_price,
                }
            )

    return pd.DataFrame.from_records(records)


def get_latest_closes_for_universe(
    all_tickers: Iterable[str],
    target_date: dt.date,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Get the latest close price up to target_date for an entire universe of tickers.

    Parameters
    ----------
    all_tickers : Iterable[str]
        All tickers from NASDAQ and S&P 500 (possibly with duplicates).
    target_date : dt.date
        Target calendar date to use as "yesterday".
    batch_size : int, optional
        Number of tickers per batch for yfinance, by default BATCH_SIZE.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["ticker", "date", "close"].
    """
    # Deduplicate tickers
    all_unique_tickers: List[str] = sorted({t for t in all_tickers if t})

    all_records: List[pd.DataFrame] = []
    for batch in chunked(all_unique_tickers, batch_size):
        try:
            batch_df = get_latest_close_for_batch(batch, target_date)
        except Exception as exc:  # noqa: BLE001
            # For robustness: log / print and continue with next batch
            print(f"Warning: batch failed for {len(batch)} tickers: {exc}")
            continue
        if not batch_df.empty:
            all_records.append(batch_df)

    if not all_records:
        raise RuntimeError("No price data could be retrieved for any ticker.")

    result: pd.DataFrame = pd.concat(all_records, ignore_index=True)
    return result


def select_tickers_below_price(
    latest_closes: pd.DataFrame,
    max_price: float = MAX_PRICE,
) -> List[str]:
    """
    Select tickers whose last close is strictly below max_price.

    Parameters
    ----------
    latest_closes : pd.DataFrame
        DataFrame with columns ["ticker", "date", "close"].
    max_price : float, optional
        Price threshold, by default MAX_PRICE.

    Returns
    -------
    List[str]
        List of tickers that satisfy close < max_price.
    """
    required_cols: Set[str] = {"ticker", "date", "close"}
    if not required_cols.issubset(set(latest_closes.columns)):
        raise ValueError(
            f"latest_closes must contain columns {required_cols}, "
            f"found {set(latest_closes.columns)}"
        )

    mask = latest_closes["close"] < max_price
    selected = latest_closes.loc[mask, "ticker"].dropna().unique().tolist()
    return sorted(selected)


def download_history_for_tickers(
    tickers: List[str],
    period: str = HISTORY_PERIOD,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, pd.DataFrame]:
    """
    Download historical daily data for a list of tickers.

    Parameters
    ----------
    tickers : List[str]
        List of tickers to download.
    period : str, optional
        Period argument passed to yfinance (e.g., "1y"), by default HISTORY_PERIOD.
    batch_size : int, optional
        Number of tickers per batch for yfinance, by default BATCH_SIZE.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping ticker -> DataFrame with OHLCV history.
    """
    ticker_to_df: Dict[str, pd.DataFrame] = {}

    for batch in chunked(tickers, batch_size):
        try:
            data = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: batch history download failed for {len(batch)} tickers: {exc}")
            continue

        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker shape: first level is ticker
            for ticker in batch:
                if ticker not in data.columns.get_level_values(0):
                    continue
                df_single = data[ticker].dropna(how="all")
                if df_single.empty:
                    continue
                df_single = df_single.copy()
                df_single.index = pd.to_datetime(df_single.index)
                ticker_to_df[ticker] = df_single
        else:
            # Single ticker shape
            if len(batch) != 1:
                raise RuntimeError(
                    "Unexpected single-ticker data shape for multiple tickers."
                )
            ticker = batch[0]
            df_single = data.dropna(how="all")
            if not df_single.empty:
                df_single = df_single.copy()
                df_single.index = pd.to_datetime(df_single.index)
                ticker_to_df[ticker] = df_single

    return ticker_to_df


def save_history_to_csv(
    history: Dict[str, pd.DataFrame],
    output_dir: Path,
    as_of_date: dt.date,
) -> None:
    """
    Save each ticker's history to a separate CSV file.

    File naming convention:
        {ticker}_history_until_{YYYYMMDD}.csv

    Parameters
    ----------
    history : Dict[str, pd.DataFrame]
        Mapping ticker -> DataFrame with history.
    output_dir : Path
        Directory where CSV files will be written.
    as_of_date : dt.date
        Date used in output file names (usually "yesterday").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str: str = as_of_date.strftime("%Y%m%d")

    for ticker, df in history.items():
        # Basic sanity check
        if df.empty:
            continue
        file_path: Path = output_dir / f"{ticker}_history_until_{date_str}.csv"
        df.to_csv(file_path, index_label="Date")
        print(f"Saved history for {ticker} to {file_path}")


# ------------------------------- Main script ---------------------------------


def main(
    max_price: float = MAX_PRICE,
    history_period: str = HISTORY_PERIOD,
    output_dir: Path = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Main entrypoint.

    Steps:
    1. Get S&P 500 and NASDAQ tickers.
    2. Compute "yesterday" date (UTC).
    3. Retrieve latest close price up to yesterday for all tickers.
    4. Filter tickers with close < MAX_PRICE.
    5. Download last year of daily history for those tickers.
    6. Save each ticker's history to CSV.
    """
    as_of_date: dt.date = get_yesterday_date()
    print(f"Using target date (yesterday): {as_of_date.isoformat()}")

    if max_price <= 0:
        raise ValueError("max_price must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    index_tickers = get_index_tickers()
    sp500 = index_tickers["sp500"]
    nasdaq = index_tickers["nasdaq"]

    all_universe: Set[str] = sp500.union(nasdaq)
    print(
        f"Total unique tickers in universe (S&P500 + NASDAQ-100): {len(all_universe)}"
    )

    latest_closes: pd.DataFrame = get_latest_closes_for_universe(
        all_tickers=all_universe,
        target_date=as_of_date,
        batch_size=batch_size,
    )

    print(f"Got latest closes for {latest_closes['ticker'].nunique()} tickers.")

    selected_tickers: List[str] = select_tickers_below_price(
        latest_closes=latest_closes,
        max_price=max_price,
    )
    print(
        f"Number of tickers with close < {max_price} USD on or before {as_of_date}: "
        f"{len(selected_tickers)}"
    )

    if not selected_tickers:
        print("No tickers met the price criterion. Exiting.")
        return

    history: Dict[str, pd.DataFrame] = download_history_for_tickers(
        tickers=selected_tickers,
        period=history_period,
        batch_size=batch_size,
    )

    if not history:
        print("No historical data downloaded for selected tickers. Exiting.")
        return

    save_history_to_csv(
        history=history,
        output_dir=output_dir,
        as_of_date=as_of_date,
    )

    print("Done.")


def parse_args() -> argparse.Namespace:
    """
    Build and parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download S&P 500 + NASDAQ-100 stocks below a target price and "
            "store one year of daily history per ticker."
        )
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=MAX_PRICE,
        help=(
            "Maximum close price (USD) a ticker must be under. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--history-period",
        type=str,
        default=HISTORY_PERIOD,
        help="yfinance period string for history downloads. Default: %(default)s.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to store CSVs. Default: %(default)s.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of tickers per yfinance batch request. Default: %(default)s.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        max_price=args.max_price,
        history_period=args.history_period,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
