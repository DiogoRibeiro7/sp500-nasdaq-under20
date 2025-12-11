"""
Download NASDAQ + S&P 500 stocks that were below a given price threshold
on the last trading day (“yesterday”) and save their last year of history.

Requirements:
    pip install yfinance pandas

Notes:
- Uses yfinance built-in helpers for S&P500 and NASDAQ tickers.
- Filters tickers by close price < MAX_PRICE on the latest available date
  up to target_date (usually yesterday).
- Saves one CSV per ticker with 1 year of daily OHLCV data.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import yfinance as yf


# ----------------------------- Configuration ---------------------------------


MAX_PRICE: float = 20.0
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


def get_index_tickers() -> Dict[str, Set[str]]:
    """
    Fetch tickers for S&P 500 and NASDAQ via yfinance.

    Returns
    -------
    Dict[str, Set[str]]
        Dictionary with keys "sp500" and "nasdaq" and sets of tickers.
    """
    sp500_tickers: List[str] = yf.tickers_sp500()
    nasdaq_tickers: List[str] = yf.tickers_nasdaq()

    # Normalize to uppercase and strip spaces
    sp500: Set[str] = {t.strip().upper() for t in sp500_tickers if t.strip()}
    nasdaq: Set[str] = {t.strip().upper() for t in nasdaq_tickers if t.strip()}

    if not sp500:
        raise RuntimeError("Could not retrieve any S&P 500 tickers from yfinance.")
    if not nasdaq:
        raise RuntimeError("Could not retrieve any NASDAQ tickers from yfinance.")

    return {"sp500": sp500, "nasdaq": nasdaq}


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


def main() -> None:
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

    index_tickers = get_index_tickers()
    sp500 = index_tickers["sp500"]
    nasdaq = index_tickers["nasdaq"]

    all_universe: Set[str] = sp500.union(nasdaq)
    print(f"Total unique tickers in universe (S&P500 + NASDAQ): {len(all_universe)}")

    latest_closes: pd.DataFrame = get_latest_closes_for_universe(
        all_tickers=all_universe,
        target_date=as_of_date,
        batch_size=BATCH_SIZE,
    )

    print(f"Got latest closes for {latest_closes['ticker'].nunique()} tickers.")

    selected_tickers: List[str] = select_tickers_below_price(
        latest_closes=latest_closes,
        max_price=MAX_PRICE,
    )
    print(
        f"Number of tickers with close < {MAX_PRICE} USD on or before {as_of_date}: "
        f"{len(selected_tickers)}"
    )

    if not selected_tickers:
        print("No tickers met the price criterion. Exiting.")
        return

    history: Dict[str, pd.DataFrame] = download_history_for_tickers(
        tickers=selected_tickers,
        period=HISTORY_PERIOD,
        batch_size=BATCH_SIZE,
    )

    if not history:
        print("No historical data downloaded for selected tickers. Exiting.")
        return

    save_history_to_csv(
        history=history,
        output_dir=OUTPUT_DIR,
        as_of_date=as_of_date,
    )

    print("Done.")


if __name__ == "__main__":
    main()
