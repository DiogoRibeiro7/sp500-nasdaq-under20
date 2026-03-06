"""
Append-only pipeline run log.

After each successful (or gracefully-skipped) pipeline run, a single row is
appended to ``data/run_log.csv`` recording what happened.  The file is created
on first use.

Schema
------
run_ts          ISO-8601 UTC timestamp of when the pipeline ran.
as_of_date      The last trading day used as the price-filter target.
universe_size   Total unique tickers in S&P 500 + NASDAQ-100 universe.
tickers_found   Number of tickers with close < max_price on as_of_date.
new_rows_added  Rows appended to the master CSV this run (0 if nothing changed).
total_rows      Total rows in the master CSV after this run.
max_price       The price ceiling used for this run.
status          "ok" | "no_tickers" | "no_history" | "validation_error"

Usage
-----
::

    from run_logger import log_run

    log_run(
        as_of_date=as_of_date,
        universe_size=len(all_universe),
        tickers_found=len(selected_tickers),
        new_rows_added=len(updated) - len(existing),
        total_rows=len(updated),
        max_price=max_price,
        status="ok",
    )
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from config import RUN_LOG_PATH

_COLUMNS = [
    "run_ts",
    "as_of_date",
    "universe_size",
    "tickers_found",
    "new_rows_added",
    "total_rows",
    "max_price",
    "status",
]


def log_run(
    as_of_date: dt.date,
    universe_size: int,
    tickers_found: int,
    new_rows_added: int,
    total_rows: int,
    max_price: float,
    status: str,
    path: Path = RUN_LOG_PATH,
) -> None:
    """
    Append one row to the pipeline run log.

    Parameters
    ----------
    as_of_date : dt.date
        Last trading day used as the price-filter target.
    universe_size : int
        Total unique tickers in the S&P 500 + NASDAQ-100 universe.
    tickers_found : int
        Tickers with close < max_price on as_of_date.
    new_rows_added : int
        Rows appended to the master CSV this run.
    total_rows : int
        Total rows in the master CSV after this run.
    max_price : float
        Price ceiling used for this run.
    status : str
        One of: "ok", "no_tickers", "no_history", "validation_error".
    path : Path
        Destination file.  Defaults to ``RUN_LOG_PATH`` from config.
    """
    row = {
        "run_ts":        dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date":    as_of_date.isoformat(),
        "universe_size": universe_size,
        "tickers_found": tickers_found,
        "new_rows_added": new_rows_added,
        "total_rows":    total_rows,
        "max_price":     max_price,
        "status":        status,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode: write header only if the file does not yet exist.
    write_header = not path.exists()
    pd.DataFrame([row], columns=_COLUMNS).to_csv(
        path,
        mode="a",
        index=False,
        header=write_header,
    )


def load_run_log(path: Path = RUN_LOG_PATH) -> pd.DataFrame:
    """
    Load the full run log as a DataFrame.

    Returns an empty DataFrame with the correct columns if the file does
    not yet exist.

    Parameters
    ----------
    path : Path
        Source file.  Defaults to ``RUN_LOG_PATH`` from config.

    Returns
    -------
    pd.DataFrame
    """
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)

    return pd.read_csv(path, parse_dates=["run_ts", "as_of_date"])
