"""
Pandera schemas for the sp500-nasdaq-under20 data pipeline.

Two schemas are defined:

``LATEST_CLOSES_SCHEMA``
    Validates the intermediate DataFrame produced by
    ``get_latest_closes_for_universe`` before the price filter is applied.

``MASTER_CSV_SCHEMA``
    Validates the final DataFrame that is written to
    ``data/under20_history.csv``.  Applied both before writing (to catch
    bad new rows early) and after reading (to detect file corruption or
    schema drift introduced by a manual edit).

Usage
-----
::

    from schemas import validate_latest_closes, validate_master_csv

    closes = get_latest_closes_for_universe(...)
    closes = validate_latest_closes(closes)          # raises on violation

    updated = _merge_and_deduplicate(existing, new_rows)
    updated = validate_master_csv(updated)           # raises on violation
    updated.to_csv(MASTER_CSV_PATH, index=False)

Both helpers return the (possibly coerced) DataFrame on success so they can
be used inline without a separate assignment.

Installing pandera
------------------
Add ``pandera>=0.18`` to ``requirements.txt``.  It is a pure-Python package
with no compiled extensions, so it installs cleanly in the GitHub Actions
environment.
"""

from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd


# ---------------------------------------------------------------------------
# Schema: latest closes (intermediate, one row per ticker)
# ---------------------------------------------------------------------------

LATEST_CLOSES_SCHEMA = DataFrameSchema(
    columns={
        "ticker": Column(
            str,
            checks=[
                Check(lambda s: s.str.len() > 0, element_wise=False,
                      error="ticker column must not contain empty strings"),
                Check(lambda s: ~s.str.contains(r"\s", regex=True), element_wise=False,
                      error="ticker symbols must not contain whitespace"),
            ],
            nullable=False,
        ),
        "date": Column(
            pa.DateTime,
            checks=Check(lambda s: s.notna().all(), error="date must not be null"),
            nullable=False,
            coerce=True,
        ),
        "close": Column(
            float,
            checks=[
                Check(lambda s: (s > 0).all(), error="close price must be positive"),
                Check(lambda s: (s < 1_000_000).all(), error="close price suspiciously large"),
            ],
            nullable=False,
            coerce=True,
        ),
    },
    strict=False,   # allow extra columns without raising
    coerce=True,
    name="LatestClosesSchema",
)


# ---------------------------------------------------------------------------
# Schema: master CSV (written to data/under20_history.csv)
# ---------------------------------------------------------------------------

MASTER_CSV_SCHEMA = DataFrameSchema(
    columns={
        "Date": Column(
            pa.DateTime,
            nullable=False,
            coerce=True,
        ),
        "Ticker": Column(
            str,
            checks=Check(lambda s: s.str.len() > 0, element_wise=False,
                         error="Ticker must not be empty"),
            nullable=False,
        ),
        "Name": Column(
            str,
            nullable=True,   # may be NA on very first run for some tickers
            coerce=True,
        ),
        "Open": Column(
            float,
            checks=[
                Check(lambda s: (s[s.notna()] > 0).all(), error="Open must be positive"),
            ],
            nullable=True,   # yfinance occasionally returns NaN for illiquid tickers
            coerce=True,
        ),
        "High": Column(
            float,
            checks=Check(lambda s: (s[s.notna()] > 0).all(), error="High must be positive"),
            nullable=True,
            coerce=True,
        ),
        "Low": Column(
            float,
            checks=Check(lambda s: (s[s.notna()] > 0).all(), error="Low must be positive"),
            nullable=True,
            coerce=True,
        ),
        "Close": Column(
            float,
            checks=[
                Check(lambda s: (s[s.notna()] > 0).all(), error="Close must be positive"),
                Check(lambda s: (s[s.notna()] < 1_000_000).all(),
                      error="Close price suspiciously large"),
            ],
            nullable=True,
            coerce=True,
        ),
        "Adj Close": Column(
            float,
            checks=Check(lambda s: (s[s.notna()] > 0).all(), error="Adj Close must be positive"),
            nullable=True,
            coerce=True,
        ),
        "Volume": Column(
            float,
            checks=Check(lambda s: (s[s.notna()] >= 0).all(), error="Volume must be non-negative"),
            nullable=True,
            coerce=True,
        ),
    },
    checks=[
        # No duplicate (Date, Ticker) pairs — the merge step should guarantee
        # this but we verify it explicitly before writing.
        Check(
            lambda df: ~df.duplicated(subset=["Date", "Ticker"]).any(),
            error="Master CSV must not contain duplicate (Date, Ticker) pairs",
        ),
    ],
    strict=False,
    coerce=True,
    name="MasterCsvSchema",
)


# ---------------------------------------------------------------------------
# Convenience wrappers with informative error messages
# ---------------------------------------------------------------------------


def validate_latest_closes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate *df* against ``LATEST_CLOSES_SCHEMA``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``get_latest_closes_for_universe``.

    Returns
    -------
    pd.DataFrame
        The validated (and possibly coerced) DataFrame.

    Raises
    ------
    pandera.errors.SchemaError
        If any column is missing, has the wrong dtype, or fails a check.
    """
    try:
        return LATEST_CLOSES_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise pa.errors.SchemaErrors(
            schema=exc.schema,
            schema_errors=exc.schema_errors,
            data=exc.data,
        ) from exc


def validate_master_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate *df* against ``MASTER_CSV_SCHEMA``.

    Call this immediately before writing to ``data/under20_history.csv``
    and immediately after reading it back, so schema drift is detected at
    the earliest possible point.

    Parameters
    ----------
    df : pd.DataFrame
        The merged master dataset.

    Returns
    -------
    pd.DataFrame
        The validated (and possibly coerced) DataFrame.

    Raises
    ------
    pandera.errors.SchemaError
        If any column is missing, has the wrong dtype, or fails a check.
    """
    try:
        return MASTER_CSV_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise pa.errors.SchemaErrors(
            schema=exc.schema,
            schema_errors=exc.schema_errors,
            data=exc.data,
        ) from exc
