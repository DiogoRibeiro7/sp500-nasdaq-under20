"""
Tests for schemas.py — pandera schema validation.

These tests verify that:
  - Valid data passes without error.
  - Specific violations (wrong dtype, negative price, duplicate keys, missing
    columns) are caught and raise SchemaErrors.

No network calls, no yfinance, no file I/O.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera.pandas
import pandera.errors
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schemas import validate_latest_closes, validate_master_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_closes(n: int = 3) -> pd.DataFrame:
    tickers = [f"T{i:02d}" for i in range(n)]
    return pd.DataFrame({
        "ticker": tickers,
        "date": pd.date_range("2024-01-02", periods=n, freq="D"),
        "close": [10.0 + i for i in range(n)],
    })


def _valid_master(n: int = 3) -> pd.DataFrame:
    tickers = [f"T{i:02d}" for i in range(n)]
    dates = pd.date_range("2024-01-02", periods=n, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Ticker": tickers,
        "Name": [f"Company {i}" for i in range(n)],
        "Open":      [9.0] * n,
        "High":      [11.0] * n,
        "Low":       [8.5] * n,
        "Close":     [10.0 + i for i in range(n)],
        "Adj Close": [10.0 + i for i in range(n)],
        "Volume":    [1_000_000.0] * n,
    })


# ---------------------------------------------------------------------------
# validate_latest_closes — happy path
# ---------------------------------------------------------------------------


class TestValidateLatestClosesValid:
    def test_valid_dataframe_passes(self):
        df = _valid_closes()
        result = validate_latest_closes(df)
        assert len(result) == 3

    def test_extra_columns_allowed(self):
        df = _valid_closes()
        df["extra"] = "ignored"
        result = validate_latest_closes(df)
        assert "extra" in result.columns

    def test_coerces_close_to_float(self):
        df = _valid_closes()
        df["close"] = df["close"].astype(int)
        result = validate_latest_closes(df)
        assert result["close"].dtype == float


# ---------------------------------------------------------------------------
# validate_latest_closes — violations
# ---------------------------------------------------------------------------


class TestValidateLatestClosesInvalid:
    def test_negative_close_raises(self):
        df = _valid_closes()
        df.loc[0, "close"] = -5.0
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_latest_closes(df)

    def test_zero_close_raises(self):
        df = _valid_closes()
        df.loc[0, "close"] = 0.0
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_latest_closes(df)

    def test_null_ticker_raises(self):
        df = _valid_closes()
        df.loc[0, "ticker"] = None
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_latest_closes(df)

    def test_missing_close_column_raises(self):
        df = _valid_closes().drop(columns=["close"])
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_latest_closes(df)

    def test_ticker_with_whitespace_raises(self):
        df = _valid_closes()
        df.loc[0, "ticker"] = "T 00"
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_latest_closes(df)


# ---------------------------------------------------------------------------
# validate_master_csv — happy path
# ---------------------------------------------------------------------------


class TestValidateMasterCsvValid:
    def test_valid_dataframe_passes(self):
        df = _valid_master()
        result = validate_master_csv(df)
        assert len(result) == 3

    def test_nullable_ohlcv_passes(self):
        df = _valid_master()
        # yfinance occasionally returns NaN for illiquid tickers — must not fail
        df.loc[0, "Open"] = float("nan")
        df.loc[1, "Volume"] = float("nan")
        result = validate_master_csv(df)
        assert len(result) == 3

    def test_nullable_name_passes(self):
        df = _valid_master()
        df.loc[0, "Name"] = None
        result = validate_master_csv(df)
        assert len(result) == 3

    def test_coerces_volume_to_float(self):
        df = _valid_master()
        df["Volume"] = df["Volume"].astype(int)
        result = validate_master_csv(df)
        assert result["Volume"].dtype == float


# ---------------------------------------------------------------------------
# validate_master_csv — violations
# ---------------------------------------------------------------------------


class TestValidateMasterCsvInvalid:
    def test_duplicate_date_ticker_raises(self):
        df = _valid_master()
        # Add a duplicate row
        duplicate = df.iloc[[0]].copy()
        df = pd.concat([df, duplicate], ignore_index=True)
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_master_csv(df)

    def test_negative_close_raises(self):
        df = _valid_master()
        df.loc[0, "Close"] = -1.0
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_master_csv(df)

    def test_negative_open_raises(self):
        df = _valid_master()
        df.loc[0, "Open"] = -0.01
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_master_csv(df)

    def test_negative_volume_raises(self):
        df = _valid_master()
        df.loc[0, "Volume"] = -100.0
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_master_csv(df)

    def test_missing_date_column_raises(self):
        df = _valid_master().drop(columns=["Date"])
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_master_csv(df)

    def test_missing_ticker_column_raises(self):
        df = _valid_master().drop(columns=["Ticker"])
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_master_csv(df)

    def test_suspiciously_large_close_raises(self):
        df = _valid_master()
        df.loc[0, "Close"] = 2_000_000.0
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_master_csv(df)
