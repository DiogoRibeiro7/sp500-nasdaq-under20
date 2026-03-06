"""
Unit tests for the sp500-nasdaq-under20 scripts.

Run with:
    pytest tests/test_under20.py -v

No network calls are made — all tests operate on synthetic DataFrames or
use fixed dates, so the suite runs offline and fast.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure the scripts/ directory is importable regardless of where pytest
# is invoked from (repo root or tests/).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from under20_stocks import (
    _is_fixed_holiday,
    _is_weekend,
    chunked,
    get_last_trading_day,
    select_tickers_below_price,
)
from update_under20_master_csv import (
    _attach_names_to_rows,
    _build_new_rows_dataframe,
    _load_existing_master,
    _merge_and_deduplicate,
    resolve_ticker_names,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_closes(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal latest-closes DataFrame from a list of dicts."""
    return pd.DataFrame(rows, columns=["ticker", "date", "close"])


def _make_ohlcv(ticker: str, dates: list[str]) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame indexed by date."""
    idx = pd.to_datetime(dates)
    n = len(dates)
    df = pd.DataFrame(
        {
            "Open": [10.0] * n,
            "High": [11.0] * n,
            "Low": [9.0] * n,
            "Close": [10.5] * n,
            "Adj Close": [10.5] * n,
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )
    return df


# ===========================================================================
# under20_stocks — date utilities
# ===========================================================================


class TestIsWeekend:
    def test_saturday(self):
        assert _is_weekend(dt.date(2024, 3, 2)) is True  # Saturday

    def test_sunday(self):
        assert _is_weekend(dt.date(2024, 3, 3)) is True  # Sunday

    def test_monday(self):
        assert _is_weekend(dt.date(2024, 3, 4)) is False

    def test_friday(self):
        assert _is_weekend(dt.date(2024, 3, 1)) is False


class TestIsFixedHoliday:
    def test_new_years_day(self):
        assert _is_fixed_holiday(dt.date(2024, 1, 1)) is True

    def test_independence_day(self):
        assert _is_fixed_holiday(dt.date(2024, 7, 4)) is True

    def test_christmas(self):
        assert _is_fixed_holiday(dt.date(2024, 12, 25)) is True

    def test_regular_day(self):
        assert _is_fixed_holiday(dt.date(2024, 6, 15)) is False

    def test_thanksgiving_not_in_fixed(self):
        # Thanksgiving is a floating holiday — not in our fixed set
        assert _is_fixed_holiday(dt.date(2024, 11, 28)) is False


class TestGetLastTradingDay:
    def test_rolls_back_over_weekend(self):
        # Tuesday 2024-03-05 → reference is Monday 2024-03-04
        # Monday is a weekday, not a holiday → returns Monday
        monday = dt.date(2024, 3, 4)
        assert get_last_trading_day(dt.date(2024, 3, 5)) == monday

    def test_rolls_back_saturday_to_friday(self):
        # Reference = Saturday 2024-03-02, last trading day = Friday 2024-03-01
        friday = dt.date(2024, 3, 1)
        assert get_last_trading_day(dt.date(2024, 3, 2)) == friday

    def test_rolls_back_sunday_to_friday(self):
        # Reference = Sunday 2024-03-03, last trading day = Friday 2024-03-01
        friday = dt.date(2024, 3, 1)
        assert get_last_trading_day(dt.date(2024, 3, 3)) == friday

    def test_rolls_back_monday_over_weekend(self):
        # Reference = Monday 2024-03-04, last trading day = Friday 2024-03-01
        friday = dt.date(2024, 3, 1)
        assert get_last_trading_day(dt.date(2024, 3, 4)) == friday

    def test_rolls_back_over_new_years(self):
        # Reference = Tuesday 2024-01-02.
        # 2024-01-01 is New Year's Day (fixed holiday) → skip.
        # 2023-12-31 is Sunday → skip.
        # 2023-12-30 is Saturday → skip.
        # 2023-12-29 is Friday → return.
        expected = dt.date(2023, 12, 29)
        assert get_last_trading_day(dt.date(2024, 1, 2)) == expected

    def test_rolls_back_over_christmas(self):
        # Reference = Wednesday 2024-12-26.
        # 2024-12-25 is Christmas (fixed holiday) → skip.
        # 2024-12-24 is Tuesday → return.
        expected = dt.date(2024, 12, 24)
        assert get_last_trading_day(dt.date(2024, 12, 26)) == expected

    def test_raises_if_max_lookback_exhausted(self):
        # With max_lookback=1 and a holiday on the only candidate, should raise.
        # 2024-01-01 is a holiday; reference = 2024-01-02, lookback = 1
        with pytest.raises(RuntimeError, match="Could not find a trading day"):
            get_last_trading_day(dt.date(2024, 1, 2), max_lookback=1)

    def test_normal_weekday_no_holiday(self):
        # Reference = Thursday 2024-06-06, last trading day = Wednesday 2024-06-05
        expected = dt.date(2024, 6, 5)
        assert get_last_trading_day(dt.date(2024, 6, 6)) == expected


# ===========================================================================
# under20_stocks — chunked
# ===========================================================================


class TestChunked:
    def test_even_split(self):
        result = list(chunked(["A", "B", "C", "D"], 2))
        assert result == [["A", "B"], ["C", "D"]]

    def test_uneven_split(self):
        result = list(chunked(["A", "B", "C"], 2))
        assert result == [["A", "B"], ["C"]]

    def test_single_chunk(self):
        result = list(chunked(["A", "B"], 10))
        assert result == [["A", "B"]]

    def test_empty(self):
        result = list(chunked([], 5))
        assert result == []

    def test_chunk_size_one(self):
        result = list(chunked(["X", "Y", "Z"], 1))
        assert result == [["X"], ["Y"], ["Z"]]


# ===========================================================================
# under20_stocks — select_tickers_below_price
# ===========================================================================


class TestSelectTickersBelowPrice:
    def test_basic_filter(self):
        closes = _make_closes([
            {"ticker": "AAA", "date": "2024-01-02", "close": 5.0},
            {"ticker": "BBB", "date": "2024-01-02", "close": 35.0},
            {"ticker": "CCC", "date": "2024-01-02", "close": 29.99},
        ])
        result = select_tickers_below_price(closes, max_price=30.0)
        assert result == ["AAA", "CCC"]

    def test_exactly_at_threshold_excluded(self):
        closes = _make_closes([
            {"ticker": "AAA", "date": "2024-01-02", "close": 30.0},
        ])
        result = select_tickers_below_price(closes, max_price=30.0)
        assert result == []

    def test_all_below(self):
        closes = _make_closes([
            {"ticker": "X", "date": "2024-01-02", "close": 1.0},
            {"ticker": "Y", "date": "2024-01-02", "close": 2.0},
        ])
        result = select_tickers_below_price(closes, max_price=30.0)
        assert result == ["X", "Y"]

    def test_none_below(self):
        closes = _make_closes([
            {"ticker": "X", "date": "2024-01-02", "close": 100.0},
        ])
        result = select_tickers_below_price(closes, max_price=30.0)
        assert result == []

    def test_result_is_sorted(self):
        closes = _make_closes([
            {"ticker": "ZZZ", "date": "2024-01-02", "close": 1.0},
            {"ticker": "AAA", "date": "2024-01-02", "close": 2.0},
            {"ticker": "MMM", "date": "2024-01-02", "close": 3.0},
        ])
        result = select_tickers_below_price(closes, max_price=30.0)
        assert result == sorted(result)

    def test_missing_required_column_raises(self):
        bad_df = pd.DataFrame({"ticker": ["A"], "close": [5.0]})  # no 'date'
        with pytest.raises(ValueError, match="must contain columns"):
            select_tickers_below_price(bad_df, max_price=30.0)

    def test_empty_input(self):
        closes = _make_closes([])
        result = select_tickers_below_price(closes, max_price=30.0)
        assert result == []


# ===========================================================================
# update_under20_master_csv — _build_new_rows_dataframe
# ===========================================================================


class TestBuildNewRowsDataframe:
    def test_basic_structure(self):
        history = {
            "AAA": _make_ohlcv("AAA", ["2024-01-02", "2024-01-03"]),
            "BBB": _make_ohlcv("BBB", ["2024-01-02"]),
        }
        df = _build_new_rows_dataframe(history)
        assert set(df["Ticker"].unique()) == {"AAA", "BBB"}
        assert len(df) == 3
        assert "Date" in df.columns
        assert "Close" in df.columns

    def test_empty_history(self):
        df = _build_new_rows_dataframe({})
        assert df.empty
        assert "Ticker" in df.columns

    def test_empty_ticker_df_skipped(self):
        history = {
            "AAA": _make_ohlcv("AAA", ["2024-01-02"]),
            "BBB": pd.DataFrame(),
        }
        df = _build_new_rows_dataframe(history)
        assert list(df["Ticker"].unique()) == ["AAA"]

    def test_adj_close_normalisation(self):
        # Simulate yfinance returning "Adj_Close" instead of "Adj Close"
        idx = pd.to_datetime(["2024-01-02"])
        raw = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5],
             "Adj_Close": [1.4], "Volume": [500]},
            index=idx,
        )
        df = _build_new_rows_dataframe({"XXX": raw})
        assert "Adj Close" in df.columns
        assert "Adj_Close" not in df.columns


# ===========================================================================
# update_under20_master_csv — _merge_and_deduplicate
# ===========================================================================


class TestMergeAndDeduplicate:
    def _make_master_row(self, ticker, date_str, name="", close=10.0):
        return {
            "Date": pd.Timestamp(date_str),
            "Ticker": ticker,
            "Name": name,
            "Open": 9.0, "High": 11.0, "Low": 8.5,
            "Close": close, "Adj Close": close, "Volume": 100_000,
        }

    def test_no_duplicates_appends_all(self):
        existing = pd.DataFrame([self._make_master_row("AAA", "2024-01-02", "Alpha Inc")])
        new_rows = pd.DataFrame([self._make_master_row("BBB", "2024-01-02", "Beta Corp")])
        result = _merge_and_deduplicate(existing, new_rows)
        assert len(result) == 2

    def test_duplicate_kept_with_name(self):
        # Existing row has no name; new row has a name → new row wins.
        existing = pd.DataFrame([self._make_master_row("AAA", "2024-01-02", name="")])
        new_rows = pd.DataFrame([self._make_master_row("AAA", "2024-01-02", name="Alpha Inc")])
        result = _merge_and_deduplicate(existing, new_rows)
        assert len(result) == 1
        assert result.iloc[0]["Name"] == "Alpha Inc"

    def test_exact_duplicate_deduplicated(self):
        row = self._make_master_row("AAA", "2024-01-02", "Alpha Inc")
        existing = pd.DataFrame([row])
        new_rows = pd.DataFrame([row])
        result = _merge_and_deduplicate(existing, new_rows)
        assert len(result) == 1

    def test_empty_new_rows_returns_existing(self):
        existing = pd.DataFrame([self._make_master_row("AAA", "2024-01-02", "Alpha Inc")])
        new_rows = pd.DataFrame(columns=existing.columns)
        result = _merge_and_deduplicate(existing, new_rows)
        assert len(result) == 1

    def test_result_sorted_by_date_then_ticker(self):
        rows = [
            self._make_master_row("ZZZ", "2024-01-03"),
            self._make_master_row("AAA", "2024-01-02"),
            self._make_master_row("MMM", "2024-01-02"),
        ]
        existing = pd.DataFrame(rows[:1])
        new_rows = pd.DataFrame(rows[1:])
        result = _merge_and_deduplicate(existing, new_rows)
        assert list(result["Ticker"]) == ["AAA", "MMM", "ZZZ"]


# ===========================================================================
# update_under20_master_csv — _attach_names_to_rows
# ===========================================================================


class TestAttachNamesToRows:
    def test_attaches_correctly(self):
        rows = pd.DataFrame({"Ticker": ["AAA", "BBB", "CCC"]})
        names = {"AAA": "Alpha Inc", "BBB": "Beta Corp"}
        result = _attach_names_to_rows(rows, names)
        assert result.loc[result["Ticker"] == "AAA", "Name"].iloc[0] == "Alpha Inc"
        assert result.loc[result["Ticker"] == "BBB", "Name"].iloc[0] == "Beta Corp"
        # CCC not in names dict → falls back to ticker symbol
        assert result.loc[result["Ticker"] == "CCC", "Name"].iloc[0] == "CCC"

    def test_empty_rows(self):
        rows = pd.DataFrame(columns=["Ticker"])
        result = _attach_names_to_rows(rows, {})
        assert "Name" in result.columns
        assert result.empty

    def test_does_not_mutate_input(self):
        rows = pd.DataFrame({"Ticker": ["AAA"]})
        _attach_names_to_rows(rows, {"AAA": "Alpha"})
        assert "Name" not in rows.columns


# ===========================================================================
# update_under20_master_csv — resolve_ticker_names
# ===========================================================================


class TestResolveTickerNames:
    def test_wikipedia_names_used_without_yfinance(self, monkeypatch):
        """When all tickers have good Wikipedia names, yfinance is never called."""
        called = []

        def fake_yf_fetch(tickers, batch_size):
            called.append(tickers)
            return {}

        monkeypatch.setattr(
            "update_under20_master_csv._fetch_names_from_yfinance", fake_yf_fetch
        )

        known = {"AAA": "Alpha Inc", "BBB": "Beta Corp"}
        result = resolve_ticker_names(["AAA", "BBB"], known_names=known)
        assert result == {"AAA": "Alpha Inc", "BBB": "Beta Corp"}
        assert called == [], "yfinance should not have been called"

    def test_fallback_to_yfinance_for_unknown(self, monkeypatch):
        """Tickers absent from known_names are passed to the yfinance fetcher."""
        def fake_yf_fetch(tickers, batch_size):
            return {t: f"YF Name for {t}" for t in tickers}

        monkeypatch.setattr(
            "update_under20_master_csv._fetch_names_from_yfinance", fake_yf_fetch
        )

        known = {"AAA": "Alpha Inc"}
        result = resolve_ticker_names(["AAA", "ZZZ"], known_names=known)
        assert result["AAA"] == "Alpha Inc"
        assert result["ZZZ"] == "YF Name for ZZZ"

    def test_ticker_used_as_name_when_wiki_name_equals_ticker(self, monkeypatch):
        """If Wikipedia name is identical to the ticker symbol, treat as missing."""
        def fake_yf_fetch(tickers, batch_size):
            return {"AAA": "Alpha Inc"}

        monkeypatch.setattr(
            "update_under20_master_csv._fetch_names_from_yfinance", fake_yf_fetch
        )

        # known name == ticker symbol → should fall through to yfinance
        known = {"AAA": "AAA"}
        result = resolve_ticker_names(["AAA"], known_names=known)
        assert result["AAA"] == "Alpha Inc"


# ===========================================================================
# update_under20_master_csv — _load_existing_master
# ===========================================================================


class TestLoadExistingMaster:
    def test_returns_empty_df_when_file_missing(self, tmp_path, monkeypatch):
        import update_under20_master_csv as m
        monkeypatch.setattr(m, "MASTER_CSV_PATH", tmp_path / "nonexistent.csv")
        df = _load_existing_master()
        assert df.empty
        assert "Date" in df.columns
        assert "Name" in df.columns

    def test_adds_name_column_to_old_csv(self, tmp_path, monkeypatch):
        import update_under20_master_csv as m
        csv_path = tmp_path / "master.csv"
        # Write a CSV without the Name column (old format)
        pd.DataFrame({
            "Date": ["2024-01-02"],
            "Ticker": ["AAA"],
            "Close": [9.5],
        }).to_csv(csv_path, index=False)

        monkeypatch.setattr(m, "MASTER_CSV_PATH", csv_path)
        df = _load_existing_master()
        assert "Name" in df.columns
