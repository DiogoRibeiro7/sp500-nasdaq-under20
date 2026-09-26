"""CSV failures identify the file and retain the underlying error."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from dataexcept import DataLoadingError, FileReadError, FileWriteError

import under20_stocks
import update_under20_master_csv
from csv_io import read_csv, write_csv
from run_logger import load_run_log, log_run


def test_missing_csv_has_file_read_error_with_original_cause(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"

    with pytest.raises(FileReadError) as caught:
        read_csv(path)

    assert caught.value.path == str(path)
    assert isinstance(caught.value.original, FileNotFoundError)
    assert caught.value.__cause__ is caught.value.original


def test_malformed_csv_has_loading_error_with_parse_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "master.csv"
    path.write_text('Date,Ticker\n"unclosed', encoding="utf-8")
    monkeypatch.setattr(update_under20_master_csv, "MASTER_CSV_PATH", path)

    with pytest.raises(DataLoadingError) as caught:
        update_under20_master_csv._load_existing_master()

    assert caught.value.source == str(path)
    assert isinstance(caught.value.original, pd.errors.ParserError)
    assert caught.value.__cause__ is caught.value.original


def test_run_log_read_failure_does_not_look_like_an_empty_log(tmp_path: Path) -> None:
    path = tmp_path / "run_log.csv"
    path.write_text('run_ts\n"unclosed', encoding="utf-8")

    with pytest.raises(DataLoadingError) as caught:
        load_run_log(path)

    assert caught.value.source == str(path)


def test_csv_write_error_preserves_original_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "master.csv"
    original = PermissionError("write denied")

    def fail_write(self: pd.DataFrame, output: Path, **options: object) -> None:
        raise original

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_write)
    with pytest.raises(FileWriteError) as caught:
        write_csv(pd.DataFrame({"value": [1]}), path)

    assert caught.value.path == str(path)
    assert caught.value.original is original
    assert caught.value.__cause__ is original


def test_log_and_ticker_cache_report_the_requested_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    log_path = blocked / "run_log.csv"

    with pytest.raises(FileWriteError) as log_error:
        log_run(dt.date(2026, 9, 25), 10, 1, 1, 1, 30.0, "ok", path=log_path)
    assert log_error.value.path == str(log_path)
    assert isinstance(log_error.value.__cause__, OSError)

    cache_path = blocked / "nasdaq_tickers.csv"
    monkeypatch.setattr(under20_stocks, "NASDAQ_CACHE_PATH", cache_path)
    with pytest.raises(FileWriteError) as cache_error:
        under20_stocks._save_cached_nasdaq_tickers(["ABC"], {"ABC": "Example"})
    assert cache_error.value.path == str(cache_path)
