"""
Tests for config.py and run_logger.py.

No network calls. Uses monkeypatch for environment variable overrides and
tmp_path for file I/O.
"""

from __future__ import annotations

import datetime as dt
import importlib
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        import config
        assert config.MAX_PRICE == 30.0
        assert config.BATCH_SIZE == 150
        assert config.NAME_BATCH_SIZE == 50
        assert config.HISTORY_PERIOD == "1y"
        assert isinstance(config.MASTER_CSV_PATH, Path)
        assert isinstance(config.NASDAQ_CACHE_PATH, Path)
        assert isinstance(config.RUN_LOG_PATH, Path)
        assert isinstance(config.OUTPUT_DIR, Path)

    def test_max_price_env_override(self, monkeypatch):
        monkeypatch.setenv("MAX_PRICE", "20.5")
        import config as cfg
        importlib.reload(cfg)
        assert cfg.MAX_PRICE == 20.5
        monkeypatch.delenv("MAX_PRICE", raising=False)
        importlib.reload(cfg)

    def test_batch_size_env_override(self, monkeypatch):
        monkeypatch.setenv("BATCH_SIZE", "75")
        import config as cfg
        importlib.reload(cfg)
        assert cfg.BATCH_SIZE == 75
        monkeypatch.delenv("BATCH_SIZE", raising=False)
        importlib.reload(cfg)

    def test_invalid_max_price_env_raises(self, monkeypatch):
        monkeypatch.setenv("MAX_PRICE", "not-a-float")
        import config as cfg
        with pytest.raises(ValueError, match="not a valid float"):
            importlib.reload(cfg)
        monkeypatch.delenv("MAX_PRICE", raising=False)
        importlib.reload(cfg)

    def test_invalid_batch_size_env_raises(self, monkeypatch):
        monkeypatch.setenv("BATCH_SIZE", "abc")
        import config as cfg
        with pytest.raises(ValueError, match="not a valid int"):
            importlib.reload(cfg)
        monkeypatch.delenv("BATCH_SIZE", raising=False)
        importlib.reload(cfg)

    def test_paths_resolve_under_repo_root(self):
        import config
        # MASTER_CSV_PATH should be data/under20_history.csv under the repo root
        assert config.MASTER_CSV_PATH.name == "under20_history.csv"
        assert config.MASTER_CSV_PATH.parent.name == "data"

    def test_nyse_fixed_holidays_is_frozenset(self):
        import config
        assert isinstance(config.NYSE_FIXED_HOLIDAYS, frozenset)
        assert (1, 1) in config.NYSE_FIXED_HOLIDAYS
        assert (12, 25) in config.NYSE_FIXED_HOLIDAYS
        assert (7, 4) in config.NYSE_FIXED_HOLIDAYS


# ---------------------------------------------------------------------------
# run_logger.py
# ---------------------------------------------------------------------------


def _sample_run(extra: dict | None = None) -> dict:
    base = {
        "as_of_date":    dt.date(2024, 3, 4),
        "universe_size": 620,
        "tickers_found": 42,
        "new_rows_added": 38,
        "total_rows":    5_000,
        "max_price":     30.0,
        "status":        "ok",
    }
    if extra:
        base.update(extra)
    return base


class TestLogRun:
    def test_creates_file_on_first_call(self, tmp_path):
        from run_logger import log_run
        path = tmp_path / "run_log.csv"
        assert not path.exists()
        log_run(**_sample_run(), path=path)
        assert path.exists()

    def test_header_written_once(self, tmp_path):
        from run_logger import log_run
        path = tmp_path / "run_log.csv"
        log_run(**_sample_run(), path=path)
        log_run(**_sample_run({"status": "no_tickers"}), path=path)
        df = pd.read_csv(path)
        # Header should appear exactly once (i.e. the file has 2 data rows)
        assert len(df) == 2
        assert list(df.columns) == [
            "run_ts", "as_of_date", "universe_size", "tickers_found",
            "new_rows_added", "total_rows", "max_price", "status",
        ]

    def test_values_recorded_correctly(self, tmp_path):
        from run_logger import log_run
        path = tmp_path / "run_log.csv"
        log_run(**_sample_run(), path=path)
        df = pd.read_csv(path)
        row = df.iloc[0]
        assert row["universe_size"] == 620
        assert row["tickers_found"] == 42
        assert row["new_rows_added"] == 38
        assert row["total_rows"] == 5_000
        assert float(row["max_price"]) == 30.0
        assert row["status"] == "ok"
        assert row["as_of_date"] == "2024-03-04"

    def test_appends_multiple_rows(self, tmp_path):
        from run_logger import log_run
        path = tmp_path / "run_log.csv"
        for i in range(5):
            log_run(**_sample_run({"new_rows_added": i}), path=path)
        df = pd.read_csv(path)
        assert len(df) == 5
        assert list(df["new_rows_added"]) == list(range(5))

    def test_status_values(self, tmp_path):
        from run_logger import log_run
        path = tmp_path / "run_log.csv"
        for status in ("ok", "no_tickers", "no_history", "validation_error"):
            log_run(**_sample_run({"status": status}), path=path)
        df = pd.read_csv(path)
        assert list(df["status"]) == [
            "ok", "no_tickers", "no_history", "validation_error"
        ]

    def test_creates_parent_directory(self, tmp_path):
        from run_logger import log_run
        path = tmp_path / "nested" / "dir" / "run_log.csv"
        log_run(**_sample_run(), path=path)
        assert path.exists()


class TestLoadRunLog:
    def test_returns_empty_df_when_missing(self, tmp_path):
        from run_logger import load_run_log
        path = tmp_path / "nonexistent.csv"
        df = load_run_log(path=path)
        assert df.empty
        assert "status" in df.columns

    def test_roundtrip(self, tmp_path):
        from run_logger import log_run, load_run_log
        path = tmp_path / "run_log.csv"
        log_run(**_sample_run(), path=path)
        df = load_run_log(path=path)
        assert len(df) == 1
        assert df.iloc[0]["status"] == "ok"
