"""
Central configuration for the sp500-nasdaq-under20 pipeline.

Every tunable constant lives here.  Scripts import from this module rather
than defining their own defaults, so a single change propagates everywhere.

Environment variable overrides
-------------------------------
Any value can be overridden at runtime without editing code::

    MAX_PRICE=20 LOG_LEVEL=DEBUG python scripts/update_under20_master_csv.py

Supported overrides:

    MAX_PRICE          float   Price ceiling in USD          (default: 30.0)
    BATCH_SIZE         int     Tickers per yfinance call     (default: 150)
    HISTORY_PERIOD     str     yfinance period string        (default: 1y)
    NAME_BATCH_SIZE    int     Tickers per name-lookup call  (default: 50)
"""

from __future__ import annotations

import os
from pathlib import Path


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} is not a valid float.") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} is not a valid int.") from exc


# ---------------------------------------------------------------------------
# Price filter
# ---------------------------------------------------------------------------

#: Maximum close price (USD). Tickers with close *strictly below* this value
#: on the last trading day are included in the pipeline.
MAX_PRICE: float = _float_env("MAX_PRICE", 30.0)

# ---------------------------------------------------------------------------
# yfinance batching
# ---------------------------------------------------------------------------

#: Number of tickers passed to ``yf.download`` in a single call.
#: Larger values are faster but more likely to trigger Yahoo rate-limits.
BATCH_SIZE: int = _int_env("BATCH_SIZE", 150)

#: Number of tickers passed to ``yf.Tickers`` in a single name-lookup call.
NAME_BATCH_SIZE: int = _int_env("NAME_BATCH_SIZE", 50)

#: yfinance ``period`` string used for historical OHLCV downloads.
HISTORY_PERIOD: str = os.environ.get("HISTORY_PERIOD", "1y").strip() or "1y"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

#: Repository root — two levels above this file (scripts/config.py → repo/).
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Consolidated master history CSV.
MASTER_CSV_PATH: Path = _REPO_ROOT / "data" / "under20_history.csv"

#: Cached NASDAQ-100 ticker list (refreshed whenever Wikipedia is reachable).
NASDAQ_CACHE_PATH: Path = _REPO_ROOT / "data" / "nasdaq_tickers.csv"

#: Append-only run log written at the end of each successful pipeline run.
RUN_LOG_PATH: Path = _REPO_ROOT / "data" / "run_log.csv"

#: Default output directory for per-ticker CSVs (``under20_stocks.py`` only).
OUTPUT_DIR: Path = _REPO_ROOT / "data_under_20"

# ---------------------------------------------------------------------------
# NYSE fixed holidays (month, day) — used by the date-rolling logic
# ---------------------------------------------------------------------------

#: Known fixed NYSE holidays as (month, day) tuples.
#: Floating holidays (Thanksgiving, MLK Day, etc.) are intentionally omitted;
#: the downstream price filter tolerates gaps by taking the last available row.
NYSE_FIXED_HOLIDAYS: frozenset[tuple[int, int]] = frozenset({
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (12, 25),  # Christmas Day
})
