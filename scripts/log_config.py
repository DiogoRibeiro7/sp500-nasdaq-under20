"""
Shared logging configuration for the sp500-nasdaq-under20 scripts.

Usage
-----
In any script that wants a configured logger::

    from scripts.log_config import get_logger
    log = get_logger(__name__)
    log.info("Starting run")
    log.warning("Something looks off: %s", detail)
    log.error("Hard failure: %s", exc)

The root logger is configured once on first import.  Subsequent calls to
``get_logger`` are cheap — they just return a child logger that inherits the
root handler and level.

Output format
-------------
Local / CI runs (no existing handlers on the root logger)::

    2024-03-04 01:00:12 UTC [INFO    ] under20_stocks        : Fetching S&P 500 tickers
    2024-03-04 01:00:13 UTC [WARNING ] update_under20_master : Could not fetch name for XYZ

If the calling process has already configured logging (e.g. pytest with
``log_cli = true``), the existing configuration is left untouched.

Log level
---------
Set the ``LOG_LEVEL`` environment variable to override the default (INFO)::

    LOG_LEVEL=DEBUG python scripts/update_under20_master_csv.py
"""

from __future__ import annotations

import logging
import os
import sys


_ROOT_LOGGER_NAME = "under20"
_DEFAULT_LEVEL = logging.INFO
_FORMAT = "%(asctime)s UTC [%(levelname)-8s] %(name)-22s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """
    Attach a StreamHandler to the root 'under20' logger if none exists yet.

    Safe to call multiple times — the guard flag ``_configured`` ensures
    handlers are only added once per process.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger(_ROOT_LOGGER_NAME)

    # If handlers are already present (e.g. pytest log_cli, Jupyter, a calling
    # framework), leave them alone.  We only set up our own handler when the
    # logger is completely unconfigured.
    if root.handlers:
        _configured = True
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, _DEFAULT_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(handler)
    # Prevent log records from bubbling up to the Python root logger and
    # being printed a second time.
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger namespaced under the 'under20' hierarchy.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.  The returned logger
        will be named ``under20.<name>`` (or just ``under20`` if *name* is
        already the root name).

    Returns
    -------
    logging.Logger
    """
    _configure_root_logger()

    if name == _ROOT_LOGGER_NAME or not name:
        return logging.getLogger(_ROOT_LOGGER_NAME)

    # Strip any leading package path so e.g. 'scripts.under20_stocks' becomes
    # 'under20.under20_stocks' rather than 'under20.scripts.under20_stocks'.
    short_name = name.split(".")[-1]
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short_name}")
