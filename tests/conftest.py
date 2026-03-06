"""
pytest configuration for the sp500-nasdaq-under20 test suite.

Adds the ``scripts/`` directory to ``sys.path`` once, at session start, so
every test module can import ``under20_stocks``, ``update_under20_master_csv``,
``schemas``, ``log_config``, and ``config`` directly without repeating the
path-insertion boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root is the parent of this file's directory (tests/conftest.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
