"""Classify CSV file failures without changing dataframe validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from dataexcept import DataLoadingError, FileReadError, FileWriteError, wrapping


def read_csv(path: Path, **options: Any) -> pd.DataFrame:
    """Read a CSV, retaining its path and the original filesystem or parse error."""

    with (
        wrapping(OSError, FileReadError, path=str(path)),
        wrapping(
            (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError),
            DataLoadingError,
            source=str(path),
        ),
    ):
        return pd.read_csv(path, **options)


def ensure_directory(path: Path) -> None:
    """Create an output directory, identifying it if creation fails."""

    with wrapping(OSError, FileWriteError, path=str(path)):
        path.mkdir(parents=True, exist_ok=True)


def write_csv(
    frame: pd.DataFrame, path: Path, *, create_parent: bool = False, **options: Any
) -> None:
    """Write a CSV, including optional parent creation in the file boundary."""

    with wrapping(OSError, FileWriteError, path=str(path)):
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, **options)
