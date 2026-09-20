"""Small reporting helpers with no optional third-party dependencies."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def markdown_table(frame: pd.DataFrame, float_digits: int = 7) -> str:
    """Render a DataFrame as a compact GitHub-flavored Markdown table."""

    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.{float_digits}g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)
