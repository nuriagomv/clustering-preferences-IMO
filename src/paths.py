"""Filesystem locations shared by the data builders and the experiment drivers."""

import os
from pathlib import Path


# src/paths.py -> repository root is the parent of src/.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

DIET_WORKBOOK = DATA_DIR / "sustainable_diet_plan_data.xlsx"


def already_computed(directory, fragment):
    """Return True if ``directory`` already holds a file whose name contains ``fragment``.

    Missing directories count as "not computed". The match is a substring
    check, same rule the original drivers used to skip finished runs.
    """
    if not directory.exists():
        return False
    return any(fragment in name for name in os.listdir(directory))


def save_path(directory, filename):
    """Create ``directory`` if needed and return ``directory / filename``."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename
