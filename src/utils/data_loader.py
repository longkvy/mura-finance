"""
Data loading utilities for MURA-Finance project.
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def load_all_dataframes(base_path: Path = Path(".")) -> Dict[str, pd.DataFrame]:
    """
    Load dev/test CSV files from the `data/` folder into a dictionary of DataFrames.

    Args:
        base_path: Project base path (the folder that contains the `data/` directory).

    Returns:
        Dictionary that may contain:
        - 'dev':  loaded from data/dev.csv (if present)
        - 'test': loaded from data/test.csv (if present)
    """
    data = {}

    # New dataset layout: data/dev.csv and data/test.csv
    data_dir = base_path / "data"
    dev_path = data_dir / "dev.csv"
    if dev_path.exists():
        data["dev"] = pd.read_csv(dev_path)
        print(f"Loaded dev split: {len(data['dev'])} rows from {dev_path}")

    test_path = data_dir / "test.csv"
    if test_path.exists():
        data["test"] = pd.read_csv(test_path)
        print(f"Loaded test split: {len(data['test'])} rows from {test_path}")

    return data
