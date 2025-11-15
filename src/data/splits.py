from __future__ import annotations
from pathlib import Path
import pandas as pd


def save_split_index(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_split_index(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

