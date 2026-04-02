"""
Data loading utilities for Inside Airbnb Dublin dataset.

Source: https://insideairbnb.com/get-the-data/
Snapshot: 16 September 2025
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_listings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load raw detailed listings (79 columns, ~6.9k rows)."""
    path = raw_dir / "listings.csv.gz"
    return pd.read_csv(path, compression="gzip", low_memory=False)


def load_calendar(raw_dir: Path = RAW_DIR, parse_dates: bool = True) -> pd.DataFrame:
    """Load raw calendar (~2.5M rows, 365 days × ~6.9k listings)."""
    path = raw_dir / "calendar.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    if parse_dates:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_reviews(raw_dir: Path = RAW_DIR, parse_dates: bool = True) -> pd.DataFrame:
    """Load raw reviews (~270k rows)."""
    path = raw_dir / "reviews.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    if parse_dates:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def save_processed(df: pd.DataFrame, name: str, fmt: str = "parquet") -> Path:
    """Save dataframe to data/processed. Default parquet for speed + types."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        path = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return path


def load_processed(name: str, fmt: str = "parquet") -> pd.DataFrame:
    """Reverse of save_processed."""
    path = PROCESSED_DIR / f"{name}.{fmt}"
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
