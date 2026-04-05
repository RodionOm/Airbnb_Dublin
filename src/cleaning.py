"""
Cleaning utilities for Inside Airbnb Dublin listings.

Key decisions:
- Price: parse "$1,234.00" -> float, drop NaN price rows (24.5% missing,
  unfixable for price prediction); keep IQR-bounded subset for modeling.
- Booleans: 't'/'f' -> bool
- Percentages: '95%' -> 0.95
- Dates: errors='coerce'
- Neighbourhood: normalize raw 'neighbourhood' to Dublin postal districts
  (D1, D2, ...) where possible; fallback to coarse 'neighbourhood_cleansed'.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------- price ----------

def parse_price(series: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0; non-parseable -> NaN."""
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


# ---------- generic ----------

def parse_tf_bool(series: pd.Series) -> pd.Series:
    """'t'/'f'/NaN -> True/False/NaN (nullable bool)."""
    return series.map({"t": True, "f": False}).astype("boolean")


def parse_percent(series: pd.Series) -> pd.Series:
    """'95%' -> 0.95"""
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
        .div(100)
    )


# ---------- neighbourhood normalization ----------

POSTAL_RE = re.compile(r"Dublin\s*(\d{1,2})", re.IGNORECASE)


def normalize_neighbourhood(raw: str | float) -> str:
    """
    Inside Airbnb 'neighbourhood' is user-entered and messy.
    Normalize to:
      'D{n}'  for Dublin postal districts (Dublin 1..24)
      'Dublin' for unspecified Dublin
      original token otherwise (Blackrock, Swords, etc.)
    """
    if not isinstance(raw, str):
        return "Unknown"
    m = POSTAL_RE.search(raw)
    if m:
        return f"D{int(m.group(1))}"
    token = raw.split(",")[0].strip()
    return token if token else "Unknown"


# ---------- main cleaning pipeline ----------

def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning to a raw listings dataframe.
    Does NOT drop missing-price rows (caller decides per task).
    """
    df = df.copy()

    # price (host-set nightly price)
    df["price"] = parse_price(df["price"])

    # booleans
    for col in ["host_is_superhost", "instant_bookable", "has_availability",
                "host_has_profile_pic", "host_identity_verified"]:
        if col in df.columns:
            df[col] = parse_tf_bool(df[col])

    # percentages
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = parse_percent(df[col])

    # dates
    for col in ["host_since", "first_review", "last_review",
                "last_scraped", "calendar_last_scraped"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # neighbourhood normalized
    df["neighbourhood_norm"] = df["neighbourhood"].apply(normalize_neighbourhood)

    # bedrooms / beds median fill (most missing are studios -> 1)
    for col in ["bedrooms", "beds"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # reviews_per_month: NaN means no reviews -> 0
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    return df


def filter_price_iqr(df: pd.DataFrame, low: float = 0.01,
                     high: float = 0.99) -> pd.DataFrame:
    """Drop rows with missing or extreme price (winsorize via filter)."""
    s = df["price"].dropna()
    lo, hi = s.quantile([low, high])
    mask = df["price"].between(lo, hi)
    return df.loc[mask].copy()


def parse_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar prep: price -> float, 'available' -> bool 'booked'."""
    df = df.copy()
    df["price"] = parse_price(df["price"])
    df["adjusted_price"] = parse_price(df["adjusted_price"])
    df["booked"] = df["available"].map({"t": False, "f": True}).astype("boolean")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
