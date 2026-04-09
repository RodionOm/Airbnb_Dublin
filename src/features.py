"""
Feature engineering for Dublin Airbnb price-prediction model.

Categories:
1. Distance features (geopy) — distance to city anchors
2. Amenity flags — parsed from JSON-like amenities string
3. Host features — tenure, response, listing portfolio size
4. Text features — description length, name caps ratio
5. Temporal — listing age proxy via first_review
"""
from __future__ import annotations

import ast
from datetime import datetime

import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Dublin landmarks (lat, lon) — chosen to cover diverse demand drivers
ANCHORS: dict[str, tuple[float, float]] = {
    "city_center":     (53.3498, -6.2603),   # O'Connell Bridge
    "temple_bar":      (53.3453, -6.2638),   # tourist core
    "trinity_college": (53.3438, -6.2546),
    "guinness":        (53.3419, -6.2867),   # Storehouse
    "airport":         (53.4213, -6.2700),   # DUB
    "ifsc":            (53.3486, -6.2406),   # business / corporate stays
}

# Hand-picked amenity flags worth modeling (high signal, common)
AMENITY_FLAGS: dict[str, list[str]] = {
    "has_wifi":         ["wifi"],
    "has_kitchen":      ["kitchen"],
    "has_washer":       ["washer"],
    "has_dryer":        ["dryer"],
    "has_workspace":    ["dedicated workspace", "workspace"],
    "has_tv":           [" tv"],
    "has_free_parking": ["free parking", "free street parking", "free residential garage"],
    "has_paid_parking": ["paid parking"],
    "has_elevator":     ["elevator"],
    "has_aircon":       ["air conditioning"],
    "has_heating":      ["heating"],
    "has_pool":         ["pool"],
    "has_gym":          ["gym"],
    "has_self_checkin": ["self check-in", "lockbox", "smart lock", "keypad"],
    "has_pets_allowed": ["pets allowed"],
    "has_long_term":    ["long term stays allowed"],
}


# ---------- distance ----------

def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds distance_to_<anchor>_km for each anchor (great-circle, km)."""
    df = df.copy()
    coords = list(zip(df["latitude"], df["longitude"]))
    for name, anchor in ANCHORS.items():
        df[f"dist_{name}_km"] = [geodesic(c, anchor).km for c in coords]
    return df


# ---------- amenities ----------

def _parse_amenities(raw) -> list[str]:
    """Amenities column is a JSON-array-like string. Tolerant parsing."""
    if not isinstance(raw, str):
        return []
    try:
        items = ast.literal_eval(raw)
        if isinstance(items, list):
            return [str(x).lower() for x in items]
    except (ValueError, SyntaxError):
        pass
    # fallback: strip brackets, split by quoted commas
    return [t.strip(' "\'').lower() for t in raw.strip("[]").split(",") if t.strip()]


def add_amenity_features(df: pd.DataFrame,
                         flags: dict[str, list[str]] = AMENITY_FLAGS) -> pd.DataFrame:
    """Add binary amenity flags + total amenities count."""
    df = df.copy()
    parsed = df["amenities"].apply(_parse_amenities)
    df["amenities_count"] = parsed.apply(len)

    for flag, keywords in flags.items():
        df[flag] = parsed.apply(
            lambda a, kws=keywords: any(any(k in item for k in kws) for item in a)
        ).astype(int)
    return df


# ---------- host ----------

def add_host_features(df: pd.DataFrame,
                      reference_date: datetime | None = None) -> pd.DataFrame:
    """Host tenure (days), listing portfolio bucket."""
    df = df.copy()
    ref = reference_date or pd.to_datetime(df["last_scraped"]).max()
    df["host_tenure_days"] = (ref - df["host_since"]).dt.days
    df["host_tenure_years"] = df["host_tenure_days"] / 365.25
    df["is_multi_listing_host"] = (df["calculated_host_listings_count"] > 1).astype(int)
    df["is_commercial_host"] = (df["calculated_host_listings_count"] >= 5).astype(int)
    return df


# ---------- text ----------

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap text signals — proxies for listing quality / effort."""
    df = df.copy()
    df["name_length"] = df["name"].fillna("").str.len()
    df["desc_length"] = df["description"].fillna("").str.len()
    df["has_description"] = (df["desc_length"] > 0).astype(int)
    name_str = df["name"].fillna("")
    df["name_caps_ratio"] = (
        name_str.str.count(r"[A-Z]") /
        name_str.str.len().replace(0, np.nan)
    ).fillna(0)
    return df


# ---------- pipeline ----------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps. Input: cleaned listings."""
    df = add_distance_features(df)
    df = add_amenity_features(df)
    df = add_host_features(df)
    df = add_text_features(df)
    return df
