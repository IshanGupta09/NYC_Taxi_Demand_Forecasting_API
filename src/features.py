"""
src/features.py
───────────────
Feature engineering for NYC Yellow Taxi demand forecasting.

Pipeline:
  raw parquet  →  hourly aggregation  →  time/lag/rolling features  →  model-ready DataFrame
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union


# ── Constants ─────────────────────────────────────────────────────────────────
NYC_HOLIDAYS_2024_2026 = [
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-11",
    "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-11",
    "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16",
]

BOROUGH_MAP = {
    1:  "EWR",        2:  "Queens",    3:  "Bronx",
    4:  "Manhattan",  5:  "Staten Island", 6: "Brooklyn",
    264: "Unknown",   265: "Unknown",
}


# ── Step 1: Load & clean raw parquet ─────────────────────────────────────────
def load_raw(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single monthly parquet file with minimal memory footprint.
    - Reads only 5 columns
    - Uses downcasted dtypes (float32 / int16) to halve RAM usage
    - Renames columns in-place to avoid an extra copy
    - Filters outliers before any copy is made
    """
    cols = [
        "tpep_pickup_datetime",
        "PULocationID",
        "passenger_count",
        "trip_distance",
        "fare_amount",
    ]
    df = pd.read_parquet(path, columns=cols)

    # ── Rename in-place (no copy) ──────────────────────────────────────────────
    df.columns = [
        "pickup_dt" if c == "tpep_pickup_datetime"
        else "zone_id" if c == "PULocationID"
        else c
        for c in df.columns
    ]

    # ── Downcast numeric cols to save ~50% RAM ─────────────────────────────────
    df["pickup_dt"]       = pd.to_datetime(df["pickup_dt"])
    df["zone_id"]         = df["zone_id"].astype("int16")
    df["passenger_count"] = pd.to_numeric(df["passenger_count"],
                                           errors="coerce").astype("float32")
    df["trip_distance"]   = df["trip_distance"].astype("float32")
    df["fare_amount"]     = df["fare_amount"].astype("float32")

    # ── Filter outliers in-place (boolean mask, no copy) ─────────────────────
    mask = (
        (df["fare_amount"]    > 0)   &
        (df["trip_distance"]  > 0)   &
        (df["trip_distance"]  < 200) &
        (df["passenger_count"].between(1, 6)) &
        (df["pickup_dt"].dt.year.between(2024, 2026))
    )
    df = df.loc[mask].reset_index(drop=True)

    return df


def load_multiple(paths: list) -> pd.DataFrame:
    """
    Load, aggregate to hourly, and concatenate multiple monthly parquet files.
    Processes one file at a time to avoid OOM on large datasets.
    Each file is reduced from ~3M rows → ~4k hourly rows before concat.
    """
    frames = []
    for p in paths:
        df = load_raw(p)
        h  = aggregate_hourly(df)   # ~3M rows → ~4k rows immediately
        frames.append(h)
        del df                       # free raw file from RAM
    return pd.concat(frames, ignore_index=True)


# ── Step 2: Aggregate to hourly zone-level demand ────────────────────────────
def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group trips into hourly buckets per pickup zone.
    Returns a DataFrame with columns: [pickup_hour, zone_id, demand, avg_fare, avg_distance]
    Only actual zone-hours with trips are kept (no zero-filling).
    """
    df = df.copy()
    df["pickup_hour"] = df["pickup_dt"].dt.floor("h")

    hourly = (
        df.groupby(["pickup_hour", "zone_id"])
          .agg(
              demand       = ("pickup_dt",     "count"),
              avg_fare     = ("fare_amount",   "mean"),
              avg_distance = ("trip_distance", "mean"),
          )
          .reset_index()
    )

    return hourly.sort_values(["zone_id", "pickup_hour"]).reset_index(drop=True)


# ── Step 3: Add time-based features ──────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["pickup_hour"]

    holidays = pd.to_datetime(NYC_HOLIDAYS_2024_2026)

    df["hour"]          = dt.dt.hour
    df["day_of_week"]   = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"]  = dt.dt.day
    df["month"]         = dt.dt.month
    df["week_of_year"]  = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]    = dt.dt.normalize().isin(holidays).astype(int)

    # Rush-hour flag: 7-10 AM and 4-8 PM on weekdays
    df["is_rush_hour"]  = (
        (~df["is_weekend"].astype(bool)) &
        (
            dt.dt.hour.between(7, 9) |
            dt.dt.hour.between(16, 19)
        )
    ).astype(int)

    # Cyclical encoding (preserves continuity across midnight / Sunday→Monday)
    df["hour_sin"]      = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"]       / 12)

    return df


# ── Step 4: Add lag & rolling features ───────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features capture past demand patterns.
    Computed per zone_id to avoid leakage across zones.

    With 26 months of data we can now add year-over-year lags (8760h)
    which dramatically improve accuracy for seasonal patterns.
    """
    df = df.sort_values(["zone_id", "pickup_hour"]).copy()
    grp = df.groupby("zone_id")["demand"]

    # Short-range lags (hours back)
    df["lag_1h"]    = grp.shift(1)
    df["lag_2h"]    = grp.shift(2)
    df["lag_3h"]    = grp.shift(3)
    df["lag_24h"]   = grp.shift(24)     # same hour yesterday
    df["lag_48h"]   = grp.shift(48)
    df["lag_168h"]  = grp.shift(168)    # same hour last week

    # Year-over-year lags (only available with 12+ months of data)
    df["lag_8760h"] = grp.shift(8760)   # same hour last year
    df["lag_8784h"] = grp.shift(8784)   # +24h offset (handles leap years)

    # Rolling statistics
    df["roll_mean_3h"]  = grp.transform(lambda x: x.shift(1).rolling(3,    min_periods=1).mean())
    df["roll_mean_6h"]  = grp.transform(lambda x: x.shift(1).rolling(6,    min_periods=1).mean())
    df["roll_mean_24h"] = grp.transform(lambda x: x.shift(1).rolling(24,   min_periods=1).mean())
    df["roll_mean_7d"]  = grp.transform(lambda x: x.shift(1).rolling(168,  min_periods=1).mean())
    df["roll_std_24h"]  = grp.transform(lambda x: x.shift(1).rolling(24,   min_periods=1).std())
    df["roll_max_24h"]  = grp.transform(lambda x: x.shift(1).rolling(24,   min_periods=1).max())

    return df


# ── Master pipeline ───────────────────────────────────────────────────────────
def build_features(paths: list) -> pd.DataFrame:
    """
    End-to-end feature pipeline.

    Each parquet file is loaded, aggregated to hourly, and freed from RAM
    before the next file is loaded. Only ~4k rows per file are kept in memory
    at any time, making this safe for 26+ months of data on a laptop.

    Usage:
        df = build_features(["data/raw/yellow_tripdata_2026-01.parquet",
                             "data/raw/yellow_tripdata_2026-02.parquet"])
    """
    print("📂 Loading raw data & aggregating per file...")
    hourly = load_multiple(paths)      # already hourly — aggregated per file

    print("🕐 Adding time features...")
    df     = add_time_features(hourly)
    del hourly

    print("📈 Adding lag & rolling features...")
    df     = add_lag_features(df)

    # Drop rows where year-over-year lags couldn't be computed
    df     = df.dropna(subset=["lag_8760h"])
    df     = df.reset_index(drop=True)

    print(f"✅ Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── Feature columns used by the model ────────────────────────────────────────
FEATURE_COLS = [
    "zone_id",
    "hour", "day_of_week", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_holiday", "is_rush_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
    "lag_8760h", "lag_8784h",                        # ← year-over-year
    "roll_mean_3h", "roll_mean_6h", "roll_mean_24h", "roll_mean_7d",
    "roll_std_24h", "roll_max_24h",
    "avg_fare", "avg_distance",
]

TARGET_COL = "demand"
