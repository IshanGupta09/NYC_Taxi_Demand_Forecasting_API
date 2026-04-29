"""
scripts/build_cache.py
──────────────────────
Run ONCE. Saves two tiny files to data/cache/:

  eda_stats.npz          ~1 MB   — pre-computed EDA summaries  → eda.py loads instantly
  train_features.parquet ~50 MB  — feature matrix with lags     → train.py loads in ~3s

Usage:
    python scripts/build_cache.py
"""

import sys, time, json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.features import load_raw, aggregate_hourly, add_time_features, add_lag_features, FEATURE_COLS, TARGET_COL

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

EDA_CACHE   = CACHE_DIR / "eda_stats.npz"
TRAIN_CACHE = CACHE_DIR / "train_features.parquet"
TOP_ZONES   = 100   # train on top-100 busiest zones only


def build():
    all_files = sorted(DATA_DIR.glob("yellow_tripdata_20*.parquet"))
    if not all_files:
        print("⚠  No parquet files. Run: python data/download_data.py")
        sys.exit(1)

    print(f"🔧 Building cache from {len(all_files)} files...")
    print(f"   This runs ONCE — eda.py and train.py will be instant after this.\n")

    # ── EDA accumulators (tiny numpy arrays) ──────────────────────────────────
    monthly_trips   = {}           # month_str → exact trip count
    hour_sum        = np.zeros(24)
    hour_cnt        = np.zeros(24)
    wday_sum        = np.zeros((2, 24))
    wday_cnt        = np.zeros((2, 24))
    zone_demand     = {}
    season_sum      = np.zeros((12, 24))
    season_cnt      = np.zeros((12, 24))
    year_sum        = {}           # year → np.zeros(24)
    year_cnt        = {}

    # ── Training accumulators (only top zones, keep hourly rows) ──────────────
    hourly_frames   = []           # for building lag features later
    zone_totals_all = {}           # to find top zones across all months

    t0 = time.time()

    for i, f in enumerate(all_files, 1):
        month = f.stem.replace("yellow_tripdata_", "")
        t = time.time()

        df = load_raw(f)
        monthly_trips[month] = len(df)

        h = aggregate_hourly(df)
        h = add_time_features(h)
        del df

        yr = int(month[:4])

        # ── Accumulate EDA stats from tiny hourly frame ────────────────────────
        for hour_val, g in h.groupby("hour"):
            hour_sum[hour_val] += g["demand"].sum()
            hour_cnt[hour_val] += len(g)

        for (wk, hv), g in h.groupby(["is_weekend", "hour"]):
            wday_sum[wk, hv] += g["demand"].sum()
            wday_cnt[wk, hv] += len(g)

        for zid, tot in h.groupby("zone_id")["demand"].sum().items():
            zone_demand[zid] = zone_demand.get(zid, 0) + tot

        for (m, hv), g in h.groupby(["month", "hour"]):
            season_sum[int(m)-1, hv] += g["demand"].sum()
            season_cnt[int(m)-1, hv] += len(g)

        if yr not in year_sum:
            year_sum[yr] = np.zeros(24)
            year_cnt[yr] = np.zeros(24)
        for hv, g in h.groupby("hour"):
            year_sum[yr][hv] += g["demand"].sum()
            year_cnt[yr][hv] += len(g)

        for zid, tot in h.groupby("zone_id")["demand"].sum().items():
            zone_totals_all[zid] = zone_totals_all.get(zid, 0) + tot

        # Keep hourly frame for training
        hourly_frames.append(h)
        del h

        elapsed = time.time() - t
        print(f"   [{i:02d}/{len(all_files)}] {month} — {monthly_trips[month]:,} trips — {elapsed:.1f}s")

    # ── Save EDA stats (tiny .npz) ─────────────────────────────────────────────
    print("\n💾 Saving EDA stats...")
    top15_zones = sorted(zone_demand, key=zone_demand.get, reverse=True)[:15]
    top15_vals  = [zone_demand[z] for z in top15_zones]

    years       = sorted(year_sum.keys())
    yoy_avg     = np.array([
        np.divide(year_sum[y], year_cnt[y], where=year_cnt[y] > 0)
        for y in years
    ])

    np.savez_compressed(
        EDA_CACHE,
        hour_avg     = np.divide(hour_sum, hour_cnt, where=hour_cnt > 0),
        wday_avg     = np.divide(wday_sum, wday_cnt, where=wday_cnt > 0),
        season_avg   = np.divide(season_sum, season_cnt, where=season_cnt > 0),
        yoy_avg      = yoy_avg,
        years        = np.array(years),
        top15_zones  = np.array(top15_zones),
        top15_vals   = np.array(top15_vals),
        month_keys   = np.array(sorted(monthly_trips.keys())),
        month_vals   = np.array([monthly_trips[m] for m in sorted(monthly_trips.keys())]),
    )
    print(f"   ✅ EDA stats → {EDA_CACHE.name} ({EDA_CACHE.stat().st_size/1e3:.0f} KB)")

    # ── Build training feature matrix ──────────────────────────────────────────
    print("\n🔧 Building training features (top 100 zones)...")
    top_zone_ids = sorted(zone_totals_all, key=zone_totals_all.get, reverse=True)[:TOP_ZONES]

    # Concat all hourly frames, filter to top zones, add lags
    all_hourly = pd.concat(hourly_frames, ignore_index=True)
    del hourly_frames
    all_hourly = all_hourly[all_hourly["zone_id"].isin(top_zone_ids)].reset_index(drop=True)
    print(f"   Adding lag features to {len(all_hourly):,} rows...")
    all_hourly = add_lag_features(all_hourly)
    all_hourly = all_hourly.dropna(subset=["lag_8760h"]).reset_index(drop=True)

    # Save only columns needed for training
    keep_cols = ["pickup_hour", "zone_id", TARGET_COL] + FEATURE_COLS
    keep_cols = list(dict.fromkeys(keep_cols))   # deduplicate, preserve order
    all_hourly[keep_cols].to_parquet(TRAIN_CACHE, index=False, compression="snappy")

    # Save top zone list
    meta = {"top_zone_ids": top_zone_ids}
    (CACHE_DIR / "zone_meta.json").write_text(json.dumps(meta, indent=2))

    total = time.time() - t0
    print(f"   ✅ Training features → {TRAIN_CACHE.name} ({TRAIN_CACHE.stat().st_size/1e6:.1f} MB)")
    print(f"      Rows: {len(all_hourly):,} | Cols: {len(keep_cols)}")
    print(f"\n✅ Cache built in {total/60:.1f} min.")
    print(f"   eda.py   → loads in <1 second")
    print(f"   train.py → loads in ~3 seconds")


if __name__ == "__main__":
    if EDA_CACHE.exists() and TRAIN_CACHE.exists():
        print(f"✅ Cache already exists:")
        print(f"   {EDA_CACHE.name}   ({EDA_CACHE.stat().st_size/1e3:.0f} KB)")
        print(f"   {TRAIN_CACHE.name} ({TRAIN_CACHE.stat().st_size/1e6:.1f} MB)")
        print(f"\n   Delete data/cache/ and re-run to rebuild.")
    else:
        build()
