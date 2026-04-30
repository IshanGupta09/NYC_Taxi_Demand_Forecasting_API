"""
src/train.py
────────────
Train XGBoost demand forecasting model on NYC Yellow Taxi data.
All experiments tracked with MLflow.

Usage:
    python src/train.py                        # uses cache (fast)
    python src/train.py --top-zones 50         # train on top 50 zones
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Project imports ────────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.features import (
    build_features,
    FEATURE_COLS, TARGET_COL
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "raw"
CACHE_DIR   = ROOT / "data" / "cache"
TRAIN_CACHE = CACHE_DIR / "train_features.parquet"
MODEL_DIR   = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── XGBoost hyper-parameters ──────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     1200,
    "max_depth":        7,
    "learning_rate":    0.04,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "reg:squarederror",
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
    }


# ── Training ───────────────────────────────────────────────────────────────────
def train(top_zones: int = 100) -> None:
    t0 = time.time()

    # 1. Load feature matrix ───────────────────────────────────────────────────
    if TRAIN_CACHE.exists():
        print(f"⚡ Loading training cache ({TRAIN_CACHE.stat().st_size/1e6:.0f} MB)...")
        df = pd.read_parquet(TRAIN_CACHE)
        df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
        print(f"   {len(df):,} rows loaded in {time.time()-t0:.1f}s")
    else:
        print("⚠  No training cache found.")
        print("   Run:  python scripts/build_cache.py\n")
        # Fallback: build from raw files (slow)
        months = [
            "2024-01","2024-02","2024-03","2024-04","2024-05","2024-06",
            "2024-07","2024-08","2024-09","2024-10","2024-11","2024-12",
            "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06",
            "2025-07","2025-08","2025-09","2025-10","2025-11","2025-12",
            "2026-01","2026-02",
        ]
        paths = [DATA_DIR / f"yellow_tripdata_{m}.parquet" for m in months
                 if (DATA_DIR / f"yellow_tripdata_{m}.parquet").exists()]
        df = build_features(paths)

    # 2. Limit to top-N zones ──────────────────────────────────────────────────
    top_zone_ids = (
        df.groupby("zone_id")["demand"].sum()
          .nlargest(top_zones).index.tolist()
    )
    df = df[df["zone_id"].isin(top_zone_ids)].reset_index(drop=True)
    print(f"🗺  Training on {len(top_zone_ids)} zones | {len(df):,} rows")

    # 3. Validate all feature columns exist ────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}\n"
                         f"Rebuild cache: python scripts/build_cache.py")

    # 4. Time-based train/test split (last 15% of timeline → test)
    #    Fixed-day windows break when cache has limited date range,
    #    so we use a percentage of unique timestamps instead.
    df = df.sort_values(["pickup_hour", "zone_id"]).reset_index(drop=True)
    unique_hours = df["pickup_hour"].sort_values().unique()
    split_hour   = unique_hours[int(len(unique_hours) * 0.85)]
    train_df = df[df["pickup_hour"] <= split_hour]
    test_df  = df[df["pickup_hour"] >  split_hour]

    print(f"📅 Train: {train_df['pickup_hour'].min().date()} → {train_df['pickup_hour'].max().date()}")
    print(f"   Test:  {test_df['pickup_hour'].min().date()}  → {test_df['pickup_hour'].max().date()}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    print(f"📊 Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # 5. MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("nyc-taxi-demand-forecasting")

    with mlflow.start_run(run_name="xgboost_2024_2026"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("top_zones",  top_zones)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows",  len(X_test))

        # 6. Train ─────────────────────────────────────────────────────────────
        print("\n🚀 Training XGBoost...")
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100,
        )

        # 7. Evaluate ──────────────────────────────────────────────────────────
        train_m = evaluate(y_train.values, model.predict(X_train))
        test_m  = evaluate(y_test.values,  model.predict(X_test))

        mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
        mlflow.log_metrics({f"test_{k}":  v for k, v in test_m.items()})

        print("\n── Results ──────────────────────────────────────────")
        print(f"  Train → MAE: {train_m['mae']:.2f} | RMSE: {train_m['rmse']:.2f} | R²: {train_m['r2']:.4f} | MAPE: {train_m['mape']:.2f}%")
        print(f"  Test  → MAE: {test_m['mae']:.2f}  | RMSE: {test_m['rmse']:.2f}  | R²: {test_m['r2']:.4f}  | MAPE: {test_m['mape']:.2f}%")

        # 8. Feature importances ───────────────────────────────────────────────
        feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
        mlflow.log_dict(feat_imp, "feature_importances.json")

        # 9. Save model + metadata ─────────────────────────────────────────────
        mlflow.xgboost.log_model(model, "model")

        # Save in both formats:
        # 1. Native UBJ format — memory-efficient, fast to load (preferred)
        ubj_path = MODEL_DIR / "xgb_demand_model.ubj"
        model.save_model(str(ubj_path))

        # 2. Pickle — fallback for compatibility
        model_path = MODEL_DIR / "xgb_demand_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        meta = {
            "top_zone_ids": top_zone_ids,
            "feature_cols": FEATURE_COLS,
            "test_mae":     test_m["mae"],
            "test_rmse":    test_m["rmse"],
            "test_r2":      test_m["r2"],
            "months":       ["2024-01→2026-02"],
        }
        meta_path = MODEL_DIR / "model_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact(str(ubj_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))

        run_id = mlflow.active_run().info.run_id
        elapsed = time.time() - t0
        print(f"\n✅ Done in {elapsed/60:.1f} min | Run ID: {run_id}")
        print(f"   Model → {model_path}")
        print("   View:    mlflow ui  →  http://localhost:5000")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-zones", type=int, default=100,
                        help="Train on top-N busiest zones (default: 100)")
    args = parser.parse_args()
    train(top_zones=args.top_zones)