"""
src/predict.py
──────────────
Inference logic — loads the trained XGBoost model and generates predictions.
Used by both the FastAPI app and standalone scripts.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "xgb_demand_model.pkl"
META_PATH  = ROOT / "models" / "model_meta.json"

# ── Holidays (same list as features.py) ───────────────────────────────────────
NYC_HOLIDAYS = pd.to_datetime([
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
])


class DemandPredictor:
    """
    Loads the trained model and generates demand predictions.

    Example:
        predictor = DemandPredictor()
        result = predictor.predict(zone_id=161, target_datetime="2026-03-15 08:00:00")
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python src/train.py` first."
            )

        # Use XGBoost's native loader — much more memory-efficient than pickle
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            # Try native format first (JSON/UBJ), fall back to pickle
            native_path = MODEL_PATH.parent / "xgb_demand_model.json"
            ubj_path    = MODEL_PATH.parent / "xgb_demand_model.ubj"

            if ubj_path.exists():
                self.model.load_model(str(ubj_path))
            elif native_path.exists():
                self.model.load_model(str(native_path))
            else:
                # Fall back to pickle with explicit gc
                import gc
                gc.collect()
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                gc.collect()
        except MemoryError:
            raise MemoryError(
                "Not enough RAM to load model. "
                "Try closing other applications and restarting."
            )

        with open(META_PATH, "r") as f:
            self.meta = json.load(f)

        self.valid_zones = set(self.meta["top_zone_ids"])
        self.feature_cols = self.meta["feature_cols"]

    def _build_feature_row(
        self,
        zone_id:         int,
        target_dt:       datetime,
        lag_values:      dict | None = None,
        avg_fare:        float = 15.0,
        avg_distance:    float = 2.5,
    ) -> pd.DataFrame:
        """
        Build a single-row feature DataFrame for inference.

        lag_values: dict with keys lag_1h, lag_2h, ..., lag_168h.
                    If None, fills with reasonable defaults (rolling avg).
        """
        dt = pd.Timestamp(target_dt)
        hour        = dt.hour
        dow         = dt.dayofweek
        month       = dt.month
        is_holiday  = int(dt.normalize() in NYC_HOLIDAYS)
        is_weekend  = int(dow >= 5)
        is_rush     = int(
            (not is_weekend) and (hour in range(7, 10) or hour in range(16, 20))
        )

        # Sensible defaults for lag features when historical data isn't supplied
        defaults = {
            "lag_1h":        30,
            "lag_2h":        28,
            "lag_3h":        25,
            "lag_24h":       35,
            "lag_48h":       33,
            "lag_168h":      38,
            "lag_8760h":     36,   # same hour last year
            "lag_8784h":     35,
            "roll_mean_3h":  28,
            "roll_mean_6h":  27,
            "roll_mean_24h": 30,
            "roll_mean_7d":  32,
            "roll_std_24h":  8,
            "roll_max_24h":  55,
        }
        if lag_values:
            defaults.update(lag_values)

        row = {
            "zone_id":       zone_id,
            "hour":          hour,
            "day_of_week":   dow,
            "day_of_month":  dt.day,
            "month":         month,
            "week_of_year":  dt.isocalendar()[1],
            "is_weekend":    is_weekend,
            "is_holiday":    is_holiday,
            "is_rush_hour":  is_rush,
            "hour_sin":      np.sin(2 * np.pi * hour  / 24),
            "hour_cos":      np.cos(2 * np.pi * hour  / 24),
            "dow_sin":       np.sin(2 * np.pi * dow   / 7),
            "dow_cos":       np.cos(2 * np.pi * dow   / 7),
            "month_sin":     np.sin(2 * np.pi * month / 12),
            "month_cos":     np.cos(2 * np.pi * month / 12),
            "avg_fare":      avg_fare,
            "avg_distance":  avg_distance,
            **defaults,
        }
        return pd.DataFrame([row])[self.feature_cols]

    def predict(
        self,
        zone_id:         int,
        target_datetime: str | datetime,
        lag_values:      dict | None = None,
        avg_fare:        float = 15.0,
        avg_distance:    float = 2.5,
    ) -> dict:
        """
        Predict taxi demand for a given zone and datetime.

        Args:
            zone_id         : NYC TLC pickup zone ID (1–265)
            target_datetime : ISO datetime string or datetime object
            lag_values      : optional dict of historical lag values
            avg_fare        : average fare in the zone (default $15)
            avg_distance    : average trip distance in miles (default 2.5)

        Returns:
            dict with prediction details
        """
        if zone_id not in self.valid_zones:
            raise ValueError(
                f"zone_id {zone_id} not in trained zones. "
                f"Valid zone IDs: {sorted(self.valid_zones)[:10]}..."
            )

        dt      = pd.Timestamp(target_datetime)
        X       = self._build_feature_row(zone_id, dt, lag_values, avg_fare, avg_distance)
        raw_pred = float(self.model.predict(X)[0])
        demand   = max(0, round(raw_pred))   # clamp negative → 0, round to int

        return {
            "zone_id":          zone_id,
            "target_datetime":  dt.isoformat(),
            "predicted_demand": demand,
            "model_version":    "xgb_v1",
            "confidence_note":  "Point estimate; ±15% typical MAPE",
        }

    def predict_next_hours(
        self,
        zone_id:    int,
        start_dt:   str | datetime,
        hours:      int = 24,
        avg_fare:   float = 15.0,
        avg_distance: float = 2.5,
    ) -> list[dict]:
        """
        Predict demand for the next `hours` hours starting from `start_dt`.

        Returns a list of hourly predictions.
        """
        start = pd.Timestamp(start_dt).floor("h")
        results = []
        for h in range(hours):
            target = start + timedelta(hours=h)
            pred   = self.predict(zone_id, target, avg_fare=avg_fare,
                                  avg_distance=avg_distance)
            results.append(pred)
        return results

    @property
    def model_info(self) -> dict:
        return {
            "model_type":    "XGBoost Regressor",
            "trained_on":    self.meta.get("months"),
            "zones_covered": len(self.valid_zones),
            "test_mae":      self.meta.get("test_mae"),
            "test_rmse":     self.meta.get("test_rmse"),
            "test_r2":       self.meta.get("test_r2"),
        }


# ── Lazy singleton ─────────────────────────────────────────────────────────────
_predictor: DemandPredictor | None = None

def get_predictor() -> DemandPredictor:
    """Return a cached DemandPredictor instance (loaded once at startup)."""
    global _predictor
    if _predictor is None:
        _predictor = DemandPredictor()
    return _predictor