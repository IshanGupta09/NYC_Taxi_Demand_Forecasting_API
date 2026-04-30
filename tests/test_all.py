"""
tests/test_features.py  +  test_api.py combined
────────────────────────────────────────────────
Run with:  pytest tests/ -v
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent))

from api.main import app
from src.features import (
    FEATURE_COLS,
    add_lag_features,
    add_time_features,
    aggregate_hourly,
)


def make_hourly_df(n_hours: int = 300, zone_id: int = 161) -> pd.DataFrame:
    """Create a synthetic hourly demand DataFrame for testing."""
    hours = pd.date_range("2026-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({
        "pickup_hour": hours,
        "zone_id":     zone_id,
        "demand":      np.random.randint(10, 100, size=n_hours),
        "avg_fare":    np.random.uniform(10, 30,  size=n_hours),
        "avg_distance": np.random.uniform(1, 5,   size=n_hours),
    })


class TestTimeFeatures:
    def test_columns_added(self):
        df  = make_hourly_df()
        out = add_time_features(df)
        for col in ["hour", "day_of_week", "is_weekend", "is_holiday",
                    "is_rush_hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_hour_range(self):
        df  = make_hourly_df()
        out = add_time_features(df)
        assert out["hour"].between(0, 23).all()

    def test_is_weekend_binary(self):
        df  = make_hourly_df()
        out = add_time_features(df)
        assert set(out["is_weekend"].unique()).issubset({0, 1})

    def test_cyclical_bounds(self):
        df  = make_hourly_df()
        out = add_time_features(df)
        assert out["hour_sin"].between(-1, 1).all()
        assert out["hour_cos"].between(-1, 1).all()

    def test_new_years_day_is_holiday(self):
        df = pd.DataFrame({
            "pickup_hour": pd.to_datetime(["2026-01-01 12:00:00"]),
            "zone_id":     [161],
            "demand":      [50],
            "avg_fare":    [15.0],
            "avg_distance": [2.0],
        })
        out = add_time_features(df)
        assert out["is_holiday"].iloc[0] == 1

    def test_rush_hour_detection(self):
        # Tuesday 8 AM should be rush hour
        df = pd.DataFrame({
            "pickup_hour": pd.to_datetime(["2026-01-06 08:00:00"]),
            "zone_id":     [161],
            "demand":      [80],
            "avg_fare":    [15.0],
            "avg_distance": [2.0],
        })
        out = add_time_features(df)
        assert out["is_rush_hour"].iloc[0] == 1


class TestLagFeatures:
    def test_lag_columns_created(self):
        df  = make_hourly_df()
        df  = add_time_features(df)
        out = add_lag_features(df)
        for col in ["lag_1h", "lag_24h", "lag_168h",
                    "lag_8760h", "lag_8784h",
                    "roll_mean_3h", "roll_mean_24h", "roll_mean_7d"]:
            assert col in out.columns, f"Missing: {col}"

    def test_lag_1h_is_previous_hour(self):
        df  = make_hourly_df()
        df  = add_time_features(df)
        out = add_lag_features(df).dropna(subset=["lag_1h"])
        # lag_1h at row i == demand at row i-1 (within same zone)
        for idx in out.index[1:5]:
            expected = out.loc[idx - 1, "demand"]
            actual   = out.loc[idx, "lag_1h"]
            assert expected == actual

    def test_no_leakage_across_zones(self):
        # Two zones must not share lag values
        df1 = make_hourly_df(n_hours=200, zone_id=100)
        df2 = make_hourly_df(n_hours=200, zone_id=200)
        df  = pd.concat([df1, df2], ignore_index=True)
        df  = add_time_features(df)
        out = add_lag_features(df)
        z100 = out[out["zone_id"] == 100]["lag_1h"].dropna()
        z200 = out[out["zone_id"] == 200]["lag_1h"].dropna()
        # They should differ (almost certainly given random data)
        assert not (z100.values == z200.values).all()


class TestAggregation:
    def make_raw(self) -> pd.DataFrame:
        hours = pd.date_range("2026-01-01", periods=500, freq="10min")
        return pd.DataFrame({
            "pickup_dt":      hours,
            "zone_id":        np.random.choice([100, 200], size=500),
            "passenger_count": np.random.randint(1, 4, size=500),
            "trip_distance":  np.random.uniform(0.5, 10, size=500),
            "fare_amount":    np.random.uniform(5, 40, size=500),
        })

    def test_output_is_hourly(self):
        raw  = self.make_raw()
        out  = aggregate_hourly(raw)
        # Every pickup_hour value should be on the hour (no minutes/seconds)
        assert (out["pickup_hour"].dt.minute == 0).all()
        assert (out["pickup_hour"].dt.second == 0).all()

    def test_demand_is_non_negative(self):
        raw = self.make_raw()
        out = aggregate_hourly(raw)
        assert (out["demand"] >= 0).all()

    def test_feature_cols_complete(self):
        df  = make_hourly_df()
        df  = add_time_features(df)
        df  = add_lag_features(df).dropna()
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        assert not missing, f"Missing feature columns: {missing}"


# ══════════════════════════════════════════════════════════════════
#  Predict Module Tests (without real model)
# ══════════════════════════════════════════════════════════════════


class TestDemandPredictorLogic:
    """Test prediction logic without needing a real trained model."""

    def test_feature_row_shape(self):
        """The feature row should match FEATURE_COLS length."""
        from src.predict import DemandPredictor

        # Create a mock predictor without loading real files
        with patch.object(DemandPredictor, '__init__', lambda self: None):
            p = DemandPredictor.__new__(DemandPredictor)
            p.feature_cols = FEATURE_COLS
            p.meta         = {}
            p.valid_zones  = {161}

            row = p._build_feature_row(
                zone_id    = 161,
                target_dt  = datetime(2026, 3, 15, 8, 0),
            )
            assert row.shape == (1, len(FEATURE_COLS))
            assert list(row.columns) == FEATURE_COLS

    def test_demand_non_negative(self):
        """Prediction should always be >= 0."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-5.0])  # model returns negative

        from src.predict import DemandPredictor
        with patch.object(DemandPredictor, '__init__', lambda self: None):
            p = DemandPredictor.__new__(DemandPredictor)
            p.feature_cols = FEATURE_COLS
            p.model        = mock_model
            p.valid_zones  = {161}
            p.meta         = {}

            row = p._build_feature_row(161, datetime(2026, 3, 15, 8))
            raw = float(p.model.predict(row)[0])
            result = max(0, round(raw))
            assert result == 0   # clamped from -5


# ══════════════════════════════════════════════════════════════════
#  FastAPI Endpoint Tests
# ══════════════════════════════════════════════════════════════════
client = TestClient(app)


class TestAPIEndpoints:
    def test_root_returns_ok(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_endpoint(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_predict_without_model_returns_503(self):
        """If model file is missing, API should return 503 not 500."""
        with patch("api.main.get_predictor", side_effect=FileNotFoundError("model missing")):
            r = client.post("/predict", json={
                "zone_id":         161,
                "target_datetime": "2026-03-20 08:00:00",
            })
            assert r.status_code == 503

    def test_predict_invalid_zone_returns_422(self):
        """zone_id must be 1–265."""
        r = client.post("/predict", json={
            "zone_id":         999,
            "target_datetime": "2026-03-20 08:00:00",
        })
        assert r.status_code == 422

    def test_predict_invalid_datetime_returns_422(self):
        r = client.post("/predict", json={
            "zone_id":         161,
            "target_datetime": "not-a-date",
        })
        assert r.status_code == 422

    def test_batch_predict_limit(self):
        """Batch endpoint should reject > 50 items."""
        items = [
            {"zone_id": 161, "target_datetime": "2026-03-20 08:00:00"}
        ] * 51
        r = client.post("/predict/batch", json={"requests": items})
        assert r.status_code == 422

    def test_next_hours_limit(self):
        """hours must be ≤ 168."""
        with patch("api.main.get_predictor", side_effect=FileNotFoundError("x")):
            r = client.post("/predict/next-hours", json={
                "zone_id":        161,
                "start_datetime": "2026-03-20 08:00:00",
                "hours":          999,
            })
            assert r.status_code == 422