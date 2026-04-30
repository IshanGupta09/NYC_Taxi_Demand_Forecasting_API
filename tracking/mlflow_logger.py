"""
tracking/mlflow_logger.py
──────────────────────────
Convenience wrapper around MLflow for consistent experiment logging
across training runs and batch inference jobs.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow

# ── Config ─────────────────────────────────────────────────────────────────────
TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT",   "nyc-taxi-demand-forecasting")


class ExperimentLogger:
    """
    Context manager + helper for logging to MLflow.

    Usage:
        with ExperimentLogger("xgboost_run") as logger:
            logger.log_params({"lr": 0.05, "depth": 7})
            model.fit(X, y)
            logger.log_metrics({"mae": 4.2, "rmse": 6.1})
            logger.log_model(model, "xgb_model")
    """

    def __init__(self, run_name: str | None = None):
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT)
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run = None

    def __enter__(self):
        self._run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, *args):
        mlflow.end_run()

    # ── Logging helpers ────────────────────────────────────────────────────────
    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_dict(self, data: dict, filename: str) -> None:
        mlflow.log_dict(data, filename)

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Auto-detects XGBoost vs sklearn and logs accordingly."""
        try:
            import xgboost as xgb
            if isinstance(model, xgb.XGBModel):
                mlflow.xgboost.log_model(model, artifact_path)
                return
        except ImportError:
            pass

        mlflow.sklearn.log_model(model, artifact_path)

    def log_feature_importances(self, importances: dict[str, float]) -> None:
        """Log feature importances as a JSON artifact and as metrics."""
        self.log_dict(importances, "feature_importances.json")
        # Also log top-10 as metrics for quick comparison in UI
        top10 = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        mlflow.log_metrics({f"fi_{k}": v for k, v in top10.items()})

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None


# ── Convenience: log inference results in batch jobs ──────────────────────────
def log_inference_run(
    zone_id:    int,
    start_dt:   str,
    hours:      int,
    predictions: list[dict],
    model_version: str = "xgb_v1",
) -> None:
    """Log a batch inference run as a lightweight MLflow run."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"{EXPERIMENT}-inference")

    avg_demand = sum(p["predicted_demand"] for p in predictions) / max(len(predictions), 1)

    with mlflow.start_run(run_name=f"inference_zone{zone_id}_{start_dt[:10]}"):
        mlflow.log_params({
            "zone_id":       zone_id,
            "start_dt":      start_dt,
            "hours":         hours,
            "model_version": model_version,
        })
        mlflow.log_metrics({
            "avg_predicted_demand": round(avg_demand, 2),
            "total_hours":          hours,
        })
        mlflow.log_dict(
            {"predictions": predictions},
            "predictions.json"
        )