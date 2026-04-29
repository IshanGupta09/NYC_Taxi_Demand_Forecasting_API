"""
api/main.py
───────────
FastAPI application serving the NYC Taxi Demand Forecasting model.

Endpoints:
  GET  /                     → health check
  GET  /model/info           → model metadata
  POST /predict              → single-hour prediction
  POST /predict/batch        → multi-zone predictions
  POST /predict/next-hours   → forecast next N hours for a zone
  GET  /zones                → list of valid zone IDs
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.predict import get_predictor

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🚕 NYC Taxi Demand Forecasting API",
    description = (
        "Predicts hourly taxi pickup demand per NYC TLC zone "
        "using an XGBoost model trained on 2026 Yellow Taxi trip data."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    zone_id:         int   = Field(...,  ge=1, le=265,
                                   description="NYC TLC pickup zone ID (1–265)",
                                   json_schema_extra={"example": 161})
    target_datetime: str   = Field(...,
                                   description="Target datetime (ISO format)",
                                   json_schema_extra={"example": "2026-03-20 08:00:00"})
    avg_fare:        float = Field(15.0, ge=0,
                                   description="Expected average fare in USD")
    avg_distance:    float = Field(2.5,  ge=0,
                                   description="Expected average trip distance in miles")
    lag_values:      Optional[dict] = Field(None,
                                   description="Optional historical lag values dict")

    @field_validator("target_datetime")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("target_datetime must be ISO format: YYYY-MM-DD HH:MM:SS")
        return v


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(
        ..., max_length=50,
        description="Up to 50 prediction requests in one call"
    )


class NextHoursRequest(BaseModel):
    zone_id:        int   = Field(...,  ge=1, le=265,
                                  description="NYC TLC pickup zone ID (1–265)")
    start_datetime: str   = Field(...,
                                  description="Start datetime (ISO format)",
                                  json_schema_extra={"example": "2026-03-20 08:00:00"})
    hours:          int   = Field(24,   ge=1, le=168,
                                  description="Hours to forecast (max 168 = 1 week)")
    avg_fare:       float = Field(15.0, ge=0)
    avg_distance:   float = Field(2.5,  ge=0)

    @field_validator("start_datetime")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("start_datetime must be ISO format: YYYY-MM-DD HH:MM:SS")
        return v


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "ok",
        "service": "NYC Taxi Demand Forecasting API",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata about the currently loaded model."""
    try:
        predictor = get_predictor()
        return predictor.model_info
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/zones", tags=["Zones"])
def list_zones():
    """Return all valid zone IDs the model was trained on."""
    try:
        predictor = get_predictor()
        return {
            "valid_zone_ids": sorted(predictor.valid_zones),
            "count":          len(predictor.valid_zones),
            "note":           "These are the top busiest NYC TLC pickup zones",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", tags=["Predictions"])
def predict(req: PredictRequest):
    """
    Predict hourly taxi demand for a single zone and datetime.

    - **zone_id**: NYC TLC Taxi Zone ID (e.g. 161 = Midtown Center)
    - **target_datetime**: The hour to forecast (e.g. "2026-03-20 08:00:00")
    """
    try:
        predictor = get_predictor()
        result    = predictor.predict(
            zone_id          = req.zone_id,
            target_datetime  = req.target_datetime,
            lag_values       = req.lag_values,
            avg_fare         = req.avg_fare,
            avg_distance     = req.avg_distance,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", tags=["Predictions"])
def predict_batch(req: BatchPredictRequest):
    """
    Predict demand for up to 50 zone/datetime pairs in one call.
    """
    try:
        predictor = get_predictor()
        results   = []
        errors    = []

        for i, r in enumerate(req.requests):
            try:
                pred = predictor.predict(
                    zone_id         = r.zone_id,
                    target_datetime = r.target_datetime,
                    lag_values      = r.lag_values,
                    avg_fare        = r.avg_fare,
                    avg_distance    = r.avg_distance,
                )
                results.append({"index": i, "status": "ok", **pred})
            except ValueError as e:
                errors.append({"index": i, "status": "error", "detail": str(e)})

        return {"predictions": results, "errors": errors, "total": len(req.requests)}

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/next-hours", tags=["Predictions"])
def predict_next_hours(req: NextHoursRequest):
    """
    Forecast demand for the next N hours for a given zone.

    Perfect for demand planning — e.g., forecast the next 24 hours
    starting from 8 AM for Midtown Manhattan (zone 161).
    """
    try:
        predictor = get_predictor()
        results   = predictor.predict_next_hours(
            zone_id      = req.zone_id,
            start_dt     = req.start_datetime,
            hours        = req.hours,
            avg_fare     = req.avg_fare,
            avg_distance = req.avg_distance,
        )
        return {
            "zone_id":     req.zone_id,
            "start":       req.start_datetime,
            "hours":       req.hours,
            "predictions": results,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
