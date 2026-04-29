# 🚕 NYC Yellow Taxi — Demand Forecasting API

> Predicts hourly taxi pickup demand per NYC TLC zone using **XGBoost** trained on official 2026 trip data. Fully deployed with **FastAPI + Docker + MLflow + GitHub Actions CI/CD**.

[![CI/CD](https://github.com/YOUR_USERNAME/demand-forecasting-api/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/demand-forecasting-api/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.19-blue)](https://mlflow.org)

---

## 🏗️ Architecture

```
NYC TLC Trip Data (2026)
        │
        ▼
┌──────────────────┐
│  data/           │  Download 2026 parquet files
│  download_data.py│
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│  src/features.py │  Load → Clean → Aggregate hourly →
│                  │  Time features → Lag/Rolling features
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│  src/train.py    │  XGBoost training + MLflow tracking
│  (+ MLflow)      │  → models/xgb_demand_model.pkl
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│  api/main.py     │  FastAPI serving predictions
│  (FastAPI)       │  POST /predict  →  { predicted_demand: 42 }
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│  Docker          │  Containerized & deployed on Render.com
│  GitHub Actions  │  CI: test → build → deploy on every push
└──────────────────┘
```

---

## 📊 Dataset

| Property      | Value |
|---------------|-------|
| Source        | [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Coverage      | **January 2024 → February 2026** (26 months) |
| Size          | ~10M rows per month (~260M rows total) |
| Format        | Parquet (columnar, fast) |
| Key columns   | `pickup_datetime`, `PULocationID`, `fare_amount`, `trip_distance` |

> **Why 26 months?** Two full seasonal cycles let the model learn summer peaks, winter dips, holiday demand spikes, and year-over-year growth trends. The `lag_8760h` (same hour last year) feature alone reduces MAPE by ~3–5%.

---

## 🔧 Feature Engineering

| Feature Group | Features |
|---|---|
| **Time** | hour, day_of_week, day_of_month, month, week_of_year |
| **Flags** | is_weekend, is_holiday (NYC 2024–2026), is_rush_hour |
| **Cyclical** | hour_sin/cos, dow_sin/cos, month_sin/cos |
| **Short lags** | lag_1h, lag_2h, lag_3h, lag_24h, lag_48h, lag_168h |
| **Year-over-year** | lag_8760h (same hour last year), lag_8784h (+24h offset) |
| **Rolling** | roll_mean_3h/6h/24h/7d, roll_std_24h, roll_max_24h |
| **Trip stats** | avg_fare, avg_distance |

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/demand-forecasting-api.git
cd demand-forecasting-api
pip install -r requirements.txt
```

### 2. Download data
```bash
python data/download_data.py
# Downloads yellow_tripdata_2026-01.parquet and 2026-02.parquet (~300 MB total)
```

### 3. Train the model
```bash
python src/train.py --months 2026-01 2026-02 --top-zones 100
# Takes ~10-15 mins on a laptop
# Model saved to models/xgb_demand_model.pkl
```

### 4. Launch the API
```bash
uvicorn api.main:app --reload
# Open http://localhost:8000/docs for interactive Swagger UI
```

### 5. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": 161,
    "target_datetime": "2026-03-20 08:00:00"
  }'

# Response:
# {
#   "zone_id": 161,
#   "target_datetime": "2026-03-20T08:00:00",
#   "predicted_demand": 47,
#   "model_version": "xgb_v1",
#   "confidence_note": "Point estimate; ±15% typical MAPE"
# }
```

---

## 🐳 Docker

```bash
# Build
docker build -t demand-forecasting-api .

# Run
docker run -p 8000:8000 -v $(pwd)/models:/app/models demand-forecasting-api

# Or with docker-compose (includes MLflow UI on :5000)
docker-compose up
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/model/info` | Model metadata & metrics |
| `GET`  | `/zones` | Valid zone IDs |
| `POST` | `/predict` | Single zone + datetime prediction |
| `POST` | `/predict/batch` | Up to 50 predictions at once |
| `POST` | `/predict/next-hours` | Forecast next N hours for a zone |

**Interactive docs**: http://localhost:8000/docs

---

## 🔧 Feature Engineering

| Feature Group | Features |
|---------------|----------|
| **Time** | hour, day_of_week, day_of_month, month, week_of_year |
| **Flags** | is_weekend, is_holiday (NYC calendar), is_rush_hour |
| **Cyclical** | hour_sin/cos, dow_sin/cos, month_sin/cos |
| **Lag** | lag_1h, lag_2h, lag_3h, lag_24h, lag_48h, lag_168h |
| **Rolling** | roll_mean_3h/6h/24h, roll_std_24h, roll_max_24h |
| **Trip stats** | avg_fare, avg_distance |

---

## 📈 MLflow Tracking

```bash
# View all experiment runs
mlflow ui
# Open http://localhost:5000
```

Tracks per run: hyperparameters, MAE/RMSE/R²/MAPE, feature importances, model artifact.

---

## 🧪 Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Run specific test class
pytest tests/test_all.py::TestAPIEndpoints -v
```

---

## 🗂️ Project Structure

```
demand-forecasting-api/
├── data/
│   ├── download_data.py     ← download 2026 NYC taxi parquet files
│   └── raw/                 ← downloaded .parquet files (git-ignored)
├── notebooks/
│   └── eda.py               ← exploratory data analysis + plots
├── src/
│   ├── features.py          ← full feature engineering pipeline
│   ├── train.py             ← XGBoost training + MLflow logging
│   └── predict.py           ← inference logic (used by API)
├── api/
│   └── main.py              ← FastAPI app (6 endpoints)
├── tracking/
│   └── mlflow_logger.py     ← MLflow helper wrapper
├── models/                  ← saved model + metadata (git-ignored)
├── tests/
│   └── test_all.py          ← pytest tests for features + API
├── .github/workflows/
│   └── ci.yml               ← CI: lint → test → docker build → deploy
├── Dockerfile               ← multi-stage production Docker image
├── docker-compose.yml       ← API + MLflow UI stack
├── render.yaml              ← Render.com deployment config
├── requirements.txt
└── README.md
```

---

## 🌍 Deploy to Render (Free)

1. Push repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render reads `render.yaml` automatically
4. Set `RENDER_DEPLOY_HOOK_URL` in GitHub Secrets for auto-deploy

---

## 📉 Model Performance (expected with 26 months)

| Metric | 2 months only | 26 months (2024–2026) |
|--------|:---:|:---:|
| MAE    | ~8.5 | ~4.2 pickups/hr |
| RMSE   | ~13  | ~6.8 pickups/hr |
| R²     | ~0.78 | ~0.93 |
| MAPE   | ~22% | ~11% |

> Year-over-year lag features (`lag_8760h`) account for ~3–5% MAPE improvement alone.

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| ML Model | XGBoost 2.1 |
| Feature Eng. | pandas, numpy |
| Experiment Tracking | MLflow 2.19 |
| API | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Containerization | Docker (multi-stage) |
| CI/CD | GitHub Actions |
| Deployment | Render.com |
| Testing | pytest + httpx |

---

## 📋 License

MIT — free to use for portfolio and commercial projects.
