# Medical Insurance Cost Prediction
## MLOps Project

An end-to-end MLOps system for predicting annual medical insurance charges using demographic and lifestyle features. Built with FastAPI, MLflow, Apache Airflow, DVC, Prometheus, Grafana, and React.

---

## Architecture Overview

```
React Frontend (3000) → FastAPI Backend (8001) → MLflow Registry (5000)
                                ↓
                        Prometheus (9090) → Grafana (3001)
                                
Airflow (8081) → DVC Pipeline → MLflow Tracking
```

---

## Prerequisites

- Docker + Docker Compose
- Python 3.10+
- Git + DVC (`pip install dvc`)

---

## Setup & Running

### Step 1 — Clone the repository

```bash
git clone <repo-url>
cd MLOPS-project-ed22b057
```

### Step 2 — Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Step 3 — Clear any stale MLflow state if any

```bash
rm -rf mlruns mlartifacts mlflow.db
```

### Step 4 — Build and start all Docker services

```bash
docker-compose build
docker-compose up -d
```

Wait 30 seconds for all services to be ready.

### Step 5 — Split the dataset (one-time only)

Run this script once to split the original dataset into initial data (80%) and holdout data (20%):

```bash
python split_data.py
```

This creates:
- `data/raw/initial_data.csv` — 1070 rows used for training
- `data/raw/holdout_data.csv` — 268 rows locked away to simulate future data

> **Note:** Only run this once. Running it again will re-split the data with a different random seed and invalidate existing DVC tracked files.

### Step 6 — Trigger the Airflow ingestion pipeline

Open `http://localhost:8081` → login: `admin / admin`

Find `ingestion_dag` → click the play button to trigger it.

Wait for all 4 tasks to turn green (validate → clean → feature_engineer → compute_baselines).

### Step 7 — Run the DVC ML pipeline

```bash
dvc repro -f
```

This runs all 4 stages: preprocess → train → evaluate → register.

To train a different model, change `active_model` in `params.yaml` and run `dvc repro` again:

```yaml
# params.yaml
active_model: xgboost   # options: linear_regression | ridge | lasso | random_forest | xgboost
```

### Step 8 — Promote the best model in MLflow UI

Open `http://localhost:5000`

Compare all model runs by RMSE. Click on the best run → go to the model version → set alias to `production`.

### Step 9 — Reload the model in the backend

```bash
curl -X POST http://localhost:8001/reload
```

Or use the "Reload Production Model" button on the Pipeline page.

### Step 10 — Import the Grafana Dashboard

1. Open `http://localhost:3001` → login: `admin / admin`
2. Go to **Connections → Data Sources → Add new data source → Prometheus**
3. Set URL to `http://prometheus:9090` → click **Save & Test**
4. Go to **Dashboards → Import**
5. Upload `monitoring/grafana_dashboard.json`
6. When prompted for the datasource, select the Prometheus datasource you just added
7. Click **Import**

The dashboard will show live metrics once predictions are made via `/predict`.

### Step 11 — Open the frontend

```
http://localhost:3000
```

---

## Running All 5 Models

To train and compare all 5 models:

```bash
for model in linear_regression ridge lasso random_forest xgboost; do
  sed -i "s/active_model: .*/active_model: $model/" params.yaml
  dvc repro -f
done
```

Then compare in MLflow UI at `http://localhost:5000` and promote the best model.

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Frontend | http://localhost:3000 | — |
| FastAPI | http://localhost:8001/docs | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8081 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --html=reports/test_report.html --self-contained-html -v
```

---

## Simulating Drift & Retraining

1. Open Airflow at `http://localhost:8081`
2. Trigger `retrain_dag` manually
3. The DAG will:
   - Load holdout data (20% of original dataset, never used in training)
   - Compute PSI drift scores against training baselines
   - Evaluate production model on holdout data
   - If drift detected or RMSE degrades → combine datasets and write `data/retrain_triggered.txt`
4. If retraining is triggered, run `dvc repro -f` on the host
5. Promote the new model in MLflow UI and reload via `/reload`

---

## Project Structure

```
├── backend/              FastAPI application
├── data/                 Raw and processed data (DVC tracked)
├── docs/                 Architecture, HLD, LLD, test plan, user manual
├── frontend/             React + Vite application
├── mlartifacts/          MLflow artifact store
├── models/               Model artifacts
├── monitoring/           Prometheus config + Grafana dashboard
├── pipeline/
│   ├── airflow/dags/     Airflow DAG definitions
│   ├── models/           One .py file per ML model
│   ├── preprocess.py     DVC Stage 1
│   ├── train.py          DVC Stage 2
│   ├── evaluate.py       DVC Stage 3
│   └── register.py       DVC Stage 4
├── tests/                pytest test files
├── docker-compose.yml    All services
├── dvc.yaml              Pipeline DAG definition
├── mlflow.Dockerfile     MLflow server image
├── params.yaml           Active model + hyperparameters
└── split_data.py         One-time holdout split script
```

---

## Documentation

All documentation is in the `docs/` folder:

| Document | Description |
|---|---|
| architecture.md | System architecture diagram with block explanations |
| HLD.md | High-level design choices and rationale |
| LLD.md | API endpoint definitions and I/O specifications |
| test_plan.md | Test cases, acceptance criteria, test report |
| user_manual.md | Non-technical user guide |

---

## Known Limitations

- **Manual model promotion**: After `dvc repro`, the best model must be promoted to Production manually in the MLflow UI. This is intentional — it allows comparing all runs before deploying.
- **dvc repro runs on host**: The DVC pipeline runs on the host machine (not inside Docker) because it needs access to the local filesystem for MLflow artifacts.