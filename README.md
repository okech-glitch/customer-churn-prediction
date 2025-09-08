# 💼 ChurnGuard AI: Customer Churn Prediction (Fullstack)

Python • React • FastAPI • scikit‑learn • Recharts • Docker • MIT License

Predict and prevent customer churn with an end‑to‑end ML application. Includes a FastAPI backend, React dashboard, training scripts, batch CSV support, real dataset stats, and portable artifacts.

## 🎯 What’s inside
- Single and Batch Churn Predictions (probabilities + confidence)
- Batch CSV Import in UI (chunked, preserves id/CustomerId)
- Server‑side CSV upload endpoint
- Real Stats API from train.csv (overall churn, per‑country, age, balance)
- Feature Importance and basic Explanations
- Trained model artifacts committed for immediate use

## 🏁 Quickstart (Local Demo)
Prereqs: Python 3.9+, Node.js 16+, Git

1) Put Kaggle files in `data/`: `train.csv`, `test.csv`, `sample_submission.csv`

2) (Optional) Retrain and create submission:
```bash
python scripts/simple_train.py
```
Artifacts: `models/*.joblib`, `data/submission.csv`

3) Start backend (API):
```bash
cd backend
python main.py  # http://localhost:8000
```

4) Start frontend (Dashboard):
```bash
cd ../frontend
npm install
npm start  # http://localhost:3000
```

## 🧭 Demo Flow
- Single Prediction tab: fill form → probability + confidence
- Batch Prediction tab: import `test.csv` → chunked processing → results preserve original `id`/`CustomerId` → Export
- Model Performance: feature importance and model summary
- Data Visualization: real stats from `/stats`

## 🔌 Backend API (local base: http://localhost:8000)
- `GET /health` → `{ status, models_loaded }`
- `GET /models` → model performance
- `GET /features` → feature importance
- `GET /stats` → dataset aggregates
- `POST /predict` → single JSON
- `POST /predict/batch` → `{ customers: [CustomerData] }`
- `POST /predict/file` → multipart `file=<csv>` (server‑side batch)
- `POST /explain` → contributions for a single JSON

CustomerData: `CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary`

Notes:
- Non‑features (id/CustomerId/Surname) are dropped and feature order is aligned to training.
- Batch APIs preserve original `id`/`CustomerId` if present.

## 🧠 Model & Training
- Pipeline: scikit‑learn Logistic Regression + Random Forest ensemble
- Encodes object columns; scales numeric; saves to `models/`
- Light tuning inside `scripts/simple_train.py`

Generate submission:
```bash
python scripts/simple_train.py  # writes data/submission.csv
```

## 📂 Structure
```
customer-churn-prediction/
├── backend/ (FastAPI: predict, batch, file, explain, features, stats, health)
├── frontend/ (React UI: batch import preserves IDs; uses /stats)
├── scripts/simple_train.py
├── data/ (CSV files; predictions.db log)
├── models/ (*.joblib, model_metadata.json)
├── docs/ (ARCHITECTURE.md, API.md, SETUP.md)
└── Dockerfile, docker-compose.yml, requirements.txt
```

## ⚙️ Environment (optional)
- Frontend: `REACT_APP_API_URL` (default http://localhost:8000)
- Backend: `PORT` (default 8000), `CORS_ORIGINS` (comma‑separated origins)

## 🧪 Tests
```bash
pytest -q
```

## 📜 License
MIT License – see `LICENSE`.


