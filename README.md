<div align="center">

# 🏦 LendGuard AI

### End-to-End Credit Risk Intelligence Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-2ECC71?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6B35?style=for-the-badge)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge)](https://mlflow.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8E44AD?style=for-the-badge)](https://shap.readthedocs.io)

<br/>

> *"In credit underwriting, a missed defaulter costs far more than a rejected good applicant. LendGuard AI is engineered around that truth."*

<br/>

**Author:** Puravi Pradhan &nbsp;|&nbsp; **Domain:** Financial Risk Analytics &nbsp;|&nbsp; **Reference:** PRCP-1006

</div>

---

## 🎬 Live Dashboard Demo
📺 **[Click here to watch the full LendGuard AI dashboard walkthrough video!](https://github.com/user-attachments/assets/db9ede29-9aff-4190-b0e8-ae099dc33830)**

---
## 📌 Overview

**LendGuard AI** is a production-grade machine learning system for predicting home loan default probability. Built on a 7-table relational dataset from the Home Credit Group, it covers the complete ML lifecycle — from raw data ingestion through deployment, monitoring, and regulatory documentation.

The pipeline is engineered around the **accuracy paradox**: with a 92:8 class imbalance, a naïve model predicts "no default" for everyone and achieves 91.9% accuracy while being completely useless. LendGuard AI addresses this directly through leak-free preprocessing, imbalance mitigation, PR-AUC-first evaluation, and decision threshold optimization.

---

## 🏗️ Architecture

```
Raw Data (7 CSVs)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 1 · Data Ingestion, EDA & Relational Aggregation │
│  • 7-table join with min/max/mean/sum/count roll-ups    │
│  • Memory optimization: 40–60% RAM reduction            │
│  • KDE distributions, correlation heatmaps, boxplots    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 2 · Preprocessing Pipeline (Leak-Free)           │
│  • 13 domain-crafted features                           │
│  • IQR capping · Log1p · OHE / Label Encoding           │
│  • SMOTEENN + Random Undersampling 2:1                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 3–4 · 8-Model Training + Hyperparameter Tuning   │
│  LR · SVM · DT · RF · AdaBoost · GBM · XGBoost · LGBM  │
│  RandomizedSearchCV · 5-Fold Stratified CV              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 5 · Explainability, Threshold & Business Reports │
│  • SHAP beeswarm, bar, waterfall plots                  │
│  • F1-optimal + Recall-constrained threshold tuning     │
│  • Model comparison radar + leaderboard matrix          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  PHASES 6–13 · Production Deployment Package            │
│  FastAPI REST · Streamlit Dashboard · MLflow Tracking   │
│  Docker Compose · Fairness Audit · PSI Drift Monitor    │
│  Model Card (Regulatory-Grade Documentation)            │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Dataset Architecture

| File | Description | Join Key |
|:-----|:------------|:---------|
| `application_train.csv` | Main table — static loan application data + TARGET | `SK_ID_CURR` |
| `bureau.csv` | External credit bureau history | `SK_ID_CURR` |
| `bureau_balance.csv` | Monthly balances of bureau credits | `SK_ID_BUREAU` |
| `previous_application.csv` | All prior credit applications | `SK_ID_CURR` |
| `POS_CASH_balance.csv` | Point-of-sale and cash loan monthly snapshots | `SK_ID_CURR` |
| `credit_card_balance.csv` | Credit card monthly snapshots | `SK_ID_CURR` |
| `installments_payments.csv` | Repayment and missed payment history | `SK_ID_CURR` |

> 300,000+ applicants × 500+ features after full relational aggregation

---

## ⚙️ Feature Engineering

### Domain-Crafted Features

| Feature | Formula | Business Rationale |
|:--------|:--------|:-------------------|
| `DEBT_BURDEN_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | % of income consumed by loan repayment |
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Total leverage relative to income |
| `EXT_SOURCE_MEAN` | `mean(EXT_SOURCE_1,2,3)` | Combined external credit score — strongest predictor |
| `EXT_SOURCE_WEIGHTED` | Weighted combination (50% SOURCE_2) | Emphasizes the most predictive bureau |
| `EMPLOYMENT_LIFE_FRACTION` | `DAYS_EMPLOYED / DAYS_BIRTH` | Employment stability as % of life |
| `GOODS_CREDIT_RATIO` | `AMT_GOODS_PRICE / AMT_CREDIT` | Over-financing indicator |
| `INCOME_PER_PERSON` | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` | Per-capita household income |

### Relational Aggregations
Each supplementary table is aggregated using **min, max, mean, sum, count** over `SK_ID_CURR`, producing a flat, single-row-per-applicant feature matrix from all 7 relational tables.

---

## 🤖 Models Trained

| Model | Notes |
|:------|:------|
| Logistic Regression | `class_weight=balanced`, SAGA solver |
| SVM (LinearSVC) | Calibrated with CalibratedClassifierCV |
| Decision Tree | Depth-constrained to prevent overfitting |
| Random Forest | n=200 → n=250 (tuned) |
| AdaBoost | n=200, lr=0.1 |
| Gradient Boosting | n=200, subsample=0.8 |
| **XGBoost** | `scale_pos_weight`, `tree_method=hist`, tuned |
| **LightGBM** ⭐ | Leaf-wise growth, GOSS sampling, tuned — **best performer** |

Hyperparameter tuning via **RandomizedSearchCV** (n_iter=20, 3-fold stratified) on RF, XGBoost, and LightGBM.

---

## 📊 Evaluation Strategy

### Why Not Accuracy?

With a 92:8 class split, accuracy is deceptive. LendGuard AI evaluates models on:

- **PR-AUC** — Primary metric; robust to class imbalance
- **Recall (Class 1)** — Catching defaulters is the core objective
- **ROC-AUC** — Secondary; overly optimistic on imbalanced data

### Decision Threshold Optimization

Rather than defaulting to `0.50`, the pipeline scans the full Precision-Recall curve to find:

| Strategy | Maximizes |
|:---------|:----------|
| F1-Optimal | Balanced precision/recall |
| Recall ≥ 0.80 constraint | Risk-averse lending policy |

### Cross-Validation Generalization Check
5-fold Stratified CV is compared against test AUC. A gap < 0.02 confirms the model generalizes and is not overfitting to the train split.

---

## 🔍 Explainability (SHAP)

Three SHAP analyses are produced for the best model:

1. **Beeswarm Summary** — Global feature importance + direction across 2,000 test samples
2. **Bar Summary** — Ranked mean |SHAP| values for top 20 features
3. **Waterfall Plot** — Individual explanation for the highest-risk applicant

> SHAP powers the `/explain` endpoint in the FastAPI service for per-applicant regulatory explanations.

---

## 🚀 Deployment Stack

### FastAPI REST Service (`deployment/app.py`)
```
GET  /health      → Model health + metadata
POST /predict     → Default probability + risk label
POST /explain     → Prediction + top-10 SHAP risk drivers
GET  /models/leaderboard → All model metrics
```

### Streamlit Dashboard (`deployment/streamlit_app.py`)
Interactive applicant scoring UI with:
- Real-time probability gauge
- Live SHAP bar chart for each prediction
- Model leaderboard with conditional formatting

### MLflow Experiment Tracking
All 11 model runs are logged with full metrics, parameters, and model artifacts. Best model is registered in the MLflow Model Registry.

### Docker Compose (Full Stack)
```bash
cd deployment && docker-compose up --build
```
Spins up three containers: **API** (`localhost:8000`) · **Dashboard** (`localhost:8501`) · **MLflow UI** (`localhost:5000`)

---

## 🛡️ Responsible AI

### Fairness Audit (Phase 10)
Audited across **Age Group**, **Gender**, and **Income Quartile**:

| Criterion | Method | Target |
|:----------|:-------|:-------|
| Disparate Impact | Predicted positive rate ratio | DIR ≥ 0.80 |
| Equalized Odds | TPR + FPR parity | Δ ≤ 5 pp |

### Data Drift Monitoring — PSI (Phase 11)
Population Stability Index tracks distribution shift over time:

| PSI Range | Status | Action |
|:----------|:-------|:-------|
| < 0.10 | ✅ Stable | No action |
| 0.10 – 0.20 | ⚠️ Monitor | Increased vigilance |
| > 0.20 | 🔴 Retrain | Trigger retraining pipeline |

### Model Card (Phase 13)
Regulatory-grade documentation covering: intended use, training data, performance, limitations, fairness audit results, and monitoring plan. Follows the Mitchell et al. (2019) Model Cards framework.

---

## 🗂️ Repository Structure

```
LendGuard-AI/
├── Exploration.ipynb           ← Full pipeline notebook (13 Phases)
└── deployment/
    ├── app.py                  ← FastAPI REST API
    ├── streamlit_app.py        ← Streamlit Dashboard
    ├── Dockerfile              ← Container definition
    ├── docker-compose.yml      ← Full stack orchestration
    ├── requirements.txt        ← Python dependencies
    ├── MODEL_CARD.md           ← Regulatory documentation
    ├── models/
    │   ├── best_model.pkl      ← Production model (LightGBM)
    │   ├── scaler.pkl          ← StandardScaler
    │   └── *.pkl               ← All trained models
    └── artifacts/
        ├── deployment_meta.json
        ├── model_leaderboard.csv
        ├── fairness_audit.png
        ├── drift_monitoring.png
        └── psi_report.csv
```

---

## ⚡ Quick Start

### Run the API locally
```bash
cd deployment
pip install -r requirements.txt
uvicorn app:app --reload
# Docs at http://localhost:8000/docs
```

### Launch the dashboard
```bash
streamlit run deployment/streamlit_app.py
```

### Full production stack (Docker)
```bash
cd deployment
docker-compose up --build
```

### Score an applicant (API call)
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "features": {
        "EXT_SOURCE_2": 0.45,
        "AMT_CREDIT": 600000,
        "AMT_INCOME_TOTAL": 120000,
        "DAYS_BIRTH": -10950,
        "DAYS_EMPLOYED": -730,
        "DEBT_BURDEN_RATIO": 0.28
    }
})
print(response.json())
# → {"default_probability": 0.712, "risk_label": "HIGH RISK — Likely Defaulter", ...}
```

---

## 🔧 Tech Stack

| Layer | Tools |
|:------|:------|
| Data & EDA | pandas, NumPy, seaborn, matplotlib |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Imbalance | imbalanced-learn (SMOTEENN, RandomUnderSampler) |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Containerization | Docker, Docker Compose |
| Serialization | joblib |

---

## 📈 Pipeline Phases at a Glance

| Phase | Description |
|:------|:------------|
| 1 | Relational Data Aggregation & EDA |
| 2 | Leak-Free Preprocessing Pipeline |
| 3 | 8-Model Training & Evaluation |
| 4 | Hyperparameter Optimization & CV Analysis |
| 5 | SHAP Explainability, Threshold Tuning, Business Reports |
| 6 | Model Serialization & Deployment Package |
| 7 | FastAPI Scoring Application |
| 8 | MLflow Experiment Tracking & Model Registry |
| 9 | Streamlit Interactive Dashboard |
| 10 | Fairness & Bias Audit |
| 11 | Data Drift Monitoring (PSI) |
| 12 | Dockerfile & Containerization |
| 13 | Model Card (Regulatory-Grade Documentation) |

---

<div align="center">

**LendGuard AI** — From raw CSV to production-ready credit risk intelligence.

*Built with rigor. Deployed with responsibility.*

</div>
