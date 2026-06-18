# 🏦 Model Card — Home Loan Default Risk Classifier
 
**Model Name:** Gradient Boosting  
**Version:** 1.0.0  
**Date:** 2026-06-08  
**Domain:** Banking & Financial Risk  
**Task:** Binary Classification — Loan Default Prediction  
**Project:** PRCP-1006
 
---
 
## Model Details
 
| Attribute | Value |
|---|---|
| Algorithm | Gradient Boosting |
| Framework | scikit-learn / LightGBM / XGBoost |
| Input Features | 325 engineered features |
| Output | Default probability ∈ [0, 1] + binary label |
| Decision Threshold | 0.1699 (F1-optimal via PR curve) |
| Primary Metric | PR-AUC (imbalance-robust) |
| Secondary Metric | Recall for class 1 (defaulter detection) |
 
---
 
## Intended Use
 
**Primary Use:**
Assist loan officers at financial institutions in identifying applicants at high risk
of default before loan disbursement.
 
**Intended Users:** Credit risk analysts, loan officers, risk management teams.
 
**Out-of-Scope Use:**
- Sole basis for loan rejection without human review
- Use outside of home loan / personal credit domain
- Deployment on populations significantly different from Home Credit Group demographics
 
---
 
## Training Data
 
| Attribute | Details |
|---|---|
| Source | Home Credit Group — Kaggle Competition Dataset |
| Tables | 7 relational tables (application, bureau, POS cash, credit card, installments) |
| Training Rows | ~246,000 applicants |
| Test Rows | ~62,000 applicants |
| Class Distribution | ~91.9% Non-Default (0) / ~8.1% Default (1) |
| Imbalance Strategy | Random Undersampling 2:1 + class_weight balanced |
 
**Feature Engineering:**
- 13+ domain-crafted features (DEBT_BURDEN_RATIO, EXT_SOURCE composites, etc.)
- Relational aggregations across 6 supplementary tables
- Log1p skewness correction, IQR outlier capping, StandardScaler
 
---
 
## Performance Metrics (Held-Out Test Set — 20% Split)
 
| Metric | Score |
|---|---|
| ROC-AUC | 0.7809 |
| PR-AUC | 0.2816 |
| Recall (Defaulter) | 0.0389 |
| Precision (Defaulter) | 0.605 |
| F1 Score (Defaulter) | 0.0731 |
| Accuracy | 0.9204 ⚠️ Deceptive in imbalanced data |
 
> **Note:** Accuracy is shown for completeness only. PR-AUC and Recall are the
> operationally meaningful metrics for this imbalanced credit risk task.
 
---
 
## Explainability
 
- **Global:** SHAP beeswarm and bar plots identifying top-20 risk drivers across all applicants
- **Local:** SHAP waterfall plots explaining individual predictions (regulatory "right to explanation")
- **Top Risk Drivers:** EXT_SOURCE composites, DEBT_BURDEN_RATIO, AMT_CREDIT, DAYS_BIRTH, DAYS_EMPLOYED
 
---
 
## Fairness & Bias
 
Audited across: **Age Group**, **Gender**, **Income Quartile**
 
| Criterion | Method | Target |
|---|---|---|
| Disparate Impact | Predicted positive rate ratio | DIR ≥ 0.80 across groups |
| Equalized Odds | TPR + FPR parity | Δ ≤ 5 percentage points |
 
See `deployment/artifacts/fairness_audit.png` for full results.
 
---
 
## Limitations & Risks
 
1. **Geographic Scope:** Trained on Home Credit Group data, primarily Eastern European
   and Southeast Asian markets. Performance may degrade on other geographies.
 
2. **Temporal Drift:** Financial behavior shifts over time (economic cycles, policy changes).
   PSI monitoring is in place — retrain when PSI > 0.20 on top features.
 
3. **Undersampling Trade-off:** Random undersampling improves recall but discards majority
   class information. Some low-risk applicants may still be misclassified.
 
4. **Protected Attributes:** Gender and age were not used as model features. However,
   proxy correlations via income and employment may still introduce indirect bias.
 
5. **Threshold Sensitivity:** The F1-optimal threshold (0.1699) may need
   adjustment based on the institution's cost-of-default vs cost-of-rejection ratio.
 
---
 
## Monitoring & Maintenance
 
| Signal | Trigger | Action |
|---|---|---|
| PSI on top features | PSI > 0.20 | Retrain model |
| Score distribution PSI | PSI > 0.20 | Investigate + retrain |
| Recall degradation | Recall drops > 5 pp | Threshold review + retrain |
| Fairness violation | DIR < 0.80 | Bias mitigation + model review |
 
**Monitoring Frequency:** Monthly PSI check; Quarterly fairness audit.
 
---
 
## Deployment
 
- **API:** FastAPI REST service (`/predict`, `/explain`, `/health`)
- **Dashboard:** Streamlit interactive scoring UI
- **Experiment Tracking:** MLflow (all 11 model runs logged)
- **Containerization:** Docker + docker-compose (API + Dashboard + MLflow)
 
---
 
## Version History
 
| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2026-06-08 | Initial production release |
 
---
 
*This model card follows the Model Cards for Model Reporting framework (Mitchell et al., 2019)
and Anthropic's responsible AI documentation practices.*
