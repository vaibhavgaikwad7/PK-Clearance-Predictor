# 💊 PK Clearance Predictor

**Predicting drug clearance from patient covariates — an end-to-end pharmacokinetics data science project.**

> 🚧 **Status: Actively in development** — Pipeline, EDA, and feature engineering complete. ML modeling and deployment in progress.

---

## What Is This?

Drug clearance — the rate at which a drug is removed from the body — varies dramatically between patients based on age, weight, genetics, smoking status, and other factors. Getting it wrong means underdosing (ineffective treatment) or overdosing (toxic side effects).

This project builds a **complete data science pipeline** to predict drug clearance from patient characteristics, using real clinical pharmacokinetics data from [PK-DB](https://pk-db.com) and molecular ADME datasets from [Therapeutics Data Commons](https://tdcommons.ai).

It covers every stage of the DS lifecycle: **data ingestion → EDA → feature engineering → ML modeling → interpretability → API serving → BI dashboard → deployment**.

---

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   PK-DB API     │────▶│   ETL        │────▶│   SQLite DB     │
│   (Clinical PK) │     │   Pipeline   │     │   (Normalized)  │
└─────────────────┘     └──────┬───────┘     └────────┬────────┘
                               │                      │
┌─────────────────┐            │              ┌───────▼────────┐
│   TDC Harvard   │────────────┘              │   Feature      │
│   (ADME Data)   │                           │   Engineering  │
└─────────────────┘                           │   (PK calcs)   │
                                              └───────┬────────┘
                                                      │
                              ┌────────────────────────┼──────────────────┐
                              │                        │                  │
                      ┌───────▼──────┐        ┌───────▼──────┐   ┌──────▼───────┐
                      │   ML Models  │        │   Streamlit  │   │   Power BI   │
                      │   XGBoost    │        │   Dashboard  │   │   Dashboard  │
                      │   LASSO      │        └──────────────┘   └──────────────┘
                      │   RF / ODE   │
                      └───────┬──────┘
                              │
                      ┌───────▼──────┐
                      │   FastAPI    │
                      │   /predict   │
                      │   + SHAP     │
                      └──────────────┘
```

---

## Data Sources

| Source | Description | Records | Access |
|--------|------------|---------|--------|
| [PK-DB](https://pk-db.com) | Clinical pharmacokinetics from 796+ studies — patient demographics, dosing, PK parameters | 160K+ individual records, 20K+ group records | REST API (CC BY 4.0) |
| [TDC](https://tdcommons.ai) | Molecular ADME benchmarks — drug clearance, half-life, bioavailability, protein binding | 1K–2K drugs per endpoint | Harvard Dataverse |

**Primary focus drug:** Caffeine (137X) — richest substance in PK-DB with 9,396 PK outputs across 105 clinical studies. Caffeine clearance is strongly influenced by smoking status, oral contraceptive use, and CYP1A2 genetic variants, making it an ideal prediction target.

---

## Project Structure

```
pk-clearance-predictor/
├── config/
│   ├── config.yaml              # Hyperparameters, paths, API config
│   └── schema.sql               # SQL schema (10 tables, 3 views)
├── data/
│   ├── raw/                     # Immutable source data (CSVs from API)
│   └── processed/               # Analysis-ready datasets + figures
├── notebooks/
│   ├── 00_api_exploration.py    # API discovery documentation
│   ├── 01_data_ingestion.py     # Multi-source ETL pipeline
│   ├── 02_eda.py                # Exploratory data analysis
│   └── 03_feature_engineering.py # PK-specific feature calculations
├── src/
│   ├── transformation/
│   │   └── pk_calculations.py   # BSA, CrCl, IBW, BMI functions
│   ├── models/                  # Training + evaluation (coming soon)
│   └── database/                # SQLAlchemy connectors (coming soon)
├── api/                         # FastAPI prediction endpoint (coming soon)
├── dashboard/                   # Streamlit multi-page app (coming soon)
├── powerbi/                     # Power BI dashboard (coming soon)
├── models/                      # Serialized model artifacts (coming soon)
└── tests/                       # pytest suite (coming soon)
```

---

## Features Engineered

| Feature | Method | Clinical Relevance |
|---------|--------|-------------------|
| **BSA** | DuBois formula | Standard metric for dose normalization across body sizes |
| **Est. CrCl** | Cockcroft-Gault equation | Renal function marker — directly determines drug elimination rate |
| **IBW** | Devine formula | Used in weight-based dosing calculations |
| **BMI** | Calculated from height/weight | Obesity affects drug distribution volume |
| **is_smoker** | Binary encoding | Smoking induces CYP1A2 — increases caffeine clearance ~1.5x |
| **is_female** | Binary encoding | Sex affects drug metabolism enzyme activity |
| **on_oc** | Binary encoding | Oral contraceptives inhibit CYP1A2 — decreases caffeine clearance |
| **age_category** | Clinical binning | Pediatric/young adult/middle-aged/elderly have different PK profiles |

---

## ML Approach (Planned)

**Target:** Drug clearance (mL/min/kg) from patient covariates

**Models to compare:**
- **XGBoost** — primary model, expected R² ~0.80–0.90
- **LASSO Regression** — interpretable baseline
- **Random Forest** — feature importance benchmark
- **One-compartment ODE** — mechanistic PK baseline

**Evaluation:** R², RMSE, MAE, Spearman correlation, % predictions within 2-fold error

**Interpretability:** SHAP values revealing which patient factors most influence clearance

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Data | pandas, NumPy, requests |
| Database | SQLite (portable) / PostgreSQL (local dev) |
| ML | scikit-learn, XGBoost, SHAP, MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Power BI |
| Visualization | matplotlib, seaborn, Plotly |
| DevOps | Docker, GitHub Actions, Makefile |
| Deployment | Streamlit Cloud, Vercel, HuggingFace Spaces |

---

## Team

| Role | Name | Focus |
|------|------|-------|
| Data Scientist | **Vaibhav Gaikwad** ([@vaibhavgaikwad7](https://github.com/vaibhavgaikwad7)) | Data pipeline, EDA, ML modeling, feature engineering, Power BI |
| Software Engineer | **Sachi** | Infrastructure, API, deployment, CI/CD, Streamlit |

---

## How to Run

```bash
# Clone
git clone https://github.com/vaibhavgaikwad7/pk-clearance-predictor.git
cd pk-clearance-predictor

# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run ingestion (pulls data from PK-DB API — takes ~15 min)
python notebooks/01_data_ingestion.py

# Run EDA
python notebooks/02_eda.py

# Run feature engineering
python notebooks/03_feature_engineering.py
```

---

## References

- Grzegorzewski et al. (2021). [PK-DB: pharmacokinetics database for individualized and stratified computational modeling](https://pubmed.ncbi.nlm.nih.gov/33151297/). *Nucleic Acids Research*, 49(D1), D1358–D1364.
- Huang et al. (2021). [Therapeutics Data Commons: Machine Learning Datasets for Therapeutics](https://tdcommons.ai). *NeurIPS 2021 Datasets and Benchmarks Track*.

---

## License

This project is for educational and portfolio purposes. PK-DB data is used under CC BY 4.0. TDC data is used under its respective licenses.
