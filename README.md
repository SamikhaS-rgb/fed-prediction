# Predicting Federal Reserve Communication Tone via NLP and Market Features

> CS439 Final Project: FOMC Statement Tone Prediction Pipeline
Author: Samikha Srinivasan 
---

## Overview

This project builds a multi-layer machine learning pipeline to predict the **tone of Federal Reserve (FOMC) statements** specifically measuring *hawkishness*, *uncertainty*, and *concern* using a combination of NLP-derived text features and market/macroeconomic signals.

We scrape and preprocess FOMC meeting statements dating back to 1994, apply LDA topic modeling and Sentence-BERT embeddings, engineer 21 market features (yield curve slope, VIX, S&P 500, HY spreads, etc.), and train XGBoost regressors across a 5-stage ablation study. Evaluation includes SHAP feature importance, PCA projections, error analysis by Fed chair era, and directional accuracy metrics.

---

## Project Structure

```
fed-prediction/
├── data/
│   ├── raw/
│   │   ├── fomc/                  # Raw FOMC statement .txt files (1994–2024)
│   │   └── market/                # Raw market CSVs (SP500, VIX, treasuries, etc.)
│   └── processed/                 # Cleaned and engineered features (CSV, NPY)
│       ├── fomc_statements_clean.csv
│       ├── market_features.csv
│       ├── tone_scores.csv
│       ├── topic_distributions.csv
│       ├── embeddings.npy
│       ├── sbert_pca.csv
│       └── master_dataset.csv
├── src/
│   ├── scraper.py          # Downloads FOMC statements from the Fed website
│   ├── preprocess.py       # Text cleaning and boilerplate removal
│   ├── market_data.py      # Fetches market data via yfinance/FRED API
│   ├── features.py         # Full feature engineering pipeline (NLP + market)
│   ├── models.py           # 3-layer XGBoost pipeline + ablation study
│   └── evaluate.py         # Visualizations: SHAP, PCA, error analysis, plots
├── outputs/
│   ├── figures/            # All generated plots (PNG)
│   ├── models/             # Saved model files (.pkl) and JSON results
│   ├── results_table.csv   # Ablation study results
│   └── shap_values.csv     # SHAP feature importances
├── requirements.txt
└── .env                    # API keys (FRED — not committed to git)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)

### Install Dependencies

```bash
git clone https://github.com/<your-username>/fed-prediction.git
cd fed-prediction
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the project root:

```
FRED_API_KEY=your_key_here
```

> ⚠️ **Never commit `.env` to GitHub.** It is already listed in `.gitignore`.

---

## Reproducing Results

Run the pipeline in order:

```bash
# 1. Scrape FOMC statements (saves to data/raw/fomc/)
python src/scraper.py

# 2. Clean raw text
python src/preprocess.py

# 3. Fetch market data (saves to data/raw/market/)
python src/market_data.py

# 4. Engineer all features and build master_dataset.csv
python src/features.py

# 5. Train models and run ablation study
python src/models.py

# 6. Generate all evaluation figures
python src/evaluate.py
```

All outputs land in `outputs/figures/` and `outputs/models/`.

---

## Methodology

### Data

- **FOMC Statements**: ~240 official statements from 1994–2024, scraped from federalreserve.gov
- **Market Data**: SP500, VIX, 2yr/10yr Treasury yields, yield curve slope, HY credit spread, DXY, inflation breakevens — sourced via `yfinance` and the FRED API

### NLP Feature Engineering (`src/features.py`)

| Feature Group | Description |
|---|---|
| **Tone Scores** | Hawkishness, uncertainty, and concern lexicon scores per statement |
| **LDA Topics** | 8-topic distribution via Latent Dirichlet Allocation (Gensim) |
| **Sentence-BERT Embeddings** | `all-MiniLM-L6-v2` embeddings → 10-dim PCA reduction |
| **Chair Era** | Categorical encoding of Greenspan / Bernanke / Yellen / Powell eras |

### Model Architecture (`src/models.py`)

A 3-layer hierarchical prediction pipeline using XGBoost regressors:

1. **Layer 1 — Tone Regression**: Predict hawkishness, uncertainty, and concern scores from market + macro features
2. **Layer 2 — Topic Prediction**: Predict 8-dim LDA topic distributions
3. **Layer 3 — Embedding Prediction**: Predict 10 SBERT PCA dimensions

All layers use `TimeSeriesSplit` cross-validation to prevent data leakage.

### Ablation Study

Five feature stages are evaluated cumulatively:

| Stage | Features Added |
|---|---|
| `baseline` | Previous statement tone only (autoregressive) |
| `macro` | + Fed chair era, FOMC meeting count, rate regime |
| `mkt_level` | + SP500, VIX, treasury yields, HY spread, DXY |
| `mkt_trend` | + 1-month rolling changes in market features |
| `full` | + Market volatility (rolling std) features |

### Key Results

| Target | Baseline MSE | Best MSE | Best Stage | Dir. Accuracy |
|---|---|---|---|---|
| Hawkishness | 0.876 | 0.606 | macro | 63.0% |
| Uncertainty | 0.387 | 0.292 | full | 60.9% |
| Concern | 0.135 | 0.109 | mkt_level | 52.2% |

Macro/regime features provide the largest single improvement for hawkishness prediction, reducing MSE by ~30% over the autoregressive baseline.

---

## Outputs & Visualizations

All figures are saved to `outputs/figures/`:

- **`shap_summary.png`** — SHAP feature importance across tone targets
- **`pca_projection.png`** — 240 FOMC statements projected to 2D, colored by hawkishness
- **`pred_vs_actual_scatter.png`** — Predicted vs. actual tone scores by Fed chair era
- **`topic_heatmap.png`** — Predicted vs. actual LDA topics (2020–2024 test set)
- **`error_by_regime.png`** — MSE decomposed by rate regime (hiking / cutting / hold)
- **`ablation_chart.png`** — MSE and directional accuracy across all 5 ablation stages

---

## Notebooks

The `notebooks/` directory contains exploratory versions of the pipeline:

| Notebook | Purpose |
|---|---|
| `01_data_collection.ipynb` | Scraping and initial data inspection |
| `02_eda.ipynb` | Exploratory data analysis and visualizations |
| `03_feature_engineering.ipynb` | Step-by-step NLP and market feature construction |
| `04_clustering.ipynb` | K-Means and regime clustering experiments |
| `05_models.ipynb` | Model training and ablation study walkthrough |
| `06_evaluation.ipynb` | Full evaluation suite with interactive plots |

> For reproducibility, use the `src/` scripts. Notebooks are provided for exploration and transparency.

---

## Dependencies

See `requirements.txt` for pinned versions. Key libraries:

- `scikit-learn`, `xgboost`, `shap` — modeling and explainability
- `sentence-transformers`, `gensim`, `nltk` — NLP pipeline
- `pandas`, `numpy`, `scipy` — data processing
- `yfinance`, `fredapi` — market data ingestion
- `matplotlib`, `seaborn` — visualization

---

## .gitignore Recommendations

Make sure these are excluded before pushing:

```
.env
venv/
__pycache__/
*.pyc
.DS_Store
data/raw/        # large data — add instructions to regenerate
outputs/models/  # large binary files — optional to exclude
```

---

## License

MIT License. Data sourced from public Federal Reserve releases and market APIs.
