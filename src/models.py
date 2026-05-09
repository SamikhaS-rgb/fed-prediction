"""
models.py
---------
Three-layer prediction pipeline from the proposal:

  Layer 1 — Tone Score Regression   (hawkishness / uncertainty / concern)
  Layer 2 — Topic Distribution Prediction  (8-dim LDA vector)
  Layer 3 — Semantic Embedding Prediction  (10 Sentence-BERT PCA dims)

Also runs the 5-stage ablation study and computes SHAP values.

HOW TO RUN:
    python src/models.py

Outputs saved to:
    outputs/models/          — trained model files (.pkl)
    outputs/results_table.csv — ablation comparison table
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

#Paths
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS    = os.path.join(ROOT, "outputs", "models")
OUTPUTS   = os.path.join(ROOT, "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TONE_TARGETS  = ["hawkishness", "uncertainty", "concern"]
N_TOPICS      = 8
TOPIC_TARGETS = [f"topic_{i}" for i in range(N_TOPICS)]
SBERT_TARGETS = [f"sbert_pca_{i}" for i in range(10)]


# Data loading helpers
def load_master() -> pd.DataFrame:
    path = os.path.join(PROCESSED, "master_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run features.py first — master_dataset.csv not found")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Returns feature column groups for the ablation study.

    Ablation stages (cumulative):
      baseline     — previous statement tone scores only (autoregressive)
      + macro      — add macro indicators (CPI, unemployment, GDP growth)
      + mkt_level  — add market level features
      + mkt_trend  — add market trend features
      + mkt_vol    — add market volatility features (= full model)
    """
    auto_cols = ["prev_hawkishness", "prev_uncertainty", "prev_concern"]

   
    cat_cols = [c for c in df.columns if
                c.startswith("chair_") or c.startswith("q_") or
                c in ("is_crisis", "is_hiking", "is_post_covid")]


    level_cols = [c for c in df.columns if c.endswith("_level")]
    trend_cols = [c for c in df.columns if c.endswith("_trend")]
    vol_cols   = [c for c in df.columns if c.endswith("_vol")]

    return {
        "baseline":    auto_cols,
        "macro":       auto_cols + cat_cols,       # cat_cols act as macro proxies
        "mkt_level":   auto_cols + cat_cols + level_cols,
        "mkt_trend":   auto_cols + cat_cols + level_cols + trend_cols,
        "full":        auto_cols + cat_cols + level_cols + trend_cols + vol_cols,
    }


def add_autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag-1 tone scores as autoregressive baseline features."""
    for col in TONE_TARGETS:
        if col in df.columns:
            df[f"prev_{col}"] = df[col].shift(1)
    return df


def get_xy(df: pd.DataFrame, feature_cols: list[str],
           target_cols: list[str]) -> tuple:
    """
    Extracts X, y matrices from the master dataset.
    Drops rows where any target is NaN. Fills feature NaNs with median.
    Returns (X, y, dates, valid_feature_cols).
    """
    subset = df.dropna(subset=target_cols).copy()

    valid_feat = [
        c for c in feature_cols
        if c in subset.columns and pd.api.types.is_numeric_dtype(subset[c])
    ]

    X      = subset[valid_feat].fillna(subset[valid_feat].median())
    y      = subset[target_cols].values
    dates  = subset["date"].values
    return X.values, y, dates, valid_feat


# Time-series cross-validation helper
def ts_cv_score(X: np.ndarray, y: np.ndarray,
                model_cls, model_kwargs: dict,
                n_splits: int = 5) -> tuple[float, float, float, float]:
    """
    Expanding-window time-series CV.
    Returns (avg_mse, std_mse, avg_dir_acc, std_dir_acc).
    dir_acc only meaningful for 1-D targets; for multi-output returns NaN.
    """
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    mses, dir_accs = [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = model_cls(**model_kwargs, random_state=RANDOM_STATE, verbosity=0)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)

        mse = mean_squared_error(y_te, preds if preds.ndim > 1 else preds.reshape(-1, 1),
                                 multioutput="uniform_average")
        mses.append(mse)

        if y_te.ndim == 1 or y_te.shape[1] == 1:
            p = preds.ravel()
            a = y_te.ravel()
            # Direction of change relative to last training value
            ref = np.concatenate([[y_tr.ravel()[-1]], a[:-1]])
            dir_accs.append(float(np.mean(np.sign(p - ref) == np.sign(a - ref))))
        else:
            dir_accs.append(float("nan"))

    return (np.mean(mses), np.std(mses),
            np.nanmean(dir_accs), np.nanstd(dir_accs))


# Layer 1 — Tone Score Regression
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)


def train_tone_regressors(df: pd.DataFrame, feature_cols: list[str],
                           label: str) -> dict:
    """
    Trains one XGBoost regressor per tone target.
    Returns a dict of evaluation results.
    """
    results = {}
    for target in TONE_TARGETS:
        if target not in df.columns:
            continue

        X, y, dates, valid_feat = get_xy(df, feature_cols, [target])
        y = y.ravel()

        avg_mse, std_mse, avg_da, std_da = ts_cv_score(
            X, y, xgb.XGBRegressor, XGB_PARAMS
        )

        # Train final model on all training data (1994–2019)
        train_mask = pd.to_datetime(dates) <= pd.Timestamp("2019-12-31")
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[~train_mask], y[~train_mask]

        final = xgb.XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE, verbosity=0)
        final.fit(X_tr, y_tr)

        # Test metrics (2020–2024)
        test_preds = final.predict(X_te)
        test_mse   = mean_squared_error(y_te, test_preds)

        ref = np.concatenate([[y_tr[-1]], y_te[:-1]])
        test_da = float(np.mean(np.sign(test_preds - ref) == np.sign(y_te - ref)))

        model_name = f"tone_{target}_{label}"
        joblib.dump(final, os.path.join(MODELS, f"{model_name}.pkl"))

        print(f"  [{label}] {target:15s}  CV MSE={avg_mse:.4f}  "
              f"Test MSE={test_mse:.4f}  Test DirAcc={test_da:.3f}")

        results[target] = {
            "label":    label,
            "target":   target,
            "cv_mse":   avg_mse,
            "cv_mse_std": std_mse,
            "test_mse": test_mse,
            "test_dir_acc": test_da,
            "feature_names": valid_feat,
        }

    return results


# Layer 2 — Topic Distribution Prediction

def train_topic_predictor(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Trains a multi-output XGBoost model to predict the 8-dim LDA topic vector.
    Evaluates with MSE and cosine similarity.
    """
    existing_topics = [c for c in TOPIC_TARGETS if c in df.columns]
    if not existing_topics:
        print("[models] No topic columns found — skipping Layer 2")
        return {}

    X, y, dates, valid_feat = get_xy(df, feature_cols, existing_topics)

    train_mask = pd.to_datetime(dates) <= pd.Timestamp("2019-12-31")
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[~train_mask], y[~train_mask]

    base = xgb.XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE, verbosity=0)
    model = MultiOutputRegressor(base)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    test_mse = mean_squared_error(y_te, preds, multioutput="uniform_average")

    from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
    cos_sims = [cosine_sim(preds[i:i+1], y_te[i:i+1])[0, 0] for i in range(len(y_te))]
    avg_cos  = float(np.mean(cos_sims))

    joblib.dump(model, os.path.join(MODELS, "topic_predictor_full.pkl"))
    print(f"  [full] topic distribution  Test MSE={test_mse:.4f}  "
          f"Avg CosSim={avg_cos:.3f}")

    return {
        "label": "topic_distribution",
        "test_mse": test_mse,
        "avg_cosine_sim": avg_cos,
        "feature_names": valid_feat,
    }


# Layer 3 — Semantic Embedding Prediction (stretch goal)
def train_embedding_predictor(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Predicts the first 10 PCA components of Sentence-BERT embeddings.
    Evaluates with cosine similarity between reconstructed embeddings.
    """
    existing_sbert = [c for c in SBERT_TARGETS if c in df.columns]
    if not existing_sbert:
        print("[models] No SBERT PCA columns found — skipping Layer 3 (stretch goal)")
        return {}

    X, y, dates, valid_feat = get_xy(df, feature_cols, existing_sbert)

    train_mask = pd.to_datetime(dates) <= pd.Timestamp("2019-12-31")
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[~train_mask], y[~train_mask]

    base  = xgb.XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE, verbosity=0)
    model = MultiOutputRegressor(base)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    test_mse = mean_squared_error(y_te, preds, multioutput="uniform_average")

    from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
    cos_sims = [cosine_sim(preds[i:i+1], y_te[i:i+1])[0, 0] for i in range(len(y_te))]
    avg_cos  = float(np.mean(cos_sims))

    joblib.dump(model, os.path.join(MODELS, "embedding_predictor_full.pkl"))
    print(f"  [full] SBERT embedding (stretch)  Test MSE={test_mse:.4f}  "
          f"Avg CosSim={avg_cos:.3f}")

    return {
        "label":          "sbert_embedding",
        "test_mse":       test_mse,
        "avg_cosine_sim": avg_cos,
        "feature_names":  valid_feat,
    }



# SHAP analysis on full model
def compute_shap_values(df: pd.DataFrame, feature_cols: list[str]):
    """
    Computes SHAP values for the full hawkishness model and saves them.
    These are used by evaluate.py for the SHAP summary plot.
    """
    if "hawkishness" not in df.columns:
        return

    model_path = os.path.join(MODELS, "tone_hawkishness_full.pkl")
    if not os.path.exists(model_path):
        print("[models] Full hawkishness model not found — skipping SHAP")
        return

    model = joblib.load(model_path)
    X, y, dates, valid_feat = get_xy(df, feature_cols, ["hawkishness"])
    X_df = pd.DataFrame(X, columns=valid_feat)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    shap_df = pd.DataFrame(shap_values, columns=valid_feat)
    shap_df.to_csv(os.path.join(OUTPUTS, "shap_values.csv"), index=False)
    print(f"[models] Saved shap_values.csv  shape={shap_df.shape}")



# Ablation study — 5 stages per proposal
ABLATION_STAGES = [
    ("baseline",   "Previous statement tone only (autoregressive)"),
    ("macro",      "+ Macro / regime features"),
    ("mkt_level",  "+ Market level features"),
    ("mkt_trend",  "+ Market trend features"),
    ("full",       "+ Market volatility features (full model)"),
]


def run_ablation_study(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all 5 ablation stages for Layer 1 tone score regression.
    Saves a results comparison table to outputs/results_table.csv.
    """
    print("\n" + "=" * 60)
    print(" Ablation Study — Layer 1: Tone Score Regression")
    print("=" * 60)

    df = add_autoregressive_features(df)
    feature_groups = get_feature_groups(df)

    all_results = []
    for stage_key, stage_desc in ABLATION_STAGES:
        feat_cols = feature_groups.get(stage_key, [])
        print(f"\n[ablation] Stage: {stage_key}  |  {stage_desc}")
        results = train_tone_regressors(df, feat_cols, label=stage_key)
        for target, r in results.items():
            all_results.append({
                "stage":       stage_key,
                "description": stage_desc,
                "target":      target,
                "cv_mse":      round(r["cv_mse"], 4),
                "test_mse":    round(r["test_mse"], 4),
                "test_dir_acc":round(r["test_dir_acc"], 3),
            })

    summary = pd.DataFrame(all_results)
    out = os.path.join(OUTPUTS, "results_table.csv")
    summary.to_csv(out, index=False)
    print(f"\n[models] Ablation results saved → {out}")
    print(summary.to_string(index=False))
    return summary



# Main
def main():
    print("\n=== FOMC Prediction — Model Training Pipeline ===\n")

    df = load_master()
    feature_groups = get_feature_groups(add_autoregressive_features(df.copy()))
    full_feat_cols = feature_groups["full"]

    ablation_summary = run_ablation_study(df)

    print("\n" + "=" * 60)
    print(" Layer 2: Topic Distribution Prediction")
    print("=" * 60)
    df2 = add_autoregressive_features(df.copy())
    topic_results = train_topic_predictor(df2, full_feat_cols)

    print("\n" + "=" * 60)
    print(" Layer 3: Semantic Embedding Prediction (stretch goal)")
    print("=" * 60)
    embed_results = train_embedding_predictor(df2, full_feat_cols)

    print("\n[models] Computing SHAP values for full hawkishness model...")
    compute_shap_values(df2, full_feat_cols)

    extra_results = {}
    if topic_results:
        extra_results["layer2_topics"] = topic_results
    if embed_results:
        extra_results["layer3_embeddings"] = embed_results
    with open(os.path.join(MODELS, "layer2_3_results.json"), "w") as f:
        json.dump(extra_results, f, indent=2, default=str)

    print("\n=== Training complete ===")
    print(f"  Models saved to {MODELS}")
    print(f"  Results table: {os.path.join(OUTPUTS, 'results_table.csv')}")


if __name__ == "__main__":
    main()
