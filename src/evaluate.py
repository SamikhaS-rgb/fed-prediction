"""
evaluate.py
-----------
Generates all evaluation outputs described in the proposal:

  1. SHAP Summary Plot
  2. Prediction vs Actual Scatter Plots (by Fed chair era)
  3. Topic Heatmap (predicted vs actual, 2020 to 2024 test meetings)
  4. PCA Projection (all 240 statements, colored by predicted hawkishness)
  5. Error Analysis by Regime
  6. Ablation Performance Chart (MSE + directional accuracy per stage)

HOW TO RUN:
    python src/evaluate.py

All figures saved to: outputs/figures/
"""

import os
import warnings
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

#Paths
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(ROOT, "data", "processed")
MODELS  = os.path.join(ROOT, "outputs", "models")
OUTPUTS = os.path.join(ROOT, "outputs")
FIGS    = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGS, exist_ok=True)

#Consistent plot style
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.dpi":      150,
})

CHAIR_COLORS = {
    "greenspan": "#1F4E79",
    "bernanke":  "#2E75B6",
    "yellen":    "#ED7D31",
    "powell":    "#A9D18E",
    "unknown":   "#BFBFBF",
}

TONE_TARGETS  = ["hawkishness", "uncertainty", "concern"]
N_TOPICS      = 8
TOPIC_TARGETS = [f"topic_{i}" for i in range(N_TOPICS)]


#Data Loading Helpers

def load_master() -> pd.DataFrame:
    path = os.path.join(PROC, "master_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run features.py first — master_dataset.csv not found")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def add_autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in TONE_TARGETS:
        if col in df.columns:
            df[f"prev_{col}"] = df[col].shift(1)
    return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    auto_cols  = [c for c in ["prev_hawkishness", "prev_uncertainty", "prev_concern"] if c in df.columns]
    cat_cols   = [c for c in df.columns if c.startswith("chair_") or c.startswith("q_")
                  or c in ("is_crisis", "is_hiking", "is_post_covid")]
    level_cols = [c for c in df.columns if c.endswith("_level")]
    trend_cols = [c for c in df.columns if c.endswith("_trend")]
    vol_cols   = [c for c in df.columns if c.endswith("_vol")]
    return {
        "full": auto_cols + cat_cols + level_cols + trend_cols + vol_cols,
    }


def load_model_predictions(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Generates out-of-sample predictions from the full trained models."""
    preds_df = df[["date", "date_str", "chair_era",
                   "is_crisis", "is_hiking", "is_post_covid"]].copy()

    for target in TONE_TARGETS:
        model_path = os.path.join(MODELS, f"tone_{target}_full.pkl")
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        subset = df.dropna(subset=[target]).copy()
        valid_feat = [c for c in feat_cols if c in subset.columns
                      and pd.api.types.is_numeric_dtype(subset[c])]
        X = subset[valid_feat].fillna(subset[valid_feat].median()).values

        pred_series = pd.Series(model.predict(X), index=subset.index)
        preds_df[f"pred_{target}"] = pred_series
        preds_df[f"actual_{target}"] = df[target]

    return preds_df


# Figure 1: SHAP Summary Plot
def plot_shap_summary(df: pd.DataFrame, feat_cols: list):
    """SHAP beeswarm plot for the full hawkishness model."""
    model_path = os.path.join(MODELS, "tone_hawkishness_full.pkl")
    shap_path  = os.path.join(OUTPUTS, "shap_values.csv")

    if not os.path.exists(model_path):
        print("[evaluate] Hawkishness model not found — skipping SHAP plot")
        return

    model = joblib.load(model_path)
    subset = df.dropna(subset=["hawkishness"]).copy()
    valid_feat = [c for c in feat_cols if c in subset.columns
                  and pd.api.types.is_numeric_dtype(subset[c])]
    X_df = subset[valid_feat].fillna(subset[valid_feat].median())

    # Use saved SHAP values if available; otherwise recompute
    if os.path.exists(shap_path):
        sv = pd.read_csv(shap_path).values
    else:
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X_df)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_df, plot_type="dot", max_display=20, show=False)
    plt.title("SHAP Feature Importance: What Market Signals Drive Fed Hawkishness?",
              pad=14, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGS, "shap_summary.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluate] Saved shap_summary.png")


# Figure 2: Prediction vs Actual Scatter Plots
def plot_pred_vs_actual(preds_df: pd.DataFrame):
    """Scatter of predicted vs actual tone scores, colored by Fed chair era."""
    targets_available = [t for t in TONE_TARGETS if f"pred_{t}" in preds_df.columns]
    if not targets_available:
        print("[evaluate] No predictions available — skipping scatter plots")
        return

    # Only test set (2020–2024)
    test = preds_df[preds_df["date"] >= "2020-01-01"].copy()

    fig, axes = plt.subplots(1, len(targets_available), figsize=(6 * len(targets_available), 5))
    if len(targets_available) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets_available):
        pred_col   = f"pred_{target}"
        actual_col = f"actual_{target}"

        for chair, color in CHAIR_COLORS.items():
            mask = test["chair_era"] == chair
            if mask.sum() == 0:
                continue
            ax.scatter(test.loc[mask, actual_col], test.loc[mask, pred_col],
                       c=color, label=chair.title(), s=60, alpha=0.8,
                       edgecolors="white", linewidths=0.5)

        # Perfect-prediction line
        lims = [
            min(test[actual_col].min(), test[pred_col].min()) - 0.05,
            max(test[actual_col].max(), test[pred_col].max()) + 0.05,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{target.title()} Score\n(2020–2024 Test Set)")
        ax.legend(fontsize=8, frameon=True)

    plt.suptitle("Predicted vs Actual Tone Scores — Test Period",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGS, "pred_vs_actual_scatter.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluate] Saved pred_vs_actual_scatter.png")


# Figure 3: Topic Heatmap (predicted vs actual, 2020-2024)

def plot_topic_heatmap(df: pd.DataFrame, feat_cols: list):
    """Heatmaps of predicted vs actual LDA topic weights for test meetings."""
    model_path = os.path.join(MODELS, "topic_predictor_full.pkl")
    topic_cols = [c for c in TOPIC_TARGETS if c in df.columns]

    if not os.path.exists(model_path) or not topic_cols:
        print("[evaluate] Topic model or data not found — skipping topic heatmap")
        return

    model  = joblib.load(model_path)
    subset = df.dropna(subset=topic_cols).copy()
    valid_feat = [c for c in feat_cols if c in subset.columns
                  and pd.api.types.is_numeric_dtype(subset[c])]
    X = subset[valid_feat].fillna(subset[valid_feat].median()).values

    preds  = model.predict(X)
    actual = subset[topic_cols].values
    dates  = subset["date"].dt.strftime("%Y-%m").values

    # Filter test set
    test_mask = subset["date"] >= "2020-01-01"
    preds_test  = preds[test_mask.values]
    actual_test = actual[test_mask.values]
    dates_test  = dates[test_mask.values]

    if len(dates_test) == 0:
        print("[evaluate] No test meetings found — skipping topic heatmap")
        return

    topic_labels = [f"T{i}" for i in range(len(topic_cols))]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(dates_test) * 0.35 + 2)))

    for ax, data, title in zip(axes, [actual_test, preds_test],
                                ["Actual Topic Weights", "Predicted Topic Weights"]):
        sns.heatmap(
            data, ax=ax,
            xticklabels=topic_labels,
            yticklabels=dates_test,
            cmap="YlOrRd",
            vmin=0, vmax=data.max(),
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Topic weight"},
            annot=len(dates_test) <= 20,
            fmt=".2f" if len(dates_test) <= 20 else "",
        )
        ax.set_title(title)
        ax.set_xlabel("LDA Topic")
        ax.set_ylabel("FOMC Meeting")
        plt.setp(ax.get_yticklabels(), fontsize=8)

    plt.suptitle("LDA Topic Distributions: Actual vs Predicted (2020–2024)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGS, "topic_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluate] Saved topic_heatmap.png")


# Figure 4: PCA Projection of all statements
def plot_pca_projection(df: pd.DataFrame, feat_cols: list):
    """
    Projects all 240 statements into 2D via PCA on Sentence-BERT embeddings,
    colored by the full model's predicted hawkishness score.
    """
    emb_path = os.path.join(PROC, "embeddings.npy")
    if not os.path.exists(emb_path):
        print("[evaluate] embeddings.npy not found — skipping PCA projection")
        return

    embeddings = np.load(emb_path)
    if len(embeddings) != len(df):
        print("[evaluate] Embedding row count mismatch — skipping PCA projection")
        return

    model_path = os.path.join(MODELS, "tone_hawkishness_full.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        valid_feat = [c for c in feat_cols if c in df.columns
                      and pd.api.types.is_numeric_dtype(df[c])]
        X = df[valid_feat].fillna(df[valid_feat].median()).values
        hawkishness_color = model.predict(X)
    else:
        hawkishness_color = df["hawkishness"].fillna(0).values

    pca  = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        proj[:, 0], proj[:, 1],
        c=hawkishness_color,
        cmap="RdYlGn_r",
        s=40, alpha=0.8, edgecolors="white", linewidths=0.3,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Predicted hawkishness score")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("PCA Projection of FOMC Statements (Sentence-BERT)\nColored by Predicted Hawkishness")

    notable = df[df["date"].dt.year.isin([2008, 2020, 2022])].index
    for i in notable[:5]:
        ax.annotate(df.loc[i, "date"].strftime("%b %Y"),
                    (proj[i, 0], proj[i, 1]),
                    fontsize=7, color="#333333",
                    xytext=(5, 3), textcoords="offset points")

    plt.tight_layout()
    out = os.path.join(FIGS, "pca_projection.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluate] Saved pca_projection.png")


# Figure 5: Error Analysis by Regime

def plot_error_by_regime(preds_df: pd.DataFrame):
    """
    Plots model residuals for hawkishness separately for:
      - Crisis periods (is_crisis = 1)
      - Hiking cycles  (is_hiking = 1)
      - Stable periods (neither)
    """
    if "pred_hawkishness" not in preds_df.columns:
        print("[evaluate] No hawkishness predictions — skipping error analysis")
        return

    preds_df = preds_df.dropna(subset=["pred_hawkishness", "actual_hawkishness"]).copy()
    preds_df["residual"] = preds_df["pred_hawkishness"] - preds_df["actual_hawkishness"]

    def regime_label(row):
        if row["is_crisis"] == 1:
            return "Crisis period\n(NBER recession)"
        elif row["is_hiking"] == 1:
            return "Hiking cycle"
        else:
            return "Stable period"

    preds_df["regime"] = preds_df.apply(regime_label, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {"Crisis period\n(NBER recession)": "#C00000",
              "Hiking cycle": "#ED7D31",
              "Stable period": "#2E75B6"}

    for ax, (regime, group) in zip(axes, preds_df.groupby("regime")):
        color = colors.get(regime, "#BFBFBF")
        ax.scatter(group["date"], group["residual"],
                   c=color, s=50, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{regime}\n(n={len(group)})")
        ax.set_xlabel("Date")
        if ax == axes[0]:
            ax.set_ylabel("Residual (predicted − actual)")
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)

        # Annotate mean absolute error
        mae = group["residual"].abs().mean()
        ax.text(0.05, 0.95, f"MAE={mae:.4f}", transform=ax.transAxes,
                fontsize=9, va="top", color=color)

    plt.suptitle("Hawkishness Prediction Errors by Regime",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGS, "error_by_regime.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluate] Saved error_by_regime.png")


# Figure 6: Ablation Performance Chart

def plot_ablation_chart():
    """Bar chart comparing MSE and directional accuracy across all 5 ablation stages."""
    results_path = os.path.join(OUTPUTS, "results_table.csv")
    if not os.path.exists(results_path):
        print("[evaluate] results_table.csv not found — skipping ablation chart")
        return

    df = pd.read_csv(results_path)

    # Average over the three tone targets per stage
    summary = (df.groupby("stage")[["test_mse", "test_dir_acc"]]
                 .mean()
                 .reset_index())

    stage_order = ["baseline", "macro", "mkt_level", "mkt_trend", "full"]
    stage_labels = {
        "baseline":  "Baseline\n(autoregressive)",
        "macro":     "+ Macro /\nregime features",
        "mkt_level": "+ Market\nlevels",
        "mkt_trend": "+ Market\ntrends",
        "full":      "+ Market\nvolatility\n(full model)",
    }
    summary["stage"] = pd.Categorical(summary["stage"], categories=stage_order, ordered=True)
    summary = summary.sort_values("stage")
    labels = [stage_labels.get(s, s) for s in summary["stage"]]

    colors = ["#ADB5BD", "#6C8EAD", "#2E75B6", "#1F4E79", "#ED7D31"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSE
    ax = axes[0]
    bars = ax.bar(range(len(summary)), summary["test_mse"],
                  color=colors[:len(summary)], edgecolor="white", width=0.6)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Test MSE by Ablation Stage\n(averaged over 3 tone targets — lower is better)")
    ax.set_ylabel("MSE")
    for bar, val in zip(bars, summary["test_mse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Directional accuracy
    ax = axes[1]
    bars = ax.bar(range(len(summary)), summary["test_dir_acc"],
                  color=colors[:len(summary)], edgecolor="white", width=0.6)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random baseline (50%)")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Directional Accuracy by Ablation Stage\n(averaged over 3 tone targets — higher is better)")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, summary["test_dir_acc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("5-Stage Ablation Study Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGS, "ablation_chart.png")
    plt.savefig(out)
    plt.close()
    print(f"[evaluate] Saved ablation_chart.png")


# Main

def main():
    print("\n=== FOMC Prediction — Evaluation & Visualisation ===\n")

    df = load_master()
    df = add_autoregressive_features(df)
    feat_cols = get_feature_groups(df)["full"]
    preds_df  = load_model_predictions(df, feat_cols)

    plot_shap_summary(df, feat_cols)
    plot_pred_vs_actual(preds_df)
    plot_topic_heatmap(df, feat_cols)
    plot_pca_projection(df, feat_cols)
    plot_error_by_regime(preds_df)
    plot_ablation_chart()

    print(f"\n=== All figures saved to {FIGS} ===")
    for f in sorted(os.listdir(FIGS)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
