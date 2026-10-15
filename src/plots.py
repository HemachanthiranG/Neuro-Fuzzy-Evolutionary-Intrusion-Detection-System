"""Comprehensive plotting module producing publication-ready PNGs and interactive Plotly figures."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from src.ann import load_ann_model, predict_ann
from src.config import AppConfig, load_config
from src.utils import load_artifact, load_json, setup_logger

logger = setup_logger("plots")

# Consistent color palette for models across all plots
MODEL_COLORS = {
    "Baseline ANN": "#1f77b4",  # Steel Blue
    "GA+ANN": "#2ca02c",        # Vibrant Green
    "MI+ANN": "#ff7f0e",        # Warm Orange
    "RF+ANN": "#9467bd",        # Purple
}


def set_plot_style():
    """Apply standard clean styling for matplotlib figures."""
    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


# ==========================================
# 1. GA Convergence Plot
# ==========================================
def plot_ga_convergence(
    history_df: pd.DataFrame,
    output_png: Optional[Path] = None,
    title_suffix: str = "",
) -> go.Figure:
    """Generate Matplotlib PNG and Plotly interactive figure for GA convergence."""
    set_plot_style()
    gens = history_df["generation"].tolist()

    if output_png:
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        ax1.plot(gens, history_df["best_fitness"], "o-", color="#1f77b4", linewidth=2.2, label="Best Fitness")
        ax1.plot(gens, history_df["mean_fitness"], "s--", color="#2ca02c", linewidth=1.6, label="Mean Fitness")
        ax1.plot(gens, history_df["worst_fitness"], "^:", color="#d62728", linewidth=1.2, label="Worst Fitness")
        ax1.set_xlabel("Generation", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Fitness (Macro-F1 - Penalty)", fontsize=11, fontweight="bold", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.plot(gens, history_df["best_features_count"], "D-.", color="#ff7f0e", linewidth=1.8, label="# Features (Best)")
        ax2.set_ylabel("Features Selected (out of 41)", fontsize=11, fontweight="bold", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax2.set_ylim(0, 45)
        ax2.grid(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)
        plt.title(f"GA Feature Selection Convergence {title_suffix}".strip(), fontsize=13, fontweight="bold")
        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_plotly.add_trace(
        go.Scatter(x=gens, y=history_df["best_fitness"], mode="lines+markers", name="Best Fitness", line=dict(color="#1f77b4", width=3)),
        secondary_y=False,
    )
    fig_plotly.add_trace(
        go.Scatter(x=gens, y=history_df["mean_fitness"], mode="lines+markers", name="Mean Fitness", line=dict(color="#2ca02c", dash="dash")),
        secondary_y=False,
    )
    fig_plotly.add_trace(
        go.Scatter(x=gens, y=history_df["worst_fitness"], mode="lines+markers", name="Worst Fitness", line=dict(color="#d62728", dash="dot")),
        secondary_y=False,
    )
    fig_plotly.add_trace(
        go.Scatter(x=gens, y=history_df["best_features_count"], mode="lines+markers", name="# Features Selected", line=dict(color="#ff7f0e", dash="dashdot")),
        secondary_y=True,
    )
    fig_plotly.update_layout(
        title=f"GA Convergence History {title_suffix}".strip(),
        xaxis_title="Generation",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"),
    )
    fig_plotly.update_yaxes(title_text="Fitness (Macro-F1 - Penalty)", secondary_y=False)
    fig_plotly.update_yaxes(title_text="Features Count (out of 41)", secondary_y=True, range=[0, 45])
    return fig_plotly


# ==========================================
# 2. Metric Comparison Grouped Bar Chart
# ==========================================
def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    output_png: Optional[Path] = None,
) -> go.Figure:
    """Grouped bar chart comparing models across primary metrics."""
    set_plot_style()
    metrics_to_show = [
        ("test_accuracy", "Accuracy"),
        ("test_macro_f1", "Macro-F1"),
        ("test_macro_precision", "Macro-Precision"),
        ("test_macro_recall", "Macro-Recall"),
    ]

    models = metrics_df["model_name"].unique().tolist()
    agg = metrics_df.groupby("model_name").mean(numeric_only=True).reset_index()

    if output_png:
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
        x = np.arange(len(metrics_to_show))
        width = 0.18
        multiplier = 0

        for model in models:
            m_row = agg[agg["model_name"] == model]
            vals = [m_row[col].iloc[0] for col, _ in metrics_to_show]
            color = MODEL_COLORS.get(model, "#333333")
            rects = ax.bar(x + (multiplier * width), vals, width, label=model, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
            ax.bar_label(rects, padding=3, fmt="%.3f", fontsize=8)
            multiplier += 1

        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title("Model Performance Comparison on Held-Out Test Set (KDDTest+)", fontsize=12, fontweight="bold")
        ax.set_xticks(x + (width * (len(models) - 1) / 2))
        ax.set_xticklabels([name for _, name in metrics_to_show], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", framealpha=0.9)
        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = go.Figure()
    for model in models:
        m_row = agg[agg["model_name"] == model]
        vals = [m_row[col].iloc[0] for col, _ in metrics_to_show]
        fig_plotly.add_trace(go.Bar(
            name=model,
            x=[name for _, name in metrics_to_show],
            y=vals,
            marker_color=MODEL_COLORS.get(model, "#333333"),
            text=[f"{v:.3f}" for v in vals],
            textposition="auto",
        ))
    fig_plotly.update_layout(
        barmode="group",
        title="Model Performance Comparison (KDDTest+)",
        yaxis_title="Metric Value",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig_plotly


# ==========================================
# 3. Per-Class F1 and Recall Comparison
# ==========================================
def plot_per_class_comparison(
    per_class_df: pd.DataFrame,
    output_png: Optional[Path] = None,
) -> go.Figure:
    """Grouped bar chart showing F1 and Recall breakdown by attack class."""
    set_plot_style()
    agg = per_class_df.groupby(["model_name", "class_name"]).mean(numeric_only=True).reset_index()
    classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    models = agg["model_name"].unique().tolist()

    if output_png:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
        x = np.arange(len(classes))
        width = 0.18

        # Left: F1 Score
        for idx, model in enumerate(models):
            m_data = agg[agg["model_name"] == model].set_index("class_name")
            f1_vals = [m_data.loc[c, "f1_score"] if c in m_data.index else 0.0 for c in classes]
            color = MODEL_COLORS.get(model, "#333333")
            axes[0].bar(x + (idx * width), f1_vals, width, label=model, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        axes[0].set_title("Per-Class F1 Score Comparison", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("F1 Score", fontsize=11)
        axes[0].set_xticks(x + (width * (len(models) - 1) / 2))
        axes[0].set_xticklabels(classes, fontsize=10)
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(loc="upper right")

        # Right: Recall / Detection Rate
        for idx, model in enumerate(models):
            m_data = agg[agg["model_name"] == model].set_index("class_name")
            rec_vals = [m_data.loc[c, "recall"] if c in m_data.index else 0.0 for c in classes]
            color = MODEL_COLORS.get(model, "#333333")
            axes[1].bar(x + (idx * width), rec_vals, width, label=model, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        axes[1].set_title("Per-Class Recall (Detection Rate) Comparison", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Recall / Detection Rate", fontsize=11)
        axes[1].set_xticks(x + (width * (len(models) - 1) / 2))
        axes[1].set_xticklabels(classes, fontsize=10)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = make_subplots(rows=1, cols=2, subplot_titles=("Per-Class F1 Score", "Per-Class Recall / Detection Rate"))
    for model in models:
        m_data = agg[agg["model_name"] == model].set_index("class_name")
        f1_vals = [m_data.loc[c, "f1_score"] if c in m_data.index else 0.0 for c in classes]
        rec_vals = [m_data.loc[c, "recall"] if c in m_data.index else 0.0 for c in classes]
        color = MODEL_COLORS.get(model, "#333333")

        fig_plotly.add_trace(go.Bar(name=model, x=classes, y=f1_vals, marker_color=color, legendgroup=model), row=1, col=1)
        fig_plotly.add_trace(go.Bar(name=model, x=classes, y=rec_vals, marker_color=color, legendgroup=model, showlegend=False), row=1, col=2)

    fig_plotly.update_layout(
        barmode="group",
        template="plotly_white",
        title="Per-Class Performance Breakdown (Note low sample counts for U2R & R2L)",
    )
    return fig_plotly


# ==========================================
# 4. Confusion Matrices Side-by-Side
# ==========================================
def plot_confusion_matrices_side_by_side(
    cm_dir: Path,
    classes: List[str],
    seed: int = 42,
    output_png: Optional[Path] = None,
) -> go.Figure:
    """Generate 4-panel normalized confusion matrix comparison."""
    set_plot_style()
    models = [
        ("Baseline ANN", f"cm_baseline_ann_seed{seed}_norm.npy"),
        ("GA+ANN", f"cm_ga_ann_seed{seed}_norm.npy"),
        ("MI+ANN", f"cm_mi_ann_seed{seed}_norm.npy"),
        ("RF+ANN", f"cm_rf_ann_seed{seed}_norm.npy"),
    ]

    cms = []
    for label, fname in models:
        p = cm_dir / fname
        if p.exists():
            cms.append((label, np.load(p)))
        else:
            cms.append((label, np.zeros((len(classes), len(classes)))))

    if output_png:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=300)
        for idx, (label, cm) in enumerate(cms):
            sns.heatmap(
                cm,
                annot=True,
                fmt=".1%",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes if idx == 0 else False,
                cbar=(idx == 3),
                ax=axes[idx],
            )
            axes[idx].set_title(label, fontsize=11, fontweight="bold")
            axes[idx].set_xlabel("Predicted Class", fontsize=10)
            if idx == 0:
                axes[idx].set_ylabel("True Class", fontsize=10)

        plt.suptitle("Normalized Confusion Matrices Side-by-Side (Recall)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = make_subplots(rows=1, cols=4, subplot_titles=[l for l, _ in cms])
    for idx, (label, cm) in enumerate(cms):
        heatmap = go.Heatmap(
            z=cm * 100,
            x=classes,
            y=classes,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            showscale=(idx == 3),
            text=[[f"{v*100:.1f}%" for v in row] for row in cm],
            texttemplate="%{text}",
        )
        fig_plotly.add_trace(heatmap, row=1, col=idx + 1)
        fig_plotly.update_xaxes(title_text="Predicted", row=1, col=idx + 1)
        if idx == 0:
            fig_plotly.update_yaxes(title_text="True Label", row=1, col=1)

    fig_plotly.update_layout(
        title="Side-by-Side Normalized Confusion Matrices",
        template="plotly_white",
        height=450,
    )
    return fig_plotly


# ==========================================
# 5. Feature Selection Heatmap & Frequencies
# ==========================================
def plot_feature_selection_frequencies(
    selected_features_path: Path,
    output_png: Optional[Path] = None,
) -> go.Figure:
    """Plot feature selection frequency across selectors."""
    set_plot_style()
    data = load_json(selected_features_path)
    all_features = list(data["ga"]["frequencies"].keys())

    # Build DataFrame
    records = []
    for feat in all_features:
        records.append({
            "Feature": feat,
            "GA Frequency": data["ga"]["frequencies"][feat]["frequency"],
            "MI Frequency": data["mi"]["frequencies"][feat]["frequency"],
            "RF Frequency": data["rf"]["frequencies"][feat]["frequency"],
        })
    df_feat = pd.DataFrame(records).sort_values("GA Frequency", ascending=False)

    if output_png:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        x = np.arange(len(df_feat))
        width = 0.28
        ax.bar(x - width, df_feat["GA Frequency"], width, label="GA", color=MODEL_COLORS["GA+ANN"], alpha=0.9)
        ax.bar(x, df_feat["MI Frequency"], width, label="Mutual Information", color=MODEL_COLORS["MI+ANN"], alpha=0.9)
        ax.bar(x + width, df_feat["RF Frequency"], width, label="Random Forest", color=MODEL_COLORS["RF+ANN"], alpha=0.9)

        ax.set_ylabel("Selection Frequency (0.0 to 1.0)", fontsize=11, fontweight="bold")
        ax.set_title("Original Feature Selection Frequency Across Search Methods", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df_feat["Feature"], rotation=90, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right")
        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Bar(name="GA", x=df_feat["Feature"], y=df_feat["GA Frequency"], marker_color=MODEL_COLORS["GA+ANN"]))
    fig_plotly.add_trace(go.Bar(name="Mutual Information", x=df_feat["Feature"], y=df_feat["MI Frequency"], marker_color=MODEL_COLORS["MI+ANN"]))
    fig_plotly.add_trace(go.Bar(name="Random Forest", x=df_feat["Feature"], y=df_feat["RF Frequency"], marker_color=MODEL_COLORS["RF+ANN"]))
    fig_plotly.update_layout(
        barmode="group",
        title="Feature Selection Frequency Across All 41 Features",
        xaxis_tickangle=-45,
        yaxis_title="Frequency",
        template="plotly_white",
    )
    return fig_plotly


# ==========================================
# 6. Feature Reduction Pareto Chart
# ==========================================
def plot_feature_reduction_pareto(
    metrics_df: pd.DataFrame,
    output_png: Optional[Path] = None,
) -> go.Figure:
    """Plot features used vs. Macro-F1 (Pareto trade-off)."""
    set_plot_style()
    agg = metrics_df.groupby("model_name").mean(numeric_only=True).reset_index()

    if output_png:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        for _, row in agg.iterrows():
            model = row["model_name"]
            x_val = row["num_features_original"]
            y_val = row["test_macro_f1"]
            color = MODEL_COLORS.get(model, "#333333")
            ax.scatter(x_val, y_val, color=color, s=250, label=model, edgecolors="black", linewidth=1.5, zorder=5)
            offset_y = 0.008 if model != "MI+ANN" else -0.012
            ax.annotate(
                f"{model}\n({int(x_val)} feats, F1: {y_val:.3f})",
                (x_val, y_val + offset_y),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlabel("Number of Original Features Used", fontsize=11, fontweight="bold")
        ax.set_ylabel("Test Macro-F1 Score", fontsize=11, fontweight="bold")
        ax.set_title("Feature Reduction vs. Macro-F1 Trade-off", fontsize=12, fontweight="bold")
        ax.set_xlim(5, 45)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", framealpha=0.9)
        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = go.Figure()
    for _, row in agg.iterrows():
        model = row["model_name"]
        fig_plotly.add_trace(go.Scatter(
            x=[row["num_features_original"]],
            y=[row["test_macro_f1"]],
            mode="markers+text",
            name=model,
            marker=dict(size=18, color=MODEL_COLORS.get(model, "#333333"), line=dict(width=1, color="black")),
            text=[f"{model}<br>({int(row['num_features_original'])} feats, F1: {row['test_macro_f1']:.3f})"],
            textposition="top center",
        ))
    fig_plotly.update_layout(
        title="Feature Reduction vs Macro-F1 Score",
        xaxis_title="Features Used (out of 41)",
        yaxis_title="Test Macro-F1",
        template="plotly_white",
    )
    return fig_plotly


# ==========================================
# 7. Training Time and Inference Latency
# ==========================================
def plot_time_comparison(
    metrics_df: pd.DataFrame,
    output_png: Optional[Path] = None,
) -> go.Figure:
    """Compare training duration and inference latency per 1000 records."""
    set_plot_style()
    agg = metrics_df.groupby("model_name").mean(numeric_only=True).reset_index()
    models = agg["model_name"].tolist()

    if output_png:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        colors = [MODEL_COLORS.get(m, "#333333") for m in models]

        # Training time
        bars1 = axes[0].bar(models, agg["training_time_sec"], color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
        axes[0].bar_label(bars1, fmt="%.1fs", padding=3, fontsize=9)
        axes[0].set_title("Model Training Time (Full Epochs)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Seconds", fontsize=10)
        axes[0].set_ylim(0, max(agg["training_time_sec"]) * 1.25)

        # Inference latency per 1000 records
        bars2 = axes[1].bar(models, agg["inference_time_per_1000_ms"], color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
        axes[1].bar_label(bars2, fmt="%.1fms", padding=3, fontsize=9)
        axes[1].set_title("Inference Latency per 1,000 Records", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Milliseconds (ms)", fontsize=10)
        axes[1].set_ylim(0, max(agg["inference_time_per_1000_ms"]) * 1.25)

        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = make_subplots(rows=1, cols=2, subplot_titles=("Training Time (Seconds)", "Inference Latency per 1,000 Records (ms)"))
    fig_plotly.add_trace(go.Bar(
        x=models,
        y=agg["training_time_sec"],
        marker_color=[MODEL_COLORS.get(m, "#333333") for m in models],
        text=[f"{v:.1f}s" for v in agg["training_time_sec"]],
        textposition="auto",
        name="Training Time",
    ), row=1, col=1)

    fig_plotly.add_trace(go.Bar(
        x=models,
        y=agg["inference_time_per_1000_ms"],
        marker_color=[MODEL_COLORS.get(m, "#333333") for m in models],
        text=[f"{v:.1f}ms" for v in agg["inference_time_per_1000_ms"]],
        textposition="auto",
        name="Latency",
    ), row=1, col=2)

    fig_plotly.update_layout(
        template="plotly_white",
        title="Efficiency Benchmark: Training Duration vs Inference Latency",
        showlegend=False,
    )
    return fig_plotly


# ==========================================
# 8. ROC Curves (One-vs-Rest Macro & per-model)
# ==========================================
def plot_roc_curves(
    config: AppConfig,
    seed: int = 42,
    output_png: Optional[Path] = None,
) -> go.Figure:
    """Generate One-vs-Rest ROC curves for all models on KDDTest+."""
    set_plot_style()
    preprocessor = load_artifact(config.paths.models_dir / "preprocessor.joblib")
    classes = config.data.classes

    # Load raw test dataset
    from src.data_loader import load_raw_nsl_kdd
    _, test_df = load_raw_nsl_kdd(config=config, raw_dir=config.paths.raw_train_path.parent)
    X_test_full, y_test = preprocessor.transform(test_df)

    models_info = [
        ("Baseline ANN", config.paths.models_dir / f"model_baseline_ann_seed{seed}.keras", np.ones(41, dtype=int)),
        ("GA+ANN", config.paths.models_dir / f"model_ga_ann_seed{seed}.keras", None),
        ("MI+ANN", config.paths.models_dir / f"model_mi_ann_seed{seed}.keras", None),
        ("RF+ANN", config.paths.models_dir / f"model_rf_ann_seed{seed}.keras", None),
    ]

    # Load chromosomes
    for idx, (label, model_path, chrom) in enumerate(models_info):
        if chrom is None:
            tag = label.split("+")[0].lower()
            mask_path = config.paths.models_dir / f"selected_mask_{tag}_seed{seed}.json"
            if mask_path.exists():
                mask_data = load_json(mask_path)
                c = np.zeros(41, dtype=int)
                c[mask_data["selected_feature_indices"]] = 1
                models_info[idx] = (label, model_path, c)

    roc_data = {}
    for label, model_path, chrom in models_info:
        if model_path.exists() and chrom is not None:
            cols = preprocessor.get_column_indices_for_chromosome(chrom)
            model = load_ann_model(model_path)
            _, probs = predict_ann(model, X_test_full[:, cols])
            
            # Binary one-hot representation for multiclass ROC
            y_test_oh = np.eye(len(classes))[y_test]
            fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
            for c_idx in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_test_oh[:, c_idx], probs[:, c_idx])
                fpr_dict[c_idx] = fpr
                tpr_dict[c_idx] = tpr
                roc_auc_dict[c_idx] = auc(fpr, tpr)

            # Micro-average ROC curve
            fpr_micro, tpr_micro, _ = roc_curve(y_test_oh.ravel(), probs.ravel())
            micro_auc = auc(fpr_micro, tpr_micro)
            roc_data[label] = {
                "fpr_micro": fpr_micro,
                "tpr_micro": tpr_micro,
                "auc_micro": micro_auc,
            }

    if output_png:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7)
        for label, info in roc_data.items():
            color = MODEL_COLORS.get(label, "#333333")
            ax.plot(
                info["fpr_micro"],
                info["tpr_micro"],
                label=f"{label} (Micro-AUC = {info['auc_micro']:.3f})",
                color=color,
                linewidth=2,
            )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
        ax.set_title("One-vs-Rest Multiclass ROC Curves on KDDTest+", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", framealpha=0.9)
        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Chance", line=dict(dash="dash", color="grey")))
    for label, info in roc_data.items():
        color = MODEL_COLORS.get(label, "#333333")
        fig_plotly.add_trace(go.Scatter(
            x=info["fpr_micro"],
            y=info["tpr_micro"],
            mode="lines",
            name=f"{label} (AUC: {info['auc_micro']:.3f})",
            line=dict(color=color, width=2.5),
        ))
    fig_plotly.update_layout(
        title="Multiclass ROC Curves (Micro-Average, Held-Out Test Set)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
    )
    return fig_plotly


# ==========================================
# 9. Class Distribution Plot (Plotly Generator)
# ==========================================
def plot_class_distribution_interactive(dataset_summary_path: Path) -> go.Figure:
    """Generate interactive Plotly grouped bar chart for class distributions."""
    data = load_json(dataset_summary_path)
    dists = data["distributions"]

    classes = data["classes"]
    train_counts = [dists["train"]["class_counts"].get(c, 0) for c in classes]
    val_counts = [dists["validation"]["class_counts"].get(c, 0) for c in classes]
    test_counts = [dists["test"]["class_counts"].get(c, 0) for c in classes]

    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Bar(name="Train Split", x=classes, y=train_counts, marker_color="#1f77b4"))
    fig_plotly.add_trace(go.Bar(name="Val Split", x=classes, y=val_counts, marker_color="#ff7f0e"))
    fig_plotly.add_trace(go.Bar(name="Held-Out Test", x=classes, y=test_counts, marker_color="#2ca02c"))
    fig_plotly.update_layout(
        barmode="group",
        title="Class Distribution Across Train, Validation, and Test Sets",
        yaxis_type="log",
        yaxis_title="Count (Log Scale)",
        template="plotly_white",
    )
    return fig_plotly


# ==========================================
# 10. False Positive Rate vs. Attack Detection Rate
# ==========================================
def plot_fpr_vs_detection_rate(
    metrics_df: pd.DataFrame,
    output_png: Optional[Path] = None,
) -> go.Figure:
    """Scatter comparison of False Positive Rate vs. Attack Detection Rate."""
    set_plot_style()
    agg = metrics_df.groupby("model_name").mean(numeric_only=True).reset_index()

    if output_png:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        for _, row in agg.iterrows():
            model = row["model_name"]
            fpr = row["test_false_positive_rate"] * 100
            dr = row["test_attack_detection_rate"] * 100
            color = MODEL_COLORS.get(model, "#333333")
            ax.scatter(fpr, dr, s=260, color=color, label=model, edgecolors="black", linewidth=1.5, zorder=5)
            ax.annotate(
                f"{model}\n(FPR: {fpr:.1f}%, DR: {dr:.1f}%)",
                (fpr, dr + 0.9),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlabel("False Positive Rate (%) [Lower is Better]", fontsize=11, fontweight="bold")
        ax.set_ylabel("Attack Detection Rate (%) [Higher is Better]", fontsize=11, fontweight="bold")
        ax.set_title("Operational Trade-off: False Alarm Rate vs. Detection Rate", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left", framealpha=0.9)
        plt.tight_layout()
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()

    # Interactive Plotly
    fig_plotly = go.Figure()
    for _, row in agg.iterrows():
        model = row["model_name"]
        fpr = row["test_false_positive_rate"] * 100
        dr = row["test_attack_detection_rate"] * 100
        fig_plotly.add_trace(go.Scatter(
            x=[fpr],
            y=[dr],
            mode="markers+text",
            name=model,
            marker=dict(size=18, color=MODEL_COLORS.get(model, "#333333"), line=dict(width=1, color="black")),
            text=[f"{model}<br>FPR: {fpr:.1f}%<br>DR: {dr:.1f}%"],
            textposition="top center",
        ))
    fig_plotly.update_layout(
        title="Operational Security Trade-off: False Positive Rate vs. Detection Rate",
        xaxis_title="False Positive Rate (%) [Lower is Better]",
        yaxis_title="Attack Detection Rate (%) [Higher is Better]",
        template="plotly_white",
    )
    return fig_plotly


# ==========================================
# Generate All Static Plots Runner
# ==========================================
def generate_all_plots(config: Optional[AppConfig] = None) -> List[Path]:
    """Generate and persist the complete suite of publication PNG plots in results/plots/."""
    if config is None:
        config = load_config()

    plots_dir = config.paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    first_seed = config.seeds[0] if config.seeds else 42

    # 1. GA Convergence Plot
    history_csv = config.paths.results_dir / f"ga_history_adaptive_seed{first_seed}.csv"
    if not history_csv.exists():
        history_csv = config.paths.results_dir / f"ga_history_seed{first_seed}.csv"
    if history_csv.exists():
        df_hist = pd.read_csv(history_csv)
        p1 = plots_dir / "1_ga_convergence.png"
        plot_ga_convergence(df_hist, output_png=p1, title_suffix=f"(Seed {first_seed})")
        generated.append(p1)

    # Metrics DataFrame
    metrics_csv = config.paths.results_dir / "metrics_all_runs.csv"
    per_class_csv = config.paths.results_dir / "per_class_metrics.csv"
    selected_feat_json = config.paths.results_dir / "selected_features.json"

    if metrics_csv.exists():
        m_df = pd.read_csv(metrics_csv)

        # 2. Metric comparison
        p2 = plots_dir / "2_metric_comparison.png"
        plot_metric_comparison(m_df, output_png=p2)
        generated.append(p2)

        # 6. Feature reduction Pareto
        p6 = plots_dir / "6_feature_reduction_pareto.png"
        plot_feature_reduction_pareto(m_df, output_png=p6)
        generated.append(p6)

        # 7. Time comparison
        p7 = plots_dir / "7_time_comparison.png"
        plot_time_comparison(m_df, output_png=p7)
        generated.append(p7)

        # 10. FPR vs Detection rate
        p10 = plots_dir / "10_fpr_vs_detection_rate.png"
        plot_fpr_vs_detection_rate(m_df, output_png=p10)
        generated.append(p10)

    # 3. Per-class comparison
    if per_class_csv.exists():
        pc_df = pd.read_csv(per_class_csv)
        p3 = plots_dir / "3_per_class_comparison.png"
        plot_per_class_comparison(pc_df, output_png=p3)
        generated.append(p3)

    # 4. Side-by-side Confusion Matrices
    p4 = plots_dir / "4_confusion_matrices_side_by_side.png"
    plot_confusion_matrices_side_by_side(config.paths.confusion_matrices_dir, config.data.classes, seed=first_seed, output_png=p4)
    generated.append(p4)

    # 5. Feature selection frequencies
    if selected_feat_json.exists():
        p5 = plots_dir / "5_feature_selection_frequencies.png"
        plot_feature_selection_frequencies(selected_feat_json, output_png=p5)
        generated.append(p5)

    # 8. ROC Curves
    p8 = plots_dir / "8_roc_curves.png"
    try:
        plot_roc_curves(config, seed=first_seed, output_png=p8)
        generated.append(p8)
    except Exception as e:
        logger.warning(f"Could not generate ROC curve plot: {e}")

    logger.info(f"Generated {len(generated)} publication PNG plots in {plots_dir}")
    return generated


if __name__ == "__main__":
    generated_paths = generate_all_plots()
    print(f"Phase 5 Plots generation complete! Generated {len(generated_paths)} plots:")
    for p in generated_paths:
        print(f" - {p.name}")
