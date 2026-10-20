"""
Evaluation utilities: metrics, SHAP summary, spatial residual analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


# ── Classification metrics ────────────────────────────────────────────────────

def classification_report(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Return a dict of evaluation metrics for binary classification.
    Threshold is chosen to maximise F1 on the evaluation data.
    """
    auc_roc = roc_auc_score(y_true, y_proba)
    avg_prec = average_precision_score(y_true, y_proba)

    # Optimal F1 threshold
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    y_pred = (y_proba >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "auc_roc":         round(float(auc_roc), 4),
        "avg_precision":   round(float(avg_prec), 4),
        "f1":              round(best_f1, 4),
        "precision":       round(float(tp / max(tp + fp, 1)), 4),
        "recall":          round(float(tp / max(tp + fn, 1)), 4),
        "threshold":       round(best_threshold, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def print_report(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}AUC-ROC: {metrics['auc_roc']:.4f}  "
          f"Avg Precision: {metrics['avg_precision']:.4f}  "
          f"F1: {metrics['f1']:.4f}  "
          f"Recall: {metrics['recall']:.4f}  "
          f"Precision: {metrics['precision']:.4f}  "
          f"@threshold={metrics['threshold']:.3f}")
    print(f"{prefix}  TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TN={metrics['tn']}")


# ── Precision-Recall curve ────────────────────────────────────────────────────

def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: str = "",
    save_path: Optional[Path] = None,
) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec, prec, lw=2, label=f"{label} (AP={ap:.3f})")
    baseline = y_true.mean()
    ax.axhline(baseline, ls="--", color="grey", lw=1, label=f"Baseline (random) = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ── SHAP summary ──────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
    save_path: Optional[Path] = None,
) -> None:
    import shap
    fig, ax = plt.subplots(figsize=(8, max_display * 0.35 + 1))
    shap.summary_plot(
        shap_values, feature_names=feature_names,
        max_display=max_display, show=False, plot_type="bar",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def top_features(shap_values: np.ndarray, feature_names: list[str], n: int = 15) -> pd.DataFrame:
    """Return a DataFrame of the top-N features by mean |SHAP|."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)


# ── Spatial residual heatmap ──────────────────────────────────────────────────

def plot_spatial_residuals(
    df_test: pd.DataFrame,
    y_proba: np.ndarray,
    y_true: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot mean residual (predicted probability - actual) per grid cell
    to reveal spatial patterns in model error.
    """
    tmp = df_test[["cell_id", "center_lat", "center_lon"]].copy()
    tmp["residual"] = y_proba - y_true

    cell_res = tmp.groupby(["cell_id", "center_lat", "center_lon"])["residual"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        cell_res["center_lon"], cell_res["center_lat"],
        c=cell_res["residual"], cmap="RdBu_r", s=15, vmin=-1, vmax=1, alpha=0.7,
    )
    plt.colorbar(sc, ax=ax, label="Mean residual (ŷ − y)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial residuals — Tier 1 LightGBM")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def moran_i(values: np.ndarray, lats: np.ndarray, lons: np.ndarray, k: int = 8) -> float:
    """
    Compute Moran's I for a set of point values with k-nearest neighbours.
    A significant positive value indicates spatial autocorrelation in residuals,
    suggesting a spatial model (Tier 3) may help.
    """
    from scipy.spatial import KDTree

    coords = np.column_stack([lats, lons])
    tree = KDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # +1 because self is included
    idx = idx[:, 1:]  # exclude self

    n = len(values)
    z = values - values.mean()

    W = np.zeros((n, n))
    for i, neighbours in enumerate(idx):
        W[i, neighbours] = 1.0

    W_sum = W.sum()
    numerator   = n * (z[:, None] * W * z[None, :]).sum()
    denominator = W_sum * (z ** 2).sum()
    return float(numerator / max(denominator, 1e-12))
