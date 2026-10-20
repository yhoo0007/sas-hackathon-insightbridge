"""
End-to-end training script.

Usage
-----
# Train Tier 1 (LightGBM) with defaults:
    python train.py

# Tune hyperparameters before training:
    python train.py --tune --n-trials 50

# Run a specific tier (only tier 1 is implemented):
    python train.py --tier 1

# Save model artefacts to a custom directory:
    python train.py --output models/run_01

Outputs (in --output directory)
--------------------------------
  tier1_lgbm/          — saved LightGBM booster + metadata
  pr_curve.png         — precision-recall curve on test set
  shap_summary.png     — SHAP bar chart (top 20 features)
  spatial_residuals.png — residual heatmap on Singapore grid
  metrics.json         — evaluation metrics (val + test)
  feature_matrix.parquet — cached feature matrix for fast re-runs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
from src.targets import build_dataset
from src.features import build_feature_matrix
from src.models.base import DengueModel
from src.models.tier1_lgbm import LGBMDengueModel
from src.evaluate import (
    classification_report,
    plot_pr_curve,
    plot_shap_summary,
    plot_spatial_residuals,
    moran_i,
    print_report,
    top_features,
)

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_END   = "2017-12-31"
VAL_START   = "2018-01-01"
VAL_END     = "2018-12-31"
TEST_START  = "2019-01-01"

TARGET_COL  = "cluster_onset_2w"

NON_FEATURE_COLS = DengueModel.NON_FEATURE_COLS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(df: pd.DataFrame):
    """Return train / val / test subsets as (X, y) tuples."""
    df = df.dropna(subset=[TARGET_COL]).copy()
    ws = pd.to_datetime(df["week_start"])

    train_mask = ws <= TRAIN_END
    val_mask   = (ws >= VAL_START) & (ws <= VAL_END)
    test_mask  = ws >= TEST_START

    def xy(mask):
        sub = df[mask]
        feat_cols = [c for c in sub.columns if c not in NON_FEATURE_COLS]
        return sub[feat_cols], sub[TARGET_COL].astype(int), sub

    return xy(train_mask), xy(val_mask), xy(test_mask)


def _load_or_build(output_dir: Path, verbose: bool) -> pd.DataFrame:
    """Load cached feature matrix or build it from scratch."""
    cache = output_dir / "feature_matrix.parquet"
    if cache.exists():
        print(f"Loading cached feature matrix from {cache} …")
        return pd.read_parquet(cache)

    print("=" * 60)
    print("Step 1/3 — Building weekly dengue target dataset")
    print("=" * 60)
    t0 = time.time()
    weekly = build_dataset(verbose=verbose)
    print(f"  Done in {time.time()-t0:.1f}s\n")

    print("=" * 60)
    print("Step 2/3 — Engineering features")
    print("=" * 60)
    t0 = time.time()
    df = build_feature_matrix(weekly, verbose=verbose)
    # Re-attach target columns that build_feature_matrix doesn't touch
    target_cols = [c for c in weekly.columns if c not in df.columns or c == TARGET_COL]
    for tc in ["cluster_onset_2w", "cases_next_2w", "log_cases_next_2w", "cases_this_week"]:
        if tc in weekly.columns and tc not in df.columns:
            df[tc] = weekly[tc].values
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Feature matrix shape: {df.shape}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"  Cached to {cache}\n")
    return df


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build / load feature matrix ───────────────────────────────────────────
    df = _load_or_build(output_dir, verbose=not args.quiet)

    # ── Train / val / test split ──────────────────────────────────────────────
    (X_train, y_train, df_train), (X_val, y_val, df_val), (X_test, y_test, df_test) = _split(df)

    print("=" * 60)
    print("Step 3/3 — Training Tier 1 LightGBM")
    print("=" * 60)
    print(f"  Train: {len(X_train):,} rows  |  positives: {y_train.sum()} ({y_train.mean():.1%})")
    print(f"  Val  : {len(X_val):,} rows    |  positives: {y_val.sum()} ({y_val.mean():.1%})")
    print(f"  Test : {len(X_test):,} rows   |  positives: {y_test.sum()} ({y_test.mean():.1%})\n")

    model = LGBMDengueModel()

    # Optional tuning pass
    if args.tune:
        print(f"  Running Optuna ({args.n_trials} trials) …")
        t0 = time.time()
        model.tune(X_train, y_train, X_val, y_val, n_trials=args.n_trials)
        print(f"  Tuning done in {time.time()-t0:.1f}s\n")

    # Final fit
    print("  Fitting model …")
    t0 = time.time()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    print(f"  Fit done in {time.time()-t0:.1f}s\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_proba  = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)

    val_metrics  = classification_report(y_val.to_numpy(),  val_proba)
    test_metrics = classification_report(y_test.to_numpy(), test_proba)

    print_report(val_metrics,  label="VAL")
    print_report(test_metrics, label="TEST")

    all_metrics = {"val": val_metrics, "test": test_metrics}
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_pr_curve(y_test.to_numpy(), test_proba, label="Tier 1 LightGBM",
                  save_path=output_dir / "pr_curve.png")
    print(f"  PR curve → {output_dir/'pr_curve.png'}")

    # SHAP (on a sample of the test set for speed)
    shap_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    shap_vals = model.compute_shap(shap_sample)
    top = top_features(shap_vals, model._feature_cols)
    print("\n  Top features by mean |SHAP|:")
    print(top.to_string(index=False))

    plot_shap_summary(shap_vals, model._feature_cols, save_path=output_dir / "shap_summary.png")
    print(f"  SHAP summary → {output_dir/'shap_summary.png'}")

    # Spatial residuals
    plot_spatial_residuals(df_test, test_proba, y_test.to_numpy(),
                           save_path=output_dir / "spatial_residuals.png")
    print(f"  Spatial residuals → {output_dir/'spatial_residuals.png'}")

    # Moran's I on residuals
    residuals = test_proba - y_test.to_numpy()
    cell_df = df_test[["cell_id", "center_lat", "center_lon"]].copy()
    cell_df["res"] = residuals
    agg = cell_df.groupby(["cell_id", "center_lat", "center_lon"])["res"].mean().reset_index()
    mi = moran_i(agg["res"].to_numpy(), agg["center_lat"].to_numpy(), agg["center_lon"].to_numpy())
    print(f"\n  Moran's I on test residuals: {mi:.4f}")
    if abs(mi) > 0.1:
        print("  ⚠  Significant spatial autocorrelation — consider Tier 3 (GNN).")
    else:
        print("  Spatial autocorrelation is low — Tier 1 captures spatial patterns well.")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = output_dir / model.name
    model.save(model_path)
    print(f"\n  Model saved → {model_path}")
    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dengue cluster prediction model")
    p.add_argument("--tier",     type=int, default=1,
                   help="Model tier to train (1=LightGBM, 2=TCN, 3=GNN). Default: 1")
    p.add_argument("--tune",     action="store_true",
                   help="Run Optuna hyperparameter search before final fit")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of Optuna trials (default: 50)")
    p.add_argument("--output",   default="models/tier1",
                   help="Output directory for model artefacts (default: models/tier1)")
    p.add_argument("--quiet",    action="store_true",
                   help="Suppress verbose progress messages")
    p.add_argument("--rebuild",  action="store_true",
                   help="Ignore cached feature matrix and rebuild from raw data")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.tier != 1:
        print(f"Tier {args.tier} is not yet implemented. Only --tier 1 is available.")
        raise SystemExit(1)

    if args.rebuild:
        cache = Path(args.output) / "feature_matrix.parquet"
        if cache.exists():
            cache.unlink()
            print(f"Deleted cached feature matrix: {cache}")

    main(args)
