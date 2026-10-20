"""
Tier 1 — LightGBM gradient boosting classifier.

Trains on the tabular feature matrix produced by src/features.py.
Supports Optuna-based hyperparameter search via tune().
Produces SHAP feature importances after fitting.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from .base import DengueModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMDengueModel(DengueModel):
    name = "tier1_lgbm"

    # Default hyperparameters — tuning overrides these
    DEFAULT_PARAMS = dict(
        objective="binary",
        metric="average_precision",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_estimators=500,
        scale_pos_weight=None,  # set automatically from data if None
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    def __init__(self, params: Optional[dict] = None):
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[lgb.LGBMClassifier] = None
        self._feature_cols: Optional[list[str]] = None
        self.shap_values_: Optional[np.ndarray] = None

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode monsoon_phase; select and order feature columns."""
        cols = self.get_feature_cols(df)
        X = df[cols].copy()
        if "monsoon_phase" in df.columns:
            dummies = pd.get_dummies(df["monsoon_phase"], prefix="monsoon", drop_first=False)
            X = pd.concat([X, dummies], axis=1)
        return X

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LGBMDengueModel":
        X_prep = self._prepare(X)
        self._feature_cols = list(X_prep.columns)

        params = dict(self._params)

        # Auto-set scale_pos_weight from class ratio if not given
        if params.get("scale_pos_weight") is None:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_prep = self._prepare(X_val)[self._feature_cols]
            eval_set = [(X_val_prep, y_val)]
        else:
            # No validation set — use a fraction of training data for early stopping
            split = int(len(X_prep) * 0.9)
            eval_set = [(X_prep.iloc[split:], y.iloc[split:])]
            X_prep = X_prep.iloc[:split]
            y = y.iloc[:split]

        self._model = lgb.LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                X_prep, y,
                eval_set=eval_set,
                callbacks=callbacks,
            )

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X_prep = self._prepare(X)
        # Align to training columns (fill any missing dummies with 0)
        for c in self._feature_cols:
            if c not in X_prep.columns:
                X_prep[c] = 0
        X_prep = X_prep[self._feature_cols]
        return self._model.predict_proba(X_prep)[:, 1]

    # ── tune ──────────────────────────────────────────────────────────────────

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> dict:
        """Optuna search over key LightGBM hyperparameters."""
        X_tr = self._prepare(X_train)
        X_v  = self._prepare(X_val)

        # Ensure consistent columns
        all_cols = list(dict.fromkeys(list(X_tr.columns) + list(X_v.columns)))
        for c in all_cols:
            if c not in X_tr.columns: X_tr[c] = 0
            if c not in X_v.columns:  X_v[c] = 0
        X_tr, X_v = X_tr[all_cols], X_v[all_cols]

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = n_neg / max(n_pos, 1)

        def objective(trial: optuna.Trial) -> float:
            p = dict(
                objective="binary",
                metric="average_precision",
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
                feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                bagging_freq=1,
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                n_estimators=500,
                scale_pos_weight=spw,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            m = lgb.LGBMClassifier(**p)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(
                    X_tr, y_train,
                    eval_set=[(X_v, y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
                )
            proba = m.predict_proba(X_v)[:, 1]
            return average_precision_score(y_val, proba)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        self._params.update(best)
        print(f"  Best AP on val: {study.best_value:.4f}  params: {best}")
        return best

    # ── SHAP ──────────────────────────────────────────────────────────────────

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for X (requires shap package)."""
        import shap
        X_prep = self._prepare(X)
        for c in self._feature_cols:
            if c not in X_prep.columns:
                X_prep[c] = 0
        X_prep = X_prep[self._feature_cols]
        explainer = shap.TreeExplainer(self._model)
        self.shap_values_ = explainer.shap_values(X_prep)
        # For binary LightGBM, shap_values_ is a list [neg_class, pos_class]
        if isinstance(self.shap_values_, list):
            self.shap_values_ = self.shap_values_[1]
        return self.shap_values_

    # ── persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(path / "booster.txt"))
        meta = {"params": self._params, "feature_cols": self._feature_cols}
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, path: Path) -> "LGBMDengueModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        self._params = meta["params"]
        self._feature_cols = meta["feature_cols"]
        booster = lgb.Booster(model_file=str(path / "booster.txt"))
        self._model = lgb.LGBMClassifier(**self._params)
        self._model._Booster = booster  # noqa: SLF001
        return self
