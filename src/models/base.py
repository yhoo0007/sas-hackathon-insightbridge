"""
Abstract base class for all dengue prediction model tiers.

Each tier implements fit(), predict_proba(), and optionally tune().
The interface is kept deliberately thin so tiers can swap in/out easily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class DengueModel(ABC):
    """
    Common interface for all tier models.

    fit()           : train on X_train / y_train
    predict_proba() : return probability of cluster onset (Target A) for X
    predict()       : return binary class at a given threshold
    tune()          : optional hyperparameter search (default: no-op, use fit defaults)
    save() / load() : persist the trained model to disk
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "DengueModel":
        """Train the model. Returns self."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1-D array of onset probabilities in [0, 1]."""

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> dict:
        """
        Run hyperparameter search. Returns the best params dict.
        Default implementation does nothing (use the model's built-in defaults).
        Override in subclasses that support tuning.
        """
        return {}

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist trained model."""

    @abstractmethod
    def load(self, path: Path) -> "DengueModel":
        """Load a saved model. Returns self."""

    # ── Shared feature-column selector ───────────────────────────────────────

    # Columns that are targets, identifiers, or metadata — never features.
    NON_FEATURE_COLS = {
        "cell_id", "week_start", "station",
        "cases_this_week", "cases_next_2w",
        "cluster_onset_2w", "log_cases_next_2w",
        "center_lat", "center_lon",
        "monsoon_phase",  # encoded via one-hot in subclass
    }

    @classmethod
    def get_feature_cols(cls, df: pd.DataFrame) -> list[str]:
        """Return the ordered list of feature column names from a DataFrame."""
        return [c for c in df.columns if c not in cls.NON_FEATURE_COLS]
