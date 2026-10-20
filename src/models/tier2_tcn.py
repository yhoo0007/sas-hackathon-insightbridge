"""
Tier 2 — Temporal Convolutional Network (TCN).

Each grid cell's weather + case history is represented as a (T, F) sequence
where T is the lookback window in days and F is the number of daily features.
The TCN learns temporal patterns without manual lag engineering.

Requirements (not installed by default):
    pip install torch  # or pip install tensorflow

TODO — Implementation checklist:
1. Prepare per-cell daily sequences:
   - Stack daily weather features + cases_this_day into shape (n_cells, T, F)
   - T = 42 days lookback; F = ~14 daily weather + case features
   - Normalise each feature channel independently (StandardScaler per channel)
2. Define TCN architecture:
   - Stacked dilated causal Conv1D layers with residual connections
   - Dilation factors: [1, 2, 4, 8, 16] to cover ~42-day receptive field
   - Output: sigmoid activation for binary onset probability
3. Training loop:
   - Binary cross-entropy loss with positive-class weight
   - AdamW optimiser, cosine annealing scheduler
   - Early stopping on validation Average Precision
4. Implement predict_proba():
   - Slice the last T days of daily features for each (cell, prediction_date)
   - Run forward pass, return sigmoid output
5. Expose tune() via Optuna on:
   - n_layers, hidden_dim, dropout, learning_rate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import DengueModel


class TCNDengueModel(DengueModel):
    name = "tier2_tcn"

    def __init__(self, lookback_days: int = 42, hidden_dim: int = 64, n_layers: int = 5):
        self.lookback_days = lookback_days
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "TCNDengueModel":
        raise NotImplementedError(
            "Tier 2 (TCN) is not yet implemented. "
            "See the TODO checklist in src/models/tier2_tcn.py."
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Tier 2 (TCN) is not yet implemented.")

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> "TCNDengueModel":
        raise NotImplementedError
