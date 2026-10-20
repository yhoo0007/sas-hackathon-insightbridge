"""
Tier 3 — Spatial-temporal Graph Neural Network (GNN + GRU).

Each grid cell is a node; edges connect Moore-neighbourhood cells (≤ 2 km).
Node features at each timestep: weather + lagged case counts.
A GRU processes the temporal sequence at each node; message passing
propagates spatial contagion between neighbours.

Requirements (not installed by default):
    pip install torch torch-geometric

TODO — Implementation checklist:
1. Build the cell graph:
   - src.grid.build_neighbour_map() → edge list
   - Convert to torch_geometric.data.Data (edge_index, edge_attr = distance)
2. Build node feature sequences:
   - Shape: (n_cells, T, F) — same daily features as Tier 2
   - Use the same StandardScaler normalisation as Tier 2 for consistency
3. Define GNN + GRU model:
   - Temporal block: GRU over T timesteps at each node → hidden state h_i
   - Spatial block: 2-layer GraphSAGE / GATConv on h_i
   - Output: MLP head → sigmoid for onset probability
4. Training loop:
   - Batch over time windows (walk-forward); shuffle cells within each window
   - Positive-class focal loss to handle imbalance
5. Decide whether to include GNN when Moran's I on Tier 1 residuals is significant
   (p < 0.05 indicates remaining spatial autocorrelation worth modelling).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import DengueModel


class GNNDengueModel(DengueModel):
    name = "tier3_gnn"

    def __init__(self, hidden_dim: int = 64, gru_layers: int = 2, gnn_layers: int = 2):
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.gnn_layers = gnn_layers
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "GNNDengueModel":
        raise NotImplementedError(
            "Tier 3 (GNN) is not yet implemented. "
            "See the TODO checklist in src/models/tier3_gnn.py."
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Tier 3 (GNN) is not yet implemented.")

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> "GNNDengueModel":
        raise NotImplementedError
