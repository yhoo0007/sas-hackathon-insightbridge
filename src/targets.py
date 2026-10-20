"""
Dengue data loading, aggregation, and target variable construction.

Steps:
1. Load dengue_combined.csv
2. Assign each address to a grid cell
3. Aggregate to weekly case counts per cell
4. Join with cell metadata (station assignment)
5. Add the forward-looking target variable (onset in next HORIZON_DAYS)

Target A (binary, default): cluster_onset_2w
  = 1 if any cases appear in the cell in the following HORIZON_DAYS days
  = 0 otherwise

Target B (regression): cases_next_2w
  = log(1 + sum of cases in the cell over the following HORIZON_DAYS days)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .grid import add_cell_id, build_cell_metadata

DATA_DIR = Path(__file__).parent.parent / "data"
DENGUE_PATH = DATA_DIR / "dengue_combined.csv"

HORIZON_DAYS = 14   # prediction horizon


def load_dengue() -> pd.DataFrame:
    """Load raw dengue CSV, parse dates."""
    df = pd.read_csv(DENGUE_PATH, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", "Number Of Cases": "n_cases"})
    # Drop duplicate rows (same address on same date can appear from multiple source files)
    df = df.drop_duplicates(subset=["date", "Latitude", "Longitude"])
    return df


def _iso_week_start(date_series: pd.Series) -> pd.Series:
    """Return the Monday (ISO week start) for each date."""
    dt = pd.to_datetime(date_series)
    return (dt - pd.to_timedelta(dt.dt.dayofweek, unit="D")).dt.normalize()


def build_weekly_dengue() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (cell_id, week_start) covering the
    full date range that appears in the dengue dataset.  Cells with zero cases
    in a given week are included so the model sees all negative examples.

    Columns: cell_id, week_start, cases_this_week, station, center_lat, center_lon
    """
    raw = load_dengue()
    raw = add_cell_id(raw)
    raw["week_start"] = _iso_week_start(raw["date"])

    # Sum cases per (cell, week)
    weekly = (
        raw.groupby(["cell_id", "week_start"])["n_cases"]
        .sum()
        .reset_index(name="cases_this_week")
    )

    # --- Build the full grid of (active_cell, week) ---
    # Only cells that ever had at least one case are kept.
    # This substantially reduces the dataset while preserving all positive examples.
    # Zeros arise naturally from the cross-join (weeks where a cell had no cases).
    active_cells = weekly["cell_id"].unique()
    all_weeks = weekly["week_start"].sort_values().unique()

    full_index = pd.MultiIndex.from_product(
        [active_cells, all_weeks], names=["cell_id", "week_start"]
    )
    weekly = (
        weekly.set_index(["cell_id", "week_start"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Join cell metadata (station, coordinates)
    cell_meta = build_cell_metadata().set_index("cell_id")
    weekly = weekly.join(cell_meta, on="cell_id")

    return weekly.sort_values(["cell_id", "week_start"]).reset_index(drop=True)


def add_targets(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward-looking target variables to the weekly DataFrame.

    Target A — cluster_onset_2w (binary):
        1 if cases > 0 in the cell during the next HORIZON_DAYS days.
        Implemented as: cases in the next ceil(HORIZON_DAYS/7) weeks > 0.

    Target B — cases_next_2w (regression, log-scaled):
        log(1 + total cases in the next HORIZON_DAYS days).

    Both targets are set to NaN for the last HORIZON_DAYS//7 weeks of the dataset
    (no future data available — these rows are dropped before training).
    """
    HORIZON_WEEKS = int(np.ceil(HORIZON_DAYS / 7))  # = 2 weeks

    df = weekly.sort_values(["cell_id", "week_start"]).copy()

    # Forward-shift: cases_next_Nw = shift(-N) on cases_this_week
    grp = df.groupby("cell_id")["cases_this_week"]
    future_cases = sum(grp.shift(-k) for k in range(1, HORIZON_WEEKS + 1))

    df["cases_next_2w"]     = future_cases
    df["cluster_onset_2w"]  = (future_cases > 0).astype(float)  # float so NaN propagates
    df["log_cases_next_2w"] = np.log1p(future_cases)

    # Last HORIZON_WEEKS rows per cell have no future data
    df.loc[df["cases_next_2w"].isna(), ["cluster_onset_2w", "log_cases_next_2w"]] = np.nan

    return df


def build_dataset(verbose: bool = True) -> pd.DataFrame:
    """
    Full pipeline: load dengue → aggregate → add targets → return.
    Targets + cases_this_week are excluded from the feature matrix
    but kept in the returned DataFrame for downstream use.
    """
    if verbose:
        print("Aggregating dengue data to weekly grid cells …")
    weekly = build_weekly_dengue()
    if verbose:
        n_cells = weekly["cell_id"].nunique()
        n_weeks = weekly["week_start"].nunique()
        print(f"  {n_cells} active cells × {n_weeks} weeks = {len(weekly):,} rows")

    if verbose:
        print("Adding target variables …")
    weekly = add_targets(weekly)

    pos = (weekly["cluster_onset_2w"] == 1).sum()
    total = weekly["cluster_onset_2w"].notna().sum()
    if verbose:
        print(f"  Positive rate: {pos}/{total} = {pos/total:.1%}")

    return weekly
