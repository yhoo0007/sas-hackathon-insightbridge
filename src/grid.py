"""
Singapore 1 km grid utilities.

Responsibilities:
- Define the grid over Singapore's bounding box
- Assign lat/lon points to a cell_id
- Assign each cell to its nearest ERA5 weather station (Voronoi)
- Compute 8-connected cell neighbours (used for spatial autoregressive features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Singapore bounding box
LAT_MIN, LAT_MAX = 1.15, 1.48
LON_MIN, LON_MAX = 103.59, 104.10

# ~1 km in degrees at Singapore's latitude
# 1° lat ≈ 111 km everywhere; 1° lon ≈ 111 × cos(1.3°) ≈ 111 km at equator
CELL_DEG = 1.0 / 111.0  # ≈ 0.009009°

# ERA5 weather grid points (from fetch_openmeteo_weather.py)
WEATHER_STATIONS = {
    "central": (1.3521, 103.8198),
    "north":   (1.4350, 103.8350),
    "east":    (1.3540, 103.9550),
    "west":    (1.3440, 103.6930),
}


def _lat_to_row(lat: float) -> int:
    return int((lat - LAT_MIN) / CELL_DEG)


def _lon_to_col(lon: float) -> int:
    return int((lon - LON_MIN) / CELL_DEG)


def latlon_to_cell_id(lat: float | np.ndarray, lon: float | np.ndarray) -> np.ndarray:
    """Return a string cell_id of the form 'r{row:03d}_c{col:03d}' for each point."""
    rows = ((np.asarray(lat) - LAT_MIN) / CELL_DEG).astype(int)
    cols = ((np.asarray(lon) - LON_MIN) / CELL_DEG).astype(int)
    # Vectorised string formatting
    vfmt = np.vectorize(lambda r, c: f"r{r:03d}_c{c:03d}")
    return vfmt(rows, cols)


def cell_id_to_latlon(cell_id: str) -> tuple[float, float]:
    """Return the center lat/lon of a cell."""
    parts = cell_id.split("_")
    row = int(parts[0][1:])
    col = int(parts[1][1:])
    lat = LAT_MIN + (row + 0.5) * CELL_DEG
    lon = LON_MIN + (col + 0.5) * CELL_DEG
    return lat, lon


def build_cell_metadata() -> pd.DataFrame:
    """
    Return a DataFrame with one row per valid cell containing:
    cell_id, center_lat, center_lon, nearest_station
    """
    n_rows = int((LAT_MAX - LAT_MIN) / CELL_DEG) + 1
    n_cols = int((LON_MAX - LON_MIN) / CELL_DEG) + 1

    records = []
    for r in range(n_rows):
        for c in range(n_cols):
            lat = LAT_MIN + (r + 0.5) * CELL_DEG
            lon = LON_MIN + (c + 0.5) * CELL_DEG
            cell_id = f"r{r:03d}_c{c:03d}"
            nearest = _nearest_station(lat, lon)
            records.append({"cell_id": cell_id, "center_lat": lat, "center_lon": lon, "station": nearest})

    return pd.DataFrame(records)


def _nearest_station(lat: float, lon: float) -> str:
    """Return the name of the nearest weather station (Euclidean on degrees — fine for Singapore scale)."""
    best, best_dist = None, float("inf")
    for name, (slat, slon) in WEATHER_STATIONS.items():
        d = (lat - slat) ** 2 + (lon - slon) ** 2
        if d < best_dist:
            best_dist = d
            best = name
    return best


def add_cell_id(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude") -> pd.DataFrame:
    """Add a cell_id column to a DataFrame that has lat/lon columns."""
    df = df.copy()
    df["cell_id"] = latlon_to_cell_id(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    return df


def get_cell_neighbours(cell_id: str) -> list[str]:
    """Return the up-to-8 Moore-neighbourhood cell_ids for a given cell."""
    parts = cell_id.split("_")
    r = int(parts[0][1:])
    c = int(parts[1][1:])
    neighbours = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if nr >= 0 and nc >= 0:
                neighbours.append(f"r{nr:03d}_c{nc:03d}")
    return neighbours


def build_neighbour_map(cell_ids: list[str]) -> dict[str, list[str]]:
    """Return {cell_id: [neighbour_cell_ids present in cell_ids]} for the given cell list."""
    cell_set = set(cell_ids)
    return {cid: [n for n in get_cell_neighbours(cid) if n in cell_set] for cid in cell_ids}
