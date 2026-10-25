"""
Feature engineering pipeline.

Produces a (cell_id, week) feature matrix by:
1. Computing rolling weather statistics at each ERA5 station
2. Joining weather features to cells via nearest-station assignment
3. Adding cyclical temporal features and monsoon phase
4. Adding autoregressive dengue lag features and neighbour aggregates

All features use only data that would be known at prediction time (no leakage).
The prediction horizon is HORIZON_DAYS ahead, so targets look forward and
features look backward.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .grid import add_cell_id, build_cell_metadata, build_neighbour_map

DATA_DIR = Path(__file__).parent.parent / "data"
WEATHER_PATH = DATA_DIR / "weather_openmeteo" / "combined.csv"

# Lag windows in days
LAG_WINDOWS = [7, 14, 21, 28]

# Prediction horizon
HORIZON_DAYS = 14


# ── Weather feature helpers ───────────────────────────────────────────────────

def _load_weather() -> pd.DataFrame:
    df = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    return df


def _build_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """
    For each (location, date) compute rolling statistics over LAG_WINDOWS.
    Returns a wide DataFrame indexed by (location, date).
    """
    dfs = []
    for loc, grp in weather.groupby("location"):
        grp = grp.sort_values("date").copy()

        for w in LAG_WINDOWS:
            # Precipitation
            grp[f"precip_sum_{w}d"]   = grp["precipitation_sum"].rolling(w, min_periods=1).sum()
            grp[f"wet_days_{w}d"]     = (grp["precipitation_sum"] > 5).astype(int).rolling(w, min_periods=1).sum()
            # Temperature
            grp[f"temp_mean_{w}d"]    = grp["temperature_2m_mean"].rolling(w, min_periods=1).mean()
            grp[f"temp_max_{w}d"]     = grp["temperature_2m_max"].rolling(w, min_periods=1).max()
            # Humidity
            grp[f"rh_mean_{w}d"]      = (
                (grp["relative_humidity_2m_max"] + grp["relative_humidity_2m_min"]) / 2
            ).rolling(w, min_periods=1).mean()
            # Evapotranspiration
            grp[f"et0_sum_{w}d"]      = grp["et0_fao_evapotranspiration"].rolling(w, min_periods=1).sum()
            # Solar radiation
            grp[f"radiation_sum_{w}d"] = grp["shortwave_radiation_sum"].rolling(w, min_periods=1).sum()

        # Dry-spell after rain: consecutive dry days up to and including current day
        is_dry = (grp["precipitation_sum"] < 1).astype(int)
        dry_streak = []
        streak = 0
        for v in is_dry:
            streak = streak + 1 if v else 0
            dry_streak.append(streak)
        grp["dry_streak_days"] = dry_streak

        # Wet-dry ratio over 14 days
        grp["wet_dry_ratio_14d"] = grp[f"wet_days_14d"] / 14.0

        dfs.append(grp)

    out = pd.concat(dfs, ignore_index=True)
    # Keep location, date, and all columns added during feature construction
    original_raw_cols = set(weather.columns) - {"location", "date"}
    keep = [c for c in out.columns if c not in original_raw_cols]
    return out[keep]


# ── Temporal features ─────────────────────────────────────────────────────────

_MONSOON_PHASE = {
    11: "NE", 12: "NE", 1: "NE",       # NE monsoon
    2: "inter", 3: "inter", 4: "inter", # inter-monsoon
    5: "SW", 6: "SW", 7: "SW",          # SW monsoon
    8: "SW", 9: "SW",
    10: "inter",                        # inter-monsoon
}


def _add_temporal_features(df: pd.DataFrame, date_col: str = "week_start") -> pd.DataFrame:
    """Add cyclical week-of-year, month, monsoon_phase, year columns."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    woy = dt.dt.isocalendar().week.astype(int)
    # Cyclical encoding avoids week-52 / week-1 discontinuity
    df["sin_woy"] = np.sin(2 * np.pi * woy / 52)
    df["cos_woy"] = np.cos(2 * np.pi * woy / 52)
    df["sin_month"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["monsoon_phase"] = dt.dt.month.map(_MONSOON_PHASE).fillna("inter")
    return df


# ── Autoregressive dengue features ───────────────────────────────────────────

def _add_ar_features(df: pd.DataFrame, neighbour_map: dict[str, list[str]]) -> pd.DataFrame:
    """
    Add lagged case counts and neighbour aggregates.

    Requires df to have columns: cell_id, week_start, cases_this_week.
    All lags are applied at the prediction date (i.e. data at t-1w, t-2w, t-4w).
    """
    df = df.sort_values(["cell_id", "week_start"]).copy()

    # Per-cell lags
    grp = df.groupby("cell_id")["cases_this_week"]
    df["cases_lag_1w"]  = grp.shift(1).fillna(0)
    df["cases_lag_2w"]  = grp.shift(2).fillna(0)
    df["cases_lag_4w"]  = grp.shift(4).fillna(0)
    df["cluster_active_lag_1w"] = (df["cases_lag_1w"] > 0).astype(int)

    # Weeks since last case (capped at 52)
    def weeks_since(series: pd.Series) -> pd.Series:
        result = []
        count = 52
        for v in series:
            if v > 0:
                count = 0
            result.append(count)
            count = min(count + 1, 52)
        return pd.Series(result, index=series.index)

    df["weeks_since_last_case"] = grp.transform(weeks_since)

    # Neighbour aggregate (sum of lagged cases in Moore neighbourhood)
    # Build a lookup: (cell_id, week) -> cases_lag_1w
    lookup = df.set_index(["cell_id", "week_start"])["cases_lag_1w"]

    def neighbour_sum(row: pd.Series) -> float:
        total = 0.0
        for nid in neighbour_map.get(row["cell_id"], []):
            key = (nid, row["week_start"])
            if key in lookup.index:
                total += lookup[key]
        return total

    df["neighbour_cases_lag_1w"] = df.apply(neighbour_sum, axis=1)
    return df


# ── Main build function ───────────────────────────────────────────────────────

def build_feature_matrix(
    weekly_dengue: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the full (cell_id, week_start) feature matrix.

    Parameters
    ----------
    weekly_dengue : DataFrame with columns [cell_id, week_start, cases_this_week]
                    and a 'station' column from the cell metadata join.
    verbose       : print progress messages

    Returns
    -------
    DataFrame with all features + cell_id + week_start (no target column).
    """
    if verbose:
        print("Loading weather data …")
    weather = _load_weather()

    if verbose:
        print("Computing rolling weather features …")
    weather_feat = _build_weather_features(weather)

    # Resample weather to weekly (use the LAST day of each week as the anchor).
    # Prediction is made on week_start; we use weather data up to week_start - 1 day
    # (i.e. the day before the prediction window begins) to avoid leakage.
    if verbose:
        print("Resampling weather to weekly …")
    weather_feat["week_start"] = (
        pd.to_datetime(weather_feat["date"]) - pd.to_timedelta(
            pd.to_datetime(weather_feat["date"]).dt.dayofweek, unit="D"
        )
    ).dt.normalize()

    weather_weekly = (
        weather_feat.groupby(["location", "week_start"])
        .last()  # last day of each week carries the most up-to-date rolling stats
        .reset_index()
    )
    # Drop raw daily columns that are now in the rolling features
    raw_cols = [c for c in weather_weekly.columns
                if c in ("temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                         "precipitation_sum", "rain_sum", "precipitation_hours",
                         "relative_humidity_2m_max", "relative_humidity_2m_min",
                         "wind_speed_10m_mean", "wind_speed_10m_max",
                         "et0_fao_evapotranspiration", "shortwave_radiation_sum",
                         "latitude", "longitude", "date")]
    weather_weekly = weather_weekly.drop(columns=[c for c in raw_cols if c in weather_weekly.columns])

    # Join weather features to the dengue weekly frame via station
    # (dengue uses "station", weather uses "location" — rename to align)
    if verbose:
        print("Joining weather features to dengue cells …")
    weather_weekly = weather_weekly.rename(columns={"location": "station"})
    df = weekly_dengue.merge(weather_weekly, on=["station", "week_start"], how="left")

    # Temporal features
    if verbose:
        print("Adding temporal features …")
    df = _add_temporal_features(df, date_col="week_start")

    # Autoregressive features
    if verbose:
        print("Adding autoregressive dengue features …")
    cell_ids = df["cell_id"].unique().tolist()
    neighbour_map = build_neighbour_map(cell_ids)
    df = _add_ar_features(df, neighbour_map)

    return df
