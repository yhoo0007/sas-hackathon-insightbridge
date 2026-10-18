"""
Weather data analysis and cleaning script.

Input : data/weather/all_stations_merged.csv
Output: data/weather/weather_cleaned.csv

Issues addressed:
  1. Leading/trailing whitespace in numeric fields
  2. '-' used as a missing-value sentinel alongside blank cells
  3. No parsed Date column
  4. Reports per-station and per-column missingness before/after cleaning
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/weather/all_stations_merged.csv")
OUT_PATH = Path("data/weather/weather_cleaned.csv")

NUMERIC_COLS = [
    "Daily Rainfall Total (mm)",
    "Highest 30 Min Rainfall (mm)",
    "Highest 60 Min Rainfall (mm)",
    "Highest 120 Min Rainfall (mm)",
    "Mean Temperature (°C)",
    "Maximum Temperature (°C)",
    "Minimum Temperature (°C)",
    "Mean Wind Speed (km/h)",
    "Max Wind Speed (km/h)",
]


# ── 1. Load ──────────────────────────────────────────────────────────────────

print("Loading raw data …")
df = pd.read_csv(RAW_PATH, dtype=str)
df.columns = df.columns.str.strip()
print(f"  Shape: {df.shape}")


# ── 2. Strip whitespace & replace '-' sentinel ────────────────────────────────

for col in df.columns:
    df[col] = df[col].str.strip()

df.replace("-", pd.NA, inplace=True)
df.replace("", pd.NA, inplace=True)


# ── 3. Convert types ──────────────────────────────────────────────────────────

df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce").astype("Int64")
df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
df["Day"]   = pd.to_numeric(df["Day"],   errors="coerce").astype("Int64")

# Drop rows where date components are missing (shouldn't happen in clean source)
before = len(df)
df = df.dropna(subset=["Year", "Month", "Day"])
if len(df) < before:
    print(f"  Dropped {before - len(df)} rows with missing date components")

df["Year"]  = df["Year"].astype(int)
df["Month"] = df["Month"].astype(int)
df["Day"]   = df["Day"].astype(int)

for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ── 4. Add a proper Date column ───────────────────────────────────────────────

df.insert(1, "Date", pd.to_datetime(df[["Year", "Month", "Day"]]))


# ── 5. Analysis: missing values ───────────────────────────────────────────────

total = len(df)

print("\n── Missing values per column ────────────────────────────────")
missing = df[NUMERIC_COLS].isna().sum().rename("missing")
missing_pct = (missing / total * 100).rename("%")
print(pd.concat([missing, missing_pct.map("{:.1f}%".format)], axis=1).to_string())

print("\n── Stations with most missing Mean Temperature ─────────────")
miss_temp = (
    df.groupby("Station")["Mean Temperature (°C)"]
    .apply(lambda s: s.isna().mean() * 100)
    .sort_values(ascending=False)
    .head(15)
    .map("{:.0f}%".format)
)
print(miss_temp.to_string())

print("\n── Date range ───────────────────────────────────────────────")
print(f"  From : {df['Date'].min().date()}")
print(f"  To   : {df['Date'].max().date()}")
print(f"  Days : {df['Date'].nunique()}")
print(f"  Stations: {df['Station'].nunique()}")

print("\n── Basic statistics for numeric columns ────────────────────")
print(df[NUMERIC_COLS].describe().round(2).to_string())

print("\n── Duplicate rows ───────────────────────────────────────────")
dupes = df.duplicated(subset=["Station", "Date"]).sum()
print(f"  Duplicate (Station, Date) pairs: {dupes}")
if dupes:
    df = df.drop_duplicates(subset=["Station", "Date"], keep="first")
    print(f"  Dropped → {len(df)} rows remain")


# ── 6. Plausibility checks (flag outliers, don't drop) ───────────────────────

print("\n── Plausibility flags ───────────────────────────────────────")
flags = {
    "Daily Rainfall < 0":    (df["Daily Rainfall Total (mm)"] < 0).sum(),
    "Mean Temp < 20°C":      (df["Mean Temperature (°C)"] < 20).sum(),
    "Mean Temp > 40°C":      (df["Mean Temperature (°C)"] > 40).sum(),
    "Mean Wind < 0":         (df["Mean Wind Speed (km/h)"] < 0).sum(),
}
for label, count in flags.items():
    print(f"  {label}: {count}")


# ── 7. Save ───────────────────────────────────────────────────────────────────

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned data → {OUT_PATH}  ({len(df):,} rows)")
