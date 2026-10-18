"""
Dengue cluster data combination and cleaning script.

Input : data/csv/*-clusters.csv  (one file per snapshot date, no headers)
Output: data/dengue_combined.csv

Issues addressed:
  1. No column headers in source files
  2. Date stored as YYMMDD integer string → parsed to proper datetime
  3. Street addresses normalised to consistent title-case
  4. Cluster Number cannot be used as unique identifier (reused across snapshots
     and changes over a cluster's lifetime) — noted in comments, kept as-is
  5. Reports basic statistics and data quality summary
"""

from pathlib import Path

import pandas as pd

CSV_DIR  = Path("data/csv")
OUT_PATH = Path("data/dengue_combined.csv")

COLUMNS = [
    "Number Of Cases",
    "Street Address",
    "Latitude",
    "Longitude",
    "Cluster Number",
    "Recent Cases In Cluster",
    "Total Cases In Cluster",
    "Date Raw",       # original YYMMDD string from filename-embedded date column
    "Month Number",
]


# ── 1. Load all snapshot files ────────────────────────────────────────────────

csv_files = sorted(CSV_DIR.glob("*-clusters.csv"))
print(f"Found {len(csv_files)} snapshot files")

frames = []
for path in csv_files:
    try:
        chunk = pd.read_csv(path, header=None, names=COLUMNS, dtype=str)
        chunk["Source File"] = path.name
        frames.append(chunk)
    except Exception as e:
        print(f"  WARNING: could not read {path.name}: {e}")

df = pd.concat(frames, ignore_index=True)
print(f"Combined shape: {df.shape}")


# ── 2. Parse Date column ──────────────────────────────────────────────────────
# Date Raw is YYMMDD, e.g. "150703" → 3 Jul 2015

def parse_yymmdd(s: str) -> pd.Timestamp:
    s = str(s).strip().zfill(6)
    return pd.Timestamp(year=2000 + int(s[:2]), month=int(s[2:4]), day=int(s[4:6]))

df["Date"] = df["Date Raw"].apply(parse_yymmdd)
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"]   = df["Date"].dt.day

df.drop(columns=["Date Raw"], inplace=True)


# ── 3. Convert numeric columns ────────────────────────────────────────────────

int_cols   = ["Number Of Cases", "Cluster Number", "Recent Cases In Cluster",
              "Total Cases In Cluster", "Month Number", "Year", "Month", "Day"]
float_cols = ["Latitude", "Longitude"]

for col in int_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

for col in float_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ── 4. Clean street addresses ─────────────────────────────────────────────────

df["Street Address"] = (
    df["Street Address"]
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)


# ── 5. Column order ───────────────────────────────────────────────────────────

df = df[[
    "Date", "Year", "Month", "Day",
    "Street Address", "Latitude", "Longitude",
    "Cluster Number", "Number Of Cases",
    "Recent Cases In Cluster", "Total Cases In Cluster",
    "Month Number", "Source File",
]]


# ── 6. Analysis ───────────────────────────────────────────────────────────────

print("\n── Date range ───────────────────────────────────────────────")
print(f"  From  : {df['Date'].min().date()}")
print(f"  To    : {df['Date'].max().date()}")
print(f"  Unique snapshot dates : {df['Date'].nunique()}")

print("\n── Missing values ───────────────────────────────────────────")
missing = df.isna().sum()
missing = missing[missing > 0]
if missing.empty:
    print("  None")
else:
    print(missing.to_string())

print("\n── Cases per snapshot (Number Of Cases sum) ─────────────────")
cases_per_date = (
    df.groupby("Date")["Number Of Cases"].sum().sort_values(ascending=False)
)
print(f"  Max total cases on a single date : {cases_per_date.max():,}  ({cases_per_date.idxmax().date()})")
print(f"  Min total cases on a single date : {cases_per_date.min():,}  ({cases_per_date.idxmin().date()})")
print(f"  Mean total cases per snapshot    : {cases_per_date.mean():.0f}")

print("\n── Active clusters per snapshot ─────────────────────────────")
clusters_per_date = df.groupby("Date")["Cluster Number"].nunique()
print(f"  Max active clusters : {clusters_per_date.max()}  ({clusters_per_date.idxmax().date()})")
print(f"  Mean active clusters: {clusters_per_date.mean():.1f}")

print("\n── Top 10 most-affected addresses (total cases across all snapshots) ──")
top_addresses = (
    df.groupby("Street Address")["Number Of Cases"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_addresses.to_string())

print("\n── Plausibility checks ──────────────────────────────────────")
print(f"  Number Of Cases <= 0    : {(df['Number Of Cases'] <= 0).sum()}")
print(f"  Latitude out of SG range: {((df['Latitude'] < 1.15) | (df['Latitude'] > 1.48)).sum()}")
print(f"  Longitude out of SG range: {((df['Longitude'] < 103.6) | (df['Longitude'] > 104.1)).sum()}")


# ── 7. Save ───────────────────────────────────────────────────────────────────

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved combined dengue data → {OUT_PATH}  ({len(df):,} rows)")
