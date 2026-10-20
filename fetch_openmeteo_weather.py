"""
Fetch historical daily weather data from Open-Meteo (ERA5 reanalysis) for Singapore.

Covers 2015-01-01 to 2020-11-30 (extra lead-in for lag feature engineering).
Uses 4 representative grid points spread across the island; Open-Meteo ERA5
resolution is ~0.25° so these capture realistic within-island variation.

Output: data/weather_openmeteo/<location_name>.csv  +  combined.csv
"""

import json
import time
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).parent / "data" / "weather_openmeteo"

# Representative points covering Singapore's extent
LOCATIONS = [
    {"name": "central",  "lat": 1.3521, "lon": 103.8198},
    {"name": "north",    "lat": 1.4350, "lon": 103.8350},
    {"name": "east",     "lat": 1.3540, "lon": 103.9550},
    {"name": "west",     "lat": 1.3440, "lon": 103.6930},
]

START_DATE = "2015-01-01"  # extra lead-in for lag features
END_DATE   = "2020-11-30"

DAILY_VARS = ",".join([
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
])

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_location(session: requests.Session, loc: dict) -> list[dict]:
    params = {
        "latitude":   loc["lat"],
        "longitude":  loc["lon"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      DAILY_VARS,
        "timezone":   "Asia/Singapore",
    }
    resp = session.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    dates = daily["time"]
    records = []
    for i, date in enumerate(dates):
        row = {
            "date":     date,
            "location": loc["name"],
            "latitude": loc["lat"],
            "longitude": loc["lon"],
        }
        for var in DAILY_VARS.split(","):
            row[var] = daily[var][i]
        records.append(row)
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    import csv
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    with requests.Session() as session:
        session.headers["User-Agent"] = "dengue-weather-fetcher/1.0"

        for loc in LOCATIONS:
            print(f"Fetching {loc['name']} ({loc['lat']}, {loc['lon']}) ...", end=" ", flush=True)
            records = fetch_location(session, loc)
            write_csv(OUTPUT_DIR / f"{loc['name']}.csv", records)
            all_records.extend(records)
            print(f"{len(records)} days saved.")
            time.sleep(1)  # be polite to the free API

    # Combined file sorted by date then location
    all_records.sort(key=lambda r: (r["date"], r["location"]))
    write_csv(OUTPUT_DIR / "combined.csv", all_records)
    print(f"\nDone. {len(all_records)} total rows → {OUTPUT_DIR / 'combined.csv'}")

    # Persist metadata
    meta = {
        "source":     "Open-Meteo ERA5 historical archive",
        "url":        BASE_URL,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "locations":  LOCATIONS,
        "variables":  DAILY_VARS.split(","),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata → {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
