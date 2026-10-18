"""
Download historical daily weather data from Meteorological Service Singapore.

URL pattern: https://www.weather.gov.sg/files/dailydata/DAILYDATA_{station_id}_{YYYYMM}.csv

Usage:
    python download_weather_data.py --start 2020-01 --end 2023-12 --output ./weather_data
    python download_weather_data.py --start 2020-01 --end 2023-12 --output ./weather_data --merge
    python download_weather_data.py --start 2020-01 --end 2023-12 --stations S24 S25 S06
"""

import argparse
import csv
import io
import os
import time
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://www.weather.gov.sg/files/dailydata/DAILYDATA_{station}_{yyyymm}.csv"

# Full station list scraped from https://www.weather.gov.sg/climate-historical-daily/
STATIONS = {
    "S06": "Paya Lebar Meteorological Station",
    "S07": "Macritchie Reservoir",
    "S08": "Lower Peirce Reservoir",
    "S11": "Jalan Bahar",
    "S23": "Tengah Meteorological Station",
    "S24": "Changi Meteorological Station",
    "S25": "Seletar Meteorological Station",
    "S29": "Pasir Ris Road",
    "S31": "Kampong Bahru Road",
    "S33": "Jurong Pier Road",
    "S35": "Old Toh Tuck Road",
    "S36": "Upper Serangoon Road",
    "S39": "Jalan Ahmad Ibrahim",
    "S40": "Mandai Lake Road",
    "S43": "Kim Chuan Road",
    "S44": "Nanyang Avenue",
    "S46": "Sime Road",
    "S50": "Clementi Road",
    "S55": "Buangkok Green",
    "S60": "Artillery Avenue (Sentosa)",
    "S61": "Chai Chee Street",
    "S63": "International Road",
    "S64": "Bukit Panjang Road",
    "S66": "Kranji Reservoir",
    "S69": "Upper Peirce Reservoir",
    "S71": "Kent Ridge Road",
    "S72": "Prince Edward Road",
    "S77": "Alexandra Road",
    "S78": "Poole Road",
    "S79": "Somerset Road",
    "S80": "Sembawang Meteorological Station",
    "S81": "Punggol Central",
    "S82": "Tuas West Road",
    "S84": "Tampines Avenue 5",
    "S86": "Chia Ping Road",
    "S88": "Toa Payoh North",
    "S89": "Tuas Road",
    "S90": "Bukit Timah Road",
    "S91": "Yishun Avenue 5",
    "S92": "South Buona Vista Road",
    "S94": "Pasir Ris Street 51",
    "S100": "Sungei Kadut Street 3",
    "S101": "Corporation Road",
    "S102": "Semakau Island",
    "S104": "Woodlands Avenue 9",
    "S105": "Admiralty Road West",
    "S106": "Jalan Noordin (Pulau Ubin)",
    "S107": "East Coast Park",
    "S108": "Marina Barrage",
    "S109": "Ang Mo Kio Avenue 5",
    "S110": "Serangoon North Avenue 1",
    "S111": "Scotts Road",
    "S112": "Lim Chu Kang Road",
    "S113": "Marine Parade Road",
    "S114": "Choa Chu Kang Avenue 4",
    "S115": "Tuas South Avenue 3",
    "S116": "Pasir Panjang Terminal",
    "S117": "Banyan Road (Jurong Island)",
    "S118": "Handy Road",
    "S119": "Nicoll Highway",
    "S120": "Botanic Garden",
    "S121": "Old Choa Chu Kang Road",
    "S122": "Sembawang Road",
    "S123": "Towner Road",
    "S127": "Kallang Practice Track",
    "S129": "Bedok Stadium",
    "S130": "Clementi Stadium",
    "S132": "Jurong West Stadium",
    "S148": "Pasir Ris Walk",
    "S180": "Taman Jurong Greens",
    "S184": "Sengkang East Avenue",
    "S187": "Coronation Road",
    "S201": "Sunset Way",
    "S203": "Pasir Panjang Road",
    "S207": "Koh Sek Lim Road",
    "S208": "Changi East Close",
    "S209": "Yishun Ring Road",
    "S210": "Woodlands Centre Road",
    "S211": "Kranji Road",
    "S213": "Coronation Walk",
    "S214": "Tanjong Rhu",
    "S215": "Geylang East Central",
    "S216": "Ang Mo Kio Avenue 10",
    "S217": "Bishan Street 13",
    "S218": "Bukit Batok Street 34",
    "S219": "Yio Chu Kang Road",
    "S220": "Compassvale Lane",
    "S221": "Hougang Avenue 1",
    "S222": "Henderson Road",
    "S223": "Margaret Drive",
    "S224": "Airport Boulevard",
    "S226": "Malan Road",
    "S227": "Woodlands Drive 62",
    "S228": "Jurong West Street 73",
    "S229": "Jurong West Street 42",
    "S230": "West Coast Road",
    "S300": "Bedok Reservoir",
    "S301": "Neo Tiew Lane 2",
    "S302": "Tengeh Reservoir",
    "S303": "Poyan Reservoir",
    "S305": "Sarimbun Reservoir",
    "S306": "Murnane Service Reservoir",
    "S307": "Pulau Tekong Reservoir",
    "S308": "Seletar Reservoir",
    "S310": "Lorong Danau",
    "S902": "Yishun Walk",
    "S903": "Lorong Kismis",
    "S904": "Holland Grove Road",
    "S905": "Hillcrest Road",
    "S907": "Rivervale Drive",
    "S908": "Fernvale Lane",
    "S909": "Seraya Road",
}


def iter_months(start: date, end: date):
    """Yield (year, month) tuples from start to end inclusive."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def download_month(session: requests.Session, station_id: str, year: int, month: int,
                   output_dir: Path, delay: float) -> tuple[bool, str | None]:
    """
    Download one station-month CSV. Returns (success, csv_text_or_None).
    Skips if the local file already exists.
    """
    yyyymm = f"{year}{month:02d}"
    filename = f"DAILYDATA_{station_id}_{yyyymm}.csv"
    file_path = output_dir / station_id / filename

    if file_path.exists():
        return True, None  # already downloaded

    url = BASE_URL.format(station=station_id, yyyymm=yyyymm)
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return False, None  # no data for this period
        resp.raise_for_status()

        # Validate: the file should start with "Station" header
        text = resp.text
        if not text.strip() or "Station" not in text.split("\n")[0]:
            return False, None

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")
        time.sleep(delay)
        return True, text

    except requests.RequestException as e:
        print(f"    ERROR {station_id} {yyyymm}: {e}")
        return False, None


def merge_csvs(output_dir: Path, merged_path: Path) -> None:
    """Merge all downloaded CSVs into a single file."""
    header_written = False
    total_rows = 0

    with merged_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        for csv_file in sorted(output_dir.rglob("DAILYDATA_*.csv")):
            if csv_file == merged_path:
                continue
            text = csv_file.read_text(encoding="utf-8")
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if not rows:
                continue
            if not header_written:
                writer.writerow(rows[0])
                header_written = True
            for row in rows[1:]:
                if row:
                    writer.writerow(row)
                    total_rows += 1

    print(f"\nMerged {total_rows} rows → {merged_path}")


def parse_ym(s: str) -> date:
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM, got: {s!r}")
    return date(int(parts[0]), int(parts[1]), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Singapore historical daily weather data from weather.gov.sg"
    )
    parser.add_argument("--start", required=True, type=parse_ym, metavar="YYYY-MM",
                        help="First month to download (e.g. 2020-01)")
    parser.add_argument("--end", required=True, type=parse_ym, metavar="YYYY-MM",
                        help="Last month to download (e.g. 2023-12)")
    parser.add_argument("--output", default="./weather_data", metavar="DIR",
                        help="Output directory (default: ./weather_data)")
    parser.add_argument("--stations", nargs="+", metavar="ID",
                        help="Limit to specific station IDs, e.g. S24 S25 S06. "
                             "Defaults to all stations.")
    parser.add_argument("--merge", action="store_true",
                        help="After downloading, merge all CSVs into a single file")
    parser.add_argument("--delay", type=float, default=0.5, metavar="SECS",
                        help="Seconds to wait between requests (default: 0.5)")
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be before or equal to --end")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_ids = args.stations if args.stations else list(STATIONS.keys())
    unknown = [s for s in station_ids if s not in STATIONS]
    if unknown:
        parser.error(f"Unknown station IDs: {', '.join(unknown)}\n"
                     f"Valid IDs: {', '.join(sorted(STATIONS))}")

    months = list(iter_months(args.start, args.end))
    total = len(station_ids) * len(months)
    print(f"Stations : {len(station_ids)}")
    print(f"Months   : {len(months)}  ({args.start.strftime('%Y-%m')} → {args.end.strftime('%Y-%m')})")
    print(f"Max files: {total}")
    print(f"Output   : {output_dir.resolve()}\n")

    downloaded = skipped = missing = 0

    with requests.Session() as session:
        session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; weather-data-downloader/1.0)"
        )

        for i, station_id in enumerate(station_ids, 1):
            name = STATIONS[station_id]
            station_downloaded = 0
            station_missing = 0

            for year, month in months:
                ok, _ = download_month(session, station_id, year, month, output_dir, args.delay)
                if ok:
                    station_downloaded += 1
                    downloaded += 1
                else:
                    station_missing += 1
                    missing += 1

            status = f"{station_downloaded} files" if station_downloaded else "no data"
            print(f"[{i:3d}/{len(station_ids)}] {station_id:5s}  {name:<45s}  {status}")

    print(f"\nDone. Downloaded: {downloaded}  |  No data / skipped: {missing}")

    if args.merge and downloaded > 0:
        merged_path = output_dir / "all_stations_merged.csv"
        merge_csvs(output_dir, merged_path)


if __name__ == "__main__":
    main()
