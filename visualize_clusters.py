"""
Visualize dengue cluster data on an interactive map of Singapore.

Usage:
    python visualize_clusters.py data/csv/180611-clusters.csv
    python visualize_clusters.py data/csv/180611-clusters.csv --output my_map.html
"""

import argparse
import sys
from pathlib import Path

import folium
import pandas as pd
import xyzservices.providers as xyz
from branca.colormap import linear

COLUMNS = [
    "num_cases",
    "address",
    "lat",
    "lon",
    "cluster_number",
    "recent_cases",
    "total_cases",
    "date",
    "month_number",
]

SINGAPORE_CENTER = [1.3521, 103.8198]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["num_cases"] = pd.to_numeric(df["num_cases"], errors="coerce")
    df["total_cases"] = pd.to_numeric(df["total_cases"], errors="coerce")
    df["recent_cases"] = pd.to_numeric(df["recent_cases"], errors="coerce")
    df.dropna(subset=["lat", "lon"], inplace=True)
    return df


def build_map(df: pd.DataFrame, title: str) -> folium.Map:
    m = folium.Map(location=SINGAPORE_CENTER, zoom_start=12, tiles=xyz.Esri.WorldGrayCanvas)

    # Color scale by total cases in each cluster (aggregate per cluster)
    cluster_totals = df.groupby("cluster_number")["total_cases"].max()
    max_total = max(cluster_totals.max(), 1)
    colormap = linear.YlOrRd_09.scale(1, max_total)
    colormap.caption = "Total cases in cluster"
    colormap.add_to(m)

    # Group markers by cluster for a LayerGroup per cluster
    cluster_groups: dict = {}
    for cluster_id, group in df.groupby("cluster_number"):
        total = int(group["total_cases"].iloc[0])
        recent = int(group["recent_cases"].iloc[0])
        layer = folium.FeatureGroup(name=f"Cluster {cluster_id} (total {total})")
        color = colormap(total)

        for _, row in group.iterrows():
            radius = max(5, row["num_cases"] ** 0.6 * 4)
            popup_html = (
                f"<b>{row['address'].title()}</b><br>"
                f"Cases at location: {int(row['num_cases'])}<br>"
                f"Recent cases (cluster): {recent}<br>"
                f"Total cases (cluster): {total}<br>"
                f"Cluster #: {cluster_id}<br>"
                f"Date: {row['date']}"
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{row['address'].title()} — {int(row['num_cases'])} case(s)",
            ).add_to(layer)

        layer.add_to(m)
        cluster_groups[cluster_id] = layer

    folium.LayerControl(collapsed=False).add_to(m)

    # Title box
    title_html = (
        f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        f"z-index:1000;background:white;padding:8px 16px;border-radius:6px;"
        f'box-shadow:0 2px 6px rgba(0,0,0,.3);font-family:sans-serif;">'
        f"<b>Dengue Clusters — {title}</b></div>"
    )
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def main():
    parser = argparse.ArgumentParser(description="Visualize dengue clusters on a Singapore map.")
    parser.add_argument("csv", help="Path to the cluster CSV file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML file path (default: <csv-stem>_map.html next to the CSV)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: file not found — {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else csv_path.with_name(csv_path.stem + "_map.html")

    print(f"Loading {csv_path} ...")
    df = load_csv(csv_path)
    print(f"  {len(df)} location rows, {df['cluster_number'].nunique()} clusters")

    date_str = str(df["date"].iloc[0]) if not df.empty else csv_path.stem
    m = build_map(df, title=date_str)
    m.save(str(output_path))
    print(f"Map saved to: {output_path}")


if __name__ == "__main__":
    main()
