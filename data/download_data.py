"""
NYC Yellow Taxi Data Downloader
Downloads monthly parquet files from NYC TLC (Jan 2024 → Feb 2026)

Why 26 months?
  - Captures 2 full seasonal cycles (summer peaks, winter dips, holidays)
  - Enables reliable lag_168h (1 week) and year-over-year features
  - Sufficient data for robust train/validation/test splits
  - ~7-8 GB on disk uncompressed; ~1.5 GB as parquet
"""

import os
import requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
RAW_DIR  = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Jan 2024 → Feb 2026  (26 months — latest available as of April 2026)
MONTHS = [
    # ── 2024 (full year) ──────────────────────────────────────────────────────
    "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
    "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    # ── 2025 (full year) ──────────────────────────────────────────────────────
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    # ── 2026 (latest available) ───────────────────────────────────────────────
    "2026-01", "2026-02",
]


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return

    print(f"  ⬇ Downloading {dest.name} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = dest.stat().st_size / (1024 ** 2)
    print(f"  ✓ Saved {dest.name} ({size_mb:.1f} MB)")


def main():
    print("=" * 55)
    print("  NYC Yellow Taxi Data Downloader")
    print("=" * 55)

    for month in MONTHS:
        fname = f"yellow_tripdata_{month}.parquet"
        url   = f"{BASE_URL}/{fname}"
        dest  = RAW_DIR / fname
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  ✗ Failed {fname}: {e}")

    print("\nAll downloads complete. Files saved to:", RAW_DIR)


if __name__ == "__main__":
    main()
