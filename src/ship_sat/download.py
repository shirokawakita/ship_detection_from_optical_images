"""Download Airbus Ship Detection from Kaggle (requires API credentials)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION = "airbus-ship-detection"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Download Kaggle competition: {COMPETITION}")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("AIRBUS_DATA_ROOT", "data/airbus_raw")),
        help="Directory to download and unzip into (default: AIRBUS_DATA_ROOT or data/airbus_raw)",
    )
    args = parser.parse_args(argv)

    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not user or not key:
        print(
            "Kaggle API credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or place ~/.kaggle/kaggle.json. See https://www.kaggle.com/docs/api",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(args.output_dir),
        "--force",
    ]
    subprocess.run(cmd, check=True)
    zip_files = list(args.output_dir.glob("*.zip"))
    if not zip_files:
        print("No zip file found after download.", file=sys.stderr)
        return 1
    for z in zip_files:
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(args.output_dir)
    print(f"Dataset material available under: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
