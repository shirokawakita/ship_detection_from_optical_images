"""Build YOLO-OBB dataset from Airbus Ship Detection CSV and images."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

from ship_sat.obb_labels import mask_to_yolo_obb_lines
from ship_sat.rle import rle_decode


def _find_default_paths(root: Path) -> tuple[Path, Path]:
    """Resolve CSV and train image directory under Airbus extract layout."""
    candidates_csv = [
        root / "train_ship_segmentations_v2.csv",
        root / "train_ship_segmentations.csv",
    ]
    csv_path = next((p for p in candidates_csv if p.is_file()), candidates_csv[0])

    candidates_img = [root / "train_v2", root / "train", root / "train_images"]
    img_dir = next((p for p in candidates_img if p.is_dir()), root / "train_v2")
    return csv_path, img_dir


def _image_path(img_dir: Path, image_id: str) -> Path:
    stem = Path(image_id).stem
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return img_dir / image_id


def prepare_dataset(
    data_root: Path,
    output_root: Path,
    val_ratio: float,
    seed: int,
    min_contour_area: float,
    class_id: int = 0,
) -> Path:
    data_root = data_root.resolve()
    output_root = output_root.resolve()

    csv_path, img_dir = _find_default_paths(data_root)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    df = pd.read_csv(csv_path)
    if "ImageId" not in df.columns or "EncodedPixels" not in df.columns:
        raise ValueError("CSV must contain ImageId and EncodedPixels columns")

    groups = df.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()
    image_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    n = len(image_ids)
    if n == 0:
        raise ValueError("No images found in CSV")
    if n == 1:
        val_set: set[str] = set()
    else:
        n_val = max(1, int(n * val_ratio))
        n_val = min(n_val, n - 1)
        val_set = set(image_ids[:n_val])

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_id in tqdm(image_ids, desc="images"):
        split = "val" if image_id in val_set else "train"
        img_path = _image_path(img_dir, image_id)
        if not img_path.is_file():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        label_lines: list[str] = []
        for rle in groups[image_id]:
            if rle is None or (isinstance(rle, float) and pd.isna(rle)):
                continue
            if isinstance(rle, str) and not str(rle).strip():
                continue
            mask = rle_decode(rle, (h, w))
            if mask.max() == 0:
                continue
            label_lines.extend(
                mask_to_yolo_obb_lines(mask, class_id=class_id, min_contour_area=min_contour_area)
            )

        out_img = output_root / "images" / split / img_path.name
        out_lbl = output_root / "labels" / split / f"{img_path.stem}.txt"
        shutil.copy2(img_path, out_img)
        out_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    data_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": {class_id: "ship"},
    }
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO-OBB dataset from Airbus CSV + images")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("AIRBUS_DATA_ROOT", "data/airbus_raw")),
        help="Unpacked Airbus Ship Detection root (CSV + train_v2)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/yolo_ship_obb"),
        help="Output YOLO dataset root (images/, labels/, data.yaml)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-contour-area", type=float, default=2.0)
    args = parser.parse_args(argv)

    yaml_path = prepare_dataset(
        data_root=args.data_root,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_contour_area=args.min_contour_area,
    )
    print(f"Wrote {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
