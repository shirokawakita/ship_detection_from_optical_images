"""SAHI sliced inference with Ultralytics OBB weights."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml
from sahi import AutoDetectionModel
from sahi.prediction import PredictionResult
from sahi.predict import get_prediction, get_sliced_prediction


def _load_defaults(config_path: Path | None) -> dict:
    if not config_path or not config_path.is_file():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def predict_sahi(
    weights: Path,
    source: Path,
    *,
    device: str = "cpu",
    confidence_threshold: float = 0.25,
    slice_width: int = 640,
    slice_height: int = 640,
    overlap_width_ratio: float = 0.2,
    overlap_height_ratio: float = 0.2,
    use_slices: bool = True,
    image_size: int | None = None,
) -> PredictionResult:
    """Load weights and run tiled or single-shot SAHI prediction."""
    weights = weights.resolve()
    source = source.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        device=device,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
    )

    if use_slices:
        return get_sliced_prediction(
            str(source),
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=0,
        )
    return get_prediction(str(source), detection_model, verbose=0)


def export_prediction_visual(
    result: PredictionResult,
    output: Path,
    *,
    hide_conf: bool = False,
) -> Path:
    """Write ``result`` to a PNG path (``hide_conf=False`` keeps score labels like CLI)."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    export_dir = str(output.parent)
    result.export_visuals(export_dir=export_dir, file_name=stem, hide_conf=hide_conf)
    written = output.parent / f"{stem}.png"
    target = output if output.suffix.lower() == ".png" else output.with_suffix(".png")
    if written.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(written), str(target))
    return target


def run_sahi_inference(
    weights: Path,
    source: Path,
    output: Path,
    *,
    device: str = "cpu",
    confidence_threshold: float = 0.25,
    slice_width: int = 640,
    slice_height: int = 640,
    overlap_width_ratio: float = 0.2,
    overlap_height_ratio: float = 0.2,
    use_slices: bool = True,
    image_size: int | None = None,
) -> Path:
    """Run SAHI prediction and write a PNG visualization to ``output``."""
    result = predict_sahi(
        weights,
        source,
        device=device,
        confidence_threshold=confidence_threshold,
        slice_width=slice_width,
        slice_height=slice_height,
        overlap_width_ratio=overlap_width_ratio,
        overlap_height_ratio=overlap_height_ratio,
        use_slices=use_slices,
        image_size=image_size,
    )
    return export_prediction_visual(result, output, hide_conf=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAHI tiled inference (Ultralytics OBB)")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/sahi/prediction.png"))
    parser.add_argument("--device", type=str, default=os.environ.get("SAHI_DEVICE", "cpu"))
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--slice-width", type=int, default=None)
    parser.add_argument("--slice-height", type=int, default=None)
    parser.add_argument("--overlap-width-ratio", type=float, default=None)
    parser.add_argument("--overlap-height-ratio", type=float, default=None)
    parser.add_argument("--no-slice", action="store_true", help="Run single-shot prediction without tiling")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional fixed inference size for the backend")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args(argv)

    cfg = _load_defaults(args.config if args.config.is_file() else None)
    sahi_cfg = cfg.get("sahi") or {}

    conf = args.confidence if args.confidence is not None else float(sahi_cfg.get("confidence_threshold", 0.25))
    sw = args.slice_width if args.slice_width is not None else int(sahi_cfg.get("slice_width", 640))
    sh = args.slice_height if args.slice_height is not None else int(sahi_cfg.get("slice_height", 640))
    owr = (
        args.overlap_width_ratio
        if args.overlap_width_ratio is not None
        else float(sahi_cfg.get("overlap_width_ratio", 0.2))
    )
    ohr = (
        args.overlap_height_ratio
        if args.overlap_height_ratio is not None
        else float(sahi_cfg.get("overlap_height_ratio", 0.2))
    )

    try:
        out = run_sahi_inference(
            args.weights,
            args.source,
            args.output,
            device=args.device,
            confidence_threshold=conf,
            slice_width=sw,
            slice_height=sh,
            overlap_width_ratio=owr,
            overlap_height_ratio=ohr,
            use_slices=not args.no_slice,
            image_size=args.imgsz,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Inference failed: {exc}", flush=True)
        return 1

    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
