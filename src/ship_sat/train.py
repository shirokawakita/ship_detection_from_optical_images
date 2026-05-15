"""Train Ultralytics YOLO-OBB on prepared data.yaml."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from ultralytics import YOLO


def _load_defaults(config_path: Path | None) -> dict:
    if not config_path or not config_path.is_file():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train YOLO-OBB (Ultralytics)")
    parser.add_argument("--data", type=Path, default=Path("data/yolo_ship_obb/data.yaml"))
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("YOLO_OBB_MODEL", "yolo11n-obb.pt"),
        help="Checkpoint or yaml, e.g. yolo11n-obb.pt or yolo26n-obb.pt",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--project",
        type=str,
        default="",
        help="Subfolder under runs/<task>/ (empty = directly under runs/obb/)",
    )
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from weights/last.pt under --project/--name (or --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit last.pt path for resume (implies resume)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Optional YAML with model/epochs/imgsz/batch/device keys",
    )
    args = parser.parse_args(argv)

    cfg = _load_defaults(args.config if args.config.is_file() else None)
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 50))
    imgsz = args.imgsz if args.imgsz is not None else int(cfg.get("imgsz", 768))
    batch = args.batch if args.batch is not None else int(cfg.get("batch", 8))
    device = args.device if args.device is not None else str(cfg.get("device", "0"))
    model_name = args.model or str(cfg.get("model", "yolo11n-obb.pt"))
    do_resume = args.resume or args.checkpoint is not None

    data_path = args.data
    if not data_path.is_file():
        print(f"data.yaml not found: {data_path}", flush=True)
        return 1

    ckpt: Path | None = args.checkpoint
    if args.checkpoint is not None and not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", flush=True)
        return 1
    if do_resume and ckpt is None:
        task_sub = Path("runs") / "obb"
        if args.project:
            task_sub = task_sub / args.project.strip("/\\")
        ckpt = task_sub / args.name / "weights" / "last.pt"
    if do_resume and ckpt is not None and not ckpt.is_file():
        # 他ランの last.pt を探索（experiment 名違い・旧 project 指定の二重パス対策）
        search_base = Path("runs") / "obb"
        if args.project:
            search_base = search_base / args.project.strip("/\\")
        weights_glob = sorted(
            search_base.glob("*/weights/last.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not weights_glob and Path("runs").is_dir():
            weights_glob = sorted(
                (p for p in Path("runs").rglob("last.pt") if p.parent.name == "weights"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if weights_glob:
            ckpt = weights_glob[0]
            print(f"Using latest checkpoint: {ckpt}", flush=True)
        else:
            print(
                "No last.pt found for resume. Starting a new training run from pretrained weights.",
                flush=True,
            )
            do_resume = False
            ckpt = None

    load_path = str(ckpt.resolve()) if ckpt is not None and ckpt.is_file() else model_name

    try:
        model = YOLO(load_path)
    except Exception as exc:  # noqa: BLE001 - surface ultralytics load errors clearly
        print(
            f"Failed to load model {load_path!r}: {exc}\n"
            "Try upgrading ultralytics (`uv sync -U ultralytics`) or use yolo11n-obb.pt.",
            flush=True,
        )
        return 1

    if do_resume and ckpt is not None and ckpt.is_file():
        # チェックポイント内の args が復元される。device / batch などは上書き可能。
        model.train(resume=str(ckpt.resolve()), device=device, batch=batch, imgsz=imgsz)
    else:
        model.train(
            data=str(data_path.resolve()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=args.project,
            name=args.name,
        )
    print("Training finished. Best weights under the run directory (weights/best.pt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
