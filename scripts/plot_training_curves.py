#!/usr/bin/env python3
"""Plot Ultralytics OBB training curves from results.csv (loss + validation metrics)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results",
        type=Path,
        default=Path("runs/obb/airbus_ship_obb/results.csv"),
        help="Path to Ultralytics results.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/training_curves.png"),
        help="Output PNG path",
    )
    args = p.parse_args()

    df = pd.read_csv(args.results)
    if "epoch" not in df.columns:
        raise SystemExit(f"Missing 'epoch' column in {args.results}")

    epochs = df["epoch"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

    # Training losses
    ax = axes[0]
    for col, label, color in (
        ("train/box_loss", "box", "#1f77b4"),
        ("train/cls_loss", "cls", "#ff7f0e"),
        ("train/dfl_loss", "dfl", "#2ca02c"),
        ("train/angle_loss", "angle", "#d62728"),
    ):
        if col in df.columns:
            ax.plot(epochs, df[col], label=label, color=color, linewidth=1.2)
    ax.set_ylabel("train loss")
    ax.set_title("Training / validation loss and metrics vs epoch")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Validation losses
    ax = axes[1]
    for col, label, color in (
        ("val/box_loss", "box", "#1f77b4"),
        ("val/cls_loss", "cls", "#ff7f0e"),
        ("val/dfl_loss", "dfl", "#2ca02c"),
        ("val/angle_loss", "angle", "#d62728"),
    ):
        if col in df.columns:
            ax.plot(epochs, df[col], label=label, color=color, linewidth=1.2)
    ax.set_ylabel("val loss")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Detection metrics (Ultralytics names these with (B) for OBB)
    ax = axes[2]
    metric_specs = [
        ("metrics/precision(B)", "precision", "#9467bd"),
        ("metrics/recall(B)", "recall", "#8c564b"),
        ("metrics/mAP50(B)", "mAP50", "#e377c2"),
        ("metrics/mAP50-95(B)", "mAP50-95", "#17becf"),
    ]
    for col, label, color in metric_specs:
        if col in df.columns:
            ax.plot(epochs, df[col], label=label, color=color, linewidth=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("metrics")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
