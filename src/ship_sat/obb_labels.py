"""Convert instance masks to YOLO OBB lines (normalized four corners)."""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_yolo_obb_lines(
    mask: np.ndarray,
    class_id: int = 0,
    min_contour_area: float = 2.0,
) -> list[str]:
    """Return YOLO-OBB label lines: ``class x1 y1 ... x4 y4`` (normalized)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 2:
            continue
        area = float(cv2.contourArea(cnt))
        if area < min_contour_area:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        box[:, 0] /= float(w)
        box[:, 1] /= float(h)
        box = np.clip(box, 0.0, 1.0)
        parts = [str(class_id)]
        for x, y in box:
            parts.append(f"{float(x):.6f}")
            parts.append(f"{float(y):.6f}")
        lines.append(" ".join(parts))
    return lines
