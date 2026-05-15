"""Tests for RLE decode and mask → YOLO OBB conversion."""

from __future__ import annotations

import cv2
import numpy as np

from ship_sat.obb_labels import mask_to_yolo_obb_lines
from ship_sat.rle import rle_decode


def test_rle_decode_empty():
    h, w = 10, 12
    m = rle_decode("", (h, w))
    assert m.shape == (h, w)
    assert m.sum() == 0


def test_rle_decode_single_run():
    h, w = 4, 3
    # Column-major flat indices 1-based: fill first 3 pixels (column 0 rows 0-2)
    m = rle_decode("1 3", (h, w))
    assert m[0, 0] == 1 and m[1, 0] == 1 and m[2, 0] == 1
    assert m[3, 0] == 0


def test_mask_to_yolo_obb_rotated_rectangle():
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    rect = ((100.0, 100.0), (80.0, 24.0), 35.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(mask, [box], 0, 1, thickness=-1)

    lines = mask_to_yolo_obb_lines(mask, class_id=0, min_contour_area=10.0)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    coords = [float(x) for x in parts[1:]]
    assert len(coords) == 8
    assert all(0.0 <= c <= 1.0 for c in coords)
    # Rough center near (0.5, 0.5) in normalized space
    xs = coords[0::2]
    ys = coords[1::2]
    assert abs(float(np.mean(xs)) - 0.5) < 0.05
    assert abs(float(np.mean(ys)) - 0.5) < 0.05


def test_rle_decode_nan_like():
    h, w = 5, 5
    m = rle_decode(float("nan"), (h, w))
    assert m.sum() == 0
