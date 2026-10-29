#!/usr/bin/env python3
"""Stage 2 - piece extraction.

Segments the 16 printed jigsaw pieces out of the rectified top-view capture
(``pieces_rectified.png``) by differencing it against the empty-table capture
(``empty_rectified.png``), fits a convex polygon with sub-pixel corners to every
piece, normalizes each piece into its own local canvas and writes the hand-off
manifest ``pieces.json`` consumed by Stage 3.

Pipeline (README section 4.2):

1. ``foreground_response`` - Lab colour difference plus gradient difference,
   lightly denoised.  The two captures share camera, lighting and table, so
   everything that changed belongs to the pieces (printing included).
2. ``foreground_mask``   - Otsu threshold (clamped), morphological open/close,
   connected components, area / border / convexity filtering.
3. ``refine_polygon``    - coarse ``approxPolyDP`` polygon, then per-edge
   sub-pixel refinement: sample the response profile across the edge normal,
   locate the half-contrast crossing, robustly fit a line through the samples
   and intersect neighbouring lines to obtain sub-pixel corners.
4. ``normalize_piece``   - uniform scale + canonical in-plane rotation (shape
   principal axis) + translation into a ``piece_output_size_px`` square canvas.
5. ``extract_pieces``    - sort by (center y, center x) into ``S01..S16``, write
   RGBA / mask / debug images and ``pieces.json``.

Conventions
-----------
* All coordinates are in the rectified canvas pixel frame, ``x`` right, ``y``
  down, origin top-left (README section 2.1).
* ``corners_px`` are ordered so that the shoelace area
  ``0.5 * sum(x_i*y_{i+1} - x_{i+1}*y_i)`` is positive (this is *clockwise* when
  the image is viewed with y pointing down) and the first corner is the
  top-most one, ties broken by the left-most one.
* ``S01..S16`` are assigned in scan order: rows top to bottom (centres whose
  ``y`` differ by less than ``SCAN_ROW_TOLERANCE_PX`` count as one row), then
  left to right inside a row.
* The JSON never contains timestamps, so two runs on the same inputs produce
  byte-identical output (``session_id`` is a deterministic hash of the inputs).

Usage
-----
    python stage2_piece_extraction/extract_pieces.py
    python stage2_piece_extraction/extract_pieces.py --config my_extraction.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(ROOT / "shared"))

from jigsaw_common.io import imread, imwrite, read_json, write_json  # noqa: E402

ALGORITHM_VERSION = "stage2-extract-1.3.0"

# Manual-review triggers that are independent of the quality score itself.
MIN_SAFE_SIDE_PX = 100.0
MAX_LOCALIZER_SHIFT_PX = 3.0
# Scan ids must not depend on sub-pixel jitter: two pieces whose centre y differ by
# less than this are treated as one row and ordered by x (README §2.1 scan order).
SCAN_ROW_TOLERANCE_PX = 2.0
DEFAULT_INPUT_PIECES = ROOT / "stage1_capture_calibration" / "output" / "rectified" / "pieces_rectified.png"
DEFAULT_INPUT_EMPTY = ROOT / "stage1_capture_calibration" / "output" / "rectified" / "empty_rectified.png"
DEFAULT_INPUT_RECTIFICATION = ROOT / "stage1_capture_calibration" / "output" / "rectified" / "rectification.json"
DEFAULT_INPUT_LAYOUT = ROOT / "stage1_capture_calibration" / "output" / "cut_pdf" / "jigsaw_layout.json"
DEFAULT_CONFIG = ROOT / "stage2_piece_extraction" / "config" / "extraction.example.json"
DEFAULT_OUTPUT_DIR = ROOT / "stage2_piece_extraction" / "output"

DEFAULT_CONFIG_VALUES: dict[str, Any] = {
    "schema_version": "1.0",
    "expected_piece_count": 16,
    "ignore_border_mm": 45.0,
    "min_piece_area_px": 2000,
    "min_quality": 0.85,
    "piece_output_size_px": 512,
    "piece_margin_px": 20.0,
    "blur_ksize": 5,
    "threshold_mode": "otsu",
    "threshold_min": 12.0,
    "threshold_max": 60.0,
    "open_ksize": 7,
    "close_ksize": 7,
    "mask_erode_px": 0.0,
    "shadow_rejection": False,
    "shadow_blur_px": 5.0,
    "shadow_window_px": 21,
    "shadow_correlation": 0.55,
    "shadow_min_texture": 1.5,
    "approx_epsilon_ratio": 0.02,
    "min_vertices": 3,
    "max_vertices": 6,
    "edge_sample_count": 25,
    "edge_profile_px": 6.0,
    "edge_profile_step_px": 0.5,
    "edge_fit_residual_px": 2.0,
    "min_edge_contrast": 20.0,
    "edge_localizer": "gradient_max",
    "half_level_response": "edge",
    "min_edge_gradient": 2.0,
    "max_side_shift_px": 3.0,
    "max_corner_shift_px": 8.0,
    "fail_on_manual_review": False,
}


# --------------------------------------------------------------------------- #
# Small geometry helpers (pure numpy, unit-tested)
# --------------------------------------------------------------------------- #
def polygon_signed_area(polygon: np.ndarray) -> float:
    """Shoelace area; positive for the ordering used by ``ordered_polygon``."""

    points = np.asarray(polygon, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_area(polygon: np.ndarray) -> float:
    return abs(polygon_signed_area(polygon))


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed_twice = float(cross.sum())
    if abs(signed_twice) < 1e-12:
        return points.mean(axis=0)
    return np.array(
        [
            float(((x + np.roll(x, -1)) * cross).sum() / (3.0 * signed_twice)),
            float(((y + np.roll(y, -1)) * cross).sum() / (3.0 * signed_twice)),
        ]
    )


def dedupe_polygon(polygon: np.ndarray, min_distance_px: float = 1.0) -> np.ndarray:
    """Drop consecutive duplicate / near-duplicate vertices."""

    points = [np.asarray(polygon, dtype=np.float64)[0]]
    for point in np.asarray(polygon, dtype=np.float64)[1:]:
        if np.linalg.norm(point - points[-1]) >= min_distance_px:
            points.append(point)
    while len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < min_distance_px:
        points.pop()
    return np.asarray(points, dtype=np.float64)


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    """2-D cross product (z component); avoids the deprecated ``np.cross`` path."""

    return float(first[0]) * float(second[1]) - float(first[1]) * float(second[0])


def is_convex_polygon(polygon: np.ndarray, tolerance: float = 1e-6) -> bool:
    points = np.asarray(polygon, dtype=np.float64)
    if len(points) < 3:
        return False
    direction = 0
    count = len(points)
    for index in range(count):
        prev = points[index]
        cur = points[(index + 1) % count]
        nxt = points[(index + 2) % count]
        cross = _cross2(cur - prev, nxt - cur)
        if abs(cross) <= tolerance:
            continue
        sign = 1 if cross > 0 else -1
        if direction == 0:
            direction = sign
        elif sign != direction:
            return False
    return direction != 0


def ordered_polygon(polygon: np.ndarray) -> np.ndarray:
    """Return vertices ordered top-most/left-most first, positive shoelace area.

    Positive shoelace in the y-down pixel frame corresponds to clockwise motion
    on screen; this is the documented Stage 2 ordering (README section 4.2 asks
    for one fixed order, the exact direction is ours to pin down).
    """

    points = dedupe_polygon(polygon, min_distance_px=1.0)
    if len(points) < 3:
        return points
    if polygon_signed_area(points) < 0:
        points = points[::-1]
    # rotate so the first vertex is top-most, then left-most.
    start = int(np.lexsort((points[:, 0], points[:, 1]))[0])
    return np.roll(points, -start, axis=0)


def scan_order_indices(
    centres: Sequence[np.ndarray], tolerance_px: float = SCAN_ROW_TOLERANCE_PX
) -> list[int]:
    """Order pieces the way a human scans them: rows top to bottom, then x.

    Rows are formed greedily from the y-sorted centres: a piece joins the current
    row while its centre y is within ``tolerance_px`` of the row's first member,
    which keeps the ids stable when sub-pixel jitter moves a piece by ~1 px but two
    pieces at the same height stay ordered by x.  Returns indices, not coordinates.
    """

    points = [np.asarray(centre, dtype=np.float64)[:2] for centre in centres]
    order = sorted(range(len(points)), key=lambda index: (float(points[index][1]), float(points[index][0])))
    rows: list[list[int]] = []
    anchor = float("nan")
    for index in order:
        y = float(points[index][1])
        if not rows or abs(y - anchor) > tolerance_px:
            rows.append([index])
            anchor = y
        else:
            rows[-1].append(index)
    return [index for row in rows for index in sorted(row, key=lambda member: float(points[member][0]))]


def prune_near_collinear(
    polygon: np.ndarray,
    sin_tolerance: float = 0.09,
    min_edge_px: float = 12.0,
) -> np.ndarray:
    """Drop spurious vertices that lie on a straight edge.

    Sub-pixel corner fitting can introduce an extra vertex when a piece edge is
    slightly bowed (soft paper edge, shadow).  Such a vertex is either nearly
    collinear with its neighbours (``|sin(turn)|`` below the tolerance) or forms
    a very short edge; both are removed until the polygon is minimal again.
    """

    points = np.asarray(polygon, dtype=np.float64)
    while len(points) > 3:
        count = len(points)
        remove_index = -1
        best_bend = None
        for index in range(count):
            prev = points[(index - 1) % count]
            cur = points[index]
            nxt = points[(index + 1) % count]
            incoming = cur - prev
            outgoing = nxt - cur
            incoming_len = float(np.linalg.norm(incoming))
            outgoing_len = float(np.linalg.norm(outgoing))
            if incoming_len < 1e-9 or outgoing_len < 1e-9:
                remove_index = index
                break
            sin_turn = abs(_cross2(incoming, outgoing)) / (incoming_len * outgoing_len)
            bend = -1.0 if min(incoming_len, outgoing_len) <= min_edge_px else sin_turn
            if best_bend is None or bend < best_bend:
                best_bend = bend
                remove_index = index
        if remove_index < 0 or best_bend is None or best_bend > sin_tolerance:
            break
        points = np.delete(points, remove_index, axis=0)
    if len(points) == 0:
        return points
    return ordered_polygon(points)


def rasterize_polygon(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points], 255)
    return mask


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Raster IoU of two polygons inside their common bounding box."""

    points = np.vstack([np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)])
    origin = np.floor(points.min(axis=0) - 2).astype(int)
    upper = np.ceil(points.max(axis=0) + 2).astype(int)
    shape = (max(2, upper[1] - origin[1]), max(2, upper[0] - origin[0]))
    a = rasterize_polygon(np.asarray(first) - origin, shape) > 0
    b = rasterize_polygon(np.asarray(second) - origin, shape) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return projected[:, :2] / projected[:, 2:3]


# --------------------------------------------------------------------------- #
# Step 1/2 - background difference and foreground mask
# --------------------------------------------------------------------------- #
def colour_response(pieces: np.ndarray, empty: np.ndarray, config: dict) -> np.ndarray:
    """Lab colour difference between the piece capture and the empty table.

    The two captures share camera, lighting and table, so everything that differs
    belongs to a piece; a plain Otsu threshold on this response isolates the 16
    bright paper pieces from the near-black bench.
    """

    ksize = int(config["blur_ksize"])
    if ksize % 2 == 0:
        ksize += 1
    blurred_pieces = cv2.GaussianBlur(pieces, (ksize, ksize), 0)
    blurred_empty = cv2.GaussianBlur(empty, (ksize, ksize), 0)
    lab_pieces = cv2.cvtColor(blurred_pieces, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_empty = cv2.cvtColor(blurred_empty, cv2.COLOR_BGR2Lab).astype(np.float32)
    return np.linalg.norm(lab_pieces - lab_empty, axis=2)


def edge_response(pieces: np.ndarray, empty: np.ndarray, colour: np.ndarray, config: dict) -> np.ndarray:
    """Colour difference plus gradient difference, used to locate piece edges.

    The gradient term adds contrast on the printed content and on the paper edge,
    which makes the half-contrast crossing sharper than the colour term alone.
    """

    ksize = int(config["blur_ksize"])
    if ksize % 2 == 0:
        ksize += 1
    blurred_pieces = cv2.GaussianBlur(pieces, (ksize, ksize), 0)
    blurred_empty = cv2.GaussianBlur(empty, (ksize, ksize), 0)
    gray_pieces = cv2.cvtColor(blurred_pieces, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_empty = cv2.cvtColor(blurred_empty, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_pieces = cv2.Laplacian(gray_pieces, cv2.CV_32F, ksize=3)
    grad_empty = cv2.Laplacian(gray_empty, cv2.CV_32F, ksize=3)
    gradient = np.abs(grad_pieces - grad_empty)
    return colour + 0.5 * np.minimum(gradient, 200.0)


def _local_correlation(first: np.ndarray, second: np.ndarray, window: int) -> np.ndarray:
    """Normalised local cross-correlation of two single-channel images."""

    size = (window, window)
    mean_first = cv2.boxFilter(first, -1, size, normalize=True)
    mean_second = cv2.boxFilter(second, -1, size, normalize=True)
    var_first = cv2.boxFilter(first * first, -1, size, normalize=True) - mean_first * mean_first
    var_second = cv2.boxFilter(second * second, -1, size, normalize=True) - mean_second * mean_second
    covariance = cv2.boxFilter(first * second, -1, size, normalize=True) - mean_first * mean_second
    scale = np.sqrt(np.maximum(var_first, 1e-6) * np.maximum(var_second, 1e-6))
    return covariance / scale


def shadow_suspicion(pieces: np.ndarray, empty: np.ndarray, config: dict) -> np.ndarray:
    """How much each pixel looks like an illumination change of the empty frame.

    A cast shadow multiplies the floor colour, so the high-pass texture of the piece
    frame stays locally correlated with the texture of the empty frame; a paper piece
    replaces the floor, so its texture is uncorrelated with it.  Pixels where the empty
    frame carries no texture (e.g. the black bench) are never judged suspicious, which
    keeps the original bench behaviour intact.
    """

    radius = int(round(float(config["shadow_blur_px"])))
    ksize = 2 * radius + 1
    window = int(config["shadow_window_px"])
    if window % 2 == 0:
        window += 1

    gray_pieces = cv2.cvtColor(pieces, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_empty = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high_pieces = gray_pieces - cv2.GaussianBlur(gray_pieces, (ksize, ksize), 0)
    high_empty = gray_empty - cv2.GaussianBlur(gray_empty, (ksize, ksize), 0)

    correlation = _local_correlation(high_pieces, high_empty, window)

    texture = cv2.boxFilter(high_empty * high_empty, -1, (window, window), normalize=True) ** 0.5
    correlation[texture < float(config["shadow_min_texture"])] = 0.0
    return np.clip(correlation, 0.0, 1.0)


def foreground_mask(
    response: np.ndarray,
    config: dict,
    ignore_border_px: int,
    shadow_suspicion: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Binary piece mask plus the threshold that produced it.

    ``shadow_suspicion`` (0..1, higher = more likely an illumination-only change) is
    only used when ``config["shadow_rejection"]`` is on; it exists so that dark and
    textured floors, where a cast shadow also clears the colour threshold, do not
    merge into the piece mask.
    """

    clipped = np.clip(response, 0.0, 255.0).astype(np.uint8)
    if config["threshold_mode"] == "otsu":
        threshold, _ = cv2.threshold(clipped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(threshold)
    else:
        threshold = float(config["threshold_min"])
    threshold = float(np.clip(threshold, config["threshold_min"], config["threshold_max"]))

    mask = (clipped >= threshold).astype(np.uint8) * 255

    if shadow_suspicion is not None and bool(config.get("shadow_rejection", False)):
        rejection = shadow_suspicion >= float(config["shadow_correlation"])
        mask[rejection] = 0

    open_ksize = int(config["open_ksize"])
    close_ksize = int(config["close_ksize"])
    if open_ksize > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    if close_ksize > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))

    # 浅色底面会带出碎片投影：阴影与纸片一起超过阈值时掩膜会外扩。把掩膜整体收缩若干像素，
    # 只用来给“粗多边形 / 逐边亚像素定位”提供初值，最终多边形仍由边缘拟合决定。
    mask_erode_px = float(config.get("mask_erode_px", 0.0))
    if mask_erode_px > 0.0:
        ksize = int(round(mask_erode_px)) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.erode(mask, kernel)

    if ignore_border_px > 0:
        mask[:ignore_border_px, :] = 0
        mask[-ignore_border_px:, :] = 0
        mask[:, :ignore_border_px] = 0
        mask[:, -ignore_border_px:] = 0
    return mask, threshold


# --------------------------------------------------------------------------- #
# Step 3 - polygon fit with sub-pixel corners
# --------------------------------------------------------------------------- #
def _bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    wx = np.clip(xs - np.floor(xs), 0.0, 1.0)
    wy = np.clip(ys - np.floor(ys), 0.0, 1.0)
    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def _coarse_polygon(contour: np.ndarray, config: dict) -> np.ndarray:
    """Convex, simplified polygon with a vertex count inside the allowed range."""

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    best = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    ratio = float(config["approx_epsilon_ratio"])
    for epsilon_ratio in [ratio, ratio * 0.75, ratio * 0.5, ratio * 0.35, ratio * 1.5, ratio * 2.0]:
        approximated = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        candidate = np.asarray(approximated, dtype=np.float64).reshape(-1, 2)
        if config["min_vertices"] <= len(candidate) <= config["max_vertices"]:
            return ordered_polygon(candidate)
        if len(candidate) < len(best):
            best = candidate
    return ordered_polygon(best)


def _fit_edge_line(
    response: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    config: dict,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """Sub-pixel line through one polygon edge.

    Returns ``(line, points, sample_count)`` where ``line`` is ``(a, b, c)`` with
    ``a*x + b*y + c = 0`` normalised to unit normal; ``None`` when the edge is not
    supported well enough.
    """

    edge = end - start
    length = float(np.linalg.norm(edge))
    if length < 8.0:
        return None, np.empty((0, 2)), 0
    direction = edge / length
    normal = np.array([direction[1], -direction[0]], dtype=np.float64)

    fractions = np.linspace(0.12, 0.88, int(config["edge_sample_count"]))
    bases = start[None, :] + fractions[:, None] * edge[None, :]

    half = float(config["edge_profile_px"])
    step = float(config["edge_profile_step_px"])
    offsets = np.arange(-half, half + 1e-9, step)
    inner_zone = offsets <= -0.4 * half
    outer_zone = offsets >= 0.4 * half
    min_contrast = float(config["min_edge_contrast"])

    points: list[np.ndarray] = []
    contrasts: list[float] = []
    for base in bases:
        profile = _bilinear(
            response,
            base[0] + normal[0] * offsets,
            base[1] + normal[1] * offsets,
        )
        inner = float(np.median(profile[inner_zone])) if np.any(inner_zone) else float(profile[0])
        outer = float(np.median(profile[outer_zone])) if np.any(outer_zone) else float(profile[-1])
        contrast = inner - outer
        if abs(contrast) < min_contrast:
            continue
        level = 0.5 * (inner + outer)
        # Printed content can make the profile non-monotonic, so take every
        # level crossing and keep the one closest to the coarse edge (offset 0).
        below = profile < level
        crossings: list[float] = []
        for index in range(len(offsets) - 1):
            if below[index] == below[index + 1]:
                continue
            first, second = float(profile[index]), float(profile[index + 1])
            span = second - first
            if abs(span) <= 1e-6:
                continue
            fraction = (level - first) / span
            if not -0.5 <= fraction <= 1.5:
                continue
            crossing = float(offsets[index] + step * fraction)
            if abs(crossing) <= half - step:
                crossings.append(crossing)
        if not crossings:
            continue
        crossing = min(crossings, key=abs)
        points.append(base + normal * crossing)
        contrasts.append(abs(contrast))

    if len(points) < max(5, int(0.3 * len(bases))):
        return None, np.asarray(points, dtype=np.float64), len(bases)

    samples = np.asarray(points, dtype=np.float64)
    line = cv2.fitLine(samples.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    norm = math.hypot(vx, vy)
    if norm < 1e-9:
        return None, samples, len(bases)
    vx, vy = vx / norm, vy / norm
    # Refit after dropping points farther than the residual budget.
    normal_vec = np.array([-vy, vx], dtype=np.float64)
    residual = np.abs((samples - np.array([x0, y0])) @ normal_vec)
    keep = residual <= float(config["edge_fit_residual_px"])
    if keep.sum() >= 6:
        samples = samples[keep]
        refined = cv2.fitLine(samples.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        vx, vy, x0, y0 = float(refined[0]), float(refined[1]), float(refined[2]), float(refined[3])
        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            return None, samples, len(bases)
        vx, vy = vx / norm, vy / norm
    # a*x + b*y + c = 0 with (a, b) = unit normal.
    a = -vy
    b = vx
    c = -(a * x0 + b * y0)
    return np.array([a, b, c], dtype=np.float64), samples, len(bases)


def _tls_line(samples: np.ndarray, residual_px: float) -> np.ndarray | None:
    """Total-least-squares line through ``samples`` with one robust refit."""

    if len(samples) < 2:
        return None
    line = cv2.fitLine(samples.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    norm = math.hypot(vx, vy)
    if norm < 1e-9:
        return None
    vx, vy = vx / norm, vy / norm
    normal_vec = np.array([-vy, vx], dtype=np.float64)
    residual = np.abs((samples - np.array([x0, y0])) @ normal_vec)
    keep = residual <= residual_px
    if keep.sum() >= 2 and keep.sum() < len(samples):
        samples = samples[keep]
        refined = cv2.fitLine(samples.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        vx, vy, x0, y0 = float(refined[0]), float(refined[1]), float(refined[2]), float(refined[3])
        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            return None
        vx, vy = vx / norm, vy / norm
    a, b = -vy, vx
    return np.array([a, b, -(a * x0 + b * y0)], dtype=np.float64)


def _fit_edge_line_gradient(
    response: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    config: dict,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """Sub-pixel line localised at the steepest intensity transition.

    ``_fit_edge_line`` uses the half-contrast level, which drifts towards the
    inside of a blurred, shadowed paper edge.  The gradient maximum sits on the
    physical paper boundary, so it is less biased and on low-contrast or short
    sides it is markedly more stable.  Same ``(line, points, sample_count)``
    contract as :func:`_fit_edge_line`.
    """

    edge = end - start
    length = float(np.linalg.norm(edge))
    if length < 8.0:
        return None, np.empty((0, 2)), 0
    direction = edge / length
    normal = np.array([direction[1], -direction[0]], dtype=np.float64)

    fractions = np.linspace(0.12, 0.88, int(config["edge_sample_count"]))
    bases = start[None, :] + fractions[:, None] * edge[None, :]

    half = float(config["edge_profile_px"])
    step = float(config["edge_profile_step_px"])
    offsets = np.arange(-half, half + 1e-9, step)
    min_gradient = float(config["min_edge_gradient"])

    points: list[np.ndarray] = []
    for base in bases:
        profile = _bilinear(
            response,
            base[0] + normal[0] * offsets,
            base[1] + normal[1] * offsets,
        )
        magnitude = np.abs(np.gradient(profile, step))
        index = int(np.argmax(magnitude))
        if index == 0 or index >= len(magnitude) - 1:
            continue
        peak = float(magnitude[index])
        if peak < min_gradient:
            continue
        # Parabolic refinement around the peak sample.
        left = float(magnitude[index - 1])
        right = float(magnitude[index + 1])
        denominator = left - 2.0 * peak + right
        delta = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        crossing = float(offsets[index] + delta * step)
        if abs(crossing) > half - step:
            continue
        points.append(base + normal * crossing)

    if len(points) < max(5, int(0.3 * len(bases))):
        return None, np.asarray(points, dtype=np.float64), len(bases)
    samples = np.asarray(points, dtype=np.float64)
    line = _tls_line(samples, float(config["edge_fit_residual_px"]))
    if line is None:
        return None, samples, len(bases)
    return line, samples, len(bases)


def _intersect_lines(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    lhs = np.array([[first[0], first[1]], [second[0], second[1]]], dtype=np.float64)
    rhs = -np.array([first[2], second[2]], dtype=np.float64)
    determinant = float(np.linalg.det(lhs))
    if abs(determinant) < 1e-9:
        return None
    return np.linalg.solve(lhs, rhs)


@dataclass
class PolygonFit:
    corners: np.ndarray
    quality: dict[str, float]
    edge_points: np.ndarray
    coarse: np.ndarray


def _fit_side_lines(
    response: np.ndarray,
    coarse: np.ndarray,
    centroid: np.ndarray,
    config: dict,
    fit_edge: Callable[..., tuple[np.ndarray | None, np.ndarray, int]],
) -> tuple[list[np.ndarray | None], list[np.ndarray], list[float]]:
    """One robust line per edge of ``coarse`` -> (lines, points, support ratios)."""

    count = len(coarse)
    lines: list[np.ndarray | None] = []
    edge_points: list[np.ndarray] = []
    support_ratio: list[float] = []
    for index in range(count):
        start = coarse[index]
        end = coarse[(index + 1) % count]
        edge = end - start
        # make sure the edge runs inside->outside with the sampled normal
        normal = np.array([edge[1], -edge[0]], dtype=np.float64)
        if np.dot(normal, 0.5 * (start + end) - centroid) < 0:
            normal = -normal
            start, end = end, start
        line, points, total = fit_edge(response, start, end, config)
        lines.append(line)
        edge_points.append(points)
        support_ratio.append(0.0 if total == 0 else len(points) / total)
    return lines, edge_points, support_ratio


def _intersect_corners(
    coarse: np.ndarray, lines: list[np.ndarray | None], config: dict
) -> np.ndarray:
    """Intersect neighbouring side lines; fall back to the coarse corner."""

    count = len(coarse)
    corners: list[np.ndarray] = []
    for index in range(count):
        previous_line = lines[(index - 1) % count]
        current_line = lines[index]
        coarse_corner = coarse[index]
        if previous_line is None or current_line is None:
            corners.append(coarse_corner)
            continue
        intersection = _intersect_lines(previous_line, current_line)
        if intersection is None or np.linalg.norm(intersection - coarse_corner) > float(config["max_corner_shift_px"]):
            corners.append(coarse_corner)
        else:
            corners.append(intersection)
    return np.asarray(corners, dtype=np.float64)


def _line_offset(first: np.ndarray, second: np.ndarray) -> float:
    """Signed distance between two unit-normal lines (positive bias direction)."""

    (a1, b1, c1), (a2, b2, c2) = first, second
    if a1 * a2 + b1 * b2 < 0.0:
        c2 = -c2
    return c1 - c2


def _merge_side_lines(
    primary_lines: list[np.ndarray | None],
    primary_points: list[np.ndarray],
    primary_support: list[float],
    fallback_lines: list[np.ndarray | None],
    fallback_points: list[np.ndarray],
    fallback_support: list[float],
    max_shift_px: float,
) -> tuple[list[np.ndarray | None], list[np.ndarray], list[float]]:
    """Keep the primary (gradient) side unless it strays too far from the fallback.

    The gradient localizer is unbiased but on a few low-contrast or very short
    sides it can jump to the wrong feature.  The half-contrast line is biased by
    a fraction of a pixel but never jumps, so it acts as a bound: any side whose
    gradient line moves further than ``max_shift_px`` from it keeps the fallback.
    """

    lines: list[np.ndarray | None] = []
    points: list[np.ndarray] = []
    support: list[float] = []
    for index, primary in enumerate(primary_lines):
        fallback = fallback_lines[index] if index < len(fallback_lines) else None
        use_primary = primary is not None
        if use_primary and fallback is not None:
            use_primary = abs(_line_offset(primary, fallback)) <= max_shift_px
        if use_primary:
            lines.append(primary)
            points.append(primary_points[index])
            support.append(primary_support[index])
        else:
            lines.append(fallback)
            points.append(fallback_points[index] if index < len(fallback_points) else np.empty((0, 2)))
            support.append(fallback_support[index] if index < len(fallback_support) else 0.0)
    return lines, points, support


def _fit_corners_around(
    response: np.ndarray,
    coarse: np.ndarray,
    centroid: np.ndarray,
    config: dict,
    fit_edge: Callable[..., tuple[np.ndarray | None, np.ndarray, int]],
) -> tuple[list[np.ndarray | None], list[np.ndarray], list[float], np.ndarray]:
    """Compatibility helper: side lines plus the corners they intersect at."""

    lines, points, support = _fit_side_lines(response, coarse, centroid, config, fit_edge)
    return lines, points, support, _intersect_corners(coarse, lines, config)


def refine_polygon(
    response: np.ndarray,
    contour: np.ndarray,
    config: dict,
    gradient_response: np.ndarray | None = None,
) -> PolygonFit:
    """Coarse polygon -> sub-pixel corners plus a quality breakdown.

    ``response`` feeds the half-contrast localizer; ``gradient_response`` feeds the
    gradient localizer (falling back to ``response``).  The Lab colour response is
    the better input for peak localisation: the Laplacian term inside
    :func:`edge_response` biases the gradient maximum ~0.7 px toward the interior.
    """

    coarse = _coarse_polygon(contour, config)
    count = len(coarse)
    if count < 3:
        corners = coarse
        return PolygonFit(corners, {"fit": 0.0, "support": 0.0, "shape": 0.0, "vertices": 0.0, "convex": 0.0}, np.empty((0, 2)), coarse)

    centroid = polygon_centroid(coarse)
    localizer = str(config.get("edge_localizer", "half_level"))
    if localizer not in ("half_level", "gradient_max"):
        raise ValueError(f"unknown edge_localizer: {localizer!r}")

    # The half-contrast fit always runs: it is the fallback the merge step bounds
    # the gradient localizer against, and the reference for the agreement metric.
    half_lines, half_points, half_support = _fit_side_lines(
        response, coarse, centroid, config, _fit_edge_line
    )
    lines, edge_points, support_ratio = half_lines, half_points, half_support
    agreement_shift = float("nan")
    if localizer == "gradient_max":
        peak_response = response if gradient_response is None else gradient_response
        peak_lines, peak_points, peak_support = _fit_side_lines(
            peak_response, coarse, centroid, config, _fit_edge_line_gradient
        )
        lines, edge_points, support_ratio = _merge_side_lines(
            peak_lines,
            peak_points,
            peak_support,
            half_lines,
            half_points,
            half_support,
            float(config["max_side_shift_px"]),
        )
        corners_array = _intersect_corners(coarse, lines, config)
        reference_corners = _intersect_corners(coarse, half_lines, config)
        if len(reference_corners) == len(corners_array):
            agreement_shift = float(np.mean(np.linalg.norm(corners_array - reference_corners, axis=1)))
    else:
        corners_array = _intersect_corners(coarse, lines, config)
    all_points = [points for points in edge_points if len(points)]

    # Residual of the fitted lines against their inlier sample points.
    residuals: list[float] = []
    for index, line in enumerate(lines):
        points = edge_points[index] if index < len(edge_points) else np.empty((0, 2))
        if line is None or not len(points):
            continue
        residual = np.abs(points @ line[:2] + line[2])
        residuals.extend(residual.tolist())

    fit_rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("nan")
    support = float(np.mean(support_ratio)) if support_ratio else 0.0

    ordered = ordered_polygon(corners_array)

    # Reject non-convex results: fall back to the convex hull of the corners.
    convex = is_convex_polygon(ordered)
    if not convex or len(ordered) < 3:
        ordered = ordered_polygon(np.asarray(cv2.convexHull(ordered.astype(np.float32))).reshape(-1, 2))
        convex = is_convex_polygon(ordered)
    if len(ordered) > config["min_vertices"]:
        ordered = prune_near_collinear(ordered)
        convex = is_convex_polygon(ordered)
    if not config["min_vertices"] <= len(ordered) <= config["max_vertices"]:
        ordered = _coarse_polygon(ordered.astype(np.float32), config)

    component_mask = rasterize_polygon(coarse, response.shape[:2])
    polygon_mask = rasterize_polygon(ordered, response.shape[:2])
    union = np.logical_or(component_mask > 0, polygon_mask > 0).sum()
    shape_iou = float(np.logical_and(component_mask > 0, polygon_mask > 0).sum() / union) if union else 0.0

    vertex_score = 1.0 if config["min_vertices"] <= len(ordered) <= config["max_vertices"] else 0.0
    side_lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    min_side_px = float(side_lengths.min()) if len(side_lengths) else 0.0
    # Peak localisation is noisier than a half-contrast crossing, so the rms that
    # still counts as "clean" is larger for the gradient localizer (3 px vs 2 px).
    fit_scale_px = 3.0 if localizer == "gradient_max" else 2.0
    quality = {
        # exp(-rms / fit_scale_px): rms 0.5 px -> 0.85, rms 2 px -> 0.51.  The
        # gradient peak jitters more than a half-contrast crossing, hence 3 px.
        "fit": float(math.exp(-max(fit_rms, 0.0) / fit_scale_px)) if math.isfinite(fit_rms) else 0.0,
        # 60% supported samples is enough for a reliable line on soft paper edges.
        "support": float(np.clip(support / 0.6, 0.0, 1.0)),
        "shape": float(np.clip(shape_iou, 0.0, 1.0)),
        "vertices": vertex_score,
        "convex": 1.0 if convex else 0.0,
        "fit_rms_px": fit_rms,
        # Diagnostic only: how far the two localizers disagree at the corners.
        "agreement_shift_px": agreement_shift,
        "edge_support": support,
        "mask_iou": shape_iou,
        "min_side_px": min_side_px,
    }
    return PolygonFit(ordered, quality, np.vstack(all_points) if all_points else np.empty((0, 2)), coarse)


def combine_quality(quality: dict[str, float]) -> float:
    """Single [0, 1] score; the weakest geometric evidence dominates."""

    weighted = (
        0.35 * quality.get("fit", 0.0)
        + 0.25 * quality.get("shape", 0.0)
        + 0.20 * quality.get("support", 0.0)
        + 0.10 * quality.get("vertices", 0.0)
        + 0.10 * quality.get("convex", 0.0)
    )
    return float(np.clip(weighted, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Step 4 - local normalization
# --------------------------------------------------------------------------- #
def normalization_matrix(corners: np.ndarray, size_px: int, margin_px: float) -> tuple[np.ndarray, float]:
    """Similarity transform mapping canvas corners into the piece canvas.

    The rotation is the shape principal axis (second-moment eigenvector with the
    non-negative x component), which makes the normalized orientation a
    deterministic function of the piece shape only.  Returns ``(M, rotation_deg)``.
    """

    corners = np.asarray(corners, dtype=np.float64)
    centroid = corners.mean(axis=0)
    centred = corners - centroid
    covariance = centred.T @ centred / max(len(corners), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    main_axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float64)
    if main_axis[0] < 0:
        main_axis = -main_axis
    phi = math.atan2(float(main_axis[1]), float(main_axis[0]))
    cos_phi, sin_phi = math.cos(-phi), math.sin(-phi)
    rotation = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]], dtype=np.float64)

    rotated = centred @ rotation.T
    lower = rotated.min(axis=0)
    upper = rotated.max(axis=0)
    extent = float(np.max(upper - lower))
    usable = float(size_px) - 2.0 * float(margin_px)
    if extent <= 1e-6 or usable <= 0:
        raise ValueError("degenerate piece polygon for normalization")
    scale = usable / extent
    rotated_centre = 0.5 * (lower + upper)
    target_centre = np.array([size_px / 2.0, size_px / 2.0], dtype=np.float64)
    # The mapping rotates about the centroid, so the linear part is scale*R and
    # the translation absorbs both the centroid offset and the rotated bbox centre.
    translation = target_centre - scale * (rotation @ centroid) - scale * rotated_centre
    matrix = np.array(
        [
            [scale * rotation[0, 0], scale * rotation[0, 1], translation[0]],
            [scale * rotation[1, 0], scale * rotation[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotation_deg = math.degrees(-phi)
    rotation_deg = float((rotation_deg + 180.0) % 360.0 - 180.0)
    return matrix, rotation_deg


def normalize_piece(
    image: np.ndarray,
    polygon: np.ndarray,
    size_px: int,
    margin_px: float,
) -> dict[str, Any]:
    """Warp one piece into its local canvas and build RGBA / mask / debug data."""

    matrix, rotation_deg = normalization_matrix(polygon, size_px, margin_px)
    size = (int(size_px), int(size_px))
    warped = cv2.warpPerspective(
        image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    polygon_mask = rasterize_polygon(_apply_transform(matrix, polygon), size)
    # Use the warped analytic polygon as alpha: straight edges, no bench pixels.
    alpha = polygon_mask
    rgba = np.dstack([warped, alpha])
    rgba[alpha == 0] = 0

    normalized_corners = _apply_transform(matrix, polygon)
    return {
        "matrix": matrix,
        "rgba": rgba,
        "mask": polygon_mask,
        "normalized_corners": normalized_corners,
        "rotation_deg": rotation_deg,
    }


def render_debug(
    rgba: np.ndarray,
    normalized_corners: np.ndarray,
    piece_id: str,
    quality: float,
) -> np.ndarray:
    """Per-piece debug image: normalized piece over a checkerboard + corner marks."""

    size = rgba.shape[0]
    checker = np.zeros((size, size, 3), dtype=np.uint8)
    cell = 32
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                checker[y : y + cell, x : x + cell] = 90
            else:
                checker[y : y + cell, x : x + cell] = 130
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    canvas = (rgba[:, :, :3].astype(np.float32) * alpha + checker.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    points = np.round(normalized_corners).astype(np.int32)
    cv2.polylines(canvas, [points], True, (0, 255, 0), 1, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(canvas, tuple(int(v) for v in point), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (int(point[0]) + 5, int(point[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, piece_id, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"q={quality:.3f} n={len(points)}",
        (10, size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


# --------------------------------------------------------------------------- #
# Step 5 - full extraction
# --------------------------------------------------------------------------- #
def _portable_path(path: str | Path | None) -> str | None:
    """Record a path so the hand-off JSON stays portable across machines.

    Paths inside the repository are written relative to the repo root with
    forward slashes; anything else falls back to an absolute POSIX-style path.
    """

    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _session_id(paths: Iterable[Path], config: dict) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).name.encode("utf-8"))
        digest.update(Path(path).read_bytes())
    digest.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    digest.update(ALGORITHM_VERSION.encode("utf-8"))
    return digest.hexdigest()[:16]


def _resolve_config(config_path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG_VALUES)
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if path and Path(path).exists():
        config.update(read_json(path))
    return config


def _ring_mm_to_px(ignore_border_mm: float, rectification: dict) -> int:
    px_per_mm = rectification.get("px_per_mm") or [0.0, 0.0]
    if not px_per_mm or not px_per_mm[0]:
        return 0
    return int(round(float(ignore_border_mm) * float(px_per_mm[0])))


def extract_pieces(
    pieces_path: str | Path,
    empty_path: str | Path,
    rectification_path: str | Path,
    layout_path: str | Path | None,
    config: dict,
    output_dir: str | Path,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the Stage 2 pipeline and return the ``pieces.json`` payload."""

    pieces_path = Path(pieces_path)
    empty_path = Path(empty_path)
    rectification_path = Path(rectification_path)

    pieces_image = imread(pieces_path)
    empty_image = imread(empty_path)
    if pieces_image.shape[:2] != empty_image.shape[:2]:
        raise ValueError(
            "pieces_rectified and empty_rectified must share the rectified canvas size, "
            f"got {pieces_image.shape[:2]} vs {empty_image.shape[:2]}"
        )
    rectification = read_json(rectification_path)
    canvas_px = rectification.get("canvas_px") or [pieces_image.shape[1], pieces_image.shape[0]]
    px_per_mm = rectification.get("px_per_mm") or [8.0, 8.0]

    layout = read_json(layout_path) if layout_path and Path(layout_path).exists() else {}
    expected_count = int(config["expected_piece_count"] or len(layout.get("pieces", [])) or 0)

    colour = colour_response(pieces_image, empty_image, config)
    response = edge_response(pieces_image, empty_image, colour, config)
    ignore_border_px = _ring_mm_to_px(float(config["ignore_border_mm"]), rectification)
    suspicion = shadow_suspicion(pieces_image, empty_image, config) if bool(config.get("shadow_rejection", False)) else None
    mask, threshold = foreground_mask(colour, config, ignore_border_px, shadow_suspicion=suspicion)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[dict[str, Any]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(config["min_piece_area_px"]):
            continue
        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        candidates.append(
            {
                "label": label,
                "area": area,
                "contour": contour,
                "centroid": np.asarray(centroids[label], dtype=np.float64),
                "component_mask": component,
            }
        )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    if expected_count and len(candidates) > expected_count:
        candidates = candidates[:expected_count]
    candidates = [candidates[index] for index in scan_order_indices([item["centroid"] for item in candidates])]

    fitted: list[dict[str, Any]] = []
    half_response = colour if str(config.get("half_level_response", "edge")) == "colour" else response
    for index, candidate in enumerate(candidates, start=1):
        fit = refine_polygon(half_response, candidate["contour"], config, gradient_response=colour)
        if len(fit.corners) < 3:
            continue
        quality_score = combine_quality(fit.quality)
        area_px = polygon_area(fit.corners)
        centre = polygon_centroid(fit.corners)
        bbox = (
            np.floor(fit.corners.min(axis=0)).astype(int).tolist(),
            np.ceil(fit.corners.max(axis=0)).astype(int).tolist(),
        )
        fitted.append(
            {
                "index": index,
                "corners": fit.corners,
                "coarse": fit.coarse,
                "quality": quality_score,
                "quality_breakdown": {key: float(value) for key, value in fit.quality.items()},
                "area_px": area_px,
                "centre_px": centre,
                "bbox": bbox,
                "component_area": candidate["area"],
            }
        )

    # A piece split by a low-contrast band would appear as extra detections; keep
    # the expected number of highest-quality candidates and report the rest.
    discarded = []
    if expected_count and len(fitted) > expected_count:
        fitted.sort(key=lambda item: item["quality"], reverse=True)
        discarded = fitted[expected_count:]
        fitted = fitted[:expected_count]
    fitted = [fitted[index] for index in scan_order_indices([item["centre_px"] for item in fitted])]

    size_px = int(config["piece_output_size_px"])
    margin_px = float(config["piece_margin_px"])
    pieces: list[dict[str, Any]] = []
    manual_review: list[str] = []
    shape_areas = []
    for order, item in enumerate(fitted, start=1):
        piece_id = f"S{order:02d}"
        normalized = normalize_piece(pieces_image, item["corners"], size_px, margin_px)
        corners = np.round(item["corners"], 3)
        normalized_corners = np.round(normalized["normalized_corners"], 3)
        normalized_short_side = float(
            min(
                np.linalg.norm(normalized_corners[(index + 1) % len(normalized_corners)] - normalized_corners[index])
                for index in range(len(normalized_corners))
            )
        )
        area_px = float(item["area_px"])
        area_mm2 = area_px / (float(px_per_mm[0]) * float(px_per_mm[1]))
        shape_areas.append(area_px)
        quality_score = float(np.clip(item["quality"], 0.0, 1.0))
        breakdown = item["quality_breakdown"]
        # Quality flag for human triage: a low score, a sliver side (< 100 px) or
        # two localizers that disagree by more than 3 px at the corners all mean
        # the sub-pixel corners deserve a second pair of eyes.
        review_reasons: list[str] = []
        if quality_score < float(config["min_quality"]):
            review_reasons.append("quality_below_min")
        if float(breakdown.get("min_side_px", float("inf"))) < MIN_SAFE_SIDE_PX:
            review_reasons.append("short_side")
        shift = float(breakdown.get("agreement_shift_px", float("nan")))
        if math.isfinite(shift) and shift > MAX_LOCALIZER_SHIFT_PX:
            review_reasons.append("localizer_disagreement")
        below_threshold = bool(review_reasons)
        if below_threshold:
            manual_review.append(piece_id)
        pieces.append(
            {
                "piece_id": piece_id,
                "rgba_path": f"pieces/{piece_id}_rgba.png",
                "mask_path": f"pieces/{piece_id}_mask.png",
                "debug_path": f"pieces/{piece_id}_debug.png",
                "corners_px": corners.tolist(),
                "corner_count": int(len(corners)),
                "center_px": [round(float(value), 3) for value in item["centre_px"]],
                "area_px": round(area_px, 3),
                "area_mm2": round(area_mm2, 3),
                "bbox_px": [int(value) for value in (*item["bbox"][0], *item["bbox"][1])],
                "perimeter_px": round(float(cv2.arcLength(item["corners"].astype(np.float32), True)), 3),
                "quality": round(quality_score, 6),
                "quality_breakdown": {
                    key: (round(float(value), 6) if math.isfinite(float(value)) else None)
                    for key, value in item["quality_breakdown"].items()
                },
                "needs_manual_review": bool(below_threshold),
                "manual_review_reasons": review_reasons,
                "normalized_corners_px": normalized_corners.tolist(),
                "normalized_short_side_px": round(normalized_short_side, 3),
                "rotation_deg": round(float(normalized["rotation_deg"]), 6),
                "rectified_to_piece_matrix": np.round(normalized["matrix"], 8).tolist(),
                "rgba": normalized["rgba"],
                "mask": normalized["mask"],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "algorithm_version": ALGORITHM_VERSION,
        "session_id": _session_id([pieces_path, empty_path, rectification_path] + ([Path(layout_path)] if layout_path else []), config),
        "coordinate_system": "rectified_px",
        "piece_size_px": [size_px, size_px],
        "piece_margin_px": margin_px,
        "canvas_px": [int(canvas_px[0]), int(canvas_px[1])],
        "px_per_mm": [float(px_per_mm[0]), float(px_per_mm[1])],
        "corner_order": "positive shoelace area in y-down pixel coordinates; first corner is top-most then left-most",
        "expected_piece_count": expected_count,
        "detected_piece_count": len(pieces),
        "inputs": {
            "pieces_rectified": _portable_path(pieces_path),
            "empty_rectified": _portable_path(empty_path),
            "rectification": _portable_path(rectification_path),
            "layout": _portable_path(layout_path),
        },
        "segmentation": {
            "threshold": round(float(threshold), 3),
            "threshold_mode": config["threshold_mode"],
            "ignore_border_px": int(ignore_border_px),
            "foreground_fraction": round(float((mask > 0).mean()), 6),
            "discarded_candidates": [
                {"centre_px": [round(float(v), 3) for v in item["centre_px"]], "quality": round(float(item["quality"]), 6)}
                for item in discarded
            ],
        },
        "summary": {
            "min_quality": round(float(min((piece["quality"] for piece in pieces), default=0.0)), 6),
            "median_quality": round(float(np.median([piece["quality"] for piece in pieces])), 6) if pieces else 0.0,
            "min_area_px": round(float(min(shape_areas)), 3) if shape_areas else 0.0,
            "max_area_px": round(float(max(shape_areas)), 3) if shape_areas else 0.0,
            "total_area_px": round(float(sum(shape_areas)), 3),
        },
        "config": {key: config[key] for key in DEFAULT_CONFIG_VALUES},
        "manual_review": manual_review,
        "pieces": pieces,
    }

    if write_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        piece_dir = output_dir / "pieces"
        piece_dir.mkdir(parents=True, exist_ok=True)
        for piece in pieces:
            imwrite(output_dir / piece["rgba_path"], piece.pop("rgba"))
            imwrite(output_dir / piece["mask_path"], piece.pop("mask"))
            debug = render_debug(
                imread(output_dir / piece["rgba_path"], cv2.IMREAD_UNCHANGED),
                np.asarray(piece["normalized_corners_px"], dtype=np.float64),
                piece["piece_id"],
                piece["quality"],
            )
            imwrite(output_dir / piece["debug_path"], debug)
        write_json(output_dir / "pieces.json", payload)
    return payload


def _format_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        f"Pieces: {payload['detected_piece_count']}/{payload['expected_piece_count']} "
        f"(canvas {payload['canvas_px'][0]}x{payload['canvas_px'][1]}, "
        f"normalized {payload['piece_size_px'][0]}x{payload['piece_size_px'][1]})",
        f"Segmentation: threshold={payload['segmentation']['threshold']} "
        f"foreground={payload['segmentation']['foreground_fraction'] * 100:.2f}% "
        f"border_ignore={payload['segmentation']['ignore_border_px']}px",
        f"Quality: min={summary['min_quality']:.4f} median={summary['median_quality']:.4f}",
        f"Area px: min={summary['min_area_px']:.0f} max={summary['max_area_px']:.0f} total={summary['total_area_px']:.0f}",
    ]
    for piece in payload["pieces"]:
        vertex_quality = piece["quality_breakdown"].get("fit_rms_px")
        fit_rms = "n/a" if vertex_quality is None else f"{vertex_quality:.3f}"
        lines.append(
            f"  {piece['piece_id']} corners={piece['corner_count']} "
            f"area={piece['area_px']:.0f}px q={piece['quality']:.3f} fit_rms={fit_rms}px "
            f"centre=({piece['center_px'][0]:.0f},{piece['center_px'][1]:.0f})"
        )
    if payload["manual_review"]:
        flagged = [
            f"{piece['piece_id']} ({', '.join(piece['manual_review_reasons'])})"
            for piece in payload["pieces"]
            if piece["needs_manual_review"]
        ]
        lines.append(f"Manual review: {'; '.join(flagged)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 piece extraction")
    parser.add_argument("--pieces", type=Path, default=DEFAULT_INPUT_PIECES, help="rectified capture with pieces")
    parser.add_argument("--empty", type=Path, default=DEFAULT_INPUT_EMPTY, help="rectified capture of the empty table")
    parser.add_argument("--rectification", type=Path, default=DEFAULT_INPUT_RECTIFICATION)
    parser.add_argument("--layout", type=Path, default=DEFAULT_INPUT_LAYOUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    config = _resolve_config(args.config)
    payload = extract_pieces(
        pieces_path=args.pieces,
        empty_path=args.empty,
        rectification_path=args.rectification,
        layout_path=args.layout,
        config=config,
        output_dir=args.output,
    )
    print(_format_report(payload))
    expected = int(payload["expected_piece_count"])
    if expected and payload["detected_piece_count"] != expected:
        print(
            f"ERROR: detected {payload['detected_piece_count']} pieces, expected {expected}. "
            "See the discarded candidates in pieces.json; tune the config or fix the capture.",
            file=sys.stderr,
        )
        return 1
    if payload["manual_review"]:
        # A quality flag is a hand-off note for a human, not a pipeline failure:
        # the geometry is still written and Stage 3 can consume it.
        flagged = [
            f"{piece['piece_id']} ({', '.join(piece['manual_review_reasons'])})"
            for piece in payload["pieces"]
            if piece["needs_manual_review"]
        ]
        print(
            "WARNING: manual review recommended for " + "; ".join(flagged),
            file=sys.stderr,
        )
        if config.get("fail_on_manual_review", False):
            return 1
    print(f"Wrote {_portable_path(args.output / 'pieces.json')} and {payload['detected_piece_count']} piece images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
