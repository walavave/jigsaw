#!/usr/bin/env python3
"""Reconstruct Stage 2 textures from Stage 3 poses and plan manual placement.

The geometry solver rotates base-edge coordinates, not PNG coordinates.  A PNG
includes a crop offset; its transform is pose @ rectified_to_base @
inverse(rectified_to_piece).  No reference pixels are pasted into the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from jigsaw_common.io import imread, imwrite, read_json  # noqa: E402

VERSION = "stage4-assembly-1.0.0"
DEFAULT_PIECES = ROOT / "stage2_piece_extraction/output/pieces.json"
DEFAULT_PLACEMENTS = ROOT / "stage3_feature_matching/quad_puzzle/examples/output.json"
DEFAULT_REFERENCE = ROOT / "stage1_capture_calibration/output/reference/reference.png"
DEFAULT_OUTPUT = ROOT / "stage4_assembly/output"
DEFAULT_CONFIG = {
    "schema_version": "1.0",
    "color_space": "Lab",
    "color_correction": True,
    "color_strength": 0.5,
    "color_gain_limits": [0.8, 1.25],
    "color_max_offset": 10.0,
    "color_sample_erode_px": 3,
    "blend_mode": "feather",
    "feather_px": 3.0,
    "edge_erode_px": 0,
    "evaluate_ssim": True,
    "adjacency_tolerance_px": 8.0,
    "write_steps": True,
    "fail_on_manual_review": False,
    "max_canvas_pixels": 25000000,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    require(
        arr.shape == shape and np.isfinite(arr).all(),
        f"{name}: expected finite {shape}",
    )
    return arr


def matrix(value: Any, name: str) -> np.ndarray:
    h = finite_array(value, (3, 3), name)
    norm = np.linalg.norm(h)
    require(
        norm > 0 and abs(np.linalg.det(h / norm)) > 1e-15, f"{name}: singular transform"
    )
    return h / norm


def transform(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = np.column_stack((points, np.ones(len(points)))) @ h.T
    d = p[:, 2]
    require(
        np.isfinite(p).all() and np.all(np.abs(d) > 1e-12),
        "transform reaches projective infinity",
    )
    require(np.all(d > 0) or np.all(d < 0), "transform crosses projective infinity")
    return p[:, :2] / d[:, None]


def polygon(value: Any, name: str) -> np.ndarray:
    p = np.asarray(value, dtype=np.float64)
    require(
        p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3 and np.isfinite(p).all(),
        f"{name}: invalid vertices",
    )
    shape = Polygon(p)
    require(
        shape.is_valid and shape.area > 1e-6,
        f"{name}: degenerate or self-intersecting polygon",
    )
    require(
        shape.convex_hull.area - shape.area < max(1e-5, shape.area * 1e-6),
        f"{name}: polygon must be convex",
    )
    return p


def signed_area(p: np.ndarray) -> float:
    return float(
        np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)) / 2
    )


def resolve_config(config: dict | None) -> dict:
    c = dict(DEFAULT_CONFIG)
    if config:
        require(
            not (set(config) - set(c)),
            f"unknown configuration keys: {sorted(set(config) - set(c))}",
        )
        c.update(config)
    require(
        c["schema_version"] == "1.0" and c["color_space"] == "Lab",
        "unsupported config version / color space",
    )
    for key in (
        "color_correction",
        "evaluate_ssim",
        "write_steps",
        "fail_on_manual_review",
    ):
        require(isinstance(c[key], bool), f"{key} must be boolean")
    for key in (
        "color_strength",
        "color_max_offset",
        "feather_px",
        "adjacency_tolerance_px",
    ):
        require(
            isinstance(c[key], (int, float))
            and not isinstance(c[key], bool)
            and math.isfinite(c[key])
            and c[key] >= 0,
            f"invalid {key}",
        )
    require(c["color_strength"] <= 1, "color_strength must be in [0,1]")
    require(
        c["blend_mode"] in {"feather", "overwrite"},
        "blend_mode must be feather or overwrite",
    )
    for key in ("color_sample_erode_px", "edge_erode_px", "max_canvas_pixels"):
        require(
            isinstance(c[key], int) and not isinstance(c[key], bool) and c[key] >= 0,
            f"invalid {key}",
        )
    limits = finite_array(c["color_gain_limits"], (2,), "color_gain_limits")
    require(0 < limits[0] <= 1 <= limits[1], "color gains must bracket 1")
    return c


def mask_polygon(p: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), np.uint8)
    # Fixed-point vertices retain subpixel accuracy; never wrap enormous inputs.
    require(np.max(np.abs(p)) < 1e7, "polygon coordinates are too large")
    cv2.fillPoly(mask, [np.round(p * 256).astype(np.int32)], 255, shift=8)
    return mask


def base_vertices(piece: dict) -> np.ndarray:
    if "base_edge_corners_px" in piece:
        return polygon(piece["base_edge_corners_px"], "base_edge_corners_px")
    b = piece["base_edge"]
    remaining = b.get("remainingVertices", [b.get("p2"), b.get("p3")])
    return polygon([[0, 0], [b["baseLength"], 0], *remaining], "base_edge")


def image_to_base(piece: dict, base: np.ndarray) -> np.ndarray:
    if (
        "rectified_to_base_edge_matrix" in piece
        and "rectified_to_piece_matrix" in piece
    ):
        h = matrix(
            piece["rectified_to_base_edge_matrix"], "rectified_to_base_edge_matrix"
        ) @ np.linalg.inv(
            matrix(piece["rectified_to_piece_matrix"], "rectified_to_piece_matrix")
        )
    else:
        image_points = finite_array(
            piece.get("image_corners_px"), base.shape, "image_corners_px"
        )
        offset = image_points - base
        require(
            np.max(np.abs(offset - offset[0])) < 0.02,
            "PNG/base frames must differ by a crop translation",
        )
        h = np.array([[1, 0, -offset[0, 0]], [0, 1, -offset[0, 1]], [0, 0, 1]], float)
    if "image_corners_px" in piece:
        image_points = finite_array(
            piece["image_corners_px"], base.shape, "image_corners_px"
        )
        require(
            np.max(np.abs(transform(h, image_points) - base)) < 0.03,
            "inconsistent PNG/base-edge corner transforms",
        )
    return h


@dataclass
class Fragment:
    id: str
    metadata: dict
    rgba: np.ndarray
    h: np.ndarray  # PNG -> assembly canvas
    target: np.ndarray
    source_id: str | None = None
    confidence: float | None = None
    match_status: str = "accepted"


def unique_records(records: Any, key: str, name: str) -> dict[str, dict]:
    require(
        isinstance(records, list) and bool(records), f"{name} must be a nonempty array"
    )
    out = {}
    for record in records:
        require(isinstance(record, dict), f"invalid {name} entry")
        id_ = record.get(key)
        require(
            isinstance(id_, str) and bool(id_) and id_ not in out,
            f"missing / duplicate {key}: {id_}",
        )
        out[id_] = record
    return out


def load_inputs(
    pieces_path: Path, placements_path: Path, layout_path: Path | None, config: dict
) -> tuple:
    manifest, result = read_json(pieces_path), read_json(placements_path)
    require(
        isinstance(manifest, dict) and isinstance(result, dict),
        "handoffs must be JSON objects",
    )
    require(
        manifest.get("schema_version") == "1.0", "unsupported pieces schema_version"
    )
    pieces = unique_records(manifest.get("pieces"), "piece_id", "pieces")
    require(
        manifest.get("detected_piece_count", len(pieces)) == len(pieces),
        "detected_piece_count does not match pieces",
    )
    require(
        manifest.get("expected_piece_count", len(pieces)) == len(pieces),
        "incomplete Stage 2 extraction",
    )
    if result.get("session_id") and manifest.get("session_id"):
        require(
            result["session_id"] == manifest["session_id"],
            "Stage 2 / 3 session_id mismatch",
        )
    warnings: list[str] = []
    if "placements" in result:
        mode = "geometry_placements"
        require(
            result.get("schema")
            in {
                "polygon-assembly-result/v2",
                "polygon-assembly-result/v1",
                "quad-assembly-result/v1",
            },
            "unsupported placement schema",
        )
        require(
            result.get("status") in {"feasible", "approximate"},
            "Stage 3 did not produce a usable layout",
        )
        require(
            result.get("coordinateSystem") == "image",
            "placements must use image coordinates",
        )
        require(
            result.get("unit") == manifest.get("coordinate_system"),
            "Stage 2 / 3 coordinate units differ",
        )
        require(
            manifest.get("piece_scale", 1) == 1,
            "geometry mode requires Stage 2 scale 1",
        )
        placements = unique_records(result["placements"], "id", "placements")
        rect = finite_array(
            [result["rectangle"]["width"], result["rectangle"]["height"]],
            (2,),
            "rectangle",
        )
        if result["status"] == "approximate" or not result.get("verification", {}).get(
            "valid", True
        ):
            warnings.append(
                "Stage 3 layout is approximate; gaps, overlaps and boundary errors remain."
            )
        layout = None
    else:
        mode = "reference_assignments"
        require(
            result.get("schema_version") == "1.0",
            "unsupported assignments schema_version",
        )
        require(not result.get("unassigned"), "Stage 3 contains unassigned pieces")
        require(layout_path is not None, "assignments require --layout")
        layout = read_json(layout_path)
        require(
            layout.get("schema_version") == "1.0", "unsupported layout schema_version"
        )
        sources = unique_records(layout["pieces"], "piece_id", "layout pieces")
        placements = unique_records(
            result["assignments"], "scan_piece_id", "assignments"
        )
        source_ids = [a.get("source_piece_id") for a in placements.values()]
        require(
            len(set(source_ids)) == len(source_ids) and set(source_ids) == set(sources),
            "source assignments must cover layout exactly once",
        )
        rect = finite_array(layout["source_size_px"], (2,), "source_size_px")
    require(
        set(pieces) == set(placements),
        "Stage 3 IDs must cover Stage 2 IDs exactly once",
    )
    require(np.all(rect > 0), "rectangle dimensions must be positive")
    require(
        math.ceil(rect[0]) * math.ceil(rect[1]) <= config["max_canvas_pixels"],
        "canvas exceeds max_canvas_pixels",
    )
    fragments = []
    for id_, p in pieces.items():
        rgba = imread(pieces_path.parent / p["rgba_path"], cv2.IMREAD_UNCHANGED)
        require(
            rgba.ndim == 3 and rgba.shape[2] == 4 and rgba.dtype == np.uint8,
            f"{id_}: expected 8-bit RGBA PNG",
        )
        if "piece_size_px" in p:
            require(
                list(rgba.shape[1::-1]) == p["piece_size_px"],
                f"{id_}: PNG size differs from manifest",
            )
        if p.get("mask_path"):
            mask = imread(pieces_path.parent / p["mask_path"], cv2.IMREAD_GRAYSCALE)
            require(mask.shape == rgba.shape[:2], f"{id_}: mask size differs from PNG")
            rgba[:, :, 3] = np.minimum(mask, rgba[:, :, 3])
        require(np.any(rgba[:, :, 3]), f"{id_}: empty alpha/mask")
        a = placements[id_]
        source_id = None
        confidence = None
        status = a.get("status", "accepted")
        require(
            status in {"accepted", "review"}, f"{id_}: unusable match status {status}"
        )
        if mode == "geometry_placements":
            base = base_vertices(p)
            origin = finite_array(a["origin"], (2,), f"{id_}.origin")
            angle = float(a["rotationDeg"])
            require(math.isfinite(angle), f"{id_}: nonfinite rotation")
            theta = math.radians(angle)
            pose = np.array(
                [
                    [math.cos(theta), -math.sin(theta), origin[0]],
                    [math.sin(theta), math.cos(theta), origin[1]],
                    [0, 0, 1],
                ]
            )
            h = pose @ image_to_base(p, base)
            target = transform(pose, base)
        else:
            source_id = a["source_piece_id"]
            target = polygon(
                sources[source_id]["polygon_px"], f"{source_id}.polygon_px"
            )
            keys = [
                k
                for k in (
                    "homography_piece_to_source",
                    "H_piece_to_source",
                    "H_scan_to_source",
                )
                if k in a
            ]
            require(
                len(keys) == 1, f"{id_}: provide exactly one supported homography field"
            )
            h = matrix(a[keys[0]], f"{id_}.{keys[0]}")
            frame = a.get("transform_source_frame", "piece_image_px")
            require(
                frame in {"piece_image_px", "rectified_px"},
                f"{id_}: invalid transform_source_frame",
            )
            if frame == "rectified_px":
                h = h @ np.linalg.inv(
                    matrix(p["rectified_to_piece_matrix"], "rectified_to_piece_matrix")
                )
            if a.get("confidence") is not None:
                confidence = float(a["confidence"])
                require(
                    math.isfinite(confidence) and 0 <= confidence <= 1,
                    f"{id_}: invalid confidence",
                )
        # Check orientation on the actual opaque support, not transparent padding.
        hull = cv2.convexHull(
            cv2.findNonZero((rgba[:, :, 3] > 0).astype(np.uint8))
        ).reshape(-1, 2)
        mapped = polygon(transform(h, hull), f"{id_}.warped_support")
        require(
            signed_area(mapped) * signed_area(hull) > 0, f"{id_}: mirrored transform"
        )
        require(
            Polygon(mapped).intersection(Polygon(target)).area > 0,
            f"{id_}: texture does not meet target polygon",
        )
        if (
            p.get("needs_manual_review")
            or id_ in manifest.get("manual_review", [])
            or status == "review"
        ):
            warnings.append(f"{id_}: upstream manual review required.")
        fragments.append(
            Fragment(id_, p, rgba, h, target, source_id, confidence, status)
        )
    return manifest, result, fragments, tuple(rect), mode, warnings


def board_rotation(
    angle: int, rect: tuple[float, float]
) -> tuple[np.ndarray, tuple[float, float]]:
    w, h = rect
    rotations = {
        0: (np.eye(3), (w, h)),
        90: (np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], float), (h, w)),
        180: (np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], float), (w, h)),
        270: (np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], float), (h, w)),
    }
    return rotations[angle]


def warp_fragment(
    fragment: Fragment, size: tuple[int, int], config: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = fragment.rgba[:, :, 3].astype(np.float32) / 255
    if config["edge_erode_px"]:
        r = config["edge_erode_px"]
        alpha = cv2.erode(
            alpha,
            np.ones((2 * r + 1, 2 * r + 1), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    # Premultiplied interpolation prevents transparent black contaminating edges.
    premul = fragment.rgba[:, :, :3].astype(np.float32) * alpha[:, :, None]
    warped_alpha = cv2.warpPerspective(alpha, fragment.h, size, flags=cv2.INTER_LINEAR)
    premul = cv2.warpPerspective(premul, fragment.h, size, flags=cv2.INTER_LINEAR)
    color = premul / np.maximum(warped_alpha[:, :, None], 1e-7)
    alpha = warped_alpha * (mask_polygon(fragment.target, size) > 0)
    weight = alpha.copy()
    if config["blend_mode"] == "feather" and config["feather_px"] > 0:
        support = (alpha > 1e-3).astype(np.uint8)
        padded = np.pad(support, 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        weight *= np.clip(distance / config["feather_px"], 0.05, 1)
    return color, alpha, weight


def finish(accumulator: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.clip(
        np.rint(accumulator / np.maximum(weight[:, :, None], 1e-7)), 0, 255
    ).astype(np.uint8)


def render_raw(
    fragments: list[Fragment], rect: tuple[float, float], config: dict
) -> tuple[np.ndarray, np.ndarray]:
    size = tuple(math.ceil(v) for v in rect)
    acc = np.zeros((size[1], size[0], 3), np.float32)
    weights = np.zeros((size[1], size[0]), np.float32)
    for f in fragments:
        color, _, weight = warp_fragment(f, size, config)
        acc += color * weight[:, :, None]
        weights += weight
    return finish(acc, weights), weights > 1e-3


def choose_orientation(
    fragments: list[Fragment],
    rect: tuple[float, float],
    reference: np.ndarray | None,
    config: dict,
    orientation: str,
) -> tuple[int, dict]:
    if orientation != "auto":
        return int(orientation), {"method": "explicit", "candidates": []}
    if reference is None:
        return 0, {"method": "preserve_solver_no_reference", "candidates": []}
    raw, covered = render_raw(fragments, rect, config)
    candidates = []
    # Compare only global quarter turns. Never move pieces relative to one another.
    for angle in (0, 90, 180, 270):
        image = np.rot90(raw, -(angle // 90))
        support = np.rot90(covered, -(angle // 90))
        ratio_error = abs(
            math.log(
                (image.shape[1] / image.shape[0])
                / (reference.shape[1] / reference.shape[0])
            )
        )
        if ratio_error > 0.08:
            continue
        size = (
            max(8, round(400 * image.shape[1] / max(image.shape[:2]))),
            max(8, round(400 * image.shape[0] / max(image.shape[:2]))),
        )
        a = cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2GRAY).astype(float)
        b = cv2.cvtColor(cv2.resize(reference, size), cv2.COLOR_BGR2GRAY).astype(float)
        valid = (
            cv2.resize(support.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            > 0
        )
        av, bv = a[valid], b[valid]
        score = (
            float(np.corrcoef(av, bv)[0, 1])
            if len(av) > 10 and av.std() > 1e-5 and bv.std() > 1e-5
            else None
        )
        candidates.append({"clockwise_deg": angle, "gray_correlation": score})
    usable = [c for c in candidates if c["gray_correlation"] is not None]
    chosen = (
        max(usable, key=lambda c: c["gray_correlation"])["clockwise_deg"]
        if usable
        else 0
    )
    scores = sorted((c["gray_correlation"] for c in usable), reverse=True)
    return chosen, {
        "method": "reference_gray_correlation",
        "candidates": candidates,
        "ambiguous": len(scores) < 2 or scores[0] < 0.4 or scores[0] - scores[1] < 0.05,
    }


def color_correct(
    color: np.ndarray, reference: np.ndarray | None, alpha: np.ndarray, config: dict
) -> tuple[np.ndarray, dict]:
    if reference is None or not config["color_correction"]:
        return color, {
            "applied": False,
            "reason": "no_reference" if reference is None else "disabled",
        }
    valid = (alpha > 0.99).astype(np.uint8)
    r = config["color_sample_erode_px"]
    if r:
        valid = cv2.erode(
            valid,
            np.ones((2 * r + 1, 2 * r + 1), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    valid = valid > 0
    if valid.sum() < 32:
        return color, {"applied": False, "reason": "insufficient_interior_samples"}
    lab = cv2.cvtColor(np.clip(color / 255, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)
    target = cv2.cvtColor(reference.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    src_mean, src_std = lab[valid].mean(axis=0), lab[valid].std(axis=0)
    dst_mean, dst_std = target[valid].mean(axis=0), target[valid].std(axis=0)
    # Low variance is not evidence for a huge contrast increase.
    gain = np.where(src_std > 0.5, dst_std / np.maximum(src_std, 0.5), 1.0)
    gain = np.clip(gain, *config["color_gain_limits"])
    strength = config["color_strength"]
    gain = 1 + strength * (gain - 1)
    offset = strength * np.clip(
        dst_mean - src_mean, -config["color_max_offset"], config["color_max_offset"]
    )
    lab = (lab - src_mean) * gain + src_mean + offset
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)
    lab[:, :, 1:] = np.clip(lab[:, :, 1:], -127, 127)
    corrected = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255
    return corrected, {
        "applied": True,
        "sample_pixels": int(valid.sum()),
        "source_mean_Lab": src_mean.tolist(),
        "target_mean_Lab": dst_mean.tolist(),
        "source_std_Lab": src_std.tolist(),
        "target_std_Lab": dst_std.tolist(),
        "gain": gain.tolist(),
        "mean_shift_Lab": offset.tolist(),
        "strength": strength,
    }


def shared_edge_length(a: np.ndarray, b: np.ndarray, tolerance: float) -> float:
    """Length of near-parallel shared sides; corner-only contact is not adjacency."""
    length = 0.0
    for p, q in zip(a, np.roll(a, -1, axis=0)):
        u = q - p
        la = np.linalg.norm(u)
        if la <= 1e-9:
            continue
        u /= la
        n = np.array([-u[1], u[0]])
        for s, t in zip(b, np.roll(b, -1, axis=0)):
            v = t - s
            lb = np.linalg.norm(v)
            if lb <= 1e-9 or abs(np.dot(u, v / lb)) < math.cos(math.radians(8)):
                continue
            x0, x1 = sorted([np.dot(s - p, u), np.dot(t - p, u)])
            lo, hi = max(0, x0), min(la, x1)
            if hi - lo < max(1, tolerance * 2):
                continue
            # Check separation at both ends of the overlapping projected interval.
            xs, xt = np.dot(s - p, u), np.dot(t - p, u)
            ys, yt = np.dot(s - p, n), np.dot(t - p, n)
            distances = [ys + (yt - ys) * (x - xs) / (xt - xs) for x in (lo, hi)]
            if max(abs(d) for d in distances) <= tolerance + 1e-6:
                length += hi - lo
    return length


def placement_order(
    fragments: list[Fragment], tolerance: float
) -> tuple[list[Fragment], dict, set[str]]:
    edges = {}
    for i, a in enumerate(fragments):
        for b in fragments[i + 1 :]:
            length = shared_edge_length(a.target, b.target, tolerance)
            if length > 0:
                edges[tuple(sorted((a.id, b.id)))] = length
    centers = {f.id: Polygon(f.target).centroid for f in fragments}
    remaining = {f.id: f for f in fragments}
    first = min(
        fragments,
        key=lambda f: (
            Polygon(f.target).distance(Point(0, 0)),
            centers[f.id].y,
            centers[f.id].x,
            f.id,
        ),
    )
    ordered = [remaining.pop(first.id)]
    component_starts = {first.id}
    while remaining:
        weights = {
            id_: sum(edges.get(tuple(sorted((id_, p.id))), 0) for p in ordered)
            for id_ in remaining
        }
        id_ = min(remaining, key=lambda k: (-weights[k], centers[k].y, centers[k].x, k))
        if weights[id_] == 0:
            component_starts.add(id_)
        ordered.append(remaining.pop(id_))
    return ordered, edges, component_starts


def geometry_report(fragments: list[Fragment], rect: tuple[float, float]) -> dict:
    shapes = [Polygon(f.target) for f in fragments]
    frame = box(0, 0, *rect)
    union = unary_union(shapes)
    total = sum(s.area for s in shapes)
    missing = frame.difference(union).area
    overlap = max(0, total - union.area)
    outside = union.difference(frame).area
    return {
        "total_piece_area": total,
        "rectangle_area": frame.area,
        "missing_area": missing,
        "overlap_excess_area": overlap,
        "outside_area": outside,
        "coverage_fraction": 1 - missing / frame.area,
        "score": (missing + overlap + outside) / total,
        "area_unit": "assembly_px_squared",
    }


def image_metrics(
    image: np.ndarray,
    reference: np.ndarray | None,
    covered: np.ndarray,
    evaluate_ssim: bool,
) -> dict:
    if reference is None:
        return {"available": False, "reason": "no_reference"}
    diff = (image.astype(float) - reference.astype(float)) ** 2
    mse = float(diff.mean())
    result = {
        "available": True,
        "mse": mse,
        "psnr_db": 10 * math.log10(255**2 / mse) if mse > 0 else None,
        "identical": mse == 0,
        "covered_psnr_db": None,
    }
    if np.any(covered):
        covered_mse = float(diff[covered].mean())
        result["covered_psnr_db"] = (
            10 * math.log10(255**2 / covered_mse) if covered_mse > 0 else None
        )
    side = min(image.shape[:2])
    if evaluate_ssim and side >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        result["ssim_gray"] = float(
            structural_similarity(
                gray,
                target,
                data_range=255,
                win_size=min(7, side if side % 2 else side - 1),
            )
        )
    else:
        result["ssim_gray"] = None
    return result


def seam_report(
    image: np.ndarray,
    reference: np.ndarray | None,
    fragments: list[Fragment],
    edges: dict,
    covered: np.ndarray,
    tolerance: float,
) -> list[dict]:
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = (
        np.gradient(gray)
        if min(gray.shape) >= 2
        else (np.zeros_like(gray), np.zeros_like(gray))
    )
    grad = np.hypot(gx, gy)
    ref_lab = (
        cv2.cvtColor(reference.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        if reference is not None
        else None
    )
    if reference is not None and min(gray.shape) >= 2:
        rx, ry = np.gradient(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        )
        ref_grad = np.hypot(rx, ry)
    else:
        ref_grad = None
    by_id = {f.id: f for f in fragments}
    output = []
    for (a, b), length in sorted(edges.items()):
        pa, pb = Polygon(by_id[a].target), Polygon(by_id[b].target)
        center = np.array(
            [pb.centroid.x - pa.centroid.x, pb.centroid.y - pa.centroid.y]
        )
        center /= max(np.linalg.norm(center), 1e-9)
        # Sample either side of A's edge where it is close to B's boundary.
        points = []
        for p, q in zip(by_id[a].target, np.roll(by_id[a].target, -1, axis=0)):
            for t in np.linspace(0.05, 0.95, max(2, int(np.linalg.norm(q - p) / 3))):
                point = p + t * (q - p)
                if pb.boundary.distance(Point(point)) <= tolerance + 1e-5:
                    points.append(point)
        pairs = []
        offset = max(2, tolerance + 2)
        for p in points:
            u, v = (
                np.rint(p - center * offset).astype(int),
                np.rint(p + center * offset).astype(int),
            )
            if (
                all(
                    0 <= xy[0] < image.shape[1] and 0 <= xy[1] < image.shape[0]
                    for xy in (u, v)
                )
                and covered[u[1], u[0]]
                and covered[v[1], v[0]]
                and pa.covers(Point(u))
                and pb.covers(Point(v))
            ):
                pairs.append((u, v))
        row = {
            "piece_a": a,
            "piece_b": b,
            "shared_edge_length_px": length,
            "sample_pairs": len(pairs),
            "mean_color_difference_deltaE76": None,
            "mean_gradient_difference": None,
            "reference_color_difference_deltaE76": None,
            "gradient_residual_vs_reference": None,
        }
        if pairs:
            uv = np.asarray(pairs)
            x1, y1, x2, y2 = uv[:, 0, 0], uv[:, 0, 1], uv[:, 1, 0], uv[:, 1, 1]
            row["mean_color_difference_deltaE76"] = float(
                np.linalg.norm(lab[y1, x1] - lab[y2, x2], axis=1).mean()
            )
            row["mean_gradient_difference"] = float(
                np.abs(grad[y1, x1] - grad[y2, x2]).mean()
            )
            if ref_lab is not None:
                row["reference_color_difference_deltaE76"] = float(
                    np.linalg.norm(ref_lab[y1, x1] - ref_lab[y2, x2], axis=1).mean()
                )
            if ref_grad is not None:
                row["gradient_residual_vs_reference"] = float(
                    (
                        np.abs(grad[y1, x1] - ref_grad[y1, x1])
                        + np.abs(grad[y2, x2] - ref_grad[y2, x2])
                    ).mean()
                    / 2
                )
        output.append(row)
    return output


def label(
    image: np.ndarray, text: str, at: tuple[int, int], scale: float = 0.65
) -> None:
    x, y = at
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = (
        max(0, min(x - w // 2, image.shape[1] - w - 2)),
        max(h + 3, min(y, image.shape[0] - 3)),
    )
    cv2.rectangle(image, (x, y - h - 3), (x + w + 2, y + 3), (250, 250, 250), -1)
    cv2.putText(
        image,
        text,
        (x + 1, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )


def save_json(path: Path, value: Any) -> None:
    # RFC-compliant output: NaN and Infinity are never serialized.
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def manual_plan(
    fragments: list[Fragment],
    edges: dict,
    component_starts: set,
    rect: tuple[float, float],
    manifest: dict,
    mode: str,
    warnings: list[str],
) -> dict:
    density = manifest.get("px_per_mm")
    mm_scale = None
    if mode == "geometry_placements" and density is not None:
        d = np.asarray(density, float)
        if (
            d.shape == (2,)
            and np.isfinite(d).all()
            and np.all(d > 0)
            and abs(d[0] - d[1]) < 1e-6
        ):
            mm_scale = float(1 / d[0])
    slots = []
    placed = []
    for index, f in enumerate(fragments, 1):
        neighbors = [p for p in placed if tuple(sorted((p, f.id))) in edges]
        center = Polygon(f.target).centroid
        h = f.h.copy()
        table_rotation = None
        table_transform = None
        if "rectified_to_piece_matrix" in f.metadata:
            table_transform = h @ matrix(
                f.metadata["rectified_to_piece_matrix"], "rectified_to_piece_matrix"
            )
            xy = np.asarray(f.metadata.get("center_px", [0, 0]), float)
            sample = transform(table_transform, np.array([xy, xy + [1, 0]]))
            delta = sample[1] - sample[0]
            table_rotation = (
                math.degrees(math.atan2(delta[1], delta[0])) + 180
            ) % 360 - 180
        row = {
            "step": index,
            "scan_piece_id": f.id,
            "source_piece_id": f.source_id,
            "target_polygon_px": f.target.tolist(),
            "target_center_px": [center.x, center.y],
            "homography_piece_to_source": (h / h[2, 2]).tolist()
            if abs(h[2, 2]) > 1e-12
            else h.tolist(),
            "H_rectified_to_assembly": table_transform.tolist()
            if table_transform is not None
            else None,
            "rotation_deg": table_rotation,
            "rotation_frame": "rectified_table_to_assembly_clockwise"
            if table_rotation is not None
            else "unavailable",
            "scan_center_rectified_px": f.metadata.get("center_px"),
            "scan_polygon_rectified_px": f.metadata.get("corners_px"),
            "align_with_placed_pieces": neighbors,
            "start_new_component": f.id in component_starts,
            "confidence": f.confidence,
            "extraction_quality": f.metadata.get("quality"),
            "needs_manual_review": bool(
                f.metadata.get("needs_manual_review")
                or f.id in manifest.get("manual_review", [])
                or f.match_status == "review"
            ),
            "step_image": f"steps/{index:02d}.png",
        }
        if mm_scale is not None:
            row["target_polygon_mm"] = (f.target * mm_scale).tolist()
            row["target_center_mm"] = [center.x * mm_scale, center.y * mm_scale]
        slots.append(row)
        placed.append(f.id)
    return {
        "schema_version": "1.0",
        "algorithm_version": VERSION,
        "session_id": manifest.get("session_id"),
        "coordinate_system": "assembly_px",
        "source_coordinate_system": "rectified_px",
        "axes": "origin at target board top-left; x right, y down; positive rotation clockwise",
        "rectangle_px": list(rect),
        "rectangle_mm": [v * mm_scale for v in rect] if mm_scale is not None else None,
        "mm_per_assembly_px": mm_scale,
        "execution": "manual",
        "order_strategy": "top-left anchor, then largest total shared edge with placed pieces",
        "warnings": warnings,
        "slot": slots,
    }


def write_guide(
    output: Path,
    plan: dict,
    fragments: list[Fragment],
    manifest: dict,
    pieces_path: Path,
) -> None:
    rows = [
        "# 人工摆放计划",
        "",
        "在空白区域建立目标外框：左上角为原点，x 向右、y 向下。按顺序图及下表逐片摆放。",
        "目标外框的坐标轴与原俯视图坐标轴保持平行；若整体转动外框，表内转角还须叠加该转角。",
        "旋转角以原始俯视照片中该片的朝向为起点；正值顺时针，负值逆时针。先核对取片图中的 S 编号。",
        "原照片中的碎片若已被移动或转动，请重新核对朝向；PNG 的基边方向不是桌面原朝向。",
        "",
    ]
    if plan["rectangle_mm"]:
        rows += [
            f"目标外框约 {plan['rectangle_mm'][0]:.2f} × {plan['rectangle_mm'][1]:.2f} mm；毫米坐标只表示该外框内的相对位置。",
            "",
        ]
    if plan["warnings"]:
        rows += ["待复核事项：", ""] + [f"- {w}" for w in plan["warnings"]] + [""]
    rows += [
        "![取片索引](piece_index.png)",
        "",
        "![目标顺序](placement_order.png)",
        "",
        "| 步骤 | 取片 | 从原桌面朝向旋转 | 目标中心 (px) | 对齐已放碎片 | 复核 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for slot in plan["slot"]:
        angle = slot["rotation_deg"]
        rotation = (
            "缺少原朝向，请看图核对"
            if angle is None
            else f"{'顺时针' if angle >= 0 else '逆时针'} {abs(angle):.2f}°"
        )
        xy = slot["target_center_px"]
        step_label = (
            f"[{slot['step']}]({slot['step_image']})"
            if slot["step_image"]
            else str(slot["step"])
        )
        rows.append(
            f"| {step_label} | {slot['scan_piece_id']} | {rotation} | ({xy[0]:.1f}, {xy[1]:.1f}) | {', '.join(slot['align_with_placed_pieces']) or '建立锚点'} | {'需要' if slot['needs_manual_review'] else '—'} |"
        )
    rows += [
        "",
        "角度来自图像平面内的变换；透视单应时，它只是中心附近的近似朝向。不要用数字拼接中的缩放或颜色校正调整实物片形。",
        "",
        "黑色缺口是未覆盖区域。参考图只用于整图朝向选择、颜色统计和评价，没有拿原图像素填补缺口。",
        "",
    ]
    cells = []
    for index, f in enumerate(fragments, 1):
        rgba = f.rgba
        scale = min(230 / rgba.shape[1], 150 / rgba.shape[0])
        small = cv2.resize(
            rgba,
            (
                max(1, round(rgba.shape[1] * scale)),
                max(1, round(rgba.shape[0] * scale)),
            ),
        )
        cell = np.full((190, 250, 3), 228, np.uint8)
        a = small[:, :, 3:4].astype(float) / 255
        x, y = (250 - small.shape[1]) // 2, 10
        cell[y : y + small.shape[0], x : x + small.shape[1]] = (
            small[:, :, :3] * a + 228 * (1 - a)
        ).astype(np.uint8)
        label(cell, f"{index:02d}: {f.id}", (125, 180))
        cells.append(cell)
    while len(cells) % 4:
        cells.append(np.full_like(cells[0], 255))
    imwrite(
        output / "piece_index.png",
        np.vstack([np.hstack(cells[i : i + 4]) for i in range(0, len(cells), 4)]),
    )
    # Original table positions identify which physical piece to pick.
    capture = manifest.get("inputs", {}).get("pieces_rectified")
    if capture:
        path = Path(capture)
        if not path.is_absolute():
            candidate = pieces_path.parent / path
            path = candidate if candidate.is_file() else ROOT / path
        if path.is_file():
            table = imread(path)
            for f in fragments:
                if f.metadata.get("center_px"):
                    label(
                        table,
                        f.id,
                        tuple(round(v) for v in f.metadata["center_px"]),
                        0.9,
                    )
            imwrite(output / "scan_locations.png", table)
            rows += ["![原桌面取片位置](scan_locations.png)", ""]
    (output / "placement_guide.md").write_text("\n".join(rows), encoding="utf-8")


def assemble(
    pieces_path: Path = DEFAULT_PIECES,
    placements_path: Path = DEFAULT_PLACEMENTS,
    output: Path = DEFAULT_OUTPUT,
    *,
    layout_path: Path | None = None,
    reference_path: Path | None = None,
    config: dict | None = None,
    orientation: str = "auto",
) -> dict:
    pieces_path, placements_path, output = (
        Path(pieces_path),
        Path(placements_path),
        Path(output),
    )
    require(orientation in {"auto", "0", "90", "180", "270"}, "invalid orientation")
    config = resolve_config(config)
    manifest, result, fragments, rect, mode, warnings = load_inputs(
        pieces_path, placements_path, layout_path, config
    )
    reference = imread(reference_path) if reference_path is not None else None
    if mode == "reference_assignments":
        require(
            orientation in {"auto", "0"},
            "assignment targets are already in the reference frame; orientation must be 0",
        )
        if reference is not None:
            require(
                list(reference.shape[1::-1]) == list(rect),
                "reference dimensions differ from layout.source_size_px",
            )
        angle, orientation_report = (
            0,
            {"method": "reference_assignments", "candidates": []},
        )
    else:
        angle, orientation_report = choose_orientation(
            fragments, rect, reference, config, orientation
        )
    global_h, rect = board_rotation(angle, rect)
    for f in fragments:
        f.h = global_h @ f.h
        f.target = transform(global_h, f.target)
    if orientation_report.get("ambiguous"):
        warnings.append(
            "Reference does not confidently resolve global orientation; check placement_order.png."
        )
    size = tuple(math.ceil(v) for v in rect)
    ref_original_size = list(reference.shape[1::-1]) if reference is not None else None
    if reference is not None:
        require(
            abs(
                math.log(
                    (rect[0] / rect[1]) / (reference.shape[1] / reference.shape[0])
                )
            )
            < 0.08,
            "reference / assembly aspect ratios differ; check orientation or reference",
        )
        reference = cv2.resize(
            reference,
            size,
            interpolation=cv2.INTER_AREA
            if reference.shape[1] > size[0]
            else cv2.INTER_LINEAR,
        )
    fragments, edges, components = placement_order(
        fragments, config["adjacency_tolerance_px"]
    )
    if len(components) > 1:
        warnings.append(
            f"Adjacency graph has {len(components)} components; separate anchors are needed."
        )
    # Validate every input before creating or replacing any result file.
    inputs = {pieces_path.resolve(), placements_path.resolve()}
    inputs.update(
        (pieces_path.parent / f.metadata[key]).resolve()
        for f in fragments
        for key in ("rgba_path", "mask_path")
        if f.metadata.get(key)
    )
    inputs.update(
        Path(p).resolve() for p in (reference_path, layout_path) if p is not None
    )
    require(
        not any(p.is_relative_to(output.resolve()) for p in inputs),
        "output directory must not contain input files",
    )
    output.mkdir(parents=True, exist_ok=True)
    # Remove only products managed by this script, so reruns with --no-steps
    # or without a reference cannot display stale results from an earlier run.
    for name in ("comparison.png", "scan_locations.png"):
        (output / name).unlink(missing_ok=True)
    steps_dir = output / "steps"
    if steps_dir.is_dir():
        for path in steps_dir.glob("*.png"):
            if path.stem.isdigit():
                path.unlink()
        if not config["write_steps"] and not any(steps_dir.iterdir()):
            steps_dir.rmdir()
    if config["write_steps"]:
        (output / "steps").mkdir(exist_ok=True)
    acc = np.zeros((size[1], size[0], 3), np.float32)
    raw_acc = np.zeros_like(acc)
    weights = np.zeros((size[1], size[0]), np.float32)
    counts = np.zeros(weights.shape, np.uint16)
    piece_reports = []
    for index, f in enumerate(fragments, 1):
        color, alpha, weight = warp_fragment(f, size, config)
        corrected, correction = color_correct(color, reference, alpha, config)
        support = alpha > 1e-3
        counts += support.astype(np.uint16)
        if config["blend_mode"] == "overwrite":
            acc[support], raw_acc[support], weights[support] = (
                corrected[support],
                color[support],
                1,
            )
        else:
            acc += corrected * weight[:, :, None]
            raw_acc += color * weight[:, :, None]
            weights += weight
        piece_reports.append(
            {
                "scan_piece_id": f.id,
                "source_piece_id": f.source_id,
                "match_confidence": f.confidence,
                "match_status": f.match_status,
                "extraction_quality": f.metadata.get("quality"),
                "covered_pixels": int(support.sum()),
                "color_correction": correction,
            }
        )
        if config["write_steps"]:
            step = finish(acc, weights)
            # Labels are guide annotations; final assembled.png has none.
            for placed in fragments[:index]:
                cv2.polylines(
                    step,
                    [np.rint(placed.target).astype(np.int32)],
                    True,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.polylines(
                step,
                [np.rint(f.target).astype(np.int32)],
                True,
                (40, 255, 40),
                3,
                cv2.LINE_AA,
            )
            c = Polygon(f.target).centroid
            label(step, f"{index:02d}: {f.id}", (round(c.x), round(c.y)))
            imwrite(output / f"steps/{index:02d}.png", step)
    assembled, raw = finish(acc, weights), finish(raw_acc, weights)
    covered = counts > 0
    overview = assembled.copy()
    seams = assembled.copy()
    for index, f in enumerate(fragments, 1):
        contour = np.rint(f.target).astype(np.int32)
        cv2.polylines(overview, [contour], True, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.polylines(seams, [contour], True, (0, 200, 255), 1, cv2.LINE_AA)
        c = Polygon(f.target).centroid
        label(overview, f"{index:02d}: {f.id}", (round(c.x), round(c.y)))
    seams[~covered] = (180, 0, 255)
    seams[counts > 1] = (255, 255, 0)
    for name, image in {
        "assembled.png": assembled,
        "assembled_raw.png": raw,
        "seams.png": seams,
        "coverage_mask.png": covered.astype(np.uint8) * 255,
        "placement_order.png": overview,
    }.items():
        imwrite(output / name, image)
    plan = manual_plan(fragments, edges, components, rect, manifest, mode, warnings)
    if not config["write_steps"]:
        for slot in plan["slot"]:
            slot["step_image"] = None
    save_json(output / "placement_plan.json", plan)
    write_guide(output, plan, fragments, manifest, pieces_path)
    if reference is not None:
        panels = []
        for title, image in (
            ("Reference (resized)", reference),
            ("Raw reconstruction", raw),
            ("Color corrected", assembled),
        ):
            factor = min(1, 720 / image.shape[1])
            small = cv2.resize(
                image,
                (
                    round(image.shape[1] * factor),
                    max(1, round(image.shape[0] * factor)),
                ),
            )
            panel = cv2.copyMakeBorder(
                small, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            label(panel, title, (panel.shape[1] // 2, 28))
            panels.append(panel)
        imwrite(output / "comparison.png", np.hstack(panels))
    report = {
        "schema_version": "1.0",
        "algorithm_version": VERSION,
        "session_id": manifest.get("session_id"),
        "input_mode": mode,
        "status": "review" if warnings else "complete",
        "warnings": warnings,
        "piece_count": len(fragments),
        "coordinate_system": "assembly_px",
        "rectangle_px": list(rect),
        "canvas_size_px": list(size),
        "board_rotation_clockwise_deg": angle,
        "H_solver_to_assembly": global_h.tolist(),
        "orientation": orientation_report,
        "reference_used": {
            "pixels_for_gap_filling": False,
            "global_orientation": orientation_report["method"]
            == "reference_gray_correlation",
            "color_statistics": reference is not None and config["color_correction"],
            "evaluation": reference is not None,
            "original_size_px": ref_original_size,
            "resized_size_px": list(size) if reference is not None else None,
        },
        "inputs": {
            "pieces_sha256": hashlib.sha256(pieces_path.read_bytes()).hexdigest(),
            "placements_sha256": hashlib.sha256(
                placements_path.read_bytes()
            ).hexdigest(),
        },
        "geometry": geometry_report(fragments, rect),
        "raster": {
            "covered_fraction": float(covered.mean()),
            "uncovered_pixels": int((~covered).sum()),
            "overlap_pixels": int((counts > 1).sum()),
        },
        "metrics": {
            "raw": image_metrics(raw, reference, covered, config["evaluate_ssim"]),
            "corrected": image_metrics(
                assembled, reference, covered, config["evaluate_ssim"]
            ),
        },
        "seams": seam_report(
            assembled,
            reference,
            fragments,
            edges,
            covered,
            config["adjacency_tolerance_px"],
        ),
        "seam_legend": {
            "yellow": "polygon boundary",
            "magenta": "uncovered pixel",
            "cyan": "overlapping textures",
        },
        "upstream_geometry_quality": result.get("quality"),
        "upstream_verification": result.get("verification"),
        "pieces": piece_reports,
        "config": config,
    }
    save_json(output / "assembly_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pieces",
        type=Path,
        default=DEFAULT_PIECES,
        help="Stage 2 full pieces.json (paths relative to this file)",
    )
    parser.add_argument(
        "--placements",
        "--assignments",
        dest="placements",
        type=Path,
        default=DEFAULT_PLACEMENTS,
        help="Stage 3 placements or assignments JSON",
    )
    parser.add_argument(
        "--layout", type=Path, help="Stage 1 layout, required for assignments"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="optional reference for orientation, conservative color correction, SSIM/PSNR",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--orientation",
        choices=["auto", "0", "90", "180", "270"],
        default="auto",
        help="global clockwise turn; auto uses reference if provided",
    )
    parser.add_argument("--no-color-correction", action="store_true")
    parser.add_argument("--no-steps", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config) if args.config else {}
        if args.no_color_correction:
            config["color_correction"] = False
        if args.no_steps:
            config["write_steps"] = False
        report = assemble(
            args.pieces,
            args.placements,
            args.output,
            layout_path=args.layout,
            reference_path=args.reference,
            config=config,
            orientation=args.orientation,
        )
        print(
            f"Assembled {report['piece_count']} pieces: {report['canvas_size_px']} px; status={report['status']}"
        )
        print(
            f"Geometry coverage: {report['geometry']['coverage_fraction']:.6%}; raster coverage: {report['raster']['covered_fraction']:.6%}"
        )
        print(f"Output: {args.output}")
        for warning in report["warnings"]:
            print(f"Review: {warning}")
        return (
            1 if report["warnings"] and report["config"]["fail_on_manual_review"] else 0
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        cv2.error,
        np.linalg.LinAlgError,
    ) as exc:
        print(f"Stage 4 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
