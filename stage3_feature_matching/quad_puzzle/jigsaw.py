#!/usr/bin/env python3
"""Single-file convex polygon puzzle assembler and step2 JSON adapter.

Input: polygon-assembly/v1, legacy quad-assembly/v1, or stage2 export JSON.
Output: clockwise degrees and the input P0's final canvas position.
Run python jigsaw.py --help. Public functions: solve, verify, validate_input,
apply_placement, normalize_input, evaluate, read_json, write_json. Best-effort
optimization also uses SciPy. No image, cutting history or original layout is
read by solving. The default returns the best complete rigid layout found,
including its measured gaps, overlaps and outside area.

See README.md for input/output fields, commands and the measured-data example.
"""
from __future__ import annotations

import argparse
import bisect
from itertools import combinations, islice
import json
import math
from numbers import Real
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

# Geometry

class ProtocolError(ValueError):
    """Input does not satisfy a supported convex polygon protocol."""

def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ProtocolError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"{name} must be finite")
    return number

def _point(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ProtocolError(f"{name} must contain exactly two coordinates")
    return [_number(value[0], name + "[0]"), _number(value[1], name + "[1]")]

def signed_area(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=float)
    p = p - p[0]
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                               - p[:, 1] * np.roll(p[:, 0], -1)))

def area(points: np.ndarray) -> float:
    return abs(signed_area(points))

def perimeter(points: np.ndarray) -> float:
    edges = np.roll(points, -1, axis=0) - points
    return float(np.sum(np.hypot(edges[:, 0], edges[:, 1])))

def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def decode_piece(piece: dict[str, Any]) -> np.ndarray:
    if not isinstance(piece, dict):
        raise ProtocolError("each piece must be an object")
    if not isinstance(piece.get("id"), str) or not piece["id"]:
        raise ProtocolError("piece id must be a nonempty string")
    for key in ("baseLength",):
        if key not in piece:
            raise ProtocolError(f"piece {piece['id']} is missing {key}")
    length = _number(piece["baseLength"], "baseLength")
    if length <= 0:
        raise ProtocolError("baseLength must be positive")
    if "remainingVertices" in piece:
        remaining = piece["remainingVertices"]
        if not isinstance(remaining, list) or not remaining:
            raise ProtocolError("remainingVertices must contain P2 through P(n-1), with n >= 3")
        rest = [_point(point, f"remainingVertices[{i}]") for i, point in enumerate(remaining)]
    else:
        rest = [_point(piece.get("p2"), "p2"), _point(piece.get("p3"), "p3")]
    p = np.array([[0.0, 0.0], [length, 0.0], *rest])
    scale = float(np.max(np.abs(p)))
    scale2 = scale * scale
    if not math.isfinite(scale2) or scale2 == 0:
        raise ProtocolError("coordinate magnitudes exceed supported float range")
    edges = np.roll(p, -1, axis=0) - p
    turns = cross2(edges, np.roll(edges, -1, axis=0))
    eps = 64 * np.finfo(float).eps * scale2
    if not np.isfinite(turns).all() or not np.all(turns > eps):
        raise ProtocolError("vertices must form a strictly convex clockwise polygon "
                            "in image coordinates; concave, crossed and degenerate "
                            "pieces are not supported by this protocol version")
    # Positive local turns alone accept a self-intersecting pentagram. Every
    # nonincident vertex must be strictly inside every directed edge half-plane.
    for i, edge in enumerate(edges):
        other = [j for j in range(len(p)) if j not in (i, (i + 1) % len(p))]
        sides = cross2(edge, p[other] - p[i])
        if not np.isfinite(sides).all() or not np.all(sides > eps):
            raise ProtocolError("vertices must describe a simple strictly convex clockwise polygon")
    if signed_area(p) <= eps:
        raise ProtocolError("piece area is zero or numerically degenerate")
    lengths = np.linalg.norm(edges, axis=1)
    if float(np.max(lengths)) > length * (1.0 + BASE_LENGTH_RTOL):
        raise ProtocolError(f"piece {piece['id']}: baseLength must be a longest edge (relative tolerance {BASE_LENGTH_RTOL})")
    return p

def extract_pieces(puzzle: Any) -> list[dict[str, Any]]:
    pieces = puzzle.get("pieces") if isinstance(puzzle, dict) else puzzle
    if not isinstance(pieces, list) or not pieces:
        raise ProtocolError("pieces must be a nonempty list")
    return pieces

def validate_pieces(puzzle: Any) -> list[np.ndarray]:
    pieces = extract_pieces(puzzle)
    polygons = []
    ids = set()
    for piece in pieces:
        polygon = decode_piece(piece)
        if piece["id"] in ids:
            raise ProtocolError(f"duplicate piece id: {piece['id']}")
        ids.add(piece["id"])
        polygons.append(polygon)
    return polygons

def _transform_radians(points: np.ndarray, tx: float, ty: float, rotation: float) -> np.ndarray:
    c, s = math.cos(rotation), math.sin(rotation)
    matrix = np.array([[c, -s], [s, c]])
    return np.asarray(points, dtype=float) @ matrix.T + np.array([tx, ty])

def _placement_pose(placement: dict[str, Any]) -> tuple[float, float, float]:
    if not isinstance(placement, dict):
        raise ProtocolError("each placement must be an object")
    for prohibited in ("scale", "scaleX", "scaleY", "reflection", "flip", "flipped",
                       "matrix", "transform"):
        if prohibited in placement:
            raise ProtocolError(f"placement field {prohibited!r} is unsupported; "
                                "only translation and rotation are allowed")
    x = placement.get("tx", placement.get("x"))
    y = placement.get("ty", placement.get("y"))
    return (_number(x, "tx"), _number(y, "ty"),
            _number(placement.get("rotation"), "rotation"))

def _verify_geometry(puzzle: Any, solution: dict[str, Any],
                    rtol: float = 1e-7) -> dict[str, Any]:
    """Verify a proposed rigid assembly without any knowledge of the cut.

    A numerical solution is accepted when normalized area and distance errors
    are below rtol. This is a tolerance-based certificate, not exact arithmetic.
    Returns valid/errors/metrics and never silently treats area alone as proof.
    """
    errors: list[str] = []
    metrics: dict[str, Any] = {}
    try:
        if not isinstance(rtol, Real) or not 0 < float(rtol) < 1:
            raise ProtocolError("rtol must be strictly between zero and one")
        pieces = extract_pieces(puzzle)
        local = validate_pieces(pieces)
        if not isinstance(solution, dict):
            raise ProtocolError("solution must be an object")
        frame = solution.get("rectangle")
        if not isinstance(frame, dict):
            raise ProtocolError("solution must contain rectangle dimensions")
        width = _number(frame.get("width"), "width")
        height = _number(frame.get("height"), "height")
        if width <= 0 or height <= 0:
            raise ProtocolError("rectangle dimensions must be positive")
        if not math.isfinite(width * height) or width * height == 0:
            raise ProtocolError("rectangle area exceeds supported float range")
        total_perimeter = sum(perimeter(p) for p in local)
        if width + height > 0.5 * total_perimeter * (1 + float(rtol)):
            raise ProtocolError("rectangle perimeter exceeds combined fragment perimeter")
        placements = solution.get("placements")
        if not isinstance(placements, list):
            raise ProtocolError("placements must be a list")
        expected = [p["id"] for p in pieces]
        actual = [p.get("id") if isinstance(p, dict) else None for p in placements]
        if len(actual) != len(expected) or any(not isinstance(i, str) for i in actual):
            raise ProtocolError("placements must contain each input id exactly once")
        if len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise ProtocolError("placements must contain each input id exactly once")
        placement_by_id = dict(zip(actual, placements))
        placed = []
        for piece, polygon in zip(pieces, local):
            pose = _placement_pose(placement_by_id[piece["id"]])
            transformed = _transform_radians(polygon, *pose)
            if not np.isfinite(transformed).all():
                raise ProtocolError("transformed coordinates must be finite")
            placed.append(transformed)
    except (ProtocolError, OverflowError, ValueError, TypeError) as exc:
        return {"valid": False, "errors": [str(exc)], "metrics": metrics}

    from shapely import set_precision
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    # All checks operate in an O(1) normalized frame so small/large units do
    # not change the meaning of rtol. Original areas are reported afterward.
    scale = max(width, height,
                *(float(np.max(np.abs(p))) for p in local))
    # GEOS floating overlay can be unstable on almost-collinear shared edges.
    # Use a grid at least 1000 times finer than the acceptance tolerance;
    # raw vertex containment and raw input areas are still checked separately.
    total_area = sum(area(p / scale) for p in local)
    raw_target_area = (width / scale) * (height / scale)
    reference_area = max(raw_target_area, total_area, np.finfo(float).tiny)
    precision_grid = min(1e-12, float(rtol) * 1e-3 * reference_area
                         / max(total_perimeter / scale, 1.0))
    normalized = [set_precision(Polygon(p / scale), precision_grid) for p in placed]
    frame_polygon = set_precision(box(0.0, 0.0, width / scale, height / scale),
                                  precision_grid)
    target_area = frame_polygon.area
    union = unary_union(normalized)
    overlaps = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            overlap = normalized[i].intersection(normalized[j]).area
            if overlap > 0:
                overlaps.append((pieces[i]["id"], pieces[j]["id"], overlap))
    overlap_sum = sum(a for _, _, a in overlaps)
    maximum_overlap = max((a for _, _, a in overlaps), default=0.0)
    missing = frame_polygon.difference(union).area
    outside = union.difference(frame_polygon).area
    symmetric_difference = union.symmetric_difference(frame_polygon).area
    area_error = abs(total_area - target_area)
    # Vertex containment detects an extremely thin protrusion that an area-only
    # tolerance could otherwise miss.
    max_outside_distance = 0.0
    for p in placed:
        q = p / scale
        violations = np.maximum.reduce([
            -q[:, 0], q[:, 0] - width / scale,
            -q[:, 1], q[:, 1] - height / scale,
            np.zeros(len(p))])
        max_outside_distance = max(max_outside_distance, float(violations.max()))
    components = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
    hole_count = sum(len(p.interiors) for p in components if p.geom_type == "Polygon")
    checks = {
        "input_area_error": area_error,
        "pairwise_overlap_sum": overlap_sum,
        "missing_area": missing,
        "outside_area": outside,
        "symmetric_difference_area": symmetric_difference,
    }
    for key, value in checks.items():
        if value > float(rtol) * reference_area:
            errors.append(f"{key} exceeds tolerance")
    if max_outside_distance > float(rtol):
        errors.append("outside_vertex_distance exceeds tolerance")
    if isinstance(puzzle, dict) and puzzle.get("target") is not None:
        target = puzzle["target"]
        try:
            tw = _number(target["width"], "target.width")
            th = _number(target["height"], "target.height")
            if abs(width - tw) > rtol * max(width, tw) or abs(height - th) > rtol * max(height, th):
                errors.append("output rectangle does not match requested target")
        except (KeyError, TypeError, ProtocolError) as exc:
            errors.append(f"invalid target: {exc}")
    unit_area = scale * scale
    metrics = {
        "piece_count": len(pieces),
        "rectangle_width": width,
        "rectangle_height": height,
        "rectangle_area": width * height,
        "input_area_sum": total_area * unit_area,
        "union_area": union.area * unit_area,
        **{key: value * unit_area for key, value in checks.items()},
        "maximum_pair_overlap": maximum_overlap * unit_area,
        "outside_vertex_distance": max_outside_distance * scale,
        "hole_count": hole_count,
        "union_component_count": len(components),
        "relative_area_error": area_error / reference_area,
        "relative_overlap_sum": overlap_sum / reference_area,
        "relative_missing_area": missing / reference_area,
        "relative_outside_area": outside / reference_area,
        "relative_symmetric_difference": symmetric_difference / reference_area,
        "relative_outside_vertex_distance": max_outside_distance,
        "rtol": float(rtol),
        "overlay_precision_grid_relative": precision_grid,
    }
    return {"valid": not errors, "errors": errors, "metrics": metrics}

# Frame Candidates

_MAX_RAW_SUMS = 2_000_000

_MAX_BATCH_SUMS = 65_536

_FLOAT_EQUIV = 64.0 * sys.float_info.epsilon

def _minimum_width(points):
    result = math.inf
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length <= 0.0:
            raise ValueError("polygons must have nonzero edges")
        normal = np.array([-edge[1], edge[0]]) / length
        result = min(result, float(np.ptp(points @ normal)))
    return result

def _edge_sum_batches(edge_lengths, terms):
    """Enumerate distinct-piece edge sums using bounded variable-size batches.

    Grouping by actual vertex counts keeps the fast quad path without padding
    triangles/pentagons with fictitious edges. Large Cartesian products are
    traversed in mixed-radix chunks, never allocated as one huge array.
    """
    piece_sets = combinations(range(len(edge_lengths)), terms)
    while True:
        chosen = list(islice(piece_sets, 256))
        if not chosen:
            return
        groups = {}
        for indices in chosen:
            counts = tuple(len(edge_lengths[i]) for i in indices)
            groups.setdefault(counts, []).append(indices)
        for counts, index_sets in groups.items():
            product = math.prod(counts)
            if product <= _MAX_BATCH_SUMS:
                batch_size = max(1, _MAX_BATCH_SUMS // product)
                for start in range(0, len(index_sets), batch_size):
                    batch = index_sets[start:start + batch_size]
                    values = np.stack([edge_lengths[indices[0]] for indices in batch])
                    for column in range(1, terms):
                        following = np.stack([edge_lengths[indices[column]] for indices in batch])
                        values = (values[:, :, None] + following[:, None, :]).reshape(len(batch), -1)
                    yield values.reshape(-1)
            else:
                for indices in index_sets:
                    for start in range(0, product, _MAX_BATCH_SUMS):
                        numbers = np.arange(start, min(start + _MAX_BATCH_SUMS, product), dtype=np.int64)
                        values = np.zeros(len(numbers))
                        for index in reversed(indices):
                            count = len(edge_lengths[index])
                            values += edge_lengths[index][numbers % count]
                            numbers //= count
                        yield values

def frame_candidates(polygons, deadline, max_candidates=192, max_terms=4):
    """Return ``([(width, height), ...], stats)`` in normalized units.

    Complete enumeration means complete only within ``max_terms`` and the
    numerical input, not a complete rectangular-tiling algorithm. Limits or
    candidate truncation must never be interpreted as infeasibility.

    Cooperative deadline checks bracket bounded NumPy batches and sorting.
    A single NumPy sort cannot be preempted; a small wall-time overrun remains
    possible. Memory is bounded by at most two million generated raw sums.
    """
    started = time.monotonic()
    if not isinstance(max_terms, int) or not 1 <= max_terms <= 8:
        raise ValueError("max_terms must be an integer from 1 to 8")
    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if math.isnan(float(deadline)):
        raise ValueError("deadline must not be NaN")
    reasons = []
    stats = {
        "boundary_subset_max_terms": max_terms,
        "boundary_sums_enumerated": 0,
        "boundary_unique_sums": 0,
        "boundary_enumeration_complete": False,
        "boundary_enumeration_truncated": False,
        "frame_candidate_count": 0,
        "frame_enumeration_truncated": False,
        "dimension_search_is_heuristic": True,
        "candidate_reciprocal_errors": [],
        "strong_frame_count": 0,
        "strong_frame_error_threshold": 1e-12,
        "boundary_sum_count_method": "one_edge_per_distinct_piece",
        "boundary_sum_storage_limit": _MAX_RAW_SUMS,
        "sum_deduplication": "exact_float_values_only",
        "frame_deduplication_relative_tolerance": _FLOAT_EQUIV,
    }

    def finish(frames):
        stats["frame_candidate_count"] = len(frames)
        stats["frame_enumeration_truncated"] = bool(reasons)
        stats["truncation_reasons"] = list(dict.fromkeys(reasons))
        stats["frame_generation_seconds"] = time.monotonic() - started
        return frames, stats

    if time.monotonic() >= deadline:
        reasons.append("deadline_before_enumeration")
        stats["boundary_enumeration_truncated"] = True
        return finish([])
    if not polygons:
        return finish([])
    arrays = [np.asarray(p, dtype=float) for p in polygons]
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all() for p in arrays):
        raise ValueError("frame_candidates expects finite polygon arrays with at least three vertices")
    lower = max(_minimum_width(p) for p in arrays)
    if not math.isfinite(lower) or lower <= 0.0:
        raise ValueError("polygons must have positive minimum width")
    upper = 1.0 / lower
    edge_lengths = [
        np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        for p in arrays
    ]
    # Leave a fraction of the available time for sorting, reciprocal lookup,
    # and ranking. This does not extend the caller's deadline.
    available = max(0.0, float(deadline) - time.monotonic())
    reserve = min(0.6, available * 0.35)
    enumerate_until = float(deadline) - reserve
    value_blocks, term_blocks = [], []
    enumerated = 0
    enumeration_complete = True
    for terms in range(1, min(max_terms, len(arrays)) + 1):
        for flat in _edge_sum_batches(edge_lengths, terms):
            if time.monotonic() >= enumerate_until:
                reasons.append("boundary_enumeration_deadline")
                enumeration_complete = False
                break
            room = _MAX_RAW_SUMS - enumerated
            if room < len(flat):
                flat = flat[:room]
                reasons.append("boundary_sum_storage_limit")
                enumeration_complete = False
            enumerated += len(flat)
            inside = flat[(flat >= lower - 1e-8) & (flat <= upper + 1e-8)]
            value_blocks.append(inside)
            term_blocks.append(np.full(len(inside), terms, dtype=np.int8))
            if not enumeration_complete:
                break
        if not enumeration_complete:
            break
    stats["boundary_sums_enumerated"] = enumerated
    stats["boundary_enumeration_complete"] = enumeration_complete
    stats["boundary_enumeration_truncated"] = not enumeration_complete
    if not value_blocks or not any(len(v) for v in value_blocks):
        return finish([])
    if time.monotonic() >= deadline:
        reasons.append("deadline_before_ranking")
        return finish([])

    # Do not round sums to decimal buckets: close but unequal sums must remain
    # available for reciprocal matching. Exact duplicates retain multiplicity.
    values = np.concatenate(value_blocks)
    terms = np.concatenate(term_blocks)
    del value_blocks, term_blocks
    values, first, counts = np.unique(values, return_index=True, return_counts=True)
    terms = terms[first]
    stats["boundary_unique_sums"] = len(values)
    if time.monotonic() >= deadline:
        reasons.append("deadline_during_sum_sort")
        return finish([])
    reciprocal = 1.0 / values
    positions = np.searchsorted(values, reciprocal)
    left = np.clip(positions - 1, 0, len(values) - 1)
    right = np.clip(positions, 0, len(values) - 1)
    nearest = np.where(np.abs(values[left] - reciprocal)
                       <= np.abs(values[right] - reciprocal), left, right)
    errors = np.abs(values[nearest] - reciprocal)
    valid = (reciprocal >= lower - 1e-8) & (reciprocal <= upper + 1e-8)
    width = np.maximum(values, reciprocal)[valid]
    height = np.minimum(values, reciprocal)[valid]
    score_terms = (terms.astype(np.int16) + terms[nearest])[valid]
    frequencies = counts[valid]
    errors = errors[valid]
    if lower <= 1.0 + 1e-8:
        square_error = float(np.min(np.abs(values - 1.0)))
        width = np.append(width, 1.0)
        height = np.append(height, 1.0)
        errors = np.append(errors, square_error)
        score_terms = np.append(score_terms, 0)
        frequencies = np.append(frequencies, 0)
    if not len(width):
        return finish([])
    if time.monotonic() >= deadline:
        reasons.append("deadline_before_candidate_sort")
        return finish([])
    # The old binary error<=1e-7 component is redundant when actual error is
    # retained immediately afterward. Ascending actual error is equivalent.
    order = np.lexsort((-frequencies, np.abs(np.log(width)), score_terms, errors))
    if time.monotonic() >= deadline:
        reasons.append("deadline_during_candidate_sort")
        return finish([])

    frames, chosen_widths, chosen_errors = [], [], []
    for counter, index in enumerate(order):
        if counter % 256 == 0 and time.monotonic() >= deadline:
            reasons.append("candidate_selection_deadline")
            break
        w = float(width[index])
        position = bisect.bisect_left(chosen_widths, w)
        adjacent = chosen_widths[max(0, position - 1):position + 1]
        if any(abs(w - old) <= _FLOAT_EQUIV * max(1.0, abs(w), abs(old))
               for old in adjacent):
            continue
        if len(frames) >= max_candidates:
            reasons.append("frame_candidate_limit")
            break
        bisect.insort(chosen_widths, w)
        frames.append((w, float(height[index])))
        chosen_errors.append(float(errors[index]))
    stats["candidate_reciprocal_errors"] = chosen_errors
    stats["strong_frame_count"] = sum(e <= 1e-12 for e in chosen_errors)
    return finish(frames)

_frame_candidates = frame_candidates

# Solver

class _BudgetExceeded(Exception):
    pass

def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])

def _target_dimensions(target, total_area):
    if target is None:
        return None
    if not isinstance(target, dict):
        raise ProtocolError("target must be an object with width and height")
    values = []
    for name in ("width", "height"):
        value = target.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ProtocolError("target dimensions must be finite positive numbers")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ProtocolError("target dimensions must be finite positive numbers")
        values.append(value)
    return tuple(values)

class _CornerSearch:
    def __init__(self, polygons, ids, width, height, deadline):
        from shapely.geometry import Polygon, box
        from shapely import set_precision
        self.Polygon = Polygon
        self.set_precision = set_precision
        self.polygons = polygons
        self.ids = ids
        self.width = width
        self.height = height
        self.deadline = deadline
        self.grid = 1e-12
        self.area_tolerance = 1e-9
        self.length_tolerance = 1e-8
        self.frame = set_precision(box(0.0, 0.0, width, height), self.grid)
        self.areas = [area(p) for p in polygons]
        self.nodes = 0
        self.candidate_tests = 0
        self.failed_states: set[tuple] = set()
        self.done = False
        self.best_placed = 0
        self.node_stop = 0
        self.angles = []
        for p in polygons:
            angles = []
            for k in range(len(p)):
                a = p[(k + 1) % len(p)] - p[k]
                b = p[(k - 1) % len(p)] - p[k]
                cosine = float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
                angles.append(math.acos(np.clip(cosine, -1.0, 1.0)))
            self.angles.append(angles)

    def _check_budget(self):
        if time.monotonic() >= self.deadline or self.nodes >= self.node_stop:
            raise _BudgetExceeded

    def _corners(self, free):
        from shapely.geometry.polygon import orient
        components = [free] if free.geom_type == "Polygon" else list(getattr(free, "geoms", []))
        corners = []
        for component in components:
            if component.geom_type != "Polygon" or component.area <= self.area_tolerance:
                continue
            ring = np.array(orient(component, sign=1.0).exterior.coords[:-1])
            if len(ring) < 3:
                continue
            # Overlay rounding can leave a long, sub-tolerance-width hairpin:
            # B -> C -> D with B almost equal to D. Its apex C is not a usable
            # corner of the remaining material. Simplify ONLY the temporary
            # corner ring; keep the actual free geometry and input unchanged.
            # Delete one vertex, then recompute neighbours, to retain genuine
            # corners and avoid the old simultaneous near-point deletion bug.
            changed = True
            while changed and len(ring) >= 3:
                changed = False
                for k in range(len(ring)):
                    if np.linalg.norm(ring[k] - ring[k - 1]) <= self.length_tolerance:
                        ring = np.delete(ring, k, axis=0)
                        changed = True
                        break
                    if np.linalg.norm(ring[(k + 1) % len(ring)] - ring[k - 1]) <= self.length_tolerance:
                        ring = np.delete(ring, k, axis=0)
                        changed = True
                        break
                if changed:
                    continue
                for k, point in enumerate(ring):
                    incoming = point - ring[(k - 1) % len(ring)]
                    outgoing = ring[(k + 1) % len(ring)] - point
                    li, lo = np.linalg.norm(incoming), np.linalg.norm(outgoing)
                    if (len(ring) > 3 and li > self.length_tolerance
                        and lo > self.length_tolerance
                        and abs(_cross(incoming, outgoing)) <= 1e-8 * li * lo
                        and np.dot(incoming, outgoing) > 0):
                        ring = np.delete(ring, k, axis=0)
                        changed = True
                        break
            if len(ring) < 3:
                continue
            for k, point in enumerate(ring):
                following = ring[(k + 1) % len(ring)] - point
                previous = ring[(k - 1) % len(ring)] - point
                la, lb = np.linalg.norm(following), np.linalg.norm(previous)
                if la <= self.length_tolerance or lb <= self.length_tolerance:
                    continue
                if _cross(following, previous) <= 1e-8 * la * lb:
                    continue
                cosine = float(np.dot(following, previous) / la / lb)
                angle = math.acos(np.clip(cosine, -1.0, 1.0))
                corners.append((angle, point, following / la))
        return sorted(corners, key=lambda x: (round(x[0], 7), x[1][1], x[1][0]))

    def _candidates(self, free, remaining, corner):
        angle, anchor, direction = corner
        candidates = []
        occupied_keys = set()
        minx, miny, maxx, maxy = free.bounds
        for i in remaining:
            p = self.polygons[i]
            for k in range(len(p)):
                if self.angles[i][k] > angle + 1e-7:
                    continue
                edge = p[(k + 1) % len(p)] - p[k]
                edge = edge / np.linalg.norm(edge)
                cosine = float(np.dot(edge, direction))
                sine = _cross(edge, direction)
                matrix = np.array([[cosine, -sine], [sine, cosine]])
                translation = anchor - matrix @ p[k]
                placed = p @ matrix.T + translation
                if (placed[:, 0].min() < minx - self.length_tolerance
                    or placed[:, 0].max() > maxx + self.length_tolerance
                    or placed[:, 1].min() < miny - self.length_tolerance
                    or placed[:, 1].max() > maxy + self.length_tolerance):
                    continue
                key = (i, tuple(sorted((float(x), float(y)) for x, y in placed)))
                if key in occupied_keys:
                    continue
                occupied_keys.add(key)
                polygon = self.Polygon(placed)
                self.candidate_tests += 1
                if polygon.difference(free).area > self.area_tolerance:
                    continue
                snap = self.set_precision(polygon, self.grid)
                if snap.is_empty or snap.geom_type != "Polygon":
                    continue
                pose = {"id": self.ids[i], "tx": float(translation[0]),
                        "ty": float(translation[1]), "rotation": math.atan2(sine, cosine)}
                # First close a corner, then prefer a larger tile.
                score = (angle - self.angles[i][k], -self.areas[i])
                candidates.append((score, i, snap, pose))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def _dfs(self, free, remaining, placements):
        self._check_budget()
        self.nodes += 1
        self.best_placed = max(self.best_placed, len(placements))
        if not remaining:
            return placements if free.area <= 5 * self.area_tolerance else None
        if free.is_empty or free.area <= self.area_tolerance:
            return None
        state = (tuple(remaining), free.normalize().wkb)
        if state in self.failed_states:
            return None
        corners = self._corners(free)
        if not corners:
            return None
        # Any convex free corner can be filled; choosing the smallest angle is
        # a branching heuristic, not a requirement that one tile fill it all.
        candidates = self._candidates(free, remaining, corners[0])
        for _, i, polygon, pose in candidates:
            self._check_budget()
            next_free = free.difference(polygon)
            result = self._dfs(next_free, tuple(j for j in remaining if j != i), placements + [pose])
            if result is not None:
                return result
        self.failed_states.add(state)
        return None

    def run(self, additional_nodes):
        self.node_stop = self.nodes + additional_nodes
        try:
            result = self._dfs(self.frame, tuple(range(len(self.polygons))), [])
            self.done = True
            return result
        except _BudgetExceeded:
            return None

def _heuristic(pieces, polygons, target, time_limit, max_boundary_edges=4):
    started = time.monotonic()
    deadline = started + time_limit
    total_area = sum(area(p) for p in polygons)
    scale = math.sqrt(total_area)
    normalized = [p / scale for p in polygons]
    if target is not None:
        frames = [(target[0] / scale, target[1] / scale)]
        dimension_stats = {"dimension_search_is_heuristic": False, "frame_candidate_count": 1}
    else:
        frames, dimension_stats = _frame_candidates(
            normalized, min(deadline, started + min(2.0, time_limit * 0.2)),
            max_terms=max_boundary_edges)
    ids = [p["id"] for p in pieces]
    searches = [_CornerSearch(normalized, ids, w, h, deadline) for w, h in frames]
    solution = None
    candidate_index = None
    # Spend the first substantial search budget on the strongest geometric
    # candidates. Approximate matches must not starve an almost-exact frame.
    # This is ordering only: weaker candidates remain in the later rounds,
    # and every stage shares the same total deadline.
    priority_count = (1 if target is not None else
                      max(1, min(4, dimension_stats.get("strong_frame_count", 0))))
    priority_count = min(priority_count, len(searches))
    stages = [(32768, range(priority_count)),
              (4096, range(min(4, len(searches))))]
    stages.extend((budget, range(len(searches)))
                  for budget in (64, 512, 4096, 32768, 262144))
    for budget, indices in stages:
        for index in indices:
            search = searches[index]
            if time.monotonic() >= deadline:
                break
            if search.done:
                continue
            placements = search.run(budget)
            if placements is not None:
                for pose in placements:
                    pose["tx"] *= scale
                    pose["ty"] *= scale
                solution = {"status": "feasible",
                            "rectangle": {"width": search.width * scale, "height": search.height * scale},
                            "placements": placements}
                candidate_index = index
                break
        if solution is not None or all(s.done for s in searches) or time.monotonic() >= deadline:
            break
    stats = {"backend": "corner_search", "seconds": time.monotonic() - started,
             "frame_schedule": "strong_frames_32768_then_progressive_rounds",
             "priority_frame_count": priority_count,
             "nodes": sum(s.nodes for s in searches),
             "candidate_tests": sum(s.candidate_tests for s in searches),
             "best_piece_count": max((s.best_placed for s in searches), default=0),
             "candidate_frame_index": candidate_index,
             "completed_frame_searches": sum(s.done for s in searches),
             **dimension_stats}
    if solution is None:
        return {"status": "unknown", "reason": "time_limit" if time.monotonic() >= deadline else "heuristic_candidates_exhausted",
                "stats": stats}
    solution["stats"] = stats
    return solution

# ---------------------------------------------------------------------------
# Public JSON protocol, degree-based poses, file IO and command-line entry point
# ---------------------------------------------------------------------------

VERSION = "3.0.1"
INPUT_SCHEMA = "polygon-assembly/v1"
LEGACY_INPUT_SCHEMA = "quad-assembly/v1"
OUTPUT_SCHEMA = "polygon-assembly-result/v1"
LEGACY_OUTPUT_SCHEMA = "quad-assembly-result/v1"
BEST_EFFORT_OUTPUT_SCHEMA = "polygon-assembly-result/v2"
COORDINATE_SYSTEM = "image"
DEFAULT_RTOL = 1e-7
BASE_LENGTH_RTOL = 1e-10
_OUTPUT_FIELDS = {"schema", "status", "unit", "coordinateSystem", "rectangle",
                  "placements", "reason", "verification", "stats"}
_STATUS_CODES = {"feasible": 0, "approximate": 0, "unknown": 2, "infeasible": 3,
                 "invalid_input": 4, "error": 5}


class DependencyError(RuntimeError):
    """A runtime dependency is missing."""


def _require_numpy():
    if np is None:
        raise DependencyError("NumPy is required; run: python -m pip install numpy==2.3.5 shapely==2.2.0 scipy==1.17.0")


def _fields(obj, required, optional, name):
    if not isinstance(obj, dict):
        raise ProtocolError(f"{name} must be an object")
    missing = set(required) - obj.keys()
    extra = obj.keys() - set(required) - set(optional)
    if missing:
        raise ProtocolError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ProtocolError(f"{name} has unsupported fields: {', '.join(sorted(map(str, extra)))}")


def _json_point(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ProtocolError(f"{name} must be a JSON array of exactly two finite numbers")
    return [_number(value[0], name + "[0]"), _number(value[1], name + "[1]")]


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a nonempty string")
    return value


def _unit_or_none(puzzle):
    value = puzzle.get("unit") if isinstance(puzzle, dict) else None
    if (isinstance(puzzle, dict) and "schema" not in puzzle
        and puzzle.get("coordinate_system") == "rectified_px"):
        value = "rectified_px"
    return value if isinstance(value, str) and value.strip() else None


def _canonical_piece(piece):
    """Copy a public piece without changing its original local frame."""
    if isinstance(piece, dict) and "base_edge" in piece:
        record = piece["base_edge"]
        if not isinstance(record, dict) or piece.get("piece_id") != record.get("id"):
            raise ProtocolError("piece_id must match base_edge.id")
        piece = record
    if isinstance(piece, dict) and "remainingVertices" in piece:
        _fields(piece, {"id", "baseLength", "remainingVertices"}, set(), "piece")
        if not isinstance(piece["remainingVertices"], list) or not piece["remainingVertices"]:
            raise ProtocolError("remainingVertices must be a nonempty array of P2 through P(n-1)")
        points = [_json_point(point, f"remainingVertices[{i}]")
                  for i, point in enumerate(piece["remainingVertices"])]
    else:
        _fields(piece, {"id", "baseLength", "p2", "p3"}, set(), "piece")
        points = [_json_point(piece["p2"], "p2"), _json_point(piece["p3"], "p3")]
    return {"id": _text(piece["id"], "piece.id"),
            "baseLength": _number(piece["baseLength"], "baseLength"),
            "remainingVertices": points}


def _adapt_input(puzzle):
    if not isinstance(puzzle, dict):
        raise ProtocolError("input must be a JSON object")
    if "schema" in puzzle:
        _fields(puzzle, {"schema", "unit", "pieces"}, {"target"}, "input")
        schema = puzzle["schema"]
        if schema not in (INPUT_SCHEMA, LEGACY_INPUT_SCHEMA):
            raise ProtocolError(f"schema must be {INPUT_SCHEMA!r} or {LEGACY_INPUT_SCHEMA!r}")
        unit = _text(puzzle["unit"], "unit")
        source_pieces = extract_pieces(puzzle)
        expected = ({"id", "baseLength", "remainingVertices"} if schema == INPUT_SCHEMA
                    else {"id", "baseLength", "p2", "p3"})
        for piece in source_pieces:
            _fields(piece, expected, set(), "piece")
    else:
        # stage2 metadata is intentionally not passed into the search. In
        # particular, canvas_px is the capture canvas, not the solved frame;
        # inputs.layout, corners_px, image paths and rectification transforms
        # supply no constraints and are never opened or used for placement.
        if puzzle.get("schema_version") != "1.0":
            raise ProtocolError("stage2 schema_version must be '1.0'")
        if puzzle.get("coordinate_system") != "rectified_px":
            raise ProtocolError("stage2 coordinate_system must be 'rectified_px'")
        source_pieces = extract_pieces(puzzle)
        unit = "rectified_px"
        compact = "unit_scale" in puzzle
        scale_key = "unit_scale" if compact else "piece_scale"
        count_key = "piece_count" if compact else "detected_piece_count"
        if _number(puzzle.get(scale_key), scale_key) != 1.0:
            raise ProtocolError(f"stage2 {scale_key} must equal 1; per-piece scaling is unsupported")
        count = puzzle.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or count != len(source_pieces):
            raise ProtocolError(f"stage2 {count_key} must equal the number of pieces")
        for piece in source_pieces:
            if compact:
                _fields(piece, {"id", "baseLength", "p2", "p3"}, set(), "stage2 piece")
            elif not isinstance(piece, dict) or "base_edge" not in piece:
                raise ProtocolError("stage2 full pieces must contain full-precision base_edge records")
    canonical = {"schema": INPUT_SCHEMA, "unit": unit,
                 "pieces": [_canonical_piece(piece) for piece in source_pieces]}
    if "schema" in puzzle and "target" in puzzle:
        _fields(puzzle["target"], {"width", "height"}, set(), "target")
        canonical["target"] = {key: _number(puzzle["target"][key], "target." + key)
                               for key in ("width", "height")}
    return canonical


def _parse_canonical(puzzle):
    pieces = puzzle["pieces"]
    polygons = validate_pieces(pieces)
    total_area = sum(area(p) for p in polygons)
    if not math.isfinite(total_area) or total_area <= 0:
        raise ProtocolError("total area must be finite and positive")
    dimensions = None
    if "target" in puzzle:
        _fields(puzzle["target"], {"width", "height"}, set(), "target")
        dimensions = _target_dimensions(puzzle["target"], total_area)
        if not math.isfinite(dimensions[0] * dimensions[1]) or dimensions[0] * dimensions[1] <= 0:
            raise ProtocolError("target area exceeds the supported floating-point range")
    return pieces, polygons, dimensions, total_area


def _parse_input(puzzle):
    _require_numpy()
    return _parse_canonical(_adapt_input(puzzle))


def normalize_input(puzzle):
    """Return validated polygon-assembly/v1 JSON with unchanged scale/P0/axes.

    Supported inputs are the native polygon protocol, legacy quadrilaterals,
    and the two full-precision stage2 export wrappers. Source-only metadata is
    discarded. This is a format conversion, not denoising or a shape repair.
    """
    _require_numpy()
    canonical = _adapt_input(puzzle)
    _parse_canonical(canonical)
    return canonical


def validate_input(puzzle):
    """Return a JSON-compatible validation report; never reframe the input."""
    report = {"schema": "polygon-assembly-validation/v1", "valid": False,
              "unit": _unit_or_none(puzzle), "pieceCount": None,
              "totalArea": None, "errors": []}
    try:
        pieces, polygons, _, total_area = _parse_input(puzzle)
        report.update(valid=True, pieceCount=len(pieces), totalArea=total_area)
        report["vertexCounts"] = [len(p) for p in polygons]
    except DependencyError as exc:
        report["errors"] = [str(exc)]
        report["errorCode"] = "dependency_error"
    except (ProtocolError, ValueError, TypeError, OverflowError) as exc:
        report["errors"] = [str(exc)]
    return report


def _parse_placement(placement, ident=None):
    _fields(placement, {"id", "rotationDeg", "origin"}, set(), "placement")
    _text(placement["id"], "placement.id")
    if ident is not None and placement["id"] != ident:
        raise ProtocolError("placement id must match the input piece id")
    degrees = _number(placement["rotationDeg"], "rotationDeg")
    if not 0.0 <= degrees < 360.0:
        raise ProtocolError("rotationDeg must be in [0, 360), measured clockwise in image coordinates")
    origin = _json_point(placement["origin"], "origin")
    return origin[0], origin[1], math.radians(degrees)


def apply_placement(piece, placement):
    """Return [P0,...,P(n-1)] in canvas coordinates as an n x 2 Python list.

    origin is the input P0's final position, not the bounding-box minimum.
    Positive rotationDeg is clockwise because image y coordinates point down.
    """
    _require_numpy()
    piece = _canonical_piece(piece)
    points = decode_piece(piece)
    moved = _transform_radians(points, *_parse_placement(placement, piece["id"]))
    if not np.isfinite(moved).all():
        raise ProtocolError("transformed coordinates exceed the supported floating-point range")
    return moved.tolist()


def _public_certificate(certificate, rtol):
    names = {
        "relative_area_error": "relativeAreaError",
        "relative_overlap_sum": "relativeOverlapArea",
        "relative_missing_area": "relativeMissingArea",
        "relative_outside_area": "relativeOutsideArea",
        "relative_symmetric_difference": "relativeSymmetricDifference",
        "relative_outside_vertex_distance": "relativeOutsideVertexDistance",
    }
    original = certificate.get("metrics", {})
    return {"valid": bool(certificate["valid"]), "errors": list(certificate["errors"]),
            "tolerance": float(rtol),
            "metrics": {public: float(original[old]) for old, public in names.items() if old in original}}


def verify(puzzle, result, rtol=DEFAULT_RTOL):
    """Recompute geometry from the public degree/origin fields.

    The result's embedded verification flag is untrusted and is never used as
    evidence that its placements are valid. This function has no source layout.
    """
    report = {"schema": "polygon-assembly-verification/v1", "valid": False,
              "errors": [], "tolerance": None, "metrics": {}}
    try:
        _require_numpy()
        rtol = _number(rtol, "rtol")
        if not 0 < rtol < 1:
            raise ProtocolError("rtol must be strictly between 0 and 1")
        report["tolerance"] = rtol
        canonical = _adapt_input(puzzle)
        pieces, _, _, _ = _parse_canonical(canonical)
        _fields(result, _OUTPUT_FIELDS, {"quality"}, "result")
        if result["schema"] not in (OUTPUT_SCHEMA, LEGACY_OUTPUT_SCHEMA, BEST_EFFORT_OUTPUT_SCHEMA):
            raise ProtocolError(f"result schema must be {BEST_EFFORT_OUTPUT_SCHEMA!r}, {OUTPUT_SCHEMA!r} or {LEGACY_OUTPUT_SCHEMA!r}")
        if result["status"] not in ("feasible", "approximate"):
            raise ProtocolError("verification needs a feasible or approximate complete layout")
        if result["unit"] != canonical["unit"]:
            raise ProtocolError("result unit must exactly match the input unit")
        if result["coordinateSystem"] != COORDINATE_SYSTEM:
            raise ProtocolError("coordinateSystem must be 'image' (x right, y down)")
        if result["status"] == "feasible" and result["reason"] is not None:
            raise ProtocolError("a feasible result must have reason=null")
        if not isinstance(result["stats"], dict):
            raise ProtocolError("result stats must be an object")
        if result["verification"] is not None and not isinstance(result["verification"], dict):
            raise ProtocolError("result verification must be an object or null")
        _fields(result["rectangle"], {"width", "height"}, set(), "rectangle")
        width = _number(result["rectangle"]["width"], "rectangle.width")
        height = _number(result["rectangle"]["height"], "rectangle.height")
        if width <= 0 or height <= 0:
            raise ProtocolError("rectangle dimensions must be positive")
        placements = result["placements"]
        if not isinstance(placements, list):
            raise ProtocolError("placements must be an array")
        legacy = []
        for item in placements:
            x, y, angle = _parse_placement(item)
            legacy.append({"id": item["id"], "tx": x, "ty": y, "rotation": angle})
        internal = {"rectangle": {"width": width, "height": height}, "placements": legacy}
        # This independent verifier uses unions/intersections/differences on
        # the serialized poses; it does not rely on the search's free region.
        certificate = _verify_geometry(canonical, internal, rtol=rtol)
        report.update(_public_certificate(certificate, rtol))
    except (DependencyError, ImportError) as exc:
        report["errors"] = [str(exc)]
        report["errorCode"] = "dependency_error"
    except (ProtocolError, ValueError, TypeError, OverflowError) as exc:
        report["errors"] = [str(exc)]
    except Exception as exc:
        report["errors"] = [f"geometry verification failed: {exc}"]
        report["errorCode"] = "geometry_error"
    return report


def _degrees(angle):
    degrees = float(math.degrees(angle) % 360.0)
    # Floating remainder of a tiny negative value can be exactly 360.0.
    return 0.0 if degrees >= 360.0 or degrees == 0.0 else degrees


def _envelope(puzzle, status, reason=None):
    schema = (LEGACY_OUTPUT_SCHEMA if isinstance(puzzle, dict)
              and puzzle.get("schema") == LEGACY_INPUT_SCHEMA else OUTPUT_SCHEMA)
    return {"schema": schema, "status": status, "unit": _unit_or_none(puzzle),
            "coordinateSystem": COORDINATE_SYSTEM, "rectangle": None,
            "placements": [], "reason": reason, "verification": None, "stats": {}}


def _public_stats(stats, elapsed):
    result = {"elapsedSeconds": float(elapsed),
            "searchSeconds": float(stats.get("seconds", 0.0)),
            "nodes": int(stats.get("nodes", 0)),
            "candidateTests": int(stats.get("candidate_tests", 0)),
            "frameCandidates": int(stats.get("frame_candidate_count", 0)),
            "selectedFrameIndex": stats.get("candidate_frame_index"),
            "bestPlacedCount": int(stats.get("best_piece_count", 0)),
            "boundaryEnumerationComplete": stats.get("boundary_enumeration_complete"),
            "candidateLimitApplied": bool(stats.get("frame_enumeration_truncated", False))}
    if "rectangle_corner_check" in stats:
        result["rectangleCornerCheck"] = stats["rectangle_corner_check"]
    if "truncation_reasons" in stats:
        result["candidateLimitReasons"] = stats["truncation_reasons"]
    return result


def _rectangle_corner_check(polygons):
    """Conservative necessary-condition witness using only supplied geometry.

    At an outer rectangular corner, incident convex tile vertex angles sum to
    90 degrees. If every angle exceeds 45 degrees, only one vertex can occupy
    each corner, hence at least four input vertices must be right angles.
    A generous angular tolerance avoids rejecting ordinary floating noise.
    Passing this check does NOT certify that an assembly exists.
    """
    angles = []
    for p in polygons:
        following = np.roll(p, -1, axis=0) - p
        previous = np.roll(p, 1, axis=0) - p
        angles.extend(np.degrees(np.arctan2(np.abs(cross2(following, previous)),
                                           np.sum(following * previous, axis=1))).tolist())
    tolerance = 1e-5
    minimum = min(angles)
    right_count = sum(abs(angle - 90.0) <= tolerance for angle in angles)
    impossible = minimum > 45.0 + tolerance and right_count < 4
    return {"possible": False if impossible else None,
            "minimumAngleDeg": float(minimum),
            "closestRightAngleErrorDeg": float(min(abs(angle - 90.0) for angle in angles)),
            "rightAngleVertexCount": right_count, "requiredRightAngleVertices": 4,
            "angleToleranceDeg": tolerance,
            "rule": "all_angles_gt_45_require_four_right_angle_vertices"}


def solve_exact(puzzle, time_limit=30.0, max_boundary_edges=4):
    """Assemble strict convex polygons from longest-edge local frames.

    Return the stable public JSON envelope. No original poses, adjacency,
    cutting tree, image, or source rectangle is required. If target is omitted,
    dimensions are inferred. Search exhaustion is unknown, not infeasible.
    """
    started = time.monotonic()
    result = _envelope(puzzle, "error")
    stats = {}
    try:
        pieces, polygons, dimensions, total_area = _parse_input(puzzle)
        corner_check = _rectangle_corner_check(polygons)
        stats["rectangle_corner_check"] = corner_check
        time_limit = _number(time_limit, "time_limit")
        if time_limit <= 0:
            raise ProtocolError("time_limit must be positive")
        if isinstance(max_boundary_edges, bool) or not isinstance(max_boundary_edges, int) or not 1 <= max_boundary_edges <= 8:
            raise ProtocolError("max_boundary_edges must be an integer from 1 to 8")
        if dimensions is not None and abs(dimensions[0] * dimensions[1] - total_area) > 1e-8 * max(total_area, dimensions[0] * dimensions[1]):
            result = _envelope(puzzle, "infeasible", "target_area_mismatch")
        elif len(polygons) == 1 and not _is_rectangle(polygons[0]):
            result = _envelope(puzzle, "infeasible", "a_single_nonrectangle_cannot_form_a_rectangle")
        elif corner_check["possible"] is False:
            result = _envelope(puzzle, "infeasible", "rectangle_corner_angle_mismatch")
        else:
            remaining = time_limit - (time.monotonic() - started)
            if remaining <= 0:
                result = _envelope(puzzle, "unknown", "time_limit")
            else:
                internal = _heuristic(pieces, polygons, dimensions, remaining, max_boundary_edges)
                stats = internal.get("stats", {})
                stats["rectangle_corner_check"] = corner_check
                if internal["status"] != "feasible":
                    result = _envelope(puzzle, "unknown", internal.get("reason", "search_exhausted"))
                else:
                    by_id = {p["id"]: p for p in internal["placements"]}
                    result = _envelope(puzzle, "feasible")
                    result["rectangle"] = {"width": float(internal["rectangle"]["width"]),
                                           "height": float(internal["rectangle"]["height"])}
                    result["placements"] = [
                        {"id": piece["id"], "rotationDeg": _degrees(by_id[piece["id"]]["rotation"]),
                         "origin": [float(by_id[piece["id"]]["tx"]), float(by_id[piece["id"]]["ty"])]}
                        for piece in pieces]
                    # Verify exactly the public fields that consumers receive.
                    certificate = verify(puzzle, result)
                    result["verification"] = {k: v for k, v in certificate.items() if k != "schema"}
                    if not certificate["valid"]:
                        result["status"] = "error" if "errorCode" in certificate else "unknown"
                        result["reason"] = certificate.get("errorCode", "candidate_failed_independent_verification")
                        result["rectangle"] = None
                        result["placements"] = []
    except (DependencyError, ImportError) as exc:
        result = _envelope(puzzle, "error", f"dependency_error: {exc}")
    except (ProtocolError, ValueError, TypeError, OverflowError) as exc:
        result = _envelope(puzzle, "invalid_input", str(exc))
    except Exception as exc:
        result = _envelope(puzzle, "error", f"solver_error: {exc}")
    result["stats"] = _public_stats(stats, time.monotonic() - started)
    return result


def _is_rectangle(points):
    if len(points) != 4:
        return False
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    dots = np.sum(edges * np.roll(edges, -1, axis=0), axis=1)
    return bool(np.max(np.abs(dots) / (lengths * np.roll(lengths, -1))) <= 1e-8)


# ---------------------------------------------------------------------------
# Best-effort rigid assembly. Residuals are objectives, never rejection gates.
# ---------------------------------------------------------------------------

def _quality_from_geometry(polygons, width, height):
    import shapely
    objects = np.asarray([shapely.Polygon(p) for p in polygons], dtype=object)
    frame = shapely.box(0.0, 0.0, width, height)
    union = shapely.union_all(objects)
    total = float(sum(shapely.area(objects)))
    frame_area = float(width * height)
    missing = float(frame.difference(union).area)
    outside = float(union.difference(frame).area)
    excess = max(0.0, total - float(union.area))
    pair_i, pair_j = np.triu_indices(len(polygons), 1)
    pair_overlap = float(np.sum(shapely.area(shapely.intersection(
        objects[pair_i], objects[pair_j])))) if len(pair_i) else 0.0
    maximum_outside = 0.0
    for p in polygons:
        maximum_outside = max(maximum_outside, float(np.max(np.maximum.reduce([
            -p[:, 0], p[:, 0] - width, -p[:, 1], p[:, 1] - height,
            np.zeros(len(p))]))))
    return {
        "objective": "(missingArea+overlapExcessArea+outsideArea)/totalPieceArea",
        "score": float((missing + excess + outside) / total),
        "pieceCount": len(polygons), "totalPieceArea": total,
        "rectangleArea": frame_area, "coveredArea": frame_area - missing,
        "unionArea": float(union.area), "missingArea": missing,
        "overlapExcessArea": excess, "outsideArea": outside,
        "pairwiseOverlapArea": pair_overlap,
        "coverageFraction": float(1.0 - missing / frame_area),
        "relativeMissingArea": float(missing / total),
        "relativeOverlapExcessArea": float(excess / total),
        "relativeOutsideArea": float(outside / total),
        "relativePairwiseOverlapArea": float(pair_overlap / total),
        "maxOutsideDistance": maximum_outside,
        "normalization": "total_input_piece_area",
    }


def evaluate(puzzle, result):
    """Recompute residual areas from all public rigid poses; no fit threshold.

    Returns a quality dictionary. Invalid numeric/ID/pose records raise
    ProtocolError; nonzero gaps, overlaps and outside area are always reported.
    Embedded scores, verification flags and source metadata are not trusted.
    """
    pieces, polygons, _, _ = _parse_input(puzzle)
    if not isinstance(result, dict):
        raise ProtocolError("result must be an object")
    rectangle = result.get("rectangle")
    _fields(rectangle, {"width", "height"}, set(), "rectangle")
    width, height = _target_dimensions(rectangle, 1.0)
    if not math.isfinite(width * height):
        raise ProtocolError("rectangle area exceeds the supported float range")
    records = result.get("placements")
    if not isinstance(records, list):
        raise ProtocolError("placements must be a list")
    by_id = {}
    for record in records:
        pose = _parse_placement(record)
        if record["id"] in by_id:
            raise ProtocolError("placements must contain each input id exactly once")
        by_id[record["id"]] = pose
    if set(by_id) != {piece["id"] for piece in pieces}:
        raise ProtocolError("placements must contain each input id exactly once")
    moved = [_transform_radians(p, *by_id[piece["id"]])
             for piece, p in zip(pieces, polygons)]
    if not all(np.isfinite(p).all() for p in moved):
        raise ProtocolError("transformed coordinates must be finite")
    # Normalize before overlays, then restore dimensional metrics. The score
    # and all fractions are invariant under one common change of units.
    scale = math.sqrt(sum(area(p) for p in polygons))
    quality = _quality_from_geometry([p / scale for p in moved], width / scale, height / scale)
    for key in ("totalPieceArea", "rectangleArea", "coveredArea", "unionArea",
                "missingArea", "overlapExcessArea", "outsideArea", "pairwiseOverlapArea"):
        quality[key] *= scale * scale
    quality["maxOutsideDistance"] *= scale
    return quality


class _ApproxTimeout(Exception):
    pass


def _approx_minkowski(a,b):
 from shapely.geometry import MultiPoint
 return np.array(MultiPoint((a[:,None,:]-b[None,:,:]).reshape(-1,2)).convex_hull.exterior.coords[:-1])
def _approx_intervals(hull,xs):
 a=hull;b=np.roll(a,-1,axis=0);dx=b[:,0]-a[:,0];nonvertical=np.abs(dx)>1e-12;a=a[nonvertical];b=b[nonvertical];dx=dx[nonvertical]
 x=np.asarray(xs)[:,None];t=(x-a[None,:,0])/dx[None,:];ys=a[None,:,1]+t*(b-a)[None,:,1];ok=(t>=-1e-10)&(t<=1+1e-10)
 return np.min(np.where(ok,ys,np.inf),axis=1),np.max(np.where(ok,ys,-np.inf),axis=1)
def _approx_orientations(p):
 e=np.roll(p,-1,axis=0)-p;angs=np.mod(-np.arctan2(e[:,1],e[:,0]),2*np.pi);angs=np.concatenate([angs,(angs+np.pi/2)%(2*np.pi)])
 out=[];seen=set()
 for a in angs:
  if round(float(a),8) in seen:continue
  seen.add(round(float(a),8));c,s=math.cos(a),math.sin(a);q=p@np.array([[c,s],[-s,c]]);shift=-q.min(axis=0);q+=shift;out.append((a,shift,q))
 return out

def _approx_strip_pack(polys,width,order,deadline=math.inf):
 os=[_approx_orientations(p) for p in polys];placed=[];poses={};height=0
 for i in order:
  if time.monotonic()>deadline:return None
  best=None
  for ang,shift,q in os[i]:
   qw=float(q[:,0].max());qh=float(q[:,1].max())
   if qw>width+1e-10:continue
   forbidden=[_approx_minkowski(p,q) for p in placed];xs=[0.,width-qw]
   for h in forbidden:xs.extend(float(x) for x in h[:,0] if 0<=x<=width-qw)
   xs=np.unique(xs);lows=[];highs=[]
   for h in forbidden:
    lo,hi=_approx_intervals(h,xs);lows.append(lo);highs.append(hi)
   ys=np.zeros(len(xs))
   if lows:
    lows=np.array(lows).T;highs=np.array(highs).T;sort=np.argsort(lows,axis=1);lows=np.take_along_axis(lows,sort,axis=1);highs=np.take_along_axis(highs,sort,axis=1)
    for j in range(lows.shape[1]):
     hit=(lows[:,j]<ys-1e-9)&(highs[:,j]>ys+1e-9)
     ys=np.where(hit,highs[:,j],ys)
   scores=ys+qh+.001*xs
   k=int(np.argmin(scores));x,y=float(xs[k]),float(ys[k]);maxh=max(height,y+qh)
   # Prioritize packed height; left bottom is secondary.
   score=maxh+.00001*(y+qh)+.0000001*x
   if best is None or score<best[0]:best=(score,ang,shift+np.array([x,y]),q+np.array([x,y]),maxh)
  if best is None:return None
  _,ang,t,q,height=best;placed.append(q);poses[i]=(float(ang),float(t[0]),float(t[1]))
 return poses,width,height

def _approx_pack(polys,seconds=3.0,seed=0,target=None):
 """Input normalized convex arrays; returns complete rigid poses + frame + stats.
 Width trials depend only on area and geometry. A shelf seed is returned even for
 zero budget; available time improves it with convex no-fit polygon placements.
 """
 from shapely.geometry import Polygon
 start=time.monotonic();deadline=start+max(0,seconds);areas=np.array([Polygon(p).area for p in polys]);A=sum(areas);rng=np.random.default_rng(seed);os=[_approx_orientations(p) for p in polys]
 # Immediate shelf seed has disjoint bounding boxes, valid for arbitrary n.
 sel=[min(opts,key=lambda v:v[2][:,0].max()*v[2][:,1].max()) for opts in os]
 width=max(math.sqrt(A*1.8),max(q[:,0].max() for _,_,q in sel));x=y=rowh=0;poses={};maxw=0
 for i in np.argsort(-areas):
  ang,shift,q=sel[i];qw=float(q[:,0].max());qh=float(q[:,1].max())
  if x+qw>width and x>0:y+=rowh;x=0;rowh=0
  t=shift+np.array([x,y]);poses[int(i)]=(float(ang),float(t[0]),float(t[1]));x+=qw;rowh=max(rowh,qh);maxw=max(maxw,x)
 best=(poses,maxw,y+rowh);score=best[1]*best[2];trials=0
 ratios=[1.8,1.5,2.,1.,2.5,3.,.75,.5]
 while time.monotonic()<deadline:
  ratio=ratios[trials%len(ratios)];W=math.sqrt(A*ratio)
  order=np.argsort(-areas) if trials<len(ratios) else rng.permutation(len(polys))
  out=_approx_strip_pack(polys,W,order,deadline)
  if out and out[1]*out[2]<score:best=out;score=out[1]*out[2]
  trials+=1
 if target is not None:
  # Keep target unchanged; seed centering is still a complete candidate.
  poses,W,H=best;tw,th=target;dx=(tw-W)/2;dy=(th-H)/2;best=({i:(a,x+dx,y+dy) for i,(a,x,y) in poses.items()},tw,th)
 return (*best,{'backend':'convex_strip_seed','trials':trials,'seconds':time.monotonic()-start})



def _four_cycle_initializers(polygons, ids, target=None, deadline=None, *,
                            partition_limit=100, configurations=3,
                            ranking_limit=100, result_limit=24,
                            length_sigma_ratio=0.008,
                            angle_sigma_deg=2.5, progress=None):
    """Return geometry-only rigid candidate assemblies and actual search stats.

    polygons: n arrays (m,2); ids: corresponding output IDs.
    target: None or {'width': ..., 'height': ...}; dimensions stay fixed if set.
    deadline: absolute time.monotonic deadline, or None.
    Configurable caps retain the best candidates already found when reached.
    If this family does not apply, or expires before a complete candidate,
    return an empty candidate list so a generic fallback can continue.

    A candidate is {'rectangle':..., 'placements':[id,rotationDeg,origin],
    'initializerScore':...}. No downstream exact-feasibility check is needed.
    """
    import heapq
    import itertools
    started=time.monotonic()
    stats={'family':'four-cycle-clusters','applicable':False,
           'groupSubsetsScored':0,'groupConfigurationsScored':0,
           'partitionsCompleted':0,'partitionsRetained':0,
           'parentConfigurationsScored':0,'completeCandidatesScored':0,
           'expired':False,'elapsedSeconds':0.0,'bestHistory':[]}
    if len(polygons)!=16 or any(len(p)!=4 for p in polygons):
        return [],stats
    stats['applicable']=True
    if len(ids)!=len(polygons):
        raise ValueError('ids and polygons must have the same length')
    for name,value in [('partition_limit',partition_limit),('configurations',configurations),
                       ('ranking_limit',ranking_limit),('result_limit',result_limit)]:
        if value is not None and (not isinstance(value,int) or value<1):
            raise ValueError(name+' must be a positive integer or None')
    configurations=min(configurations or 1536,1536)
    if angle_sigma_deg<=0 or length_sigma_ratio<=0:
        raise ValueError('noise scales must be positive')
    class Expired(Exception):pass
    def check():
        if deadline is not None and time.monotonic()>=deadline:raise Expired
    qs=[np.asarray(p,dtype=float) for p in polygons]
    total=sum(abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))))*.5 for p in qs)
    scale=math.sqrt(total)
    length_sigma=length_sigma_ratio*scale
    states=np.array(list(itertools.product(range(4),repeat=4)),int)
    permutations=np.array([(0,*p) for p in itertools.permutations((1,2,3))])
    def rotate(q,t):
        c,s=np.cos(t),np.sin(t)
        return q@np.array([[c,s],[-s,c]])
    def measurements(pp):
        lengths=[];aa=[]
        for q in pp:
            e=np.roll(q,-1,axis=0)-q;l=np.linalg.norm(e,axis=1)
            lengths.append(l)
            aa.append(np.degrees(np.arccos(np.clip(np.sum(e*(-np.roll(e,1,axis=0)),axis=1)/(l*np.roll(l,1)),-1,1))))
        return np.array(lengths),np.array(aa)
    def configs(pp,indices,top,length_noise,rectangle=False):
        L,A=measurements(pp);out=[]
        for perm in permutations:
            check();order=np.array(indices)[perm];c=states
            aa=A[order[None,:],c]
            prevlength=L[order[None,:],(c-1)%4]
            nextlength=L[order[None,:],c]
            lengtherr=(prevlength-np.roll(nextlength,-1,axis=1))/length_noise
            ap=A[order[None,:],(c-1)%4];an=A[order[None,:],(c+1)%4]
            joinerr=(ap+np.roll(an,-1,axis=1)-180)/angle_sigma_deg
            centererr=(aa.sum(axis=1)-360)/angle_sigma_deg
            scores=np.sum(lengtherr**2+joinerr**2,axis=1)+centererr**2
            if rectangle:
                outer=A[order[None,:],(c+2)%4]
                scores+=2*np.sum(((outer-90)/angle_sigma_deg)**2,axis=1)
            if rectangle:stats['parentConfigurationsScored']+=len(scores)
            else:stats['groupConfigurationsScored']+=len(scores)
            ii=np.argsort(scores,kind='stable')[:top]
            out.extend((float(scores[k]),order.tolist(),c[k].tolist()) for k in ii)
        out.sort(key=lambda x:x[0]);return out[:top]
    def cluster(pp,order,cs):
        _,a=measurements(pp)
        aa=np.array([a[i,c] for i,c in zip(order,cs)])
        sector=aa+(360-aa.sum())/4
        theta=np.r_[0,np.cumsum(sector[:-1])]*np.pi/180
        poses=[];corners=[]
        for i,c,t in zip(order,cs,theta):
            q=pp[i];v=q[(c+1)%4]-q[c]
            alpha=t-np.arctan2(v[1],v[0]);translation=-rotate(q[[c]],alpha)[0]
            corners.append((rotate(q,alpha)+translation)[(c+2)%4])
            poses.append((float(alpha),translation))
        return {'indices':list(order),'poses':poses,'quad':np.array(corners)}
    from shapely.geometry import Polygon,box
    from shapely.ops import unary_union
    from scipy.optimize import minimize
    def fit(poses):
        points=np.vstack([rotate(q,a)+t for q,(a,t) in zip(qs,poses)])
        hull=np.asarray(Polygon(points).convex_hull.exterior.coords)[:-1]
        edges=np.roll(hull,-1,axis=0)-hull
        theta=-np.arctan2(edges[:,1],edges[:,0])
        extent=[np.ptp(rotate(points,t),axis=0) for t in theta]
        if target is None:
            k=int(np.argmin([v.prod() for v in extent]));angle=float(theta[k])
        else:
            wh=np.array([target['width'],target['height']],float)
            candidates=[(float(t),np.ptp(rotate(points,t),axis=0)) for t in np.r_[theta,theta+np.pi/2]]
            angle=min(candidates,key=lambda tv:float(np.sum((tv[1]-wh)**2)))[0]
        pts=rotate(points,angle);lo=pts.min(axis=0);wh=np.ptp(pts,axis=0)
        transformed=[(a+angle,rotate(t[None,:],angle)[0]-lo) for a,t in poses]
        union=unary_union([Polygon(rotate(q,a)+t) for q,(a,t) in zip(qs,transformed)])
        excess=total-union.area
        if target is None:
            start=np.array([0.,0.,*wh])
            def loss(x):
                check();x0,y0,x1,y1=x
                if x1<=x0 or y1<=y0:return 1e3
                return (box(x0,y0,x1,y1).symmetric_difference(union).area+excess)/total
            bestx=start.copy();bestv=loss(start)
            def f(x):
                nonlocal bestx,bestv
                v=loss(x)
                if v<bestv:bestx=x.copy();bestv=v
                return v
            try:minimize(f,start,method='Powell',options={'maxiter':50,'xtol':1e-5,'ftol':1e-8})
            except Expired:pass
            x0,y0,x1,y1=bestx;wh=np.array([x1-x0,y1-y0]);translation=np.array([x0,y0])
        else:
            wh=np.array([target['width'],target['height']],float)
            start=(np.array(union.bounds)[2:]-wh)/2
            bestx=start.copy();bestv=float((box(*start,*(start+wh)).symmetric_difference(union).area+excess)/total)
            def f(x):
                nonlocal bestx,bestv
                check();v=(box(*x,*(x+wh)).symmetric_difference(union).area+excess)/total
                if v<bestv:bestx=x.copy();bestv=v
                return v
            try:minimize(f,start,method='Powell',options={'maxiter':50,'xtol':1e-5,'ftol':1e-8})
            except Expired:pass
            translation=bestx
        transformed=[(a,t-translation) for a,t in transformed]
        return transformed,float(wh[0]),float(wh[1]),float(bestv)
    results=[];best_score=float('inf')
    try:
        groups=list(itertools.combinations(range(16),4));records=[]
        # Geometry is measured once globally; repeated small arrays remain cheap.
        for group in groups:
            check();records.append(configs(qs,group,configurations,length_sigma))
            stats['groupSubsetsScored']+=1
        costs=np.array([r[0][0] for r in records]);masks=[sum(1<<i for i in g) for g in groups]
        bypiece=[[] for _ in range(16)]
        for j,g in enumerate(groups):
            for i in g:bypiece[i].append(j)
        for arr in bypiece:arr.sort(key=lambda j:costs[j])
        heap=[];full=(1<<16)-1
        def visit(mask,chosen,cost):
            check()
            if mask==full:
                stats['partitionsCompleted']+=1;item=(-float(cost),tuple(chosen))
                if partition_limit is None or len(heap)<partition_limit:heapq.heappush(heap,item)
                elif cost<-heap[0][0]:heapq.heapreplace(heap,item)
                return
            if partition_limit is not None and len(heap)==partition_limit and cost>=-heap[0][0]:return
            low=((~mask)&full)&-((~mask)&full);i=low.bit_length()-1
            for j in bypiece[i]:
                nc=cost+costs[j]
                if partition_limit is not None and len(heap)==partition_limit and nc>=-heap[0][0]:break
                if masks[j]&mask==0:visit(mask|masks[j],chosen+[j],nc)
        visit(0,[],0)
        partitions=sorted([(-c,list(p)) for c,p in heap]);stats['partitionsRetained']=len(partitions)
        cache={};rank=[]
        # Assemble the best partition first, so early deadlines still yield poses.
        for pi,(_,part) in enumerate(partitions):
            for variants in itertools.product(range(configurations),repeat=4):
                check();gs=[];localcost=0.
                for j,v in zip(part,variants):
                    key=(j,v)
                    if key not in cache:
                        rr=records[j][v];cache[key]=cluster(qs,rr[1],rr[2])
                    gs.append(cache[key]);localcost+=records[j][v][0]
                parent=configs([g['quad'] for g in gs],(0,1,2,3),1,2*length_sigma,True)[0]
                rank.append((float(localcost+parent[0]),gs,parent))
            # First partition provides usable output before global ranking ends.
            if pi==0:
                provisional=sorted(rank,key=lambda r:r[0])[:min(4,len(rank))]
                for rank_score,gs,parent in provisional:
                    c=cluster([g['quad'] for g in gs],parent[1],parent[2]);poses=[None]*16
                    for gindex,(a,t) in zip(c['indices'],c['poses']):
                        g=gs[gindex]
                        for i,(la,lt) in zip(g['indices'],g['poses']):poses[i]=(a+la,rotate(lt[None,:],a)[0]+t)
                    pp,w,h,score=fit(poses);stats['completeCandidatesScored']+=1
                    r={'rectangle':{'width':w,'height':h},'placements':[{'id':ids[i],'rotationDeg':float(np.degrees(a)%360),'origin':t.tolist()} for i,(a,t) in enumerate(pp)],'initializerScore':score,'fourCycleFitScore':rank_score}
                    results.append(r)
                    if score<best_score:
                        best_score=score;stats['bestHistory'].append({'seconds':time.monotonic()-started,'score':score,'candidate':stats['completeCandidatesScored']})
                        if progress is not None:progress(r,stats)
        rank.sort(key=lambda r:r[0])
        for rank_score,gs,parent in (rank if ranking_limit is None else rank[:ranking_limit]):
            check();c=cluster([g['quad'] for g in gs],parent[1],parent[2]);poses=[None]*16
            for gindex,(a,t) in zip(c['indices'],c['poses']):
                g=gs[gindex]
                for i,(la,lt) in zip(g['indices'],g['poses']):poses[i]=(a+la,rotate(lt[None,:],a)[0]+t)
            pp,w,h,score=fit(poses);stats['completeCandidatesScored']+=1
            r={'rectangle':{'width':w,'height':h},'placements':[{'id':ids[i],'rotationDeg':float(np.degrees(a)%360),'origin':t.tolist()} for i,(a,t) in enumerate(pp)],'initializerScore':score,'fourCycleFitScore':rank_score}
            results.append(r)
            if score<best_score:
                best_score=score;stats['bestHistory'].append({'seconds':time.monotonic()-started,'score':score,'candidate':stats['completeCandidatesScored']})
                if progress is not None:progress(r,stats)
    except Expired:
        stats['expired']=True
    results.sort(key=lambda r:r['initializerScore'])
    # Exact duplicate poses from the provisional pass add no optimization value.
    unique=[];seen=set()
    for r in results:
        sig=tuple(round(float(v),8) for p in r['placements'] for v in [p['rotationDeg'],*p['origin']])
        if sig in seen:continue
        seen.add(sig);unique.append(r)
    stats['elapsedSeconds']=time.monotonic()-started
    stats['candidatesReturned']=len(unique if result_limit is None else unique[:result_limit])
    stats['caps']={'partitions':partition_limit,'configurations':configurations,'ranked':ranking_limit,'results':result_limit}
    return (unique if result_limit is None else unique[:result_limit]),stats



def _approx_hierarchical(polygons, deadline):
    records, stats = _four_cycle_initializers(
        polygons, [str(i) for i in range(len(polygons))], deadline=deadline)
    # A good local contact fit can start with more area error yet converge to
    # a better assembly. Preserve that alternative to raw area ranking.
    candidates = []
    for record in sorted(records, key=lambda r: r["fourCycleFitScore"]):
        by_id = {int(p["id"]): p for p in record["placements"]}
        poses = [(math.radians(by_id[i]["rotationDeg"]), *by_id[i]["origin"])
                 for i in range(len(polygons))]
        candidates.append((poses, record["rectangle"]["width"], record["rectangle"]["height"]))
    return candidates, stats

def _contact_lsq_initializer(polygons, ids, candidate, target=None, deadline=None,
                            *, contact_distance_ratio=0.025,
                            angle_tolerance_deg=12.0,
                            full_edge_distance_ratio=0.018,
                            frame_distance_ratio=0.035,
                            line_weight=2.0, robust=True, max_nfev=150):
    """Infer nearby edge contacts from a candidate and fit only rigid poses.

    Requires no grouping, cutting tree, original placements, image, or IDs with
    semantic meaning. Works on variable-vertex convex polygons. It returns the
    better-by-area of the original candidate and the contact-refined candidate,
    plus factual diagnostics. Known target dimensions remain fixed.

    Positional thresholds are normalized by sqrt(sum(piece areas)). They guide
    an optional initializer; failure to infer contacts never rejects a puzzle.
    """
    from scipy.optimize import least_squares
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    started=time.monotonic()
    stats={'method':'geometry-contact-least-squares','edgePairsConsidered':0,
           'contactsDetected':0,'fullEdgePairs':0,'lineGroups':0,
           'frameEdges':0,'expired':False,'accepted':False}
    local=[np.asarray(p,dtype=float) for p in polygons]
    total=sum(Polygon(p).area for p in local);scale=math.sqrt(total)
    center=np.array([p.mean(axis=0)/scale for p in local])
    centered=[p/scale-c for p,c in zip(local,center)]
    n=len(local);lookup={p['id']:p for p in candidate['placements']}
    poses=[]
    for i,ident in enumerate(ids):
        p=lookup[ident];a=math.radians(p['rotationDeg']);c,s=math.cos(a),math.sin(a)
        poses.append([*(np.asarray(p['origin'])/scale+np.array([[c,-s],[s,c]])@center[i]),a])
    poses=np.array(poses)
    initial_wh=np.array([candidate['rectangle']['width'],candidate['rectangle']['height']])/scale
    known=target is not None
    if known:initial_wh=np.array([target['width'],target['height']],float)/scale
    homogeneous=len(set(map(len,local)))==1
    array=np.array(centered) if homogeneous else None
    def points(x):
        a=x[:3*n].reshape(n,3);c=np.cos(a[:,2]);s=np.sin(a[:,2])
        if homogeneous:
            xy=np.empty_like(array)
            xy[:,:,0]=array[:,:,0]*c[:,None]-array[:,:,1]*s[:,None]+a[:,None,0]
            xy[:,:,1]=array[:,:,0]*s[:,None]+array[:,:,1]*c[:,None]+a[:,None,1]
            return xy
        return [q@np.array([[c[i],s[i]],[-s[i],c[i]]])+a[i,:2] for i,q in enumerate(centered)]
    initial_points=points(poses.ravel())
    edges=[]
    for i,pp in enumerate(initial_points):
        for j in range(len(pp)):
            a,b=pp[j],pp[(j+1)%len(pp)];v=b-a;length=float(np.linalg.norm(v));u=v/length
            edges.append((i,j,(j+1)%len(pp),a,b,u,length))
    # A rectangle boundary is inferred from proximity and direction, not exact
    # right angles. An edge may fail this heuristic and remain unconstrained.
    cos_tol=math.cos(math.radians(angle_tolerance_deg));frame=[];is_frame=set()
    w,h=initial_wh
    for e,(i,j,k,a,b,u,ll) in enumerate(edges):
        distances=[max(abs(a[0]),abs(b[0])),max(abs(a[0]-w),abs(b[0]-w)),
                   max(abs(a[1]),abs(b[1])),max(abs(a[1]-h),abs(b[1]-h))]
        side=int(np.argmin(distances));parallel=abs(u[1 if side<2 else 0])
        if distances[side]<=frame_distance_ratio and parallel>=cos_tol:
            frame.append((i,j,k,side));is_frame.add(e)
    stats['frameEdges']=len(frame)
    parent=list(range(len(edges)))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    contacts=[]
    for i,a in enumerate(edges):
        if i in is_frame:continue
        for j in range(i+1,len(edges)):
            b=edges[j]
            if a[0]==b[0] or j in is_frame:continue
            stats['edgePairsConsidered']+=1
            if float(a[5]@b[5])>-cos_tol:continue
            proj=np.array([(b[3]-a[3])@a[5],(b[4]-a[3])@a[5]])
            lo=max(0.,float(proj.min()));hi=min(a[6],float(proj.max()))
            if hi-lo<0.10*min(a[6],b[6]):continue
            midpoint=a[3]+.5*(lo+hi)*a[5]
            normal=np.array([-b[5][1],b[5][0]])
            if abs(float((midpoint-b[3])@normal))>contact_distance_ratio:continue
            contacts.append((i,j));union(i,j)
    stats['contactsDetected']=len(contacts)
    if not contacts and not frame:
        stats['elapsedSeconds']=time.monotonic()-started
        return candidate,stats
    components={}
    for i,j in contacts:
        components.setdefault(find(i),set()).update([i,j])
    full_pairs=[];line_groups=[];line_parameters=[]
    for group in components.values():
        group=sorted(group)
        if len(group)==2:
            a,b=edges[group[0]],edges[group[1]]
            endpoint_error=max(float(np.linalg.norm(a[3]-b[4])),float(np.linalg.norm(a[4]-b[3])))
            if endpoint_error<=full_edge_distance_ratio:
                full_pairs.extend([((a[0],a[1]),(b[0],b[2])),((a[0],a[2]),(b[0],b[1]))]);stats['fullEdgePairs']+=1
                continue
        refs=[(edges[e][0],v) for e in group for v in edges[e][1:3]]
        xy=np.array([initial_points[i][j] for i,j in refs]);mean=xy.mean(axis=0)
        _,vectors=np.linalg.eigh((xy-mean).T@(xy-mean));normal=vectors[:,0]
        line_groups.append(refs);line_parameters.extend([math.atan2(normal[1],normal[0]),float(mean@normal)])
    stats['lineGroups']=len(line_groups)
    # Endpoint coincidence is used only for an almost complete paired edge;
    # partial contacts retain their sliding degree of freedom along a line.
    frame_offset=3*n;line_offset=3*n+(0 if known else 2)
    x=np.r_[poses.ravel(),[] if known else initial_wh,line_parameters]
    initial=x.copy();best_x=x.copy();best_residual=float('inf');calls=0
    def dimensions(x):return initial_wh if known else x[frame_offset:frame_offset+2]
    class Expired(Exception):pass
    def residual(x):
        nonlocal best_x,best_residual,calls
        if deadline is not None and time.monotonic()>=deadline:raise Expired
        calls+=1;pp=points(x);wh=dimensions(x);r=[]
        for (i,j),(k,l) in full_pairs:r.extend(pp[i][j]-pp[k][l])
        for g,refs in enumerate(line_groups):
            theta,d=x[line_offset+2*g:line_offset+2*g+2];normal=np.array([math.cos(theta),math.sin(theta)])
            r.extend(line_weight*(float(pp[i][j]@normal)-d) for i,j in refs)
        for i,j,k,side in frame:
            coord=side//2;boundary=0. if side%2==0 else wh[coord]
            r.extend([line_weight*(pp[i][j][coord]-boundary),line_weight*(pp[i][k][coord]-boundary)])
        # Tiny pose prior only removes underdetermined tangential drift.
        r.extend(0.002*(x[:3*n]-initial[:3*n]))
        values=np.array(r);value=float(np.sum(values**2))
        if value<best_residual:best_residual=value;best_x=x.copy()
        return values
    try:
        opt=least_squares(residual,x,max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10,
                          loss='soft_l1' if robust else 'linear',f_scale=.003)
        best_x=opt.x
        stats['optimizerEvaluations']=int(opt.nfev)
        stats['optimizerSuccess']=bool(opt.success)
    except Expired:
        stats['expired']=True
    stats['residualEvaluations']=calls
    def area_score(x):
        pp=points(x);wh=dimensions(x)
        if not np.isfinite(wh).all() or np.any(wh<=0):return float('inf')
        union=unary_union([Polygon(p) for p in pp]);rectangle=box(0,0,*wh)
        return float(rectangle.symmetric_difference(union).area+1.-union.area)
    before=area_score(initial);after=area_score(best_x)
    stats['beforeScore']=before;stats['afterScore']=after
    if after<before:
        a=best_x[:3*n].reshape(n,3);wh=dimensions(best_x)*scale;placements=[]
        for i,(cx,cy,theta) in enumerate(a):
            c,s=math.cos(theta),math.sin(theta)
            origin=(np.array([cx,cy])-np.array([[c,-s],[s,c]])@center[i])*scale
            placements.append({'id':ids[i],'rotationDeg':float(math.degrees(theta)%360),'origin':origin.tolist()})
        result={**candidate,'rectangle':{'width':float(wh[0]),'height':float(wh[1])},'placements':placements}
        result['initializerScore']=after;stats['accepted']=True
    else:result=candidate
    stats['elapsedSeconds']=time.monotonic()-started
    return result,stats


class _ApproxOptimizer:
    """Nonsmooth rigid packing optimization with a complete incumbent."""
    def __init__(self, polygons, ids, target, deadline, seed):
        import shapely
        self.shapely = shapely
        self.polygons = polygons
        self.ids = ids
        self.n = len(polygons)
        self.target = target
        self.deadline = deadline
        self.rng = np.random.default_rng(seed)
        self.centers = np.asarray([np.mean(p, axis=0) for p in polygons])
        self.centered = [p - c for p, c in zip(polygons, self.centers)]
        self.homogeneous = len({len(p) for p in polygons}) == 1
        if self.homogeneous:
            self.arr = np.asarray(self.centered)
        self.pair_i, self.pair_j = np.triu_indices(self.n, 1)
        self.total = sum(area(p) for p in polygons)
        self.evaluations = 0
        self.accepted = 0
        self.best = None
        self.best_score = math.inf
        self.best_loss = math.inf

    def encode(self, poses, width, height):
        values = []
        for i, (angle, tx, ty) in enumerate(poses):
            c, s = math.cos(angle), math.sin(angle)
            center = np.array([[c, -s], [s, c]]) @ self.centers[i] + [tx, ty]
            values.extend([*center, angle])
        if self.target is not None:
            dx, dy = (self.target[0] - width) / 2, (self.target[1] - height) / 2
            for i in range(self.n):
                values[3 * i] += dx
                values[3 * i + 1] += dy
            width, height = self.target
        return np.asarray([*values, math.log(width), math.log(height)])

    def decode(self, x):
        poses = x[:3 * self.n].reshape((-1, 3))
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        if self.homogeneous:
            points = np.empty_like(self.arr)
            points[:, :, 0] = self.arr[:, :, 0] * c[:, None] - self.arr[:, :, 1] * s[:, None] + poses[:, None, 0]
            points[:, :, 1] = self.arr[:, :, 0] * s[:, None] + self.arr[:, :, 1] * c[:, None] + poses[:, None, 1]
            objects = self.shapely.polygons(points)
        else:
            points = [p @ np.array([[c[i], s[i]], [-s[i], c[i]]]) + poses[i, :2]
                      for i, p in enumerate(self.centered)]
            objects = np.asarray([self.shapely.Polygon(p) for p in points], dtype=object)
        wh = np.asarray(self.target) if self.target is not None else np.exp(x[-2:])
        return points, objects, wh

    def public_poses(self, x, scale):
        rows = x[:3 * self.n].reshape((-1, 3))
        result = []
        for ident, center, row in zip(self.ids, self.centers, rows):
            angle = row[2]
            c, s = math.cos(angle), math.sin(angle)
            origin = (row[:2] - np.array([[c, -s], [s, c]]) @ center) * scale
            result.append({"id": ident, "rotationDeg": _degrees(angle),
                           "origin": [float(origin[0]), float(origin[1])]})
        return result

    def loss(self, x):
        self.evaluations += 1
        if not np.isfinite(x).all() or np.any(np.abs(x[-2:]) > 40):
            return 1e50
        _, objects, wh = self.decode(x)
        frame = self.shapely.box(0.0, 0.0, *wh)
        outside = float(np.sum(self.shapely.area(self.shapely.difference(objects, frame))))
        b = self.shapely.bounds(objects)
        i, j = self.pair_i, self.pair_j
        mask = ((b[i, 0] < b[j, 2]) & (b[j, 0] < b[i, 2]) &
                (b[i, 1] < b[j, 3]) & (b[j, 1] < b[i, 3]))
        overlap = float(np.sum(self.shapely.area(self.shapely.intersection(
            objects[i[mask]], objects[j[mask]])))) if np.any(mask) else 0.0
        # This inexpensive surrogate equals the stated union objective where
        # there is no triple coverage and no overlapping area outside the box.
        # The retained candidate is always ranked by the actual union score.
        value = float((wh[0] * wh[1] - self.total + 2.0 * (outside + overlap)) / self.total)
        if value < self.best_loss:
            self.best_loss = value
            union = self.shapely.union_all(objects)
            score = float((frame.difference(union).area + union.difference(frame).area
                           + max(0.0, self.total - union.area)) / self.total)
            if score < self.best_score:
                self.best_score = score
                self.best = x.copy()
                self.accepted += 1
        return value

    def consider(self, x):
        # Compare every distinct complete seed by the true area objective,
        # even when it is not better under the optimization surrogate.
        points, _, wh = self.decode(x)
        quality = _quality_from_geometry(points, *wh)
        score = quality["score"]
        if self.best is None or score < self.best_score:
            self.best, self.best_score = x.copy(), score
            self.accepted += 1
        loss = self.loss(x)
        return score, loss

    def optimize(self, x, seconds, method="L-BFGS-B"):
        from scipy.optimize import minimize
        stop = min(self.deadline, time.monotonic() + max(0.0, seconds))
        if stop <= time.monotonic():
            return
        local_best, local_value = x.copy(), math.inf
        def objective(y):
            nonlocal local_best, local_value
            if time.monotonic() >= stop:
                raise _ApproxTimeout
            value = self.loss(y)
            if value < local_value:
                local_best, local_value = y.copy(), value
            return value
        options = ({"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7,
                    "maxls": 30, "maxfun": 1000000}
                   if method == "L-BFGS-B" else
                   {"maxiter": 1000, "ftol": 1e-10, "xtol": 1e-7, "maxfev": 1000000})
        try:
            minimize(objective, x, method=method, options=options)
        except _ApproxTimeout:
            pass
        self.consider(local_best)

    def polish(self, seconds):
        """Increase covered union area with single/pair/whole-layout motions.

        For fixed input area A and rectangle F, the stated score equals
        (area(F)+A-2*area(union intersect F))/A. Maximizing newly covered free
        area therefore optimizes the true score, including triple overlaps.
        """
        from scipy.optimize import minimize
        stop = min(self.deadline, time.monotonic() + max(0.0, seconds))
        if stop <= time.monotonic():
            return
        x = self.best.copy()
        rounds = 0
        while time.monotonic() < stop:
            previous = self.best_score
            points, objects, wh = self.decode(x)
            frame = self.shapely.box(0.0, 0.0, *wh)
            groups = [(int(i),) for i in self.rng.permutation(self.n)]
            if rounds % 2 == 0:
                pairs = [(int(i), int(j)) for i, j in zip(self.pair_i, self.pair_j)
                         if objects[i].distance(objects[j]) < 0.01]
                self.rng.shuffle(pairs)
                groups.extend(pairs)
            groups.append(tuple(range(self.n)))
            for group in groups:
                if time.monotonic() >= stop:
                    break
                others = self.shapely.union_all([p for i, p in enumerate(objects) if i not in group])
                free = frame.difference(others)
                old = [np.asarray(points[i]) for i in group]
                center = np.mean([np.mean(p, axis=0) for p in old], axis=0)
                before = self.shapely.union_all(objects[list(group)]).intersection(free).area
                best_gain, step = before, np.zeros(3)
                def group_loss(v):
                    nonlocal best_gain, step
                    if time.monotonic() >= stop:
                        raise _ApproxTimeout
                    self.evaluations += 1
                    c, s = math.cos(v[0] * 3), math.sin(v[0] * 3)
                    matrix = np.array([[c, -s], [s, c]])
                    moved = [self.shapely.Polygon((p - center) @ matrix.T + center + v[1:]) for p in old]
                    gain = self.shapely.union_all(moved).intersection(free).area
                    if gain > best_gain:
                        best_gain, step = gain, np.asarray(v).copy()
                    return -gain
                try:
                    minimize(group_loss, np.zeros(3), method="Powell",
                             bounds=[(-0.01, 0.01), (-0.015, 0.015), (-0.015, 0.015)],
                             options={"xtol": 1e-7, "ftol": 1e-10, "maxiter": 6, "maxfev": 500})
                except _ApproxTimeout:
                    pass
                if best_gain > before + 1e-12:
                    rows = x[:3 * self.n].reshape((-1, 3))
                    c, s = math.cos(step[0] * 3), math.sin(step[0] * 3)
                    matrix = np.array([[c, -s], [s, c]])
                    for i in group:
                        rows[i, :2] = (rows[i, :2] - center) @ matrix.T + center + step[1:]
                        rows[i, 2] += step[0] * 3
                    points, objects, wh = self.decode(x)
                    # Record every accepted complete incumbent, even if the
                    # deadline arrives before this coordinate round ends.
                    quality = _quality_from_geometry(points, *wh)
                    if quality["score"] < self.best_score:
                        self.best, self.best_score = x.copy(), quality["score"]
                        self.accepted += 1
            if self.target is None and time.monotonic() < stop:
                union = self.shapely.union_all(objects)
                best_wh, best_value = wh.copy(), math.inf
                def frame_loss(v):
                    nonlocal best_wh, best_value
                    if time.monotonic() >= stop:
                        raise _ApproxTimeout
                    self.evaluations += 1
                    if min(v) <= 0:
                        return 1e50
                    value = v[0] * v[1] + self.total - 2 * union.intersection(self.shapely.box(0, 0, *v)).area
                    if value < best_value:
                        best_wh, best_value = np.asarray(v).copy(), value
                    return value
                frame_loss(wh)
                try:
                    minimize(frame_loss, wh, method="Powell",
                             bounds=[(max(1e-9, v - 0.01), v + 0.01) for v in wh],
                             options={"xtol": 1e-8, "ftol": 1e-10, "maxiter": 8, "maxfev": 500})
                except _ApproxTimeout:
                    pass
                x[-2:] = np.log(best_wh)
            self.consider(x)
            x = self.best.copy()
            rounds += 1
            if previous - self.best_score < 1e-8:
                break


def solve(puzzle, time_limit=30.0, max_boundary_edges=4, mode="best_effort", seed=0):
    """Return the best complete rigid layout found within the time budget.

    The default optimizes gaps, overlaps and outside area instead of requiring
    exact tiling. No right-angle, target-area-equality or perfect-fit gate is
    applied. Use mode='exact' for the previous strict solver and its protocol.
    """
    if mode == "exact":
        return solve_exact(puzzle, time_limit, max_boundary_edges)
    started = time.monotonic()
    result = _envelope(puzzle, "error")
    result.update(schema=BEST_EFFORT_OUTPUT_SCHEMA, quality=None)
    try:
        if mode not in ("best_effort", "best-effort"):
            raise ProtocolError("mode must be best_effort or exact")
        time_limit = _number(time_limit, "time_limit")
        if time_limit < 0:
            raise ProtocolError("time_limit must be nonnegative")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ProtocolError("seed must be a nonnegative integer")
        pieces, polygons, dimensions, total_area = _parse_input(puzzle)
        scale = math.sqrt(total_area)
        normalized = [p / scale for p in polygons]
        target = tuple(v / scale for v in dimensions) if dimensions is not None else None
        deadline = started + time_limit
        optimizer = _ApproxOptimizer(normalized, [p["id"] for p in pieces], target, deadline, seed)
        # A complete rigid shelf layout is constructed before any time-bound
        # search. Even a zero-second request therefore returns every piece.
        poses, width, height, _ = _approx_pack(normalized, seconds=0, seed=seed)
        x = optimizer.encode([poses[i] for i in range(len(polygons))], width, height)
        optimizer.consider(x)
        candidates = [x]
        exact_solution = None
        if time.monotonic() < deadline and time_limit >= 1:
            # Try the old search as one source of candidates; its failure is
            # not propagated and no pre-search impossibility check is called.
            seconds = min(1.5, time_limit * 0.04, deadline - time.monotonic())
            internal = _heuristic(pieces, polygons, dimensions, seconds, max_boundary_edges=4)
            if internal.get("status") == "feasible":
                by_id = {p["id"]: p for p in internal["placements"]}
                poses = [(by_id[p["id"]]["rotation"], by_id[p["id"]]["tx"] / scale,
                          by_id[p["id"]]["ty"] / scale) for p in pieces]
                x = optimizer.encode(poses, internal["rectangle"]["width"] / scale,
                                     internal["rectangle"]["height"] / scale)
                optimizer.consider(x)
                if optimizer.best_score <= 1e-10:
                    exact_solution = True
        if exact_solution is None and time.monotonic() < deadline:
            if len(polygons) == 16 and all(len(p) == 4 for p in polygons):
                search_end = min(deadline, time.monotonic() + min(40.0, time_limit * 0.4))
                starts, _ = _approx_hierarchical(normalized, search_end)
                for poses, width, height in starts:
                    x = optimizer.encode(poses, width, height)
                    optimizer.consider(x)
                    candidates.append(x)
            else:
                seconds = max(0.0, min(3.0, time_limit * 0.15, deadline - time.monotonic()))
                poses, width, height, _ = _approx_pack(normalized, seconds, seed)
                x = optimizer.encode([poses[i] for i in range(len(polygons))], width, height)
                optimizer.consider(x)
                candidates.append(x)
        if exact_solution is None and time.monotonic() < deadline:
            contact_end = min(deadline, time.monotonic() + min(8.0, time_limit * 0.15))
            known = ({"width": target[0], "height": target[1]} if target is not None else None)
            for index in range(1, len(candidates)):
                if time.monotonic() >= contact_end:
                    break
                x = candidates[index]
                _, _, wh = optimizer.decode(x)
                candidate = {"rectangle": {"width": float(wh[0]), "height": float(wh[1])},
                             "placements": optimizer.public_poses(x, 1.0)}
                fitted, _ = _contact_lsq_initializer(
                    normalized, optimizer.ids, candidate, known, contact_end)
                poses = [(math.radians(p["rotationDeg"]), *p["origin"]) for p in fitted["placements"]]
                x = optimizer.encode(poses, fitted["rectangle"]["width"], fitted["rectangle"]["height"])
                optimizer.consider(x)
                candidates[index] = x
        if exact_solution is None and time.monotonic() < deadline:
            # The current best is refined first. Then other complete starts
            # and small rigid perturbations can improve it; no candidate may
            # overwrite a better retained layout.
            remaining = deadline - time.monotonic()
            pool = ([candidates[1]] if len(candidates) > 1 else []) + [optimizer.best.copy()] + candidates[2:8]
            probe_seconds = min(6.0, remaining * 0.3 / max(1, len(pool)))
            for initial in pool:
                optimizer.optimize(initial, probe_seconds)
            optimizer.optimize(optimizer.best.copy(), (deadline - time.monotonic()) * 0.55)
            optimizer.polish((deadline - time.monotonic()) * 0.75)
            while time.monotonic() < deadline and optimizer.best_score > 1e-10:
                x = optimizer.best.copy()
                rows = x[:3 * len(polygons)].reshape((-1, 3))
                rows[:, :2] += optimizer.rng.normal(0, 0.0015, (len(polygons), 2))
                rows[:, 2] += optimizer.rng.normal(0, math.radians(0.25), len(polygons))
                optimizer.optimize(x, min(12.0, deadline - time.monotonic()))
        _, _, wh = optimizer.decode(optimizer.best)
        result.update(status="approximate", reason="best_found_within_budget",
                      rectangle={"width": float(wh[0] * scale), "height": float(wh[1] * scale)},
                      placements=optimizer.public_poses(optimizer.best, scale))
        result["quality"] = evaluate(puzzle, result)
        certificate = verify(puzzle, result)
        result["verification"] = {k: v for k, v in certificate.items() if k != "schema"}
        if certificate["valid"]:
            result.update(status="feasible", reason=None)
        result["stats"] = {
            "mode": "best_effort", "seed": seed, "timeLimitSeconds": time_limit,
            "elapsedSeconds": time.monotonic() - started,
            "bestPlacedCount": len(polygons), "completeCandidateCount": len(candidates),
            "objectiveEvaluations": optimizer.evaluations, "improvements": optimizer.accepted,
            "optimalityProven": False, "inputGeometryChanged": False,
            "usedImageContent": False, "usedOriginalPositions": False,
        }
    except (DependencyError, ImportError) as exc:
        result.update(status="error", reason=f"dependency_error: {exc}")
    except (ProtocolError, ValueError, TypeError, OverflowError) as exc:
        result.update(status="invalid_input", reason=str(exc))
    except Exception as exc:
        result.update(status="error", reason=f"solver_error: {exc}")
    return result


def _reject_constant(value):
    raise ProtocolError(f"non-finite JSON constant is not allowed: {value}")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_json(path):
    """Read strict UTF-8 JSON from a path or '-' for standard input."""
    options = {"parse_constant": _reject_constant, "object_pairs_hook": _unique_json_object}
    if str(path) == "-":
        return json.load(sys.stdin, **options)
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle, **options)


def write_json(value, path="-"):
    """Write full-precision JSON atomically, or print one JSON document."""
    rendered = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path is None or str(path) == "-":
        sys.stdout.write(rendered)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ProtocolError(message)


def _parser():
    parser = _ArgumentParser(description="Assemble convex polygons from longest-edge frames; accepts stage2 exports.")
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    solving = commands.add_parser("solve", help="Return the best complete rigid layout using local piece geometry.")
    solving.add_argument("input", help="Input JSON path, or - for stdin.")
    solving.add_argument("-o", "--output", default="-", help="Result JSON path (default: stdout).")
    solving.add_argument("--time-limit", type=float, default=30.0, help="Search budget in seconds (default: 30).")
    solving.add_argument("--mode", choices=("best-effort", "exact"), default="best-effort", help="Default: minimize residual errors and return all pieces.")
    solving.add_argument("--seed", type=int, default=0, help="Random restart seed (default: 0).")
    solving.add_argument("--max-boundary-edges", type=int, default=4, help="Only for --mode exact: edge terms per side, 1-8.")
    checking = commands.add_parser("verify", help="Independently verify a public result JSON.")
    checking.add_argument("input")
    checking.add_argument("result")
    checking.add_argument("-o", "--output", default="-")
    checking.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    evaluating = commands.add_parser("evaluate", help="Measure gaps, overlaps and outside area without a fit threshold.")
    evaluating.add_argument("input")
    evaluating.add_argument("result")
    evaluating.add_argument("-o", "--output", default="-")
    validating = commands.add_parser("validate", help="Validate the longest-edge input protocol.")
    validating.add_argument("input")
    validating.add_argument("-o", "--output", default="-")
    normalizing = commands.add_parser("normalize", help="Convert legacy or stage2 input to validated polygon-assembly/v1 JSON.")
    normalizing.add_argument("input")
    normalizing.add_argument("-o", "--output", default="-")
    return parser


def main(argv=None):
    """Command-line entry point; stdout is JSON except for --help/--version."""
    args = None
    puzzle = None
    destination = "-"
    try:
        args = _parser().parse_args(argv)
        destination = args.output
        input_paths = [args.input] + ([args.result] if args.command in {"verify", "evaluate"} else [])
        if input_paths.count("-") > 1:
            raise ProtocolError(f"{args.command} cannot read both input and result from stdin")
        if destination != "-" and any(p != "-" and Path(destination).resolve() == Path(p).resolve() for p in input_paths):
            destination = "-"
            raise ProtocolError("output must not overwrite an input file")
        puzzle = read_json(args.input)
        if args.command == "solve":
            result = solve(puzzle, time_limit=args.time_limit, max_boundary_edges=args.max_boundary_edges,
                           mode=args.mode, seed=args.seed)
            code = _STATUS_CODES[result["status"]]
        elif args.command == "verify":
            result = verify(puzzle, read_json(args.result), rtol=args.rtol)
            code = 5 if "errorCode" in result else (0 if result["valid"] else 1)
        elif args.command == "evaluate":
            result = evaluate(puzzle, read_json(args.result))
            code = 0
        elif args.command == "normalize":
            result = normalize_input(puzzle)
            code = 0
        else:
            result = validate_input(puzzle)
            code = 5 if "errorCode" in result else (0 if result["valid"] else 1)
    except (ProtocolError, json.JSONDecodeError, ValueError, UnicodeError) as exc:
        if args is not None and args.command in {"verify", "validate"}:
            result = {"schema": "polygon-assembly-check/v1", "valid": False, "errors": [str(exc)]}
            code = 1
        else:
            result = _envelope(puzzle, "invalid_input", str(exc))
            code = 4
    except OSError as exc:
        result = _envelope(puzzle, "error", f"io_error: {exc}")
        code = 5
    except (DependencyError, ImportError) as exc:
        result = _envelope(puzzle, "error", f"dependency_error: {exc}")
        code = 5
    try:
        write_json(result, destination)
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Cannot write result JSON: {exc}\n")
        if destination != "-":
            write_json(_envelope(puzzle, "error", f"output_write_error: {exc}"))
        return 5
    return code


if __name__ == "__main__":
    raise SystemExit(main())
