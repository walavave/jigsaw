"""Tests for Stage 2 piece extraction (``stage2_piece_extraction/extract_pieces.py``).

The suite has three layers:

1. Pure geometry / response helpers, exercised on small hand-checked arrays.
2. A fully synthetic end-to-end board: a dark bench plus 16 drawn convex quads
   with a bright low-frequency texture, so the colour difference from the empty
   capture is large.  ``extract_pieces`` runs twice (once per output directory)
   to check the byte-for-byte determinism of ``pieces.json`` and of the minimal
   ``pieces_base_edge.json`` hand-off.  Every piece is cropped 1:1 in its own
   base-edge frame (longest side P0=(0,0) -> P1=(baseLength,0), clockwise in the
   y-down image frame), so the assertions are on that frame, not on a rescaled
   square canvas.
3. The real rectified capture from Stage 1, skipped when it is not available.

All artefacts live in ``tempfile.TemporaryDirectory()``; nothing is written into
the repository tree.  Images are read/written with the unicode-safe helpers from
``shared/jigsaw_common/io.py`` because the repository path contains non-ASCII
characters (``cv2.imread``/``cv2.imwrite`` fail on such paths on Windows).
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "stage2_piece_extraction"))

from extract_pieces import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_VALUES,
    MAX_LOCALIZER_SHIFT_PX,
    MIN_SAFE_SIDE_PX,
    SCAN_ROW_TOLERANCE_PX,
    _apply_transform,
    _resolve_config,
    base_edge_frame,
    base_edge_record,
    colour_response,
    combine_quality,
    dedupe_polygon,
    edge_response,
    extract_pieces,
    foreground_mask,
    is_convex_polygon,
    normalize_piece,
    ordered_polygon,
    polygon_area,
    polygon_centroid,
    polygon_iou,
    polygon_signed_area,
    prune_near_collinear,
    rasterize_polygon,
    render_debug,
    scan_order_indices,
)
from jigsaw_common.io import imread, imwrite, write_json  # noqa: E402

REAL_RECTIFIED_DIR = ROOT / "stage1_capture_calibration/output/rectified"
REAL_PIECES = REAL_RECTIFIED_DIR / "pieces_rectified.png"
REAL_EMPTY = REAL_RECTIFIED_DIR / "empty_rectified.png"
REAL_RECTIFICATION = REAL_RECTIFIED_DIR / "rectification.json"
REAL_LAYOUT = ROOT / "stage1_capture_calibration/output/cut_pdf/jigsaw_layout.json"
EXAMPLE_CONFIG = ROOT / "stage2_piece_extraction/config/extraction.example.json"

# Synthetic capture geometry.  1400x1000 px at 5 px/mm is a 280x200 mm bench,
# large enough for 16 pieces while keeping the whole run around one second.
SYNTH_CANVAS_W = 1400
SYNTH_CANVAS_H = 1000
SYNTH_PX_PER_MM = 5.0
SYNTH_IGNORE_BORDER_MM = 9.0  # 45 px border at 5 px/mm
SYNTH_EDGE_PROFILE_PX = 2.0
SYNTH_SUPERSAMPLE = 4
SYNTH_SEED = 20261028
SYNTH_BG = (18, 18, 18)

# The 16 placed quads are matched to detections within this tolerance.
SYNTH_CORNER_TOLERANCE_PX = 2.0

REQUIRED_PIECE_FIELDS = {
    "piece_id",
    "rgba_path",
    "mask_path",
    "debug_path",
    "corners_px",
    "corner_count",
    "center_px",
    "area_px",
    "area_mm2",
    "bbox_px",
    "perimeter_px",
    "quality",
    "quality_breakdown",
    "needs_manual_review",
    "manual_review_reasons",
    "min_side_px",
    # v1.4.0 base-edge frame (1:1 crop in the rectified canvas unit).
    "base_edge",
    "base_edge_corners_px",
    "base_length_px",
    "base_side_index",
    "rotation_deg",
    "piece_size_px",
    "image_corners_px",
    "rectified_to_piece_matrix",
    "rectified_to_base_edge_matrix",
}
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "algorithm_version",
    "session_id",
    "coordinate_system",
    "piece_frame",
    "piece_scale",
    "piece_margin_px",
    "canvas_px",
    "px_per_mm",
    "corner_order",
    "expected_piece_count",
    "detected_piece_count",
    "inputs",
    "segmentation",
    "summary",
    "config",
    "manual_review",
    "pieces",
}
# Per-piece fields that were replaced by the base-edge frame in Stage 2 1.4.0.
REMOVED_PIECE_FIELDS = {"normalized_corners_px", "normalized_short_side_px"}
REMOVED_TOP_LEVEL_FIELDS = {"piece_size_px"}
# Keys of one record in ``pieces_base_edge.json`` (the minimal assembly hand-off).
BASE_EDGE_RECORD_FIELDS = {"id", "baseLength", "p2", "p3"}
# Documented vocabulary for ``manual_review_reasons`` (Stage 2 >= 1.1.0).
REVIEW_REASON_VOCABULARY = frozenset(
    {"quality_below_min", "short_side", "localizer_disagreement"}
)


# ---------------------------------------------------------------------------
# small geometry helpers used only by the tests
# ---------------------------------------------------------------------------


def _square(size: float = 4.0) -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]],
        dtype=np.float64,
    )


def _rotated_rectangle(
    centre: tuple[float, float],
    half_width: float,
    half_height: float,
    angle_rad: float,
) -> np.ndarray:
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    local = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float64,
    )
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return local @ rotation.T + np.asarray(centre, dtype=np.float64)


def _perimeter(polygon: np.ndarray) -> float:
    points = np.asarray(polygon, dtype=np.float64)
    return float(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum())


def _best_cyclic_vertex_error(detected: np.ndarray, target: np.ndarray) -> float:
    """Max vertex distance under the best cyclic shift (either winding)."""

    best = math.inf
    for orientation in (detected, detected[::-1]):
        count = len(orientation)
        if count != len(target):
            raise AssertionError("vertex counts differ")
        for shift in range(count):
            rolled = np.roll(orientation, -shift, axis=0)
            best = min(best, float(np.linalg.norm(rolled - target, axis=1).max()))
    return best


def _ordered_edge_lengths(points) -> list[float]:
    """Lengths of the polygon edges in the given vertex order."""

    points = np.asarray(points, dtype=np.float64)
    return [
        float(np.linalg.norm(points[(index + 1) % len(points)] - points[index]))
        for index in range(len(points))
    ]


def _assert_base_edge_piece_contract(
    test_case: unittest.TestCase, piece: dict, margin_px: float
) -> None:
    """Check the Stage 2 >= 1.4.0 base-edge frame of one piece.

    The hand-off contract is: the longest side is the base, ``P0 = (0, 0)`` and
    ``P1 = (baseLength, 0)``, ``P2``/``P3`` keep the clockwise (y-down) order and
    the frame is a pure rotation (unit scale) of ``corners_px``.  The crop is the
    frame bounding box plus ``margin_px`` on every side, at 1:1 canvas scale.
    Everything here is derived from the payload itself, so it also works for a
    capture whose corner coordinates the test does not know in advance.
    """

    piece_id = piece["piece_id"]
    corners = np.asarray(piece["corners_px"], dtype=np.float64)
    frame = np.asarray(piece["base_edge_corners_px"], dtype=np.float64)
    record = piece["base_edge"]
    base_length = float(piece["base_length_px"])
    base_index = int(piece["base_side_index"])

    test_case.assertEqual(frame.shape, (4, 2), msg=piece_id)

    # 1) Frame definition: P0/P1 on the +x axis, P2/P3 below it, clockwise.
    test_case.assertAlmostEqual(float(frame[0][0]), 0.0, places=9, msg=piece_id)
    test_case.assertAlmostEqual(float(frame[0][1]), 0.0, places=9, msg=piece_id)
    test_case.assertAlmostEqual(float(frame[1][1]), 0.0, places=9, msg=piece_id)
    test_case.assertGreater(float(frame[2][1]), 0.0, msg=piece_id)
    test_case.assertGreater(float(frame[3][1]), 0.0, msg=piece_id)
    test_case.assertGreater(
        float(polygon_signed_area(frame)), 0.0, msg=f"{piece_id}: not clockwise"
    )
    test_case.assertGreater(base_length, 0.0, msg=piece_id)

    # 2) The minimal hand-off record agrees with the frame.
    test_case.assertEqual(set(record), BASE_EDGE_RECORD_FIELDS, msg=piece_id)
    test_case.assertEqual(record["id"], piece_id, msg=piece_id)
    test_case.assertAlmostEqual(
        float(record["baseLength"]), base_length, places=6, msg=piece_id
    )
    test_case.assertAlmostEqual(
        float(record["baseLength"]), float(frame[1][0]), places=6, msg=piece_id
    )
    np.testing.assert_allclose(
        np.asarray(record["p2"], dtype=np.float64), frame[2], atol=1e-5, err_msg=piece_id
    )
    np.testing.assert_allclose(
        np.asarray(record["p3"], dtype=np.float64), frame[3], atol=1e-5, err_msg=piece_id
    )

    # 3) The base really is the longest side, in both frames (the payload
    #    rounds corners to 3 decimals, hence the 1e-2 tolerances).
    frame_edges = _ordered_edge_lengths(frame)
    corner_edges = _ordered_edge_lengths(corners)
    test_case.assertAlmostEqual(
        max(frame_edges), frame_edges[0], places=9, msg=f"{piece_id}: base not longest"
    )
    test_case.assertAlmostEqual(
        frame_edges[0], base_length, places=6, msg=piece_id
    )
    test_case.assertAlmostEqual(
        max(corner_edges), base_length, delta=1e-2, msg=f"{piece_id}: base not longest"
    )
    np.testing.assert_allclose(
        sorted(frame_edges), sorted(corner_edges), atol=1e-2, err_msg=piece_id
    )
    test_case.assertAlmostEqual(
        polygon_area(frame),
        float(piece["area_px"]),
        delta=0.01 * float(piece["area_px"]),
        msg=f"{piece_id}: area changed by the frame",
    )

    # 4) Both matrices are rigid unit-scale rotations.
    matrix = np.asarray(piece["rectified_to_piece_matrix"], dtype=np.float64)
    base_matrix = np.asarray(piece["rectified_to_base_edge_matrix"], dtype=np.float64)
    for name, candidate in (
        ("rectified_to_piece_matrix", matrix),
        ("rectified_to_base_edge_matrix", base_matrix),
    ):
        test_case.assertEqual(candidate.shape, (3, 3), msg=f"{piece_id} {name}")
        np.testing.assert_allclose(
            candidate[2], [0.0, 0.0, 1.0], atol=1e-9, err_msg=f"{piece_id} {name}"
        )
        linear = candidate[:2, :2]
        np.testing.assert_allclose(
            linear @ linear.T, np.eye(2), atol=1e-6, err_msg=f"{piece_id} {name}"
        )
        test_case.assertAlmostEqual(
            float(np.linalg.det(linear)), 1.0, places=6, msg=f"{piece_id} {name}"
        )

    # ``corners_px`` starts at the original polygon vertex 0 while the frame
    # starts at the base edge: roll by ``base_side_index`` before comparing.
    mapped = np.roll(_apply_transform(base_matrix, corners), -base_index, axis=0)
    np.testing.assert_allclose(mapped, frame, atol=5e-3, err_msg=piece_id)
    # The inverse transform must return the original polygon vertices (the
    # payload's ``corners_px`` keeps the fitted vertex order, which is a
    # permutation of the ordered frame vertices, so match as a point set).
    inverse = np.linalg.inv(base_matrix)
    back = _apply_transform(inverse, frame)
    for point in back:
        test_case.assertLess(
            float(np.linalg.norm(corners - point, axis=1).min()),
            5e-3,
            msg=f"{piece_id}: inverse frame does not return a corner",
        )

    # 5) Crop = frame bbox + 2 * margin, 1:1 scale, base horizontal in crop.
    margin = float(margin_px)
    width, height = (int(value) for value in piece["piece_size_px"])
    test_case.assertGreaterEqual(width, 2, msg=piece_id)
    test_case.assertGreaterEqual(height, 2, msg=piece_id)
    extent = frame.max(axis=0) - frame.min(axis=0)
    test_case.assertLessEqual(
        abs(width - math.ceil(float(extent[0]) + 2.0 * margin)), 1, msg=piece_id
    )
    test_case.assertLessEqual(
        abs(height - math.ceil(float(extent[1]) + 2.0 * margin)), 1, msg=piece_id
    )
    image_corners = np.asarray(piece["image_corners_px"], dtype=np.float64)
    test_case.assertEqual(image_corners.shape, (4, 2), msg=piece_id)
    test_case.assertAlmostEqual(
        float(image_corners[0][1]), float(image_corners[1][1]), places=6,
        msg=f"{piece_id}: base edge is not horizontal in the crop",
    )
    np.testing.assert_allclose(
        np.roll(_apply_transform(matrix, corners), -base_index, axis=0),
        image_corners,
        atol=5e-3,
        err_msg=piece_id,
    )
    offset = image_corners - frame
    np.testing.assert_allclose(
        offset, np.repeat(offset[0:1], 4, axis=0), atol=2e-3, err_msg=piece_id
    )
    lower = frame.min(axis=0)
    np.testing.assert_allclose(
        offset[0], [margin - lower[0], margin - lower[1]], atol=1e-2, err_msg=piece_id
    )
    translation_delta = matrix[:2, 2] - base_matrix[:2, 2]
    np.testing.assert_allclose(
        translation_delta, offset[0], atol=1e-3, err_msg=piece_id
    )
    test_case.assertGreaterEqual(
        float(image_corners.min()), margin - 1e-3, msg=f"{piece_id}: outside margin"
    )
    test_case.assertLessEqual(
        float(image_corners[:, 0].max()), width - margin + 1e-3, msg=piece_id
    )
    test_case.assertLessEqual(
        float(image_corners[:, 1].max()), height - margin + 1e-3, msg=piece_id
    )

    # 6) The per-piece minimum side mirrors the quality breakdown.
    breakdown = piece["quality_breakdown"]
    test_case.assertIn("min_side_px", breakdown, msg=piece_id)
    if piece["min_side_px"] is not None:
        test_case.assertAlmostEqual(
            float(piece["min_side_px"]),
            float(breakdown["min_side_px"]),
            places=5,
            msg=piece_id,
        )
    # 7) The old normalized-canvas fields are gone.
    for removed in REMOVED_PIECE_FIELDS:
        test_case.assertNotIn(removed, piece, msg=piece_id)


def _assert_manual_review_contract(
    test_case: unittest.TestCase, payload: dict, config: dict
) -> None:
    """Check the Stage 2 >= 1.1.0 triage contract.

    ``needs_manual_review`` is an advisory flag, so no capture-specific piece id
    may be hard-coded: instead the flags must be *complete* (no silent
    offenders), *explainable* (reasons from the documented vocabulary) and
    *consistent* with the payload-level ``manual_review`` list.
    """

    min_quality = float(config["min_quality"])
    pieces = payload["pieces"]
    piece_ids = [piece["piece_id"] for piece in pieces]

    flagged = sorted(
        piece["piece_id"] for piece in pieces if piece["needs_manual_review"]
    )
    test_case.assertEqual(payload["manual_review"], flagged)
    test_case.assertLessEqual(set(payload["manual_review"]), set(piece_ids))

    for piece in pieces:
        piece_id = piece["piece_id"]
        reasons = piece["manual_review_reasons"]
        test_case.assertIsInstance(reasons, list, msg=piece_id)
        test_case.assertLessEqual(set(reasons), REVIEW_REASON_VOCABULARY, msg=piece_id)

        breakdown = piece["quality_breakdown"]
        test_case.assertIn("min_side_px", breakdown, msg=piece_id)
        test_case.assertIn("agreement_shift_px", breakdown, msg=piece_id)
        quality = float(piece["quality"])
        min_side = float(breakdown["min_side_px"])
        shift = breakdown["agreement_shift_px"]

        expected_reasons: list[str] = []
        if quality < min_quality:
            expected_reasons.append("quality_below_min")
        if min_side < MIN_SAFE_SIDE_PX:
            expected_reasons.append("short_side")
        if shift is not None and float(shift) > MAX_LOCALIZER_SHIFT_PX:
            expected_reasons.append("localizer_disagreement")
        test_case.assertEqual(reasons, expected_reasons, msg=piece_id)
        test_case.assertEqual(bool(reasons), piece["needs_manual_review"], msg=piece_id)

        if not piece["needs_manual_review"]:
            # Flags are complete: a clean piece really satisfies every triage
            # rule, so there are no silent offenders.
            test_case.assertGreaterEqual(quality, min_quality, msg=piece_id)
            test_case.assertGreaterEqual(min_side, MIN_SAFE_SIDE_PX, msg=piece_id)
            if shift is not None:
                test_case.assertLessEqual(
                    float(shift), MAX_LOCALIZER_SHIFT_PX, msg=piece_id
                )


def _assert_scan_order_contract(test_case: unittest.TestCase, centres) -> None:
    """The piece ids must be in the documented scan order (Stage 2 >= 1.2.0).

    Ids are row-clustered: rows run top to bottom, and within a row pieces are
    ordered by ``x`` (so a sub-pixel ``y`` jitter inside a row cannot reorder
    them).  Applying :func:`scan_order_indices` to an already-ordered list is
    therefore the identity.  This is contract-derived, so it does not encode the
    piece layout of a particular capture.
    """

    points = [np.asarray(centre, dtype=np.float64) for centre in centres]
    test_case.assertEqual(
        scan_order_indices(points),
        list(range(len(points))),
        msg="piece ids are not in scan order (rows top-to-bottom, then x)",
    )


# ---------------------------------------------------------------------------
# pure helper tests
# ---------------------------------------------------------------------------


class Stage2HelperTests(unittest.TestCase):
    def test_polygon_area_signed_area_and_centroid(self) -> None:
        square = _square(2.0)
        self.assertAlmostEqual(polygon_signed_area(square), 4.0, places=9)
        self.assertAlmostEqual(polygon_area(square), 4.0, places=9)
        # Reversing the winding flips the sign but not the magnitude.
        self.assertAlmostEqual(polygon_signed_area(square[::-1]), -4.0, places=9)
        self.assertAlmostEqual(polygon_area(square[::-1]), 4.0, places=9)

        triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
        self.assertAlmostEqual(polygon_area(triangle), 6.0, places=9)
        np.testing.assert_allclose(polygon_centroid(triangle), [4.0 / 3.0, 1.0])
        np.testing.assert_allclose(polygon_centroid(square), [1.0, 1.0])

        # Degenerate polygons fall back to the mean of the vertices.
        np.testing.assert_allclose(
            polygon_centroid(np.array([[0.0, 0.0], [1.0, 1.0]])),
            [0.5, 0.5],
        )
        self.assertEqual(polygon_area(np.array([[0.0, 0.0]])), 0.0)

    def test_ordered_polygon_starts_top_left_with_positive_area(self) -> None:
        # A valid (non self-intersecting) square in an arbitrary cyclic order.
        shuffled = np.array(
            [[12.0, 1.0], [12.0, 3.0], [10.0, 3.0], [10.0, 1.0]],
            dtype=np.float64,
        )
        expected = np.array(
            [[10.0, 1.0], [12.0, 1.0], [12.0, 3.0], [10.0, 3.0]],
            dtype=np.float64,
        )
        for candidate in (shuffled, shuffled[::-1], np.roll(shuffled, 2, axis=0)):
            ordered = ordered_polygon(candidate)
            np.testing.assert_allclose(ordered, expected)
            self.assertGreater(polygon_signed_area(ordered), 0.0)
            # First vertex is top-most, then left-most.
            top = ordered[np.lexsort((ordered[:, 0], ordered[:, 1]))[0]]
            np.testing.assert_allclose(ordered[0], top)

    def test_dedupe_polygon_removes_duplicate_vertices(self) -> None:
        polygon = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        deduped = dedupe_polygon(polygon)
        self.assertEqual(len(deduped), 4)
        self.assertAlmostEqual(polygon_area(deduped), 1.0, places=9)

        near_duplicate = np.array(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        self.assertEqual(len(dedupe_polygon(near_duplicate)), 4)

    def test_prune_near_collinear_drops_spurious_vertices(self) -> None:
        # Extra vertex exactly on the top edge (split edge).
        split_square = np.array(
            [[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]]
        )
        pruned = prune_near_collinear(split_square)
        self.assertEqual(len(pruned), 4)
        self.assertAlmostEqual(polygon_area(pruned), 1600.0, places=6)
        self.assertTrue(is_convex_polygon(pruned))

        # Extra vertex creating a short (sub-min-edge) edge.  The short-edge
        # rule may drop either end of the sliver, so only pin down the count,
        # convexity and an area close to the original 1600 px^2.
        short_edge = np.array(
            [[0.0, 0.0], [1.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]]
        )
        pruned = prune_near_collinear(short_edge)
        self.assertEqual(len(pruned), 4)
        self.assertTrue(is_convex_polygon(pruned))
        self.assertAlmostEqual(polygon_area(pruned), 1600.0, delta=25.0)

    def test_convexity_detection(self) -> None:
        self.assertTrue(is_convex_polygon(_square(4.0)))
        self.assertTrue(is_convex_polygon(_square(4.0)[::-1]))
        self.assertTrue(is_convex_polygon(np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 3.0]])))
        # Concave "dart": the third vertex lies inside the outer triangle.
        dart = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 1.0], [0.0, 4.0]])
        self.assertFalse(is_convex_polygon(dart))
        # Collinear vertices are tolerated, not treated as a bend.
        self.assertTrue(
            is_convex_polygon(
                np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
            )
        )
        self.assertFalse(is_convex_polygon(np.array([[0.0, 0.0], [1.0, 1.0]])))

    def test_polygon_iou_bounds(self) -> None:
        square = _square(4.0)
        self.assertAlmostEqual(polygon_iou(square, square.copy()), 1.0, places=9)
        # Overlapping copy: 0 < IoU < 1 and symmetric.  The exact value is a
        # little above the analytic 1/3 because the rasteriser also paints the
        # boundary pixels, so only loose bounds are asserted here.
        shifted = square + np.array([2.0, 0.0])
        iou = polygon_iou(square, shifted)
        self.assertGreater(iou, 0.3)
        self.assertLess(iou, 0.5)
        self.assertAlmostEqual(iou, polygon_iou(shifted, square), places=12)
        further = square + np.array([4.0, 0.0])
        self.assertLess(polygon_iou(square, further), iou)
        disjoint = square + np.array([50.0, 50.0])
        self.assertAlmostEqual(polygon_iou(square, disjoint), 0.0, places=9)

    def test_rasterize_polygon_matches_analytic_area(self) -> None:
        polygon = np.array([[5.0, 5.0], [205.0, 5.0], [205.0, 205.0], [5.0, 205.0]])
        mask = rasterize_polygon(polygon, (230, 230))
        self.assertEqual(mask.shape, (230, 230))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertLessEqual(set(np.unique(mask)), {0, 255})
        analytic_area = polygon_area(polygon)  # 200 x 200 = 40000 px^2
        self.assertAlmostEqual(analytic_area, 40000.0, places=6)
        # Boundary pixels are painted too, so allow roughly half a pixel of
        # error per unit of perimeter.
        self.assertLessEqual(abs(int((mask > 0).sum()) - analytic_area), 0.5 * _perimeter(polygon) + 2.0)
        self.assertGreater(int(mask[100, 100]), 0)  # interior is set
        self.assertEqual(int(mask[220, 220]), 0)  # outside stays clear
        # Fewer than three points rasterizes to an empty mask.
        self.assertEqual(int(rasterize_polygon(np.array([[0.0, 0.0], [1.0, 0.0]]), (10, 10)).sum()), 0)

    def test_apply_transform_affine_and_projective(self) -> None:
        affine = np.array([[2.0, 0.0, 3.0], [0.0, 3.0, 4.0], [0.0, 0.0, 1.0]])
        transformed = _apply_transform(affine, np.array([[1.0, 1.0], [0.0, 0.0]]))
        np.testing.assert_allclose(transformed, [[5.0, 7.0], [3.0, 4.0]])
        identity = np.eye(3)
        points = np.array([[12.5, -3.0]])
        np.testing.assert_allclose(_apply_transform(identity, points), points)
        # The homogeneous divide is honoured.
        projective = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1e-3, 0.0, 1.0]])
        np.testing.assert_allclose(
            _apply_transform(projective, np.array([[1000.0, 0.0]])),
            [[500.0, 0.0]],
        )

    def test_base_edge_frame_puts_the_longest_side_on_the_x_axis(self) -> None:
        angle = 0.25
        corners = _rotated_rectangle((700.0, 900.0), 200.0, 125.0, angle)
        frame = base_edge_frame(corners)

        self.assertEqual(
            set(frame),
            {"matrix", "corners", "base_length", "base_side_index", "rotation_deg", "ordered"},
        )
        frame_corners = np.asarray(frame["corners"], dtype=np.float64)
        self.assertEqual(frame_corners.shape, (4, 2))
        # The base is the longest side (2 * 200 px), put on the +x axis.
        self.assertAlmostEqual(frame["base_length"], 400.0, places=6)
        np.testing.assert_allclose(frame_corners[0], [0.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(float(frame_corners[1][1]), 0.0, places=9)
        self.assertAlmostEqual(
            float(frame_corners[1][0]), frame["base_length"], places=9
        )
        # Clockwise in the y-down image frame: P2/P3 below the base edge.
        self.assertGreater(float(frame_corners[2][1]), 0.0)
        self.assertGreater(float(frame_corners[3][1]), 0.0)
        self.assertAlmostEqual(
            polygon_signed_area(frame_corners), polygon_area(corners), places=6
        )
        # The frame keeps the polygon order, starting at the base side.
        np.testing.assert_allclose(frame["ordered"], ordered_polygon(corners))
        base_index = int(frame["base_side_index"])
        # Unit scale: pure rotation + translation.
        matrix = np.asarray(frame["matrix"], dtype=np.float64)
        self.assertEqual(matrix.shape, (3, 3))
        np.testing.assert_allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12)
        linear = matrix[:2, :2]
        np.testing.assert_allclose(linear @ linear.T, np.eye(2), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(linear)), 1.0, places=12)
        # 1:1 transform: the frame is the polygon itself, just rotated.
        np.testing.assert_allclose(
            sorted(_ordered_edge_lengths(frame_corners)),
            sorted(_ordered_edge_lengths(corners)),
            atol=1e-9,
        )
        mapped = np.roll(_apply_transform(matrix, frame["ordered"]), -base_index, axis=0)
        np.testing.assert_allclose(mapped, frame_corners, atol=1e-9)
        # ``rotation_deg`` describes the matrix as a y-down rotation.
        theta = math.radians(frame["rotation_deg"])
        np.testing.assert_allclose(
            linear,
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            atol=1e-12,
        )
        self.assertGreaterEqual(frame["rotation_deg"], -180.0)
        self.assertLessEqual(frame["rotation_deg"], 180.0)
        self.assertAlmostEqual(
            abs(frame["rotation_deg"]), math.degrees(angle), places=6
        )

    def test_base_edge_frame_rejects_degenerate_polygons(self) -> None:
        with self.assertRaisesRegex(ValueError, "longest edge has zero length"):
            base_edge_frame(np.full((4, 2), 5.0))
        # All-collinear vertices: every "edge" has zero area, so the result
        # cannot be expressed as a clockwise quad.
        with self.assertRaisesRegex(ValueError, "clockwise base-edge quad"):
            base_edge_frame(
                np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
            )

    def test_base_edge_record_is_minimal_and_full_precision(self) -> None:
        corners = np.array(
            [[0.0, 0.0], [400.0, 0.0], [400.25, 250.5], [0.125, 250.0]],
            dtype=np.float64,
        )
        record = base_edge_record("S07", corners, 400.0)
        self.assertEqual(set(record), BASE_EDGE_RECORD_FIELDS)
        self.assertEqual(record["id"], "S07")
        self.assertIsInstance(record["baseLength"], float)
        self.assertEqual(record["baseLength"], 400.0)
        # P2/P3 are copied at full float precision, not scaled or re-ordered.
        self.assertEqual(record["p2"], [400.25, 250.5])
        self.assertEqual(record["p3"], [0.125, 250.0])
        for value in (*record["p2"], *record["p3"]):
            self.assertIsInstance(value, float)


    def test_normalize_piece_crops_1to1_in_base_edge_frame(self) -> None:
        margin_px = 10.0
        image = np.full((300, 320, 3), 20, dtype=np.uint8)
        polygon = _rotated_rectangle((160.0, 150.0), 80.0, 50.0, 0.25)
        cv2.fillPoly(
            image,
            [np.round(polygon).astype(np.int32)],
            (230, 240, 250),
        )
        normalized = normalize_piece(image, polygon, margin_px)

        self.assertEqual(
            set(normalized),
            {
                "matrix",
                "base_edge_matrix",
                "base_edge_corners",
                "image_corners",
                "base_length",
                "base_side_index",
                "rotation_deg",
                "size",
                "rgba",
                "mask",
            },
        )
        width, height = (int(value) for value in normalized["size"])
        frame = np.asarray(normalized["base_edge_corners"], dtype=np.float64)
        image_corners = np.asarray(normalized["image_corners"], dtype=np.float64)
        extent = frame.max(axis=0) - frame.min(axis=0)
        self.assertEqual(width, math.ceil(float(extent[0]) + 2.0 * margin_px))
        self.assertEqual(height, math.ceil(float(extent[1]) + 2.0 * margin_px))
        self.assertEqual(width, 180)
        self.assertEqual(height, 120)
        self.assertAlmostEqual(float(normalized["base_length"]), 160.0, places=6)

        # 1:1 crop: lengths are preserved exactly (no per-piece rescaling).
        np.testing.assert_allclose(
            sorted(_ordered_edge_lengths(image_corners)),
            sorted(_ordered_edge_lengths(polygon)),
            atol=1e-6,
        )
        self.assertAlmostEqual(
            polygon_area(image_corners), polygon_area(polygon), places=6
        )
        # Canonical base-edge frame: base along +x on top, P2/P3 below.
        np.testing.assert_allclose(frame[0], [0.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(float(frame[1][1]), 0.0, places=9)
        self.assertAlmostEqual(float(frame[1][0]), 160.0, places=6)
        self.assertGreater(float(frame[2][1]), 0.0)
        self.assertGreater(float(frame[3][1]), 0.0)
        self.assertGreater(float(polygon_signed_area(frame)), 0.0)
        # The crop corners are the frame translated by the margin.
        lower = frame.min(axis=0)
        offset = image_corners - frame
        np.testing.assert_allclose(offset, np.repeat(offset[0:1], 4, axis=0), atol=1e-9)
        np.testing.assert_allclose(
            offset[0], [margin_px - lower[0], margin_px - lower[1]], atol=1e-9
        )
        # ``matrix`` maps rectified corners to crop pixels; the base is horizontal.
        base_index = int(normalized["base_side_index"])
        np.testing.assert_allclose(
            np.roll(_apply_transform(normalized["matrix"], polygon), -base_index, axis=0),
            image_corners,
            atol=1e-6,
        )
        translation = (
            np.asarray(normalized["matrix"], dtype=np.float64)[:2, 2]
            - np.asarray(normalized["base_edge_matrix"], dtype=np.float64)[:2, 2]
        )
        np.testing.assert_allclose(translation, offset[0], atol=1e-9)
        self.assertAlmostEqual(
            float(normalized["rotation_deg"]), -math.degrees(0.25), places=6
        )

        rgba = normalized["rgba"]
        mask = normalized["mask"]
        self.assertEqual(rgba.shape, (height, width, 4))
        self.assertEqual(mask.shape, (height, width))
        self.assertEqual(rgba.dtype, np.uint8)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertLessEqual(set(np.unique(mask)), {0, 255})
        np.testing.assert_array_equal(rgba[..., 3], mask)
        self.assertTrue(bool(np.all(rgba[mask == 0] == 0)))
        # The warped brightness inside the mask matches the drawn paper.
        self.assertGreater(float(np.median(rgba[mask > 0][:, 0])), 180.0)
        # The analytic polygon and its rasterized alpha describe the same area.
        analytic = polygon_area(image_corners)
        mask_area = float((mask > 0).sum())
        self.assertLessEqual(
            abs(mask_area - analytic),
            0.01 * analytic + 0.5 * _perimeter(image_corners),
            msg=f"mask area {mask_area} vs analytic {analytic}",
        )

    def test_render_debug_supports_non_square_crops(self) -> None:
        width, height = 140, 60
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = 200
        rgba[..., 3] = 255
        corners = np.array(
            [[10.0, 10.0], [120.0, 10.0], [120.0, 50.0], [10.0, 50.0]],
            dtype=np.float64,
        )
        debug = render_debug(rgba, corners, "S01", 0.91)
        self.assertEqual(debug.shape, (height, width, 3))
        self.assertEqual(debug.dtype, np.uint8)
        # The base edge P0 -> P1 is highlighted in orange, BGR (0, 165, 255).
        base_band = debug[7:14, 40:100]
        self.assertGreater(int(base_band[..., 2].max()), 200)
        self.assertGreater(int(base_band[..., 1].max()), 120)
        # The background outside the piece is the checkerboard, not uniform white.
        self.assertGreater(
            int(np.unique(debug[55:58, 0:140].reshape(-1, 3), axis=0).shape[0]), 1
        )

    def test_combine_quality_weighted_score(self) -> None:
        quality = {"fit": 0.8, "shape": 0.6, "support": 0.4, "vertices": 1.0, "convex": 1.0}
        self.assertAlmostEqual(combine_quality(quality), 0.71, places=9)
        self.assertEqual(combine_quality({}), 0.0)
        self.assertAlmostEqual(combine_quality({key: 2.0 for key in quality}), 1.0, places=9)
        self.assertAlmostEqual(combine_quality({"fit": -1.0}), 0.0, places=9)
        # The fit term carries the largest weight.
        base = {"fit": 0.5, "shape": 0.5, "support": 0.5, "vertices": 0.5, "convex": 0.5}
        stronger_fit = dict(base, fit=0.9)
        self.assertGreater(combine_quality(stronger_fit), combine_quality(base))

    def test_colour_and_edge_response(self) -> None:
        config = {"blur_ksize": 5}
        empty = np.full((200, 200, 3), SYNTH_BG, dtype=np.uint8)
        pieces = empty.copy()
        pieces[60:140, 60:140] = (225, 235, 245)

        colour = colour_response(pieces, empty, config)
        self.assertEqual(colour.shape, (200, 200))
        self.assertEqual(colour.dtype, np.float32)
        self.assertTrue(bool(np.all(np.isfinite(colour))))
        self.assertTrue(bool(np.all(colour >= 0.0)))
        self.assertGreater(float(colour[100, 100]), 50.0)
        self.assertLess(float(colour[5, 5]), 1e-6)
        # Identical captures carry no piece signal at all.
        np.testing.assert_array_equal(colour_response(empty, empty, config), np.zeros_like(colour))

        response = edge_response(pieces, empty, colour, config)
        self.assertEqual(response.shape, (200, 200))
        self.assertEqual(response.dtype, np.float32)
        self.assertTrue(bool(np.all(response >= colour - 1e-5)))
        self.assertLess(float(response[5, 5]), 1e-6)
        # The gradient term adds a sharp bump at the paper edge.
        self.assertGreater(float((response - colour).max()), 10.0)
        empty_colour = colour_response(empty, empty, config)
        np.testing.assert_array_equal(
            edge_response(empty, empty, empty_colour, config),
            np.zeros_like(empty_colour),
        )

    def test_foreground_mask_threshold_and_border_guard(self) -> None:
        response = np.zeros((120, 120), dtype=np.float32)
        response[20:80, 20:80] = 150.0  # 60x60 piece well inside the border
        response[108:118, 108:118] = 150.0  # cut by the border guard
        response[50, 50] = 255.0  # isolated speck, removed by the opening
        config = dict(DEFAULT_CONFIG_VALUES)
        config.update(
            {
                "threshold_mode": "otsu",
                "threshold_min": 10.0,
                "threshold_max": 60.0,
                "open_ksize": 5,
                "close_ksize": 1,
            }
        )
        mask, threshold = foreground_mask(response, config, ignore_border_px=12)
        self.assertEqual(mask.shape, response.shape)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertLessEqual(set(np.unique(mask)), {0, 255})
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 10.0)
        self.assertLessEqual(threshold, 60.0)
        self.assertEqual(int((mask > 0).sum()), 60 * 60)
        components = cv2.connectedComponents(mask)[0]
        self.assertEqual(components, 2)  # background + the single surviving square
        self.assertEqual(int(mask[:12].sum()) + int(mask[-12:].sum()), 0)
        self.assertEqual(int(mask[:, :12].sum()) + int(mask[:, -12:].sum()), 0)

    def test_resolve_config_merges_defaults_and_overrides(self) -> None:
        # ``_resolve_config(None)`` starts from the code defaults and then
        # layers the shipped example config on top when it exists.
        expected = dict(DEFAULT_CONFIG_VALUES)
        if DEFAULT_CONFIG.exists():
            expected.update(json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8")))
        self.assertEqual(_resolve_config(None), expected)
        self.assertEqual(DEFAULT_CONFIG_VALUES["expected_piece_count"], 16)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps({"expected_piece_count": 7, "min_quality": 0.5}),
                encoding="utf-8",
            )
            merged = _resolve_config(config_path)
            self.assertEqual(merged["expected_piece_count"], 7)
            self.assertEqual(merged["min_quality"], 0.5)
            # v1.4.0 dropped the normalized-canvas output size; the crop is 1:1
            # and its padding is ``piece_margin_px`` instead.
            self.assertNotIn("piece_output_size_px", merged)
            self.assertNotIn("piece_output_size_px", DEFAULT_CONFIG_VALUES)
            self.assertEqual(merged["piece_margin_px"], 20.0)
            # Existing defaults are not mutated by the merge.
            self.assertEqual(DEFAULT_CONFIG_VALUES["expected_piece_count"], 16)

            missing = _resolve_config(Path(temp_dir) / "does_not_exist.json")
            self.assertEqual(missing, DEFAULT_CONFIG_VALUES)

    @unittest.skipUnless(EXAMPLE_CONFIG.exists(), f"example config missing: {EXAMPLE_CONFIG}")
    def test_example_config_loads_tuned_values(self) -> None:
        resolved = _resolve_config(EXAMPLE_CONFIG)
        self.assertEqual(set(resolved), set(DEFAULT_CONFIG_VALUES))
        self.assertEqual(resolved["expected_piece_count"], 16)
        self.assertEqual(resolved["threshold_max"], 25.0)
        self.assertEqual(resolved["ignore_border_mm"], 5.0)
        self.assertNotIn("piece_output_size_px", resolved)
        self.assertEqual(resolved["piece_margin_px"], 20.0)

    def test_scan_order_indices_groups_rows_then_orders_by_x(self) -> None:
        """Scan order clusters rows by y (2 px tolerance), then sorts each row by x."""

        def scan(centres: list[tuple[float, float]]) -> list[int]:
            return scan_order_indices(
                [np.asarray(centre, dtype=np.float64) for centre in centres]
            )

        self.assertAlmostEqual(SCAN_ROW_TOLERANCE_PX, 2.0)

        # (a) Equal centre heights are one row: ordered by x.
        self.assertEqual(scan([(300.0, 50.0), (100.0, 50.0)]), [1, 0])
        self.assertEqual(scan([(100.0, 50.0), (300.0, 50.0)]), [0, 1])

        # (b) A 1.9 px y gap is still one row, so the x order beats the y jitter.
        self.assertEqual(scan([(300.0, 50.0), (100.0, 51.9)]), [1, 0])
        self.assertEqual(scan([(100.0, 50.0), (300.0, 51.9)]), [0, 1])

        # (c) A 2.1 px y gap splits the rows: y order wins even though piece 1
        # has the smaller x (a plain x sort would say [1, 0]).
        self.assertEqual(scan([(300.0, 50.0), (100.0, 52.1)]), [0, 1])

        # (d) Permuting the input does not change the scan order.
        labels = [(row, column) for row in range(3) for column in range(4)]
        rng = np.random.default_rng(11)
        centres = [
            (
                column * 300.0 + rng.uniform(-40.0, 40.0),
                row * 200.0 + rng.uniform(-0.9, 0.9),
            )
            for row, column in labels
        ]
        row_major = list(labels)
        arrangements = (
            row_major,
            row_major[::-1],
            row_major[::2] + row_major[1::2],
        )
        for arrangement in arrangements:
            permutation = [row_major.index(label) for label in arrangement]
            order = scan([centres[index] for index in permutation])
            self.assertEqual([arrangement[index] for index in order], row_major)


# ---------------------------------------------------------------------------
# synthetic end-to-end tests
# ---------------------------------------------------------------------------


def _render_synthetic_board() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Draw 16 bright textured quads on a dark bench.

    The empty capture is the bench without the pieces; the pieces capture is the
    same bench with a piece texture composited through a 4x supersampled
    coverage map, which gives clean anti-aliased edges.
    """

    width, height = SYNTH_CANVAS_W, SYNTH_CANVAS_H
    rng = np.random.default_rng(SYNTH_SEED)

    empty = np.full((height, width, 3), SYNTH_BG, dtype=np.uint8)

    # Low-frequency paper texture (the wavelength is well above the 2 px edge
    # profile, so it contributes little to the edge response).
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    pattern = np.stack(
        [
            np.clip(205.0 + 28.0 * np.sin(xx / 27.0) + 14.0 * np.cos(yy / 17.0), 0.0, 255.0),
            np.clip(215.0 + 22.0 * np.cos(yy / 23.0) + 10.0 * np.sin(xx / 31.0), 0.0, 255.0),
            np.clip(198.0 + 24.0 * np.sin((xx + yy) / 33.0), 0.0, 255.0),
        ],
        axis=2,
    ).astype(np.float32)

    quads: list[np.ndarray] = []
    for row in range(4):
        for column in range(4):
            centre_x = 45.0 + 20.0 + column * 325.0 + 145.0
            centre_y = 45.0 + 20.0 + row * 220.0 + 100.0
            angle = rng.uniform(-0.30, 0.30)
            half_width = 115.0 + rng.uniform(-8.0, 8.0)
            half_height = 78.0 + rng.uniform(-6.0, 6.0)
            quads.append(
                _rotated_rectangle(
                    (centre_x, centre_y), half_width, half_height, angle
                )
            )

    coverage_big = np.zeros(
        (height * SYNTH_SUPERSAMPLE, width * SYNTH_SUPERSAMPLE), dtype=np.uint8
    )
    for quad in quads:
        cv2.fillPoly(
            coverage_big,
            [np.round(quad * SYNTH_SUPERSAMPLE).astype(np.int32)],
            255,
        )
    coverage = (
        cv2.resize(coverage_big, (width, height), interpolation=cv2.INTER_AREA)
        .astype(np.float32)
        / 255.0
    )
    pieces = (
        coverage[..., None] * pattern
        + (1.0 - coverage[..., None]) * np.asarray(SYNTH_BG, dtype=np.float32)
    ).astype(np.uint8)

    # A few darker printed marks inside each piece, far enough from the edges
    # that the 2 px edge profile never touches them.
    for quad in quads:
        centre = quad.mean(axis=0)
        cx, cy = int(centre[0]), int(centre[1])
        cv2.circle(pieces, (cx - 45, cy - 20), 13, (60, 80, 130), -1, cv2.LINE_AA)
        cv2.circle(pieces, (cx + 48, cy + 18), 11, (40, 110, 70), -1, cv2.LINE_AA)
        cv2.line(pieces, (cx - 70, cy + 34), (cx + 25, cy - 34), (35, 90, 35), 4, cv2.LINE_AA)

    return pieces, empty, quads


def _synthetic_config() -> dict:
    """Defaults tuned only where the synthetic capture differs from the bench.

    ``ignore_border_mm`` is scaled to the synthetic ``px_per_mm`` (45 px border)
    and ``edge_profile_px`` is tightened because the synthetic edges are clean
    4x anti-aliased edges rather than the soft, shadowed paper edges the 6 px
    default was tuned for.
    """

    config = dict(DEFAULT_CONFIG_VALUES)
    config.update(
        {
            "expected_piece_count": 16,
            "ignore_border_mm": SYNTH_IGNORE_BORDER_MM,
            "edge_profile_px": SYNTH_EDGE_PROFILE_PX,
        }
    )
    return config


class SyntheticExtractionTests(unittest.TestCase):
    """End-to-end run on a fully synthetic, fully known board."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._temp_dir.cleanup)
        base = Path(cls._temp_dir.name)
        input_dir = base / "input"
        input_dir.mkdir()
        pieces_image, empty_image, placed_quads = _render_synthetic_board()
        imwrite(input_dir / "pieces_rectified.png", pieces_image)
        imwrite(input_dir / "empty_rectified.png", empty_image)
        write_json(
            input_dir / "rectification.json",
            {
                "schema_version": "1.0",
                "canvas_px": [SYNTH_CANVAS_W, SYNTH_CANVAS_H],
                "px_per_mm": [SYNTH_PX_PER_MM, SYNTH_PX_PER_MM],
            },
        )
        cls.input_dir = input_dir
        cls.placed_quads = placed_quads
        cls.config = _synthetic_config()
        cls.first_output = base / "output_a"
        cls.second_output = base / "output_b"
        cls.payload = cls._extract(cls.first_output)
        cls._extract(cls.second_output)

    @classmethod
    def _extract(cls, output_dir: Path) -> dict:
        return extract_pieces(
            cls.input_dir / "pieces_rectified.png",
            cls.input_dir / "empty_rectified.png",
            cls.input_dir / "rectification.json",
            None,
            cls.config,
            output_dir,
            write_outputs=True,
        )

    def test_detects_sixteen_pieces_sorted_by_center(self) -> None:
        payload = self.payload
        for field in REQUIRED_TOP_LEVEL_FIELDS:
            self.assertIn(field, payload)
        self.assertEqual(payload["detected_piece_count"], 16)
        self.assertEqual(payload["expected_piece_count"], 16)
        self.assertEqual(len(payload["pieces"]), 16)
        self.assertEqual(payload["canvas_px"], [SYNTH_CANVAS_W, SYNTH_CANVAS_H])
        self.assertAlmostEqual(float(payload["piece_scale"]), 1.0)
        self.assertNotIn("piece_size_px", payload)
        self.assertIsInstance(payload["piece_frame"], str)
        self.assertIn("base_edge", payload["piece_frame"])
        self.assertEqual(payload["piece_margin_px"], DEFAULT_CONFIG_VALUES["piece_margin_px"])
        self.assertAlmostEqual(payload["px_per_mm"][0], SYNTH_PX_PER_MM)
        _assert_manual_review_contract(self, payload, self.config)
        self.assertEqual(payload["coordinate_system"], "rectified_px")

        piece_ids = [piece["piece_id"] for piece in payload["pieces"]]
        self.assertEqual(piece_ids, [f"S{index:02d}" for index in range(1, 17)])
        centres = [tuple(float(value) for value in piece["center_px"]) for piece in payload["pieces"]]
        _assert_scan_order_contract(self, centres)
        # The synthetic 4x4 grid has rows ~220 px apart, so the documented
        # semantics can be asserted directly: pieces whose centre y differ by at
        # most SCAN_ROW_TOLERANCE_PX share a row and ascend in x, while a new
        # row may only start below the previous one.
        for previous, current in zip(centres, centres[1:]):
            delta_y = current[1] - previous[1]
            if abs(delta_y) <= SCAN_ROW_TOLERANCE_PX:
                self.assertGreater(
                    current[0], previous[0], msg="same row must be ordered by x"
                )
            else:
                self.assertGreater(
                    delta_y, SCAN_ROW_TOLERANCE_PX, msg="rows must ascend in y"
                )

        expected_files = {"pieces.json", "pieces_base_edge.json"}
        for index in range(1, 17):
            for suffix in ("rgba", "mask", "debug"):
                expected_files.add(f"pieces/S{index:02d}_{suffix}.png")
        written = {
            path.relative_to(self.first_output).as_posix()
            for path in self.first_output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(written, expected_files)

    def test_piece_outputs_are_consistent(self) -> None:
        pieces = self.payload["pieces"]
        for piece in pieces:
            for field in REQUIRED_PIECE_FIELDS:
                self.assertIn(field, piece, msg=piece["piece_id"])
            piece_id = piece["piece_id"]
            corners = np.asarray(piece["corners_px"], dtype=np.float64)
            self.assertEqual(piece["corner_count"], len(corners))
            self.assertGreaterEqual(piece["corner_count"], 3)
            self.assertLessEqual(piece["corner_count"], 6)
            self.assertTrue(is_convex_polygon(corners), msg=piece_id)
            self.assertAlmostEqual(
                float(piece["area_px"]), polygon_area(corners), delta=0.5, msg=piece_id
            )
            self.assertAlmostEqual(
                float(piece["area_mm2"]),
                float(piece["area_px"]) / (SYNTH_PX_PER_MM ** 2),
                delta=0.05,
                msg=piece_id,
            )
            # bbox_px is floor(min)..ceil(max), so it can be up to 2 px wider
            # than the extent of the float corners.
            self.assertAlmostEqual(
                float(np.ptp(corners[:, 0])),
                float(piece["bbox_px"][2] - piece["bbox_px"][0]),
                delta=2.0,
            )
            self.assertAlmostEqual(
                float(np.ptp(corners[:, 1])),
                float(piece["bbox_px"][3] - piece["bbox_px"][1]),
                delta=2.0,
            )
            self.assertGreaterEqual(float(piece["quality"]), 0.0)
            self.assertLessEqual(float(piece["quality"]), 1.0)
            self.assertFalse(piece["needs_manual_review"])
            self.assertIn("fit", piece["quality_breakdown"])

            rgba_path = self.first_output / piece["rgba_path"]
            mask_path = self.first_output / piece["mask_path"]
            debug_path = self.first_output / piece["debug_path"]
            self.assertTrue(rgba_path.is_file(), msg=piece_id)
            self.assertTrue(mask_path.is_file(), msg=piece_id)
            self.assertTrue(debug_path.is_file(), msg=piece_id)
            rgba = imread(rgba_path, cv2.IMREAD_UNCHANGED)
            mask = imread(mask_path, cv2.IMREAD_UNCHANGED)
            debug = imread(debug_path, cv2.IMREAD_UNCHANGED)
            width, height = (int(value) for value in piece["piece_size_px"])
            self.assertEqual(rgba.shape, (height, width, 4), msg=piece_id)
            self.assertEqual(mask.shape, (height, width), msg=piece_id)
            self.assertEqual(debug.shape, (height, width, 3), msg=piece_id)
            np.testing.assert_array_equal(rgba[..., 3], mask)
            self.assertLessEqual(set(np.unique(mask)), {0, 255})
            self.assertGreater(int((mask > 0).sum()), 0)
            # 1.4.0: the crop is the piece's own base-edge frame at 1:1 scale.
            _assert_base_edge_piece_contract(
                self, piece, float(self.payload["piece_margin_px"])
            )
            # The alpha mask and the rasterized crop polygon describe the same
            # region (cv2.fillPoly quantisation is sub-pixel, so IoU ~1).
            image_corners = np.asarray(piece["image_corners_px"], dtype=np.float64)
            expected_mask = rasterize_polygon(image_corners, (height, width))
            intersection = int(np.count_nonzero((mask > 0) & (expected_mask > 0)))
            union = int(np.count_nonzero((mask > 0) | (expected_mask > 0)))
            self.assertGreater(intersection / max(union, 1), 0.995, msg=piece_id)

        summary = self.payload["summary"]
        self.assertGreaterEqual(summary["min_quality"], float(self.config["min_quality"]))
        self.assertGreater(summary["min_area_px"], 0.0)
        self.assertLessEqual(summary["min_area_px"], summary["max_area_px"])
        self.assertAlmostEqual(
            summary["total_area_px"],
            sum(float(piece["area_px"]) for piece in pieces),
            places=1,
        )

    def test_base_edge_handoff_file_is_minimal_and_exact(self) -> None:
        handoff = json.loads(
            (self.first_output / "pieces_base_edge.json").read_text(encoding="utf-8")
        )
        payload = self.payload
        self.assertEqual(
            set(handoff),
            {
                "schema_version",
                "algorithm_version",
                "session_id",
                "coordinate_system",
                "unit_scale",
                "px_per_mm",
                "base_edge",
                "piece_count",
                "pieces",
            },
        )
        self.assertEqual(handoff["algorithm_version"], payload["algorithm_version"])
        self.assertEqual(handoff["session_id"], payload["session_id"])
        self.assertEqual(handoff["coordinate_system"], "rectified_px")
        self.assertAlmostEqual(float(handoff["unit_scale"]), 1.0)
        self.assertIsInstance(handoff["base_edge"], str)
        self.assertIn("baseLength", handoff["base_edge"])
        self.assertEqual(handoff["piece_count"], 16)
        self.assertEqual(len(handoff["pieces"]), 16)

        pieces = payload["pieces"]
        self.assertEqual(
            [record["id"] for record in handoff["pieces"]],
            [piece["piece_id"] for piece in pieces],
        )
        for record, piece in zip(handoff["pieces"], pieces):
            piece_id = piece["piece_id"]
            self.assertEqual(set(record), BASE_EDGE_RECORD_FIELDS, msg=piece_id)
            # The minimal hand-off carries the per-piece base-edge record
            # verbatim, at full float precision (pieces.json also round-trips it).
            self.assertEqual(record, piece["base_edge"], msg=piece_id)
            self.assertAlmostEqual(
                float(record["baseLength"]),
                float(piece["base_length_px"]),
                places=6,
                msg=piece_id,
            )
            np.testing.assert_allclose(
                np.asarray(record["p2"], dtype=np.float64),
                np.asarray(piece["base_edge_corners_px"], dtype=np.float64)[2],
                atol=1e-5,
                err_msg=piece_id,
            )
            np.testing.assert_allclose(
                np.asarray(record["p3"], dtype=np.float64),
                np.asarray(piece["base_edge_corners_px"], dtype=np.float64)[3],
                atol=1e-5,
                err_msg=piece_id,
            )
            for value in (*record["p2"], *record["p3"]):
                self.assertIsInstance(value, float, msg=piece_id)

    def test_detected_corners_match_placed_quads(self) -> None:
        placed = [ordered_polygon(quad) for quad in self.placed_quads]
        detected = [np.asarray(piece["corners_px"], dtype=np.float64) for piece in self.payload["pieces"]]
        centres = [
            np.asarray(piece["center_px"], dtype=np.float64) for piece in self.payload["pieces"]
        ]
        matched: set[int] = set()
        for quad in placed:
            distances = [float(np.linalg.norm(centre - quad.mean(axis=0))) for centre in centres]
            best = int(np.argmin(distances))
            self.assertLess(distances[best], 20.0, msg="no detection near a placed quad")
            self.assertNotIn(best, matched, msg="one detection was matched twice")
            matched.add(best)
            piece = self.payload["pieces"][best]
            self.assertEqual(piece["corner_count"], len(quad), msg=piece["piece_id"])
            error = _best_cyclic_vertex_error(detected[best], quad)
            self.assertLess(
                error,
                SYNTH_CORNER_TOLERANCE_PX,
                msg=f"{piece['piece_id']}: corner error {error:.3f}px",
            )
        self.assertEqual(len(matched), 16)

    def test_rerun_is_deterministic(self) -> None:
        first_json = (self.first_output / "pieces.json").read_bytes()
        second_json = (self.second_output / "pieces.json").read_bytes()
        self.assertEqual(first_json, second_json)
        first_handoff = (self.first_output / "pieces_base_edge.json").read_bytes()
        second_handoff = (self.second_output / "pieces_base_edge.json").read_bytes()
        self.assertEqual(first_handoff, second_handoff)
        first_files = {
            path.relative_to(self.first_output).as_posix()
            for path in self.first_output.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(self.second_output).as_posix()
            for path in self.second_output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first_files, second_files)
        # pieces.json + pieces_base_edge.json + 16 * (rgba, mask, debug)
        self.assertEqual(len(first_files), 50)


# ---------------------------------------------------------------------------
# real-data integration test (skipped when Stage 1 output is unavailable)
# ---------------------------------------------------------------------------


def _real_data_available() -> bool:
    return all(
        path.exists()
        for path in (REAL_PIECES, REAL_EMPTY, REAL_RECTIFICATION, EXAMPLE_CONFIG)
    )


def _real_fingerprint(payload: dict) -> str:
    """Stable JSON projection of a payload, excluding the per-piece images."""

    fields = (
        "piece_id",
        "corner_count",
        "corners_px",
        "center_px",
        "area_px",
        "area_mm2",
        "bbox_px",
        "perimeter_px",
        "quality",
        "quality_breakdown",
        "needs_manual_review",
        "manual_review_reasons",
        "base_edge",
        "base_edge_corners_px",
        "base_length_px",
        "base_side_index",
        "image_corners_px",
        "piece_size_px",
        "rotation_deg",
        "rectified_to_piece_matrix",
        "rectified_to_base_edge_matrix",
    )
    return json.dumps(
        {
            "session_id": payload["session_id"],
            "detected_piece_count": payload["detected_piece_count"],
            "pieces": [{key: piece[key] for key in fields} for piece in payload["pieces"]],
        },
        sort_keys=True,
        default=float,
    )


@unittest.skipUnless(
    _real_data_available(),
    f"real rectified capture not available: {REAL_PIECES}",
)
class RealDataIntegrationTests(unittest.TestCase):
    """Run against the current rectified capture when Stage 1 has produced it."""

    def test_real_capture_yields_sixteen_convex_pieces(self) -> None:
        config = _resolve_config(EXAMPLE_CONFIG)
        layout_path = REAL_LAYOUT if REAL_LAYOUT.exists() else None

        def run_into(output_dir: Path, write_outputs: bool) -> dict:
            return extract_pieces(
                REAL_PIECES,
                REAL_EMPTY,
                REAL_RECTIFICATION,
                layout_path,
                config,
                output_dir,
                write_outputs=write_outputs,
            )

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        output_dir = Path(temp_dir.name)
        # ``write_outputs=True`` exercises the on-disk artifacts (1:1 crops and
        # the minimal base-edge hand-off) against the real capture.
        payload = run_into(output_dir, write_outputs=True)
        # A second run must reproduce the same session id and geometry.  (The
        # synthetic test additionally pins the byte-for-byte determinism of the
        # written pieces.json / piece images.)
        payload_again = run_into(output_dir / "again", write_outputs=False)
        self.assertEqual(_real_fingerprint(payload), _real_fingerprint(payload_again))

        expected_count = int(config["expected_piece_count"])
        self.assertAlmostEqual(float(payload["piece_scale"]), 1.0)
        self.assertNotIn("piece_size_px", payload)
        self.assertIsInstance(payload["piece_frame"], str)
        margin = float(payload["piece_margin_px"])
        self.assertEqual(payload["detected_piece_count"], expected_count)
        self.assertEqual(payload["expected_piece_count"], expected_count)
        self.assertEqual(len(payload["pieces"]), expected_count)
        rectification = json.loads(REAL_RECTIFICATION.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["canvas_px"], [int(value) for value in rectification["canvas_px"]]
        )
        self.assertEqual(
            payload["px_per_mm"],
            [float(value) for value in rectification["px_per_mm"]],
        )
        # 1.1.0 contract: manual review is an advisory triage flag, so assert
        # completeness / explainability / payload consistency instead of
        # hard-coding the flagged ids of the current capture.
        _assert_manual_review_contract(self, payload, config)
        self.assertRegex(payload["session_id"], r"^[0-9a-f]{16}$")
        self.assertEqual(
            [piece["piece_id"] for piece in payload["pieces"]],
            [f"S{index:02d}" for index in range(1, expected_count + 1)],
        )
        centres = [tuple(float(value) for value in piece["center_px"]) for piece in payload["pieces"]]
        # 1.2.0: ids follow the row-clustered scan order, not a naive (y, x) sort.
        _assert_scan_order_contract(self, centres)

        min_quality = float(config["min_quality"])
        for piece in payload["pieces"]:
            piece_id = piece["piece_id"]
            corners = np.asarray(piece["corners_px"], dtype=np.float64)
            self.assertEqual(piece["corner_count"], len(corners))
            self.assertGreaterEqual(piece["corner_count"], 3)
            self.assertLessEqual(piece["corner_count"], 6)
            self.assertTrue(is_convex_polygon(corners), msg=piece_id)
            # needs_manual_review and the per-piece quality gate are checked in
            # bulk by _assert_manual_review_contract above.
            self.assertGreaterEqual(float(piece["quality"]), 0.0)
            self.assertLessEqual(float(piece["quality"]), 1.0)

            rgba_path = output_dir / piece["rgba_path"]
            mask_path = output_dir / piece["mask_path"]
            debug_path = output_dir / piece["debug_path"]
            self.assertTrue(rgba_path.is_file(), msg=piece_id)
            self.assertTrue(mask_path.is_file(), msg=piece_id)
            self.assertTrue(debug_path.is_file(), msg=piece_id)
            rgba = imread(rgba_path, cv2.IMREAD_UNCHANGED)
            mask = imread(mask_path, cv2.IMREAD_UNCHANGED)
            debug = imread(debug_path, cv2.IMREAD_UNCHANGED)
            width, height = (int(value) for value in piece["piece_size_px"])
            self.assertEqual(rgba.shape, (height, width, 4), msg=piece_id)
            self.assertEqual(mask.shape, (height, width), msg=piece_id)
            self.assertEqual(debug.shape, (height, width, 3), msg=piece_id)
            np.testing.assert_array_equal(rgba[..., 3], mask)
            self.assertLessEqual(set(np.unique(mask)), {0, 255})
            self.assertGreater(int((mask > 0).sum()), 0)

            # 1.4.0: the piece is cropped in its own base-edge frame at 1:1.
            _assert_base_edge_piece_contract(self, piece, margin)

            # The alpha mask and the rasterized crop polygon agree.  Because the
            # crop is 1:1, the polygon area and the mask area differ only by
            # cv2.fillPoly quantisation (bounded by roughly half a pixel per
            # perimeter pixel).
            image_corners = np.asarray(piece["image_corners_px"], dtype=np.float64)
            expected_mask = rasterize_polygon(image_corners, (height, width))
            intersection = int(np.count_nonzero((mask > 0) & (expected_mask > 0)))
            union = int(np.count_nonzero((mask > 0) | (expected_mask > 0)))
            self.assertGreater(intersection / max(union, 1), 0.995, msg=piece_id)
            analytic = polygon_area(image_corners)
            mask_area = float((mask > 0).sum())
            allowance = 0.01 * analytic + 0.5 * _perimeter(image_corners) + 1e-6
            self.assertLessEqual(
                abs(mask_area - analytic),
                allowance,
                msg=f"{piece_id}: mask {mask_area:.1f} vs polygon {analytic:.1f}px",
            )

        # The minimal hand-off file is written next to pieces.json and carries
        # the same base-edge records at full float precision.
        handoff = json.loads(
            (output_dir / "pieces_base_edge.json").read_text(encoding="utf-8")
        )
        self.assertEqual(handoff["piece_count"], expected_count)
        self.assertEqual(handoff["session_id"], payload["session_id"])
        self.assertEqual(handoff["coordinate_system"], "rectified_px")
        self.assertEqual(
            [record["id"] for record in handoff["pieces"]],
            [piece["piece_id"] for piece in payload["pieces"]],
        )
        for record, piece in zip(handoff["pieces"], payload["pieces"]):
            self.assertEqual(set(record), BASE_EDGE_RECORD_FIELDS)
            self.assertEqual(record, piece["base_edge"], msg=piece["piece_id"])
            self.assertAlmostEqual(
                float(record["baseLength"]),
                float(piece["base_length_px"]),
                places=6,
                msg=piece["piece_id"],
            )

        summary = payload["summary"]
        # 1.1.0: the global min quality MAY sit below config["min_quality"] when
        # that piece is honestly flagged; the contract above guarantees that a
        # piece below min_quality is never silently clean.  Keep the summary
        # consistent with the per-piece qualities, and only allow the global
        # minimum to drop below the bar for an explained piece.
        self.assertAlmostEqual(
            summary["min_quality"],
            min(float(piece["quality"]) for piece in payload["pieces"]),
            places=5,
        )
        lowest = min(payload["pieces"], key=lambda piece: float(piece["quality"]))
        if float(summary["min_quality"]) < min_quality:
            self.assertTrue(lowest["needs_manual_review"], msg=lowest["piece_id"])
            self.assertIn("quality_below_min", lowest["manual_review_reasons"])
        self.assertGreaterEqual(float(summary["median_quality"]), 0.0)
        self.assertLessEqual(float(summary["median_quality"]), 1.0)


if __name__ == "__main__":
    unittest.main()
