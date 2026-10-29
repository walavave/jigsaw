"""Tests for Stage 2 piece extraction (``stage2_piece_extraction/extract_pieces.py``).

The suite has three layers:

1. Pure geometry / response helpers, exercised on small hand-checked arrays.
2. A fully synthetic end-to-end board: a dark bench plus 16 drawn convex quads
   with a bright low-frequency texture, so the colour difference from the empty
   capture is large.  ``extract_pieces`` runs twice (once per output directory)
   to check the byte-for-byte determinism of ``pieces.json``.
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
    colour_response,
    combine_quality,
    dedupe_polygon,
    edge_response,
    extract_pieces,
    foreground_mask,
    is_convex_polygon,
    normalization_matrix,
    normalize_piece,
    ordered_polygon,
    polygon_area,
    polygon_centroid,
    polygon_iou,
    polygon_signed_area,
    prune_near_collinear,
    rasterize_polygon,
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
    "normalized_corners_px",
    "rotation_deg",
    "rectified_to_piece_matrix",
}
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "algorithm_version",
    "session_id",
    "coordinate_system",
    "piece_size_px",
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

    def test_normalization_matrix_maps_corners_inside_margin(self) -> None:
        size_px, margin_px = 512, 20.0
        angle = 0.30
        corners = _rotated_rectangle((700.0, 900.0), 200.0, 125.0, angle)
        matrix, rotation_deg = normalization_matrix(corners, size_px, margin_px)

        self.assertEqual(matrix.shape, (3, 3))
        np.testing.assert_allclose(matrix[2], [0.0, 0.0, 1.0])
        transformed = _apply_transform(matrix, corners)
        usable = size_px - 2.0 * margin_px
        # All corners stay inside the margin and the longer bbox side fills the
        # usable box exactly (the shorter side keeps the aspect ratio).
        self.assertGreaterEqual(float(transformed.min()), margin_px - 1e-9)
        self.assertLessEqual(float(transformed.max()), size_px - margin_px + 1e-9)
        extents = np.ptp(transformed, axis=0)
        self.assertAlmostEqual(float(extents.max()), usable, places=9)
        self.assertLessEqual(float(extents.min()), float(extents.max()) + 1e-9)
        np.testing.assert_allclose(
            transformed.min(axis=0) + transformed.max(axis=0),
            [size_px, size_px],
            atol=1e-9,
        )
        # Uniform scale: the linear part is s * R with R a rotation.
        linear = matrix[:2, :2]
        np.testing.assert_allclose(linear.T @ linear, np.eye(2) * float(np.linalg.det(linear)), atol=1e-9)
        self.assertGreater(float(np.linalg.det(linear)), 0.0)
        self.assertGreaterEqual(rotation_deg, -180.0)
        self.assertLessEqual(rotation_deg, 180.0)
        self.assertAlmostEqual(rotation_deg, -math.degrees(angle), places=6)
        # Principal axis of the normalized shape is horizontal.
        centred = transformed - transformed.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if axis[0] < 0:
            axis = -axis
        self.assertAlmostEqual(float(axis[1]), 0.0, places=6)

    def test_normalization_matrix_rejects_degenerate_polygon(self) -> None:
        degenerate = np.full((4, 2), 5.0)
        with self.assertRaises(ValueError):
            normalization_matrix(degenerate, 512, 20.0)

    def test_normalize_piece_produces_consistent_rgba(self) -> None:
        size_px, margin_px = 128, 10.0
        image = np.full((300, 320, 3), 20, dtype=np.uint8)
        polygon = _rotated_rectangle((160.0, 150.0), 80.0, 50.0, 0.25)
        cv2.fillPoly(
            image,
            [np.round(polygon).astype(np.int32)],
            (230, 240, 250),
        )
        normalized = normalize_piece(image, polygon, size_px, margin_px)

        rgba = normalized["rgba"]
        mask = normalized["mask"]
        self.assertEqual(rgba.shape, (size_px, size_px, 4))
        self.assertEqual(mask.shape, (size_px, size_px))
        self.assertEqual(rgba.dtype, np.uint8)
        self.assertLessEqual(set(np.unique(mask)), {0, 255})
        np.testing.assert_array_equal(rgba[..., 3], mask)
        self.assertTrue(bool(np.all(rgba[mask == 0] == 0)))

        normalized_corners = np.asarray(normalized["normalized_corners"])
        transformed = _apply_transform(normalized["matrix"], polygon)
        np.testing.assert_allclose(normalized_corners, transformed, atol=1e-9)
        self.assertGreaterEqual(float(transformed.min()), margin_px - 1e-9)
        self.assertLessEqual(float(transformed.max()), size_px - margin_px + 1e-9)
        np.testing.assert_allclose(
            (transformed.min(axis=0) + transformed.max(axis=0)) / 2.0,
            [size_px / 2.0, size_px / 2.0],
            atol=1e-9,
        )
        # The warped brightness inside the mask matches the drawn paper.
        self.assertGreater(float(np.median(rgba[mask > 0][:, 0])), 180.0)
        # The analytic polygon and its rasterized alpha describe the same area.
        analytic = polygon_area(transformed)
        mask_area = float((mask > 0).sum())
        self.assertLessEqual(
            abs(mask_area - analytic),
            0.01 * analytic + 0.5 * _perimeter(transformed),
            msg=f"mask area {mask_area} vs analytic {analytic}",
        )
        self.assertAlmostEqual(float(normalized["rotation_deg"]), -math.degrees(0.25), places=6)

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
            self.assertEqual(merged["piece_output_size_px"], 512)
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
            "piece_output_size_px": 256,
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
        self.assertEqual(payload["piece_size_px"], [256, 256])
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

        expected_files = {"pieces.json"}
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
            self.assertEqual(rgba.shape, (256, 256, 4), msg=piece_id)
            self.assertEqual(mask.shape, (256, 256), msg=piece_id)
            np.testing.assert_array_equal(rgba[..., 3], mask)
            self.assertLessEqual(set(np.unique(mask)), {0, 255})
            self.assertGreater(int((mask > 0).sum()), 0)

            matrix = np.asarray(piece["rectified_to_piece_matrix"], dtype=np.float64)
            self.assertEqual(matrix.shape, (3, 3))
            np.testing.assert_allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-9)
            normalized = np.asarray(piece["normalized_corners_px"], dtype=np.float64)
            mapped = _apply_transform(matrix, corners)
            np.testing.assert_allclose(mapped, normalized, atol=5e-3)
            margin = float(self.payload["piece_margin_px"])
            size = float(self.payload["piece_size_px"][0])
            self.assertGreaterEqual(float(normalized.min()), margin - 1e-3, msg=piece_id)
            self.assertLessEqual(float(normalized.max()), size - margin + 1e-3, msg=piece_id)

        summary = self.payload["summary"]
        self.assertGreaterEqual(summary["min_quality"], float(self.config["min_quality"]))
        self.assertGreater(summary["min_area_px"], 0.0)
        self.assertLessEqual(summary["min_area_px"], summary["max_area_px"])
        self.assertAlmostEqual(
            summary["total_area_px"],
            sum(float(piece["area_px"]) for piece in pieces),
            places=1,
        )

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
        self.assertEqual(len(first_files), 49)


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
        "normalized_corners_px",
        "rotation_deg",
        "rectified_to_piece_matrix",
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

        def run_into(output_dir: str) -> dict:
            return extract_pieces(
                REAL_PIECES,
                REAL_EMPTY,
                REAL_RECTIFICATION,
                layout_path,
                config,
                Path(output_dir),
                write_outputs=False,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            payload = run_into(temp_dir)
        # A second run must reproduce the same session id and geometry.  (The
        # synthetic test additionally pins the byte-for-byte determinism of the
        # written pieces.json / piece images.)
        with tempfile.TemporaryDirectory() as second_dir:
            payload_again = run_into(second_dir)
        self.assertEqual(_real_fingerprint(payload), _real_fingerprint(payload_again))

        expected_count = int(config["expected_piece_count"])
        output_size = int(config["piece_output_size_px"])
        self.assertEqual(payload["detected_piece_count"], expected_count)
        self.assertEqual(payload["expected_piece_count"], expected_count)
        self.assertEqual(len(payload["pieces"]), expected_count)
        rectification = json.loads(REAL_RECTIFICATION.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["canvas_px"], [int(value) for value in rectification["canvas_px"]]
        )
        self.assertEqual(payload["piece_size_px"], [output_size, output_size])
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

            rgba = np.asarray(piece["rgba"])
            mask = np.asarray(piece["mask"])
            self.assertEqual(rgba.shape, (output_size, output_size, 4), msg=piece_id)
            self.assertEqual(mask.shape, (output_size, output_size), msg=piece_id)
            np.testing.assert_array_equal(rgba[..., 3], mask)
            self.assertLessEqual(set(np.unique(mask)), {0, 255})

            matrix = np.asarray(piece["rectified_to_piece_matrix"], dtype=np.float64)
            normalized = np.asarray(piece["normalized_corners_px"], dtype=np.float64)
            np.testing.assert_allclose(
                _apply_transform(matrix, corners), normalized, atol=5e-3
            )
            margin = float(payload["piece_margin_px"])
            size = float(payload["piece_size_px"][0])
            self.assertGreaterEqual(float(normalized.min()), margin - 1e-3, msg=piece_id)
            self.assertLessEqual(float(normalized.max()), size - margin + 1e-3, msg=piece_id)

            # The alpha mask and the polygon must describe the same area.  The
            # allowance covers cv2.fillPoly pixel quantisation, which is bounded
            # by roughly half a pixel per perimeter pixel.
            warped = _apply_transform(matrix, corners)
            analytic = polygon_area(warped)
            mask_area = float((mask > 0).sum())
            allowance = 0.01 * analytic + 0.5 * _perimeter(warped) + 1e-6
            self.assertLessEqual(
                abs(mask_area - analytic),
                allowance,
                msg=f"{piece_id}: mask {mask_area:.1f} vs polygon {analytic:.1f}px",
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
