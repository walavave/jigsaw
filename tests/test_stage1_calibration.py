import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1_capture_calibration"))

from calibration import (  # noqa: E402
    calibrate_from_images,
    chessboard_object_points,
)
from checkerboard_plane import build_checkerboard_homography  # noqa: E402
from export_realsense_intrinsics import build_intrinsics_payload  # noqa: E402
from rectification import (  # noqa: E402
    MarkerDetection,
    build_homography,
    detect_marker_centres,
    rectify_image,
    validate_marker_layout,
)


def render_chessboard_view(transform: np.ndarray) -> np.ndarray:
    cols, rows, square_px = 9, 6, 60
    board = np.full(
        ((rows + 1) * square_px, (cols + 1) * square_px),
        255,
        dtype=np.uint8,
    )
    for row in range(rows + 1):
        for col in range(cols + 1):
            if (row + col) % 2 == 0:
                board[
                    row * square_px : (row + 1) * square_px,
                    col * square_px : (col + 1) * square_px,
                ] = 0
    return cv2.warpPerspective(
        board,
        transform,
        (800, 700),
        borderValue=255,
    )


class CalibrationTests(unittest.TestCase):
    def test_chessboard_object_points_use_physical_millimetres(self) -> None:
        points = chessboard_object_points(cols=9, rows=6, square_mm=18.0)
        self.assertEqual(points.shape, (54, 3))
        self.assertAlmostEqual(float(points[1, 0] - points[0, 0]), 18.0)
        np.testing.assert_allclose(points[:, 2], 0.0)

    def test_calibrates_from_synthetic_chessboard_views(self) -> None:
        transforms = [
            np.array([[1, 0, 20], [0, 1, 20], [0, 0, 1]], dtype=np.float32),
            np.array([[0.9, 0.1, 80], [-0.05, 1, 50], [0.0001, 0, 1]], dtype=np.float32),
            np.array([[1.1, -0.1, 20], [0.1, 0.9, 80], [0, 0.0001, 1]], dtype=np.float32),
            np.array([[0.8, 0.2, 180], [-0.15, 1, 30], [0.0002, 0, 1]], dtype=np.float32),
            np.array([[1, -0.2, 100], [0.15, 0.8, 100], [0, 0.00015, 1]], dtype=np.float32),
            np.array([[0.9, 0.15, 40], [-0.15, 0.85, 150], [0.00015, 0, 1]], dtype=np.float32),
            np.array([[0.75, 0.25, 250], [0.05, 0.8, 60], [0.0001, 0.0001, 1]], dtype=np.float32),
            np.array([[1.2, 0.05, 30], [-0.05, 0.7, 120], [0, 0, 1]], dtype=np.float32),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths: list[Path] = []
            for index, transform in enumerate(transforms):
                path = Path(temporary_directory) / f"view_{index:02d}.png"
                self.assertTrue(
                    cv2.imwrite(str(path), render_chessboard_view(transform))
                )
                paths.append(path)

            calibration = calibrate_from_images(paths, cols=9, rows=6, square_mm=18)

        self.assertEqual(len(calibration.accepted_images), len(transforms))
        self.assertEqual(calibration.camera_matrix.shape, (3, 3))
        self.assertEqual(len(calibration.per_view_errors_px), len(transforms))
        self.assertTrue(math.isfinite(calibration.rms_reprojection_px))
        self.assertLess(calibration.rms_reprojection_px, 3.0)

    def test_homography_recovers_a_known_planar_transform(self) -> None:
        markers = [
            {"id": 0, "center_mm": [25, 25]},
            {"id": 1, "center_mm": [375, 25]},
            {"id": 2, "center_mm": [375, 255]},
            {"id": 3, "center_mm": [25, 255]},
        ]
        workspace_mm = (400.0, 280.0)
        canvas_px = (3200, 2240)
        scale = canvas_px[0] / workspace_mm[0]
        targets = np.array(
            [[x * scale, y * scale] for _, x, y in (
                (0, 25, 25),
                (1, 375, 25),
                (2, 375, 255),
                (3, 25, 255),
            )],
            dtype=np.float64,
        )
        target_to_source = cv2.getPerspectiveTransform(
            targets.astype(np.float32),
            np.array(
                [[90, 70], [3090, 110], [3150, 2170], [60, 2110]],
                dtype=np.float32,
            ),
        )
        sources = cv2.perspectiveTransform(
            targets.reshape(-1, 1, 2),
            target_to_source,
        ).reshape(-1, 2)
        detections = [
            MarkerDetection(
                marker_id=index,
                centre_px=tuple(float(value) for value in point),
                corners_px=np.zeros((4, 2), dtype=np.float32),
            )
            for index, point in enumerate(sources)
        ]

        result = build_homography(
            detections,
            markers,
            workspace_mm,
            canvas_px,
        )

        self.assertLess(result.rms_reprojection_px, 1e-8)
        expected = np.linalg.inv(target_to_source)
        expected /= expected[2, 2]
        np.testing.assert_allclose(
            result.homography,
            expected,
            atol=1e-6,
        )

    def test_aruco_detection_and_rectification_end_to_end(self) -> None:
        markers = [
            {"id": 0, "center_mm": [25, 25]},
            {"id": 1, "center_mm": [375, 25]},
            {"id": 2, "center_mm": [375, 255]},
            {"id": 3, "center_mm": [25, 255]},
        ]
        canvas_px = (800, 560)
        px_per_mm = canvas_px[0] / 400.0
        target = np.full((canvas_px[1], canvas_px[0]), 255, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_size_px = round(40 * px_per_mm)
        for marker in markers:
            marker_image = cv2.aruco.generateImageMarker(
                dictionary,
                marker["id"],
                marker_size_px,
            )
            centre_x = round(marker["center_mm"][0] * px_per_mm)
            centre_y = round(marker["center_mm"][1] * px_per_mm)
            half = marker_size_px // 2
            target[
                centre_y - half : centre_y + half,
                centre_x - half : centre_x + half,
            ] = marker_image

        target_to_source = np.array(
            [
                [1.02, 0.07, 120],
                [0.03, 0.96, 90],
                [0.00004, 0.00001, 1],
            ],
            dtype=np.float64,
        )
        source = cv2.warpPerspective(
            target,
            target_to_source,
            (1200, 840),
            borderValue=255,
        )
        source_bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

        detections = detect_marker_centres(source_bgr, "DICT_4X4_50")
        self.assertEqual({item.marker_id for item in detections}, {0, 1, 2, 3})
        result = build_homography(
            detections,
            markers,
            (400, 280),
            canvas_px,
        )
        self.assertLess(result.rms_reprojection_px, 1e-6)

        rectified = rectify_image(source_bgr, result.homography, canvas_px)
        rectified_detections = detect_marker_centres(
            rectified,
            "DICT_4X4_50",
        )
        self.assertEqual(
            {item.marker_id for item in rectified_detections},
            {0, 1, 2, 3},
        )

    def test_checkerboard_plane_rectification_end_to_end(self) -> None:
        cols, rows = 9, 6
        canvas_px = (800, 560)
        workspace_mm = (400.0, 280.0)
        px_per_mm = canvas_px[0] / workspace_mm[0]
        square_px = round(18 * px_per_mm)
        origin_px = (round(110 * px_per_mm), round(77 * px_per_mm))
        target = np.full((canvas_px[1], canvas_px[0]), 255, dtype=np.uint8)
        for row in range(rows + 1):
            for col in range(cols + 1):
                if (row + col) % 2 == 0:
                    left = origin_px[0] + col * square_px
                    top = origin_px[1] + row * square_px
                    target[
                        top : top + square_px,
                        left : left + square_px,
                    ] = 0

        target_to_source = np.array(
            [
                [1.01, 0.05, 70],
                [-0.02, 0.97, 55],
                [0.00003, 0.00001, 1],
            ],
            dtype=np.float64,
        )
        source = cv2.warpPerspective(
            target,
            target_to_source,
            (800, 700),
            borderValue=255,
        )
        result = build_checkerboard_homography(
            image=source,
            cols=cols,
            rows=rows,
            square_mm=18,
            workspace_mm=workspace_mm,
            canvas_px=canvas_px,
            board_outer_origin_mm=(110, 77),
        )

        self.assertLess(result.rms_reprojection_px, 0.5)
        expected = np.linalg.inv(target_to_source)
        expected /= expected[2, 2]
        np.testing.assert_allclose(
            result.homography,
            expected,
            atol=0.5,
        )

    def test_marker_layout_rejects_duplicate_ids(self) -> None:
        layout = {
            "dictionary": "DICT_4X4_50",
            "workspace_mm": [400, 280],
            "canvas_px": [3200, 2240],
            "markers": [
                {"id": 0, "center_mm": [25, 25]},
                {"id": 0, "center_mm": [375, 25]},
                {"id": 2, "center_mm": [375, 255]},
                {"id": 3, "center_mm": [25, 255]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_marker_layout(layout)

    def test_realsense_brown_conrady_payload_matches_project_schema(self) -> None:
        payload = build_intrinsics_payload(
            width=1280,
            height=720,
            fx=910.5,
            fy=911.0,
            ppx=642.25,
            ppy=361.75,
            model="Brown-Conrady",
            coeffs=[0.12, -0.04, 0.001, -0.002, 0.01],
        )

        self.assertEqual(payload["image_size_px"], [1280, 720])
        self.assertEqual(
            payload["calibration_type"],
            "realsense_factory_color_intrinsics",
        )
        self.assertEqual(np.asarray(payload["camera_matrix"]).shape, (3, 3))
        self.assertEqual(len(payload["distortion_coefficients"]), 5)
        self.assertEqual(payload["distortion_coefficients"][0], 0.12)

    def test_realsense_inverse_brown_conrady_is_rejected_if_distorted(self) -> None:
        with self.assertRaisesRegex(ValueError, "cv2.undistort"):
            build_intrinsics_payload(
                width=1280,
                height=720,
                fx=910.5,
                fy=911.0,
                ppx=642.25,
                ppy=361.75,
                model="Inverse Brown Conrady",
                coeffs=[0.12, -0.04, 0.001, -0.002, 0.01],
            )


if __name__ == "__main__":
    unittest.main()
