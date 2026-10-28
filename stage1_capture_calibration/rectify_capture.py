#!/usr/bin/env python3
"""Undistort and rectify a photographed puzzle workspace.

The default method uses one chessboard image on the table plane.  The legacy
four-marker method remains available through ``--aruco-layout``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from jigsaw_common.io import imread, imwrite, read_json, write_json  # noqa: E402

try:
    from .calibration import undistort_image  # noqa: E402
    from .checkerboard_plane import (  # noqa: E402
        build_checkerboard_homography,
        draw_checkerboard_debug,
        validate_checkerboard_layout,
    )
    from .rectification import (  # noqa: E402
        build_homography,
        detect_marker_centres,
        draw_detection_debug,
        rectify_image,
        validate_marker_layout,
    )
except ImportError:  # Direct script execution.
    from calibration import undistort_image  # noqa: E402
    from checkerboard_plane import (  # noqa: E402
        build_checkerboard_homography,
        draw_checkerboard_debug,
        validate_checkerboard_layout,
    )
    from rectification import (  # noqa: E402
        build_homography,
        detect_marker_centres,
        draw_detection_debug,
        rectify_image,
        validate_marker_layout,
    )


def _require_image_size(
    image: np.ndarray,
    expected: tuple[int, int],
    label: str,
) -> None:
    actual = (image.shape[1], image.shape[0])
    if actual != expected:
        raise ValueError(
            f"{label} resolution does not match camera intrinsics: "
            f"{actual} != {expected}. Reuse the calibration resolution or "
            "export matching RealSense factory intrinsics."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", type=Path, required=True)
    parser.add_argument(
        "--empty",
        type=Path,
        required=True,
        help="Same workspace after pieces are removed.",
    )
    parser.add_argument(
        "--plane",
        type=Path,
        help=(
            "One photo of the chessboard lying flat in the workspace. "
            "Required for the default checkerboard method."
        ),
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        required=True,
        help="Camera intrinsics JSON from chessboard or RealSense factory calibration.",
    )
    parser.add_argument(
        "--checkerboard-layout",
        type=Path,
        default=(
            ROOT
            / "stage1_capture_calibration"
            / "config"
            / "checkerboard_layout.json"
        ),
    )
    parser.add_argument(
        "--aruco-layout",
        type=Path,
        help="Optional legacy four-marker layout. Overrides checkerboard mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "stage1_capture_calibration"
            / "output"
            / "rectified"
        ),
    )
    args = parser.parse_args()

    intrinsics = read_json(args.intrinsics)
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        intrinsics["distortion_coefficients"],
        dtype=np.float64,
    )
    expected_size = tuple(int(value) for value in intrinsics["image_size_px"])
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must be a 3 x 3 matrix.")

    pieces_raw = imread(args.pieces)
    empty_raw = imread(args.empty)
    if pieces_raw.shape[:2] != empty_raw.shape[:2]:
        raise ValueError("pieces and empty photos must have the same resolution.")
    _require_image_size(pieces_raw, expected_size, "pieces image")
    _require_image_size(empty_raw, expected_size, "empty image")

    pieces_undistorted = undistort_image(
        pieces_raw,
        camera_matrix,
        distortion,
    )
    empty_undistorted = undistort_image(
        empty_raw,
        camera_matrix,
        distortion,
    )

    if args.aruco_layout is not None:
        layout = read_json(args.aruco_layout)
        validate_marker_layout(layout)
        detections = detect_marker_centres(
            pieces_undistorted,
            layout["dictionary"],
        )
        workspace_mm = tuple(float(value) for value in layout["workspace_mm"])
        canvas_px = tuple(int(value) for value in layout["canvas_px"])
        result = build_homography(
            detections,
            layout["markers"],
            workspace_mm,
            canvas_px,
        )
        homography = result.homography
        debug = draw_detection_debug(pieces_undistorted, detections)
        debug_name = "marker_detection_debug.png"
        metadata = {
            "rectification_method": "aruco_markers",
            "dictionary": layout["dictionary"],
            "source_marker_centres_px": result.source_centres_px.tolist(),
            "target_marker_centres_px": result.target_centres_px.tolist(),
            "rms_reprojection_px": result.rms_reprojection_px,
            "detected_marker_ids": [item.marker_id for item in detections],
        }
    else:
        if args.plane is None:
            raise ValueError(
                "The default checkerboard method requires --plane with one "
                "chessboard image on the table plane."
            )
        layout = read_json(args.checkerboard_layout)
        validate_checkerboard_layout(layout)
        plane_raw = imread(args.plane)
        _require_image_size(plane_raw, expected_size, "plane image")
        plane_undistorted = undistort_image(
            plane_raw,
            camera_matrix,
            distortion,
        )
        workspace_mm = tuple(float(value) for value in layout["workspace_mm"])
        canvas_px = tuple(int(value) for value in layout["canvas_px"])
        board_origin = layout.get("board_outer_origin_mm")
        board_origin_mm = (
            None
            if board_origin is None
            else (float(board_origin[0]), float(board_origin[1]))
        )
        result = build_checkerboard_homography(
            image=plane_undistorted,
            cols=int(layout["cols"]),
            rows=int(layout["rows"]),
            square_mm=float(layout["square_mm"]),
            workspace_mm=workspace_mm,
            canvas_px=canvas_px,
            board_outer_origin_mm=board_origin_mm,
        )
        homography = result.homography
        debug = draw_checkerboard_debug(
            plane_undistorted,
            int(layout["cols"]),
            int(layout["rows"]),
            result,
        )
        debug_name = "checkerboard_detection_debug.png"
        metadata = {
            "rectification_method": "checkerboard_plane",
            "checkerboard": {
                "cols": int(layout["cols"]),
                "rows": int(layout["rows"]),
                "square_mm": float(layout["square_mm"]),
                "outer_origin_mm": list(result.board_outer_origin_mm),
            },
            "source_checkerboard_corners_px": result.source_corners_px.tolist(),
            "target_checkerboard_corners_px": result.target_corners_px.tolist(),
            "rms_reprojection_px": result.rms_reprojection_px,
        }

    pieces_rectified = rectify_image(
        pieces_undistorted,
        homography,
        canvas_px,
    )
    empty_rectified = rectify_image(
        empty_undistorted,
        homography,
        canvas_px,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imwrite(args.output_dir / "pieces_rectified.png", pieces_rectified)
    imwrite(args.output_dir / "empty_rectified.png", empty_rectified)
    imwrite(args.output_dir / debug_name, debug)
    write_json(
        args.output_dir / "rectification.json",
        {
            "schema_version": "1.0",
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": distortion.tolist(),
            "workspace_mm": list(workspace_mm),
            "canvas_px": list(canvas_px),
            "px_per_mm": [
                canvas_px[0] / workspace_mm[0],
                canvas_px[1] / workspace_mm[1],
            ],
            "homography_raw_undistorted_to_rectified": homography.tolist(),
            "inverse_homography": np.linalg.inv(homography).tolist(),
            "input_pieces": str(args.pieces),
            "input_empty": str(args.empty),
            "input_plane": None if args.plane is None else str(args.plane),
            **metadata,
        },
    )

    print(
        "Method: "
        + (
            "ArUco markers"
            if args.aruco_layout is not None
            else "checkerboard plane"
        )
    )
    print(f"Homography RMS: {metadata['rms_reprojection_px']:.3f} px")
    print(f"Output directory: {args.output_dir}")
    if metadata["rms_reprojection_px"] > 2.0:
        print("WARNING: homography RMS > 2 px. Check board flatness and layout.")


if __name__ == "__main__":
    main()
