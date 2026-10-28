#!/usr/bin/env python3
"""Calibrate camera intrinsics from a directory of chessboard images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from jigsaw_common.io import imread, imwrite, iter_images, write_json  # noqa: E402

try:
    from .calibration import (  # noqa: E402
        calibrate_from_images,
        calibration_to_dict,
        undistort_image,
    )
except ImportError:  # Direct script execution.
    from calibration import (  # noqa: E402
        calibrate_from_images,
        calibration_to_dict,
        undistort_image,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing 15-25 chessboard photos.",
    )
    parser.add_argument("--cols", type=int, default=6, help="Inner corners per row.")
    parser.add_argument("--rows", type=int, default=8, help="Inner corners per column.")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "stage1_capture_calibration"
            / "output"
            / "calibration"
            / "intrinsics.json"
        ),
    )
    parser.add_argument(
        "--undistorted-example",
        type=Path,
        default=None,
        help="Optional path for an undistorted sample image.",
    )
    args = parser.parse_args()

    paths = iter_images(args.images)
    if not paths:
        raise FileNotFoundError(f"No calibration images found in {args.images}")

    calibration = calibrate_from_images(
        paths,
        cols=args.cols,
        rows=args.rows,
        square_mm=args.square_mm,
    )
    payload = calibration_to_dict(
        calibration,
        cols=args.cols,
        rows=args.rows,
        square_mm=args.square_mm,
    )
    write_json(args.output, payload)

    example_path = args.undistorted_example
    if example_path is None:
        example_path = args.output.parent / "undistorted_example.png"
    example = undistort_image(
        imread(calibration.accepted_images[0]),
        calibration.camera_matrix,
        calibration.distortion,
    )
    imwrite(example_path, example)

    print()
    print(f"Accepted views: {len(calibration.accepted_images)}")
    print(f"RMS reprojection: {calibration.rms_reprojection_px:.4f} px")
    print(f"Calibration JSON: {args.output}")
    print(f"Undistorted sample: {example_path}")
    if calibration.rms_reprojection_px > 1.0:
        print("WARNING: RMS > 1.0 px. Retake blurry or low-tilt views.")


if __name__ == "__main__":
    main()

