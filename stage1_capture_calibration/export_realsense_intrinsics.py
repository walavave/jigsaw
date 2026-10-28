#!/usr/bin/env python3
"""Export Intel RealSense factory colour intrinsics for OpenCV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from jigsaw_common.io import write_json  # noqa: E402


def build_intrinsics_payload(
    width: int,
    height: int,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    model: str,
    coeffs: list[float],
) -> dict:
    """Convert librealsense values into the project's intrinsics schema."""

    coefficients = [float(value) for value in coeffs]
    while len(coefficients) < 5:
        coefficients.append(0.0)
    coefficients = coefficients[:5]
    model_key = model.lower().replace("_", " ").replace("-", " ")
    has_nonzero_distortion = any(abs(value) > 1e-12 for value in coefficients)

    supported = "brown conrady" in model_key and "inverse" not in model_key
    if has_nonzero_distortion and not supported:
        raise ValueError(
            "librealsense reports a distortion model that cv2.undistort cannot "
            f"consume directly: {model}. Use the factory SDK point/colour APIs, "
            "or calibrate the colour camera with the chessboard instead."
        )

    return {
        "schema_version": "1.0",
        "calibration_type": "realsense_factory_color_intrinsics",
        "image_size_px": [int(width), int(height)],
        "camera_matrix": [
            [float(fx), 0.0, float(ppx)],
            [0.0, float(fy), float(ppy)],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": coefficients,
        "distortion_model": model,
        "rms_reprojection_px": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
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
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise SystemExit(
            "pyrealsense2 is not installed. Install Intel librealsense Python "
            "bindings, or use calibrate_intrinsics.py with chessboard photos."
        ) from error

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )

    try:
        profile = pipeline.start(config)
        pipeline.wait_for_frames()
        video_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = video_profile.get_intrinsics()
        payload = build_intrinsics_payload(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            ppx=intrinsics.ppx,
            ppy=intrinsics.ppy,
            model=str(intrinsics.model),
            coeffs=list(intrinsics.coeffs),
        )
        payload["requested_stream"] = {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
        }
        write_json(args.output, payload)
        print(f"RealSense colour intrinsics: {args.output}")
        print(f"Stream: {intrinsics.width} x {intrinsics.height}")
        print(f"Distortion model: {intrinsics.model}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
