"""Camera intrinsic calibration helpers for stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass
class IntrinsicCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    rms_reprojection_px: float
    per_view_errors_px: list[float]
    accepted_images: list[str]


def chessboard_object_points(
    cols: int,
    rows: int,
    square_mm: float,
) -> np.ndarray:
    """Create 3-D object points for an inner-corner chessboard."""

    points = np.zeros((rows * cols, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return points


def detect_chessboard(
    gray: np.ndarray,
    cols: int,
    rows: int,
) -> np.ndarray | None:
    """Detect a chessboard with the subpixel detector and a classic fallback."""

    pattern = (cols, rows)
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    if found and corners is not None:
        return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=classic_flags)
    if not found or corners is None:
        return None

    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        1e-4,
    )
    cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return corners


def calibrate_from_images(
    image_paths: Sequence[Path],
    cols: int,
    rows: int,
    square_mm: float,
) -> IntrinsicCalibration:
    """Estimate camera intrinsics and distortion from chessboard views."""

    if cols < 3 or rows < 3:
        raise ValueError("Chessboard inner-corner counts must be at least 3 x 3.")
    if square_mm <= 0:
        raise ValueError("square_mm must be positive.")
    if len(image_paths) < 6:
        raise ValueError(
            "At least 6 valid calibration views are recommended; 15-25 is better."
        )

    object_template = chessboard_object_points(cols, rows, square_mm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[str] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        buffer = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[skip] cannot read: {path}")
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(f"[skip] inconsistent size: {path} {current_size} != {image_size}")
            continue

        corners = detect_chessboard(image, cols, rows)
        if corners is None:
            print(f"[skip] chessboard not found: {path.name}")
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted.append(str(path))
        print(f"[ok]   {path.name}")

    if image_size is None or len(object_points) < 6:
        raise RuntimeError(
            f"Only {len(object_points)} valid views found; at least 6 are required."
        )

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=0,
    )

    per_view_errors: list[float] = []
    for object_view, image_view, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(
            object_view,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        error = cv2.norm(image_view, projected, cv2.NORM_L2) / np.sqrt(
            len(projected)
        )
        per_view_errors.append(float(error))

    return IntrinsicCalibration(
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1),
        image_size=image_size,
        rms_reprojection_px=float(rms),
        per_view_errors_px=per_view_errors,
        accepted_images=accepted,
    )


def calibration_to_dict(
    calibration: IntrinsicCalibration,
    cols: int,
    rows: int,
    square_mm: float,
) -> dict:
    return {
        "schema_version": "1.0",
        "calibration_type": "chessboard_intrinsics",
        "image_size_px": list(calibration.image_size),
        "pattern": {
            "cols_inner_corners": cols,
            "rows_inner_corners": rows,
            "square_mm": square_mm,
        },
        "camera_matrix": calibration.camera_matrix.tolist(),
        "distortion_coefficients": calibration.distortion.tolist(),
        "rms_reprojection_px": calibration.rms_reprojection_px,
        "per_view_errors_px": calibration.per_view_errors_px,
        "accepted_images": calibration.accepted_images,
        "image_count": len(calibration.accepted_images),
    }


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    return cv2.undistort(image, camera_matrix, distortion)

