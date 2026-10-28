"""Rectify a table plane from one chessboard image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

try:
    from .calibration import detect_chessboard
except ImportError:  # Direct script execution.
    from calibration import detect_chessboard


@dataclass
class CheckerboardPlaneResult:
    homography: np.ndarray
    source_corners_px: np.ndarray
    target_corners_px: np.ndarray
    rms_reprojection_px: float
    board_outer_origin_mm: tuple[float, float]
    workspace_mm: tuple[float, float]
    canvas_px: tuple[int, int]


def validate_checkerboard_layout(layout: Mapping) -> None:
    required = {
        "cols",
        "rows",
        "square_mm",
        "workspace_mm",
        "canvas_px",
    }
    missing = required - set(layout)
    if missing:
        raise ValueError(f"Checkerboard layout is missing: {sorted(missing)}")

    cols = int(layout["cols"])
    rows = int(layout["rows"])
    square_mm = float(layout["square_mm"])
    workspace_mm = tuple(float(value) for value in layout["workspace_mm"])
    canvas_px = tuple(int(value) for value in layout["canvas_px"])
    if cols < 3 or rows < 3:
        raise ValueError("Checkerboard inner-corner counts must be at least 3 x 3.")
    if square_mm <= 0:
        raise ValueError("square_mm must be positive.")
    if len(workspace_mm) != 2 or min(workspace_mm) <= 0:
        raise ValueError("workspace_mm must contain two positive dimensions.")
    if len(canvas_px) != 2 or min(canvas_px) <= 0:
        raise ValueError("canvas_px must contain two positive dimensions.")

    scale_x = canvas_px[0] / workspace_mm[0]
    scale_y = canvas_px[1] / workspace_mm[1]
    if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.01:
        raise ValueError(
            "canvas_px and workspace_mm have inconsistent aspect ratios: "
            f"{scale_x:.3f} vs {scale_y:.3f} px/mm."
        )


def checkerboard_outer_size_mm(
    cols: int,
    rows: int,
    square_mm: float,
) -> tuple[float, float]:
    """Return the physical outer size, including the outer square ring."""

    return ((cols + 1) * square_mm, (rows + 1) * square_mm)


def centered_board_origin_mm(
    cols: int,
    rows: int,
    square_mm: float,
    workspace_mm: tuple[float, float],
) -> tuple[float, float]:
    board_width, board_height = checkerboard_outer_size_mm(cols, rows, square_mm)
    return (
        (workspace_mm[0] - board_width) / 2.0,
        (workspace_mm[1] - board_height) / 2.0,
    )


def orient_chessboard_corners(
    corners: np.ndarray,
    cols: int,
    rows: int,
) -> np.ndarray:
    """Return corners row-major from the image's top-left board corner.

    A plain chessboard has no ID or arrow, so OpenCV may return the same grid
    rotated by 180 degrees.  For a normally mounted top-down camera, choosing
    the interpretation nearest the image's top-left preserves workspace
    orientation.
    """

    grid = np.asarray(corners, dtype=np.float32).reshape(rows, cols, 2)
    first = grid[0, 0]
    last = grid[-1, -1]
    if first[0] + first[1] > last[0] + last[1]:
        grid = grid[::-1, ::-1]
    return grid.reshape(-1, 1, 2)


def build_checkerboard_homography(
    image: np.ndarray,
    cols: int,
    rows: int,
    square_mm: float,
    workspace_mm: tuple[float, float],
    canvas_px: tuple[int, int],
    board_outer_origin_mm: tuple[float, float] | None = None,
) -> CheckerboardPlaneResult:
    """Map one flat chessboard image into a fixed top-view workspace.

    The board must be aligned with the desired workspace axes.  Its physical
    top-left outer corner is specified in workspace millimetres; by default a
    board that fits inside the workspace is centred automatically.
    """

    layout = {
        "cols": cols,
        "rows": rows,
        "square_mm": square_mm,
        "workspace_mm": list(workspace_mm),
        "canvas_px": list(canvas_px),
    }
    validate_checkerboard_layout(layout)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners = detect_chessboard(gray, cols, rows)
    if corners is None:
        raise RuntimeError(
            "Chessboard not found in the plane-calibration image. "
            "Use the same inner-corner counts and keep the full board visible."
        )
    corners = orient_chessboard_corners(corners, cols, rows)

    board_width, board_height = checkerboard_outer_size_mm(
        cols,
        rows,
        square_mm,
    )
    if board_outer_origin_mm is None:
        board_outer_origin_mm = centered_board_origin_mm(
            cols,
            rows,
            square_mm,
            workspace_mm,
        )

    origin_x, origin_y = board_outer_origin_mm
    if (
        origin_x < 0
        or origin_y < 0
        or origin_x + board_width > workspace_mm[0] + 1e-6
        or origin_y + board_height > workspace_mm[1] + 1e-6
    ):
        raise ValueError(
            "The chessboard outer bounds must lie inside workspace_mm; "
            f"board size is {board_width:g} x {board_height:g} mm."
        )

    source = corners.reshape(-1, 2).astype(np.float64)
    scale_x = canvas_px[0] / workspace_mm[0]
    scale_y = canvas_px[1] / workspace_mm[1]
    target = np.array(
        [
            [
                (origin_x + (col + 1) * square_mm) * scale_x,
                (origin_y + (row + 1) * square_mm) * scale_y,
            ]
            for row in range(rows)
            for col in range(cols)
        ],
        dtype=np.float64,
    )

    homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if homography is None:
        raise RuntimeError("Could not estimate the chessboard homography.")

    projected = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))

    return CheckerboardPlaneResult(
        homography=homography,
        source_corners_px=source,
        target_corners_px=target,
        rms_reprojection_px=rms,
        board_outer_origin_mm=(
            float(board_outer_origin_mm[0]),
            float(board_outer_origin_mm[1]),
        ),
        workspace_mm=workspace_mm,
        canvas_px=canvas_px,
    )


def draw_checkerboard_debug(
    image: np.ndarray,
    cols: int,
    rows: int,
    result: CheckerboardPlaneResult,
) -> np.ndarray:
    debug = image.copy()
    cv2.drawChessboardCorners(
        debug,
        (cols, rows),
        result.source_corners_px.astype(np.float32).reshape(-1, 1, 2),
        True,
    )
    cv2.putText(
        debug,
        f"Chessboard RMS {result.rms_reprojection_px:.3f}px",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return debug
