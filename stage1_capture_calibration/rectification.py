"""Planar rectification using four or more ArUco marker centres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np


@dataclass
class MarkerDetection:
    marker_id: int
    centre_px: tuple[float, float]
    corners_px: np.ndarray


@dataclass
class RectificationResult:
    homography: np.ndarray
    source_centres_px: np.ndarray
    target_centres_px: np.ndarray
    rms_reprojection_px: float
    detections: list[MarkerDetection]


def _dictionary(name: str):
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def detect_marker_centres(
    image: np.ndarray,
    dictionary_name: str,
) -> list[MarkerDetection]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = _dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # pragma: no cover - compatibility with older OpenCV 4.x
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        return []

    detections: list[MarkerDetection] = []
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        centre = points.mean(axis=0)
        detections.append(
            MarkerDetection(
                marker_id=int(marker_id),
                centre_px=(float(centre[0]), float(centre[1])),
                corners_px=points,
            )
        )
    return detections


def build_homography(
    detections: list[MarkerDetection],
    markers: list[dict],
    workspace_mm: tuple[float, float],
    canvas_px: tuple[int, int],
) -> RectificationResult:
    by_id = {detection.marker_id: detection for detection in detections}
    required_ids = [int(marker["id"]) for marker in markers]
    missing = [marker_id for marker_id in required_ids if marker_id not in by_id]
    if missing:
        raise RuntimeError(
            "Missing ArUco markers: "
            f"{missing}. Detected IDs: {sorted(by_id)}. "
            "Keep all four markers fully visible and in focus."
        )

    scale_x = canvas_px[0] / workspace_mm[0]
    scale_y = canvas_px[1] / workspace_mm[1]
    if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.01:
        raise ValueError(
            "canvas_px and workspace_mm have inconsistent aspect ratios: "
            f"{scale_x:.3f} vs {scale_y:.3f} px/mm."
        )

    ordered = sorted(markers, key=lambda item: int(item["id"]))
    source = np.array(
        [by_id[int(marker["id"])].centre_px for marker in ordered],
        dtype=np.float64,
    )
    target = np.array(
        [
            [
                float(marker["center_mm"][0]) * scale_x,
                float(marker["center_mm"][1]) * scale_y,
            ]
            for marker in ordered
        ],
        dtype=np.float64,
    )

    method = cv2.RANSAC if len(source) >= 5 else 0
    homography, _ = cv2.findHomography(source, target, method, 3.0)
    if homography is None:
        raise RuntimeError("Could not estimate the rectification homography.")

    source_h = source.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source_h, homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))

    return RectificationResult(
        homography=homography,
        source_centres_px=source,
        target_centres_px=target,
        rms_reprojection_px=rms,
        detections=detections,
    )


def rectify_image(
    image: np.ndarray,
    homography: np.ndarray,
    canvas_px: tuple[int, int],
) -> np.ndarray:
    return cv2.warpPerspective(
        image,
        homography,
        canvas_px,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def draw_detection_debug(
    image: np.ndarray,
    detections: list[MarkerDetection],
) -> np.ndarray:
    debug = image.copy()
    if detections:
        corners = [detection.corners_px.reshape(1, 4, 2) for detection in detections]
        ids = np.array([[detection.marker_id] for detection in detections], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
        for detection in detections:
            x, y = detection.centre_px
            cv2.circle(debug, (round(x), round(y)), 4, (0, 0, 255), -1)
            cv2.putText(
                debug,
                f"ID {detection.marker_id}",
                (round(x) + 7, round(y) - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return debug


def validate_marker_layout(layout: Mapping) -> None:
    required = {"dictionary", "workspace_mm", "canvas_px", "markers"}
    missing = required - set(layout)
    if missing:
        raise ValueError(f"Marker layout is missing: {sorted(missing)}")
    workspace = layout["workspace_mm"]
    canvas = layout["canvas_px"]
    if len(workspace) != 2 or min(float(value) for value in workspace) <= 0:
        raise ValueError("workspace_mm must contain two positive dimensions.")
    if len(canvas) != 2 or min(int(value) for value in canvas) <= 0:
        raise ValueError("canvas_px must contain two positive dimensions.")
    if len(layout["markers"]) < 4:
        raise ValueError("At least four ArUco markers are required.")
    ids = [int(marker["id"]) for marker in layout["markers"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Marker IDs must be unique.")
    for marker in layout["markers"]:
        if "center_mm" not in marker or len(marker["center_mm"]) != 2:
            raise ValueError(f"Marker {marker.get('id')} has no valid center_mm.")
        x, y = (float(value) for value in marker["center_mm"])
        width, height = (float(value) for value in workspace)
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(
                f"Marker {marker['id']} center ({x}, {y}) is outside "
                f"the {width} x {height} mm workspace."
            )

