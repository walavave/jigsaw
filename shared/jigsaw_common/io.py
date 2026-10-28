"""Filesystem helpers that work with non-ASCII project paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image via a byte buffer so Chinese paths work on all platforms."""

    file_path = Path(path)
    buffer = np.fromfile(file_path, dtype=np.uint8)
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")
    return image


def imwrite(path: str | Path, image: np.ndarray) -> None:
    """Write an image while preserving non-ASCII paths."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    extension = file_path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Could not encode image for: {file_path}")
    encoded.tofile(file_path)


def iter_images(directory: str | Path) -> list[Path]:
    """Return supported images in stable filename order."""

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: object) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return float("nan")
    return sum(materialized) / len(materialized)

