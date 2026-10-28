#!/usr/bin/env python3
"""Generate physically sized ArUco markers as PNG and printable PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


MM = 72.0 / 25.4
FONT = "STSong-Light"


def mm(value: float) -> float:
    return value * MM


def parse_ids(value: str) -> list[int]:
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(ids) < 4:
        raise argparse.ArgumentTypeError("At least four marker IDs are required.")
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("Marker IDs must be unique.")
    return ids


def make_marker_image(dictionary_name: str, marker_id: int, size_mm: float) -> Image.Image:
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    pixels = max(64, round(size_mm / 25.4 * 600))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels, borderBits=1)
    return Image.fromarray(np.asarray(marker, dtype=np.uint8), mode="L").convert("RGB")


def draw_marker_page(
    c: canvas.Canvas,
    marker: Image.Image,
    dictionary_name: str,
    marker_id: int,
    size_mm: float,
) -> None:
    marker_size = mm(size_mm)
    cut_size = mm(size_mm + 10)
    left = (A4[0] - marker_size) / 2.0
    top = (A4[1] + marker_size) / 2.0

    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont(FONT, 14)
    c.drawCentredString(A4[0] / 2, A4[1] - mm(38), "ArUco 标定标记")
    c.setFont(FONT, 10)
    c.drawCentredString(
        A4[0] / 2,
        A4[1] - mm(47),
        f"{dictionary_name} | ID {marker_id} | {size_mm:g} mm",
    )

    c.drawImage(
        ImageReader(marker),
        left,
        top - marker_size,
        width=marker_size,
        height=marker_size,
        preserveAspectRatio=True,
        mask="auto",
    )

    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    c.setLineWidth(0.5)
    c.setDash(3, 3)
    c.rect(
        left - mm(5),
        top - marker_size - mm(5),
        cut_size,
        cut_size,
        stroke=1,
        fill=0,
    )
    c.setDash()

    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.setFont(FONT, 7)
    c.drawCentredString(
        A4[0] / 2,
        top - marker_size - mm(14),
        "沿外侧虚线裁下，保留标记周围白边；按 100% 实际大小打印。",
    )
    c.drawCentredString(
        A4[0] / 2,
        mm(18),
        "同一相机、同一焦距拍摄；四个 ID 的标记全部保持在画面内。",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--ids", type=parse_ids, default=[0, 1, 2, 3])
    parser.add_argument("--size-mm", type=float, default=40.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "output"
            / "calibration_targets"
            / "aruco_markers"
        ),
    )
    args = parser.parse_args()

    if args.size_mm <= 10:
        raise ValueError("Marker size must be greater than 10 mm.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    pdf_path = args.output_dir / f"aruco_{args.dictionary}_{args.size_mm:g}mm.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.setTitle("Printable ArUco calibration markers")

    for marker_id in args.ids:
        marker = make_marker_image(args.dictionary, marker_id, args.size_mm)
        png_path = args.output_dir / f"{args.dictionary}_id{marker_id}_{args.size_mm:g}mm.png"
        marker.save(png_path, dpi=(600, 600))
        draw_marker_page(c, marker, args.dictionary, marker_id, args.size_mm)
        c.showPage()

    c.save()
    print(f"PDF: {pdf_path}")
    print(f"PNG directory: {args.output_dir}")
    print(f"Marker size: {args.size_mm:g} mm")


if __name__ == "__main__":
    main()

