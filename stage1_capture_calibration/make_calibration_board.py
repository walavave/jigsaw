#!/usr/bin/env python3
"""Generate a physically sized chessboard PDF for camera calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


MM = 72.0 / 25.4
FONT = "STSong-Light"
PAGE_MARGIN_X_MM = 15.0
BOARD_AREA_BOTTOM_MM = 38.0
BOARD_AREA_TOP_MARGIN_MM = 29.0


def mm(value: float) -> float:
    return value * MM


def draw_checkerboard(
    c: canvas.Canvas,
    cols_inner: int,
    rows_inner: int,
    square_mm: float,
) -> tuple[float, float, float, float]:
    squares_x = cols_inner + 1
    squares_y = rows_inner + 1
    board_w = squares_x * square_mm
    board_h = squares_y * square_mm
    left = (A4[0] - mm(board_w)) / 2.0
    area_bottom = mm(BOARD_AREA_BOTTOM_MM)
    area_top = A4[1] - mm(BOARD_AREA_TOP_MARGIN_MM)
    bottom = area_bottom + (area_top - area_bottom - mm(board_h)) / 2.0

    c.setStrokeColorRGB(0.65, 0.65, 0.65)
    c.setLineWidth(0.4)
    c.rect(
        left - mm(2),
        bottom - mm(2),
        mm(board_w + 4),
        mm(board_h + 4),
        stroke=1,
        fill=0,
    )

    c.setFillColorRGB(0, 0, 0)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                c.rect(
                    left + mm(col * square_mm),
                    bottom + mm(row * square_mm),
                    mm(square_mm),
                    mm(square_mm),
                    stroke=0,
                    fill=1,
                )

    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont(FONT, 15)
    c.drawCentredString(
        A4[0] / 2,
        A4[1] - mm(11),
        "相机内参标定棋盘格",
    )
    c.setFont(FONT, 8)
    c.drawCentredString(
        A4[0] / 2,
        A4[1] - mm(18),
        f"内角点 {cols_inner} x {rows_inner} | 方格 {square_mm:g} mm | 打印比例 100%",
    )
    return (left, bottom, mm(board_w), mm(board_h))


def draw_instructions(c: canvas.Canvas, cols: int, rows: int, square_mm: float) -> None:
    c.setFillColorRGB(0.15, 0.15, 0.15)
    lines = [
        "打印：A4 / 100% / 实际大小，关闭“适合页面”和自动缩放。",
        "拍摄：固定焦距和手动对焦，改变位置与倾斜角拍 15-25 张。",
        f"标定参数：--cols {cols} --rows {rows} --square-mm {square_mm:g}",
    ]
    y = mm(32)
    for index, text in enumerate(lines):
        c.setFont(FONT, 7.5)
        c.drawString(mm(18), y - mm(index * 6), text)

    line_y = mm(9)
    c.setStrokeColorRGB(0.1, 0.1, 0.1)
    c.setLineWidth(0.8)
    c.line(mm(55), line_y, mm(155), line_y)
    c.line(mm(55), line_y - mm(1.5), mm(55), line_y + mm(1.5))
    c.line(mm(155), line_y - mm(1.5), mm(155), line_y + mm(1.5))
    c.setFont(FONT, 6.5)
    c.drawCentredString(A4[0] / 2, line_y + mm(2.2), "100 mm 校准线")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cols", type=int, default=6, help="Inner corners per row.")
    parser.add_argument("--rows", type=int, default=8, help="Inner corners per column.")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to checkerboard_<cols>x<rows>_<square>mm.pdf.",
    )
    args = parser.parse_args()

    board_w = (args.cols + 1) * args.square_mm
    board_h = (args.rows + 1) * args.square_mm
    available_w = (A4[0] / MM) - 2 * PAGE_MARGIN_X_MM
    available_h = (
        A4[1] / MM - BOARD_AREA_TOP_MARGIN_MM - BOARD_AREA_BOTTOM_MM
    )
    if board_w > available_w or board_h > available_h:
        raise ValueError(
            f"Board {board_w:g} x {board_h:g} mm is too large for the A4 "
            f"printable area {available_w:g} x {available_h:g} mm."
        )

    output = args.output
    if output is None:
        output = (
            Path(__file__).resolve().parent
            / "output"
            / "calibration_targets"
            / f"checkerboard_{args.cols}x{args.rows}_{args.square_mm:g}mm.pdf"
        )

    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=A4)
    c.setTitle("Camera calibration chessboard")
    draw_checkerboard(c, args.cols, args.rows, args.square_mm)
    draw_instructions(c, args.cols, args.rows, args.square_mm)
    c.showPage()
    c.save()
    print(f"PDF: {output}")
    print(f"Inner corners: {args.cols} x {args.rows}")
    print(f"Square: {args.square_mm:g} mm")
    print(f"Board outer size: {board_w:g} x {board_h:g} mm")


if __name__ == "__main__":
    main()

