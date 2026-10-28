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


def mm(value: float) -> float:
    return value * MM


def draw_checkerboard(
    c: canvas.Canvas,
    cols_inner: int,
    rows_inner: int,
    square_mm: float,
) -> None:
    squares_x = cols_inner + 1
    squares_y = rows_inner + 1
    board_w = squares_x * square_mm
    board_h = squares_y * square_mm
    left = (A4[0] - mm(board_w)) / 2.0
    bottom = mm(82)

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
        bottom + mm(board_h) + mm(16),
        "相机内参标定棋盘格",
    )
    c.setFont(FONT, 8)
    c.drawCentredString(
        A4[0] / 2,
        bottom + mm(board_h) + mm(10),
        f"内角点 {cols_inner} x {rows_inner} | 方格 {square_mm:g} mm | 打印比例 100%",
    )


def draw_instructions(c: canvas.Canvas, cols: int, rows: int, square_mm: float) -> None:
    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.setFont(FONT, 9)
    y = mm(70)
    lines = [
        "使用方法",
        f"1. 按 A4、100% 实际大小打印，不要选择“适合页面”。",
        f"2. 量下方 100 mm 校准线；长度不符时重新打印。",
        f"3. 用同一台相机、同一分辨率、同一焦距和手动对焦拍摄 15-25 张。",
        f"4. 让棋盘格覆盖画面中心/边缘，并改变倾斜角度；避免反光和运动模糊。",
        f"5. 运行 calibrate_intrinsics.py，参数使用 --cols {cols} --rows {rows} --square-mm {square_mm:g}。",
    ]
    for index, text in enumerate(lines):
        c.setFont(FONT, 10 if index == 0 else 8)
        c.drawString(mm(18), y - mm(index * 6.2), text)

    line_y = mm(20)
    c.setStrokeColorRGB(0.1, 0.1, 0.1)
    c.setLineWidth(0.8)
    c.line(mm(18), line_y, mm(118), line_y)
    c.line(mm(18), line_y - mm(1.5), mm(18), line_y + mm(1.5))
    c.line(mm(118), line_y - mm(1.5), mm(118), line_y + mm(1.5))
    c.setFont(FONT, 7)
    c.drawString(mm(18), line_y + mm(2.5), "100 mm 校准线")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cols", type=int, default=9, help="Inner corners per row.")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners per column.")
    parser.add_argument("--square-mm", type=float, default=18.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "output"
            / "calibration_targets"
            / "checkerboard_9x6_18mm.pdf"
        ),
    )
    args = parser.parse_args()

    board_w = (args.cols + 1) * args.square_mm
    board_h = (args.rows + 1) * args.square_mm
    if board_w > 190 or board_h > 190:
        raise ValueError(
            f"Board {board_w:g} x {board_h:g} mm is too large for A4 printing."
        )

    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.output), pagesize=A4)
    c.setTitle("Camera calibration chessboard")
    draw_checkerboard(c, args.cols, args.rows, args.square_mm)
    draw_instructions(c, args.cols, args.rows, args.square_mm)
    c.showPage()
    c.save()
    print(f"PDF: {args.output}")
    print(f"Inner corners: {args.cols} x {args.rows}")
    print(f"Square: {args.square_mm:g} mm")


if __name__ == "__main__":
    main()

