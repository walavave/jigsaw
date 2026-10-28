#!/usr/bin/env python3
"""Generate a print-ready PDF for the recursive 16-piece cut layout."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from jigsaw_common.cut_layout import (  # noqa: E402
    CutLayout,
    ExplodedPiece,
    build_exploded_pieces,
    build_cut_layout,
    line_segments,
    minimum_boundary_stagger,
    polygon_points,
)


MM = 72.0 / 25.4
PAGE_W, PAGE_H = landscape(A4)
FONT_NAME = "STSong-Light"

INK = (0.12, 0.13, 0.15)
MUTED = (0.40, 0.42, 0.45)
PAPER = (1.0, 1.0, 1.0)
ACCENT = (0.88, 0.03, 0.36)
ACCENT_SOFT = (1.0, 0.86, 0.92)
CUT_BLUE = (0.02, 0.37, 0.78)


def mm(value: float) -> float:
    return value * MM


@dataclass(frozen=True)
class PageTransform:
    image_left: float
    image_top: float
    image_width: float
    image_height: float
    source_width: float
    source_height: float

    def point(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            self.image_left + (x / self.source_width) * self.image_width,
            self.image_top - (y / self.source_height) * self.image_height,
        )


def register_fonts() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))


def draw_page_header(
    c: canvas.Canvas,
    title: str,
    subtitle: str,
    page_number: str,
) -> None:
    c.setFillColorRGB(*INK)
    c.setFont(FONT_NAME, 15)
    c.drawString(mm(14), PAGE_H - mm(14), title)
    c.setFont(FONT_NAME, 7.5)
    c.setFillColorRGB(*MUTED)
    c.drawString(mm(14), PAGE_H - mm(19), subtitle)
    c.drawRightString(PAGE_W - mm(14), PAGE_H - mm(14), page_number)
    c.setStrokeColorRGB(0.82, 0.83, 0.85)
    c.setLineWidth(0.6)
    c.line(mm(14), PAGE_H - mm(22), PAGE_W - mm(14), PAGE_H - mm(22))


def draw_footer(c: canvas.Canvas, page_number: int) -> None:
    c.setStrokeColorRGB(0.82, 0.83, 0.85)
    c.setLineWidth(0.5)
    c.line(mm(14), mm(17), PAGE_W - mm(14), mm(17))
    c.setFillColorRGB(*MUTED)
    c.setFont(FONT_NAME, 6.5)
    c.drawString(mm(14), mm(10), "打印设置：A4 横向 / 100% / 实际大小 / 关闭适应页面")
    c.drawRightString(PAGE_W - mm(14), mm(10), f"{page_number} / 2")


def polygon_path(c: canvas.Canvas, points: list[tuple[float, float]]):
    if not points:
        raise ValueError("A polygon path needs at least one point.")
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()
    return path


def exploded_bbox(
    exploded_pieces: list[ExplodedPiece],
) -> tuple[float, float, float, float]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for piece in exploded_pieces:
        for x, y in polygon_points(piece.polygon):
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    return (min_x, min_y, max_x, max_y)


def fit_exploded_scale(
    exploded_pieces: list[ExplodedPiece],
    available_width_mm: float,
    available_height_mm: float,
) -> tuple[float, tuple[float, float, float, float]]:
    """Find the largest shared scale in millimetres per source pixel."""

    bbox = exploded_bbox(exploded_pieces)
    width_px = bbox[2] - bbox[0]
    height_px = bbox[3] - bbox[1]
    scale = min(available_width_mm / width_px, available_height_mm / height_px)
    return (scale, bbox)


def minimum_exploded_gap(exploded_pieces: list[ExplodedPiece]) -> float:
    minimum = float("inf")
    for first_index in range(len(exploded_pieces)):
        for second_index in range(first_index + 1, len(exploded_pieces)):
            minimum = min(
                minimum,
                exploded_pieces[first_index].polygon.distance(
                    exploded_pieces[second_index].polygon
                ),
            )
    return minimum


def render_cut_sheet(
    c: canvas.Canvas,
    layout: CutLayout,
    image_path: Path,
    requested_gap_mm: float,
) -> float:
    left = mm(14)
    right = PAGE_W - mm(14)
    bottom = mm(31)
    top = PAGE_H - mm(26)
    available_width_mm = (right - left) / MM
    available_height_mm = (top - bottom) / MM

    # Estimate a target gap in source pixels, then refine it if fitting to the
    # page causes the physical gap to become smaller than requested.
    target_gap_px = requested_gap_mm / 0.20
    exploded: list[ExplodedPiece] = []
    scale = 0.0
    bbox = (0.0, 0.0, 0.0, 0.0)
    for _ in range(8):
        exploded = build_exploded_pieces(layout, target_gap_px)
        scale, bbox = fit_exploded_scale(
            exploded,
            available_width_mm,
            available_height_mm,
        )
        actual_gap_mm = minimum_exploded_gap(exploded) * scale
        if actual_gap_mm >= requested_gap_mm * 0.98:
            break
        target_gap_px *= (requested_gap_mm / actual_gap_mm) * 1.08
    else:
        raise RuntimeError("Could not fit pieces with the requested cutting gap.")

    bbox_width = (bbox[2] - bbox[0]) * scale * MM
    bbox_height = (bbox[3] - bbox[1]) * scale * MM
    bbox_left = left + ((right - left) - bbox_width) / 2.0
    bbox_top = top - ((top - bottom) - bbox_height) / 2.0

    def to_page(point: tuple[float, float]) -> tuple[float, float]:
        return (
            bbox_left + (point[0] - bbox[0]) * scale * MM,
            bbox_top - (point[1] - bbox[1]) * scale * MM,
        )

    image_reader = ImageReader(str(image_path))
    image_height_mm = layout.height * scale
    image_width_mm = layout.width * scale

    # Draw each translated piece by clipping the same source image to its
    # polygon.  The pieces remain separated by a small white cutting gap.
    for piece in exploded:
        page_points = [to_page(point) for point in polygon_points(piece.polygon)]
        piece_path = polygon_path(c, page_points)

        c.saveState()
        c.clipPath(piece_path, stroke=0, fill=0)
        image_left = to_page(piece.shift)[0]
        image_top = to_page(piece.shift)[1]
        c.drawImage(
            image_reader,
            image_left,
            image_top - image_height_mm * MM,
            width=image_width_mm * MM,
            height=image_height_mm * MM,
            preserveAspectRatio=False,
            mask="auto",
        )
        c.restoreState()

        piece_path = polygon_path(c, page_points)
        c.setStrokeColorRGB(*ACCENT)
        c.setLineWidth(0.52)
        c.drawPath(piece_path, stroke=1, fill=0)

    # A compact scale bar makes print scaling errors immediately visible.
    bar_y = mm(23)
    bar_start = mm(14)
    c.setStrokeColorRGB(*INK)
    c.setLineWidth(0.8)
    c.line(bar_start, bar_y, bar_start + mm(100), bar_y)
    c.line(bar_start, bar_y - mm(1.4), bar_start, bar_y + mm(1.4))
    c.line(bar_start + mm(100), bar_y - mm(1.4), bar_start + mm(100), bar_y + mm(1.4))
    c.setFillColorRGB(*MUTED)
    c.setFont(FONT_NAME, 6.5)
    c.drawString(bar_start, bar_y + mm(2.2), "100 mm 校准线")
    c.drawString(bar_start + mm(101.5), bar_y - mm(0.6), "打印后先量此线")

    return scale


def draw_cut_legend(c: canvas.Canvas) -> None:
    x = mm(14)
    y = mm(19.2)
    c.setFont(FONT_NAME, 6.5)
    c.setFillColorRGB(*MUTED)
    c.circle(x + mm(1.2), y + mm(0.6), mm(1.0), fill=1, stroke=0)
    c.setFillColorRGB(*ACCENT)
    c.circle(x + mm(1.2), y + mm(0.6), mm(1.0), fill=1, stroke=0)
    c.setFillColorRGB(*MUTED)
    c.drawString(x + mm(3.0), y, "洋红色边界 = 沿此线裁剪；碎片之间已留出可下刀间隙")


def render_assembly_guide(
    c: canvas.Canvas,
    layout: CutLayout,
    image_path: Path,
) -> None:
    image_width = mm(210)
    image_height = image_width if layout.width == layout.height else image_width * layout.height / layout.width
    image_left = (PAGE_W - image_width) / 2.0
    image_top = PAGE_H - mm(28)
    image_bottom = image_top - image_height
    transform = PageTransform(
        image_left=image_left,
        image_top=image_top,
        image_width=image_width,
        image_height=image_height,
        source_width=layout.width,
        source_height=layout.height,
    )

    c.drawImage(
        ImageReader(str(image_path)),
        image_left,
        image_bottom,
        width=image_width,
        height=image_height,
        preserveAspectRatio=False,
        mask="auto",
    )

    # De-duplicate shared boundaries so every cut is a single crisp line.
    drawn: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    c.setStrokeColorRGB(*ACCENT)
    c.setLineWidth(0.72)
    for piece in layout.pieces:
        for start, end in line_segments(piece.polygon):
            a = (round(start[0], 3), round(start[1], 3))
            b = (round(end[0], 3), round(end[1], 3))
            key = tuple(sorted((a, b)))  # type: ignore[arg-type]
            if key in drawn:
                continue
            drawn.add(key)  # type: ignore[arg-type]
            pa = transform.point(start)
            pb = transform.point(end)
            c.line(pa[0], pa[1], pb[0], pb[1])

    # Piece IDs.
    c.setFont(FONT_NAME, 6.6)
    for piece in layout.pieces:
        px, py = transform.point(piece.centroid)
        c.setFillColorRGB(*PAPER)
        c.setStrokeColorRGB(*INK)
        c.setLineWidth(0.35)
        c.circle(px, py, mm(3.1), fill=1, stroke=1)
        c.setFillColorRGB(*INK)
        label = piece.piece_id[1:]
        c.drawCentredString(px, py - mm(1.25), label)

    # First-level centre and the four second-level centres.
    c.setFillColorRGB(*ACCENT)
    p = transform.point(layout.first_split.center)
    c.circle(p[0], p[1], mm(1.7), fill=1, stroke=0)
    c.setFillColorRGB(*PAPER)
    c.setFont(FONT_NAME, 5.5)
    c.drawCentredString(p[0], p[1] - mm(0.9), "A")

    c.setFont(FONT_NAME, 6.0)
    for index, region in enumerate(layout.regions, start=1):
        q = transform.point(region.split.center)
        c.setFillColorRGB(*CUT_BLUE)
        c.setStrokeColorRGB(*PAPER)
        c.setLineWidth(0.35)
        c.rect(q[0] - mm(1.5), q[1] - mm(1.5), mm(3.0), mm(3.0), fill=1, stroke=1)
        c.setFillColorRGB(*PAPER)
        c.drawCentredString(q[0], q[1] - mm(0.9), chr(64 + index))

    # Legend panel.
    panel_top = image_bottom - mm(8)
    c.setFillColorRGB(*INK)
    c.setFont(FONT_NAME, 10.5)
    c.drawString(mm(14), panel_top, "装配编号图")
    c.setFont(FONT_NAME, 6.8)
    c.setFillColorRGB(*MUTED)
    c.drawString(
        mm(14),
        panel_top - mm(5.0),
        "此页用于核对编号和切割结构，不用于裁剪。第 1 页才是一比一裁剪页。",
    )

    legend_y = panel_top - mm(12.0)
    c.setFillColorRGB(*ACCENT)
    c.circle(mm(16.5), legend_y + mm(0.8), mm(1.3), fill=1, stroke=0)
    c.setFillColorRGB(*INK)
    c.setFont(FONT_NAME, 7.0)
    c.drawString(mm(20.0), legend_y, "A：一级切分点，从该点向外延伸两条相交直线")

    legend_y -= mm(6.2)
    c.setFillColorRGB(*CUT_BLUE)
    c.rect(mm(15.2), legend_y - mm(0.5), mm(2.6), mm(2.6), fill=1, stroke=0)
    c.setFillColorRGB(*INK)
    c.drawString(mm(20.0), legend_y, "B-E：四个一级区域各自的二级切分点")

    legend_y -= mm(6.2)
    c.setStrokeColorRGB(*ACCENT)
    c.setLineWidth(1.0)
    c.line(mm(14.5), legend_y + mm(0.8), mm(18.0), legend_y + mm(0.8))
    c.setFillColorRGB(*INK)
    c.drawString(mm(20.0), legend_y, "洋红色折线：16 块碎片的最终边界")

    legend_y -= mm(6.2)
    c.setFillColorRGB(*PAPER)
    c.setStrokeColorRGB(*INK)
    c.setLineWidth(0.35)
    c.circle(mm(16.2), legend_y + mm(0.8), mm(2.2), fill=1, stroke=1)
    c.setFillColorRGB(*INK)
    c.setFont(FONT_NAME, 5.5)
    c.drawCentredString(mm(16.2), legend_y + mm(0.1), "01")
    c.setFont(FONT_NAME, 7.0)
    c.drawString(mm(20.0), legend_y, "编号 01-16：碎片 ID，与 JSON 清单一致")

    # Geometry summary sits beside the subtitle instead of competing with the
    # page footer.
    c.setFillColorRGB(*MUTED)
    c.setFont(FONT_NAME, 6.2)
    c.drawRightString(
        PAGE_W - mm(14),
        panel_top - mm(5.0),
        f"原图 {layout.width} x {layout.height}px | 随机种子 {layout.seed} | 一级中心 "
        f"({layout.first_split.center[0]:.0f}, {layout.first_split.center[1]:.0f})px",
    )


def write_json(layout: CutLayout, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(layout.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_source_assets(
    layout: CutLayout,
    image_path: Path,
    output_dir: Path,
) -> None:
    """Write one RGBA crop and one mask for each source polygon."""

    output_dir.mkdir(parents=True, exist_ok=True)
    padding = 2
    with Image.open(image_path).convert("RGBA") as source:
        source.save(output_dir.parent / "reference.png")
        for piece in layout.pieces:
            min_x, min_y, max_x, max_y = piece.polygon.bounds
            left = max(0, math.floor(min_x) - padding)
            top = max(0, math.floor(min_y) - padding)
            right = min(layout.width, math.ceil(max_x) + padding)
            bottom = min(layout.height, math.ceil(max_y) + padding)
            crop = source.crop((left, top, right, bottom))

            mask = Image.new("L", crop.size, 0)
            draw = ImageDraw.Draw(mask)
            points = [
                (x - left, y - top)
                for x, y in polygon_points(piece.polygon)
            ]
            draw.polygon(points, fill=255)

            rgba = crop.copy()
            rgba.putalpha(mask)
            rgba.save(output_dir / f"{piece.piece_id}_rgba.png")
            mask.save(output_dir / f"{piece.piece_id}_mask.png")


def parse_center(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Centre must look like 0.52,0.48")
    x, y = (float(part) for part in parts)
    if not (0.0 < x < 1.0 and 0.0 < y < 1.0):
        raise argparse.ArgumentTypeError("Centre coordinates must be between 0 and 1")
    return (x, y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a print-ready PDF for recursive 16-piece cuts.",
    )
    parser.add_argument("--image", type=Path, default=ROOT / "pic.jpg")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "stage1_capture_calibration"
            / "output"
            / "cut_pdf"
            / "jigsaw_16_pieces.pdf"
        ),
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        default=(
            ROOT
            / "stage1_capture_calibration"
            / "output"
            / "cut_pdf"
            / "jigsaw_layout.json"
        ),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=(
            ROOT
            / "stage1_capture_calibration"
            / "output"
            / "reference"
            / "source_pieces"
        ),
    )
    parser.add_argument(
        "--centre",
        type=parse_center,
        default=(0.52, 0.48),
        help="First-level centre as fractions of width,height.",
    )
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument(
        "--stagger-ratio",
        type=float,
        default=0.10,
        help=(
            "Minimum offset between second-level cuts on a shared first-level "
            "boundary, as a fraction of the shorter image side."
        ),
    )
    parser.add_argument(
        "--gap-mm",
        type=float,
        default=3.0,
        help="Minimum white gap between two cut pieces.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.gap_mm <= 0:
        raise ValueError("--gap-mm must be positive")
    if args.stagger_ratio <= 0:
        raise ValueError("--stagger-ratio must be positive")

    with Image.open(args.image) as image:
        width, height = image.size

    layout = build_cut_layout(
        width=width,
        height=height,
        centre_fraction=args.centre,
        seed=args.seed,
        min_boundary_stagger_px=min(width, height) * args.stagger_ratio,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(layout, args.layout_json)
    write_source_assets(layout, args.image, args.assets_dir)
    register_fonts()

    c = canvas.Canvas(str(args.output), pagesize=landscape(A4))
    c.setTitle("16 片几何拼图裁剪页")
    c.setAuthor("jigsaw vision pipeline")

    draw_page_header(
        c,
        "16 片几何拼图 | 裁剪页",
        "先按 100% 实际大小打印；沿洋红色边界逐块剪下。碎片已做微小分离，方便直接下刀。",
        "PAGE 01",
    )
    scale = render_cut_sheet(c, layout, args.image, args.gap_mm)
    draw_cut_legend(c)
    draw_footer(c, 1)
    c.showPage()

    draw_page_header(
        c,
        "16 片几何拼图 | 编号与结构图",
        "用于核对 01-16 的编号、一级切分点和二级切分点。不要用本页裁剪。",
        "PAGE 02",
    )
    render_assembly_guide(c, layout, args.image)
    draw_footer(c, 2)
    c.showPage()

    c.save()

    size_mm = (width * scale, height * scale)
    print(f"Layout: {args.layout_json}")
    print(f"Assets: {args.assets_dir}")
    print(f"PDF:    {args.output}")
    print(f"Scale:  {scale:.6f} mm/px")
    print(f"Base:   {size_mm[0]:.1f} x {size_mm[1]:.1f} mm")
    print(f"Pieces: {len(layout.pieces)}")
    print(f"Boundary stagger: {minimum_boundary_stagger(layout):.1f}px")


if __name__ == "__main__":
    main()

