"""Recursive two-line polygon partitioning for a 4 x 4 style puzzle.

The important detail is that the pieces are not a regular rectangular grid.
The source rectangle is first cut by two crossing lines through a point near
the image centre.  Each of the four resulting convex regions is then cut by
two more crossing lines through a point near that region's centre.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from shapely.affinity import translate
from shapely.geometry import LineString, Point as ShapelyPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import split


Point = tuple[float, float]


@dataclass(frozen=True)
class SplitSpec:
    """The two crossing lines used at one recursion level."""

    center: Point
    angle_a_deg: float
    angle_b_deg: float


@dataclass
class Piece:
    piece_id: str
    parent_region_id: str
    polygon: Polygon
    centroid: Point


@dataclass
class Region:
    region_id: str
    polygon: Polygon
    split: SplitSpec
    piece_ids: list[str] = field(default_factory=list)


@dataclass
class CutLayout:
    width: int
    height: int
    seed: int
    centre_fraction: Point
    first_split: SplitSpec
    regions: list[Region]
    pieces: list[Piece]

    def to_dict(self) -> dict:
        def point(value: Point) -> list[float]:
            return [round(float(value[0]), 4), round(float(value[1]), 4)]

        def polygon(value: Polygon) -> list[list[float]]:
            return [point((x, y)) for x, y in list(value.exterior.coords)[:-1]]

        return {
            "schema_version": "1.0",
            "source_size_px": [self.width, self.height],
            "seed": self.seed,
            "centre_fraction": point(self.centre_fraction),
            "first_split": {
                "center_px": point(self.first_split.center),
                "angle_a_deg": round(self.first_split.angle_a_deg, 4),
                "angle_b_deg": round(self.first_split.angle_b_deg, 4),
            },
            "regions": [
                {
                    "region_id": region.region_id,
                    "polygon_px": polygon(region.polygon),
                    "center_px": point(region.split.center),
                    "angles_deg": [
                        round(region.split.angle_a_deg, 4),
                        round(region.split.angle_b_deg, 4),
                    ],
                    "piece_ids": region.piece_ids,
                }
                for region in self.regions
            ],
            "pieces": [
                {
                    "piece_id": piece.piece_id,
                    "parent_region_id": piece.parent_region_id,
                    "polygon_px": polygon(piece.polygon),
                    "centroid_px": point(piece.centroid),
                }
                for piece in self.pieces
            ],
        }


@dataclass
class ExplodedPiece:
    """A rigidly translated piece used for the printable cut sheet."""

    piece_id: str
    polygon: Polygon
    shift: Point


def _flatten_polygons(geometry: BaseGeometry) -> list[Polygon]:
    """Return every non-empty Polygon contained in a Shapely geometry."""

    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        polygons: list[Polygon] = []
        for child in geometry.geoms:
            polygons.extend(_flatten_polygons(child))
        return polygons
    return []


def _line_for_polygon(polygon: Polygon, point: Point, angle_deg: float) -> LineString:
    """Build a segment long enough to cross the whole polygon."""

    min_x, min_y, max_x, max_y = polygon.bounds
    diagonal = math.hypot(max_x - min_x, max_y - min_y)
    half_length = diagonal * 2.5 + 1.0
    angle = math.radians(angle_deg)
    dx = math.cos(angle) * half_length
    dy = math.sin(angle) * half_length
    px, py = point
    return LineString([(px - dx, py - dy), (px + dx, py + dy)])


def _split_by_line(polygon: Polygon, point: Point, angle_deg: float) -> list[Polygon]:
    line = _line_for_polygon(polygon, point, angle_deg)
    result = split(polygon, line)
    parts = [
        part
        for part in _flatten_polygons(result)
        if part.area > polygon.area * 1e-10
    ]
    return parts


def split_into_four(
    polygon: Polygon,
    point: Point,
    angle_a_deg: float,
    angle_b_deg: float,
) -> list[Polygon]:
    """Split a convex polygon with two crossing lines through ``point``.

    The function returns exactly four valid convex polygons, or an empty list
    when a line grazes a vertex or the split is numerically unstable.
    """

    if not polygon.is_valid or polygon.area <= 0:
        return []

    from shapely.geometry import Point as ShapelyPoint

    if not polygon.contains(ShapelyPoint(point)):
        return []

    first = _split_by_line(polygon, point, angle_a_deg)
    if len(first) != 2:
        return []

    parts: list[Polygon] = []
    for half in first:
        subparts = _split_by_line(half, point, angle_b_deg)
        if len(subparts) != 2:
            return []
        parts.extend(subparts)

    if len(parts) != 4:
        return []

    if any(part.area <= polygon.area * 1e-5 for part in parts):
        return []

    union_area = sum(part.area for part in parts)
    if abs(union_area - polygon.area) > polygon.area * 1e-6:
        return []

    for part in parts:
        if part.convex_hull.area - part.area > polygon.area * 1e-6:
            return []

    return parts


def _compactness(polygon: Polygon) -> float:
    perimeter = polygon.length
    if perimeter <= 0:
        return 0.0
    return max(0.0, min(1.0, 4.0 * math.pi * polygon.area / (perimeter**2)))


def choose_balanced_split(
    polygon: Polygon,
    point: Point,
    preferred_angle_deg: float,
    min_area_fraction: float = 0.06,
    constraint: Callable[[SplitSpec], float] | None = None,
    constraint_min: float = 0.0,
) -> tuple[list[Polygon], SplitSpec] | None:
    """Search nearby line orientations and prefer four reasonably shaped pieces."""

    best: tuple[float, list[Polygon], SplitSpec] | None = None

    for offset in range(-32, 33, 2):
        angle_a = (preferred_angle_deg + offset) % 180.0
        for separation in (78, 82, 86, 90, 94, 98, 102):
            angle_b = (angle_a + separation) % 180.0
            parts = split_into_four(polygon, point, angle_a, angle_b)
            if not parts:
                continue

            fractions = [part.area / polygon.area for part in parts]
            minimum = min(fractions)
            if minimum < min_area_fraction:
                continue

            # Prefer balanced areas, compact pieces, and close-to-perpendicular
            # intersections.  The separation term only breaks near-ties.
            score = (
                minimum
                + 0.025 * min(_compactness(part) for part in parts)
                - 0.00035 * abs(separation - 90)
            )
            spec = SplitSpec(
                center=(float(point[0]), float(point[1])),
                angle_a_deg=angle_a,
                angle_b_deg=angle_b,
            )
            if constraint is not None:
                constraint_value = constraint(spec)
                if constraint_value < constraint_min:
                    continue
                if math.isfinite(constraint_value) and constraint_min > 0:
                    score += 0.04 * min(2.0, constraint_value / constraint_min)
            if best is None or score > best[0]:
                best = (score, parts, spec)

    return None if best is None else (best[1], best[2])


def _sort_top_left(values: Iterable[Polygon]) -> list[Polygon]:
    return sorted(
        values,
        key=lambda polygon: (
            round(polygon.centroid.y, 3),
            round(polygon.centroid.x, 3),
        ),
    )


def _flatten_points(geometry: BaseGeometry) -> list[tuple[float, float]]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [(float(geometry.x), float(geometry.y))]
    if geometry.geom_type == "MultiPoint":
        return [(float(point.x), float(point.y)) for point in geometry.geoms]
    if hasattr(geometry, "geoms"):
        points: list[tuple[float, float]] = []
        for child in geometry.geoms:
            points.extend(_flatten_points(child))
        return points
    return []


def _shared_boundary(first: Polygon, second: Polygon) -> BaseGeometry | None:
    shared = first.boundary.intersection(second.boundary)
    if shared.is_empty or shared.length <= 1e-6:
        return None
    return shared


def _line_endpoints(geometry: BaseGeometry) -> list[tuple[float, float]]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        coordinates = list(geometry.coords)
        return [
            (float(coordinates[0][0]), float(coordinates[0][1])),
            (float(coordinates[-1][0]), float(coordinates[-1][1])),
        ]
    if hasattr(geometry, "geoms"):
        points: list[tuple[float, float]] = []
        for child in geometry.geoms:
            points.extend(_line_endpoints(child))
        return points
    return []


def _intersections_with_shared_boundary(
    region_polygon: Polygon,
    split: SplitSpec,
    shared_boundary: BaseGeometry,
) -> list[tuple[float, float]]:
    endpoints = _line_endpoints(shared_boundary)
    points: list[tuple[float, float]] = []

    for angle in (split.angle_a_deg, split.angle_b_deg):
        line = _line_for_polygon(region_polygon, split.center, angle)
        intersection = line.intersection(shared_boundary)
        if intersection.is_empty:
            continue
        if intersection.geom_type in {"LineString", "MultiLineString"}:
            # A second-level cut must cross the first-level boundary, not run
            # along it.
            return []
        for x, y in _flatten_points(intersection):
            if endpoints and min(math.hypot(x - ex, y - ey) for ex, ey in endpoints) < 1.0:
                continue
            points.append((x, y))
    return points


def _minimum_stagger_distance(
    region_polygon: Polygon,
    candidate: SplitSpec,
    previous_regions: Sequence[Region],
) -> float:
    """Smallest distance between second-level cut hits on shared boundaries."""

    minimum = float("inf")
    for previous in previous_regions:
        shared = _shared_boundary(region_polygon, previous.polygon)
        if shared is None:
            continue
        candidate_points = _intersections_with_shared_boundary(
            region_polygon,
            candidate,
            shared,
        )
        previous_points = _intersections_with_shared_boundary(
            previous.polygon,
            previous.split,
            shared,
        )
        for first in candidate_points:
            for second in previous_points:
                minimum = min(
                    minimum,
                    math.hypot(first[0] - second[0], first[1] - second[1]),
                )
    return minimum


def minimum_boundary_stagger(layout: CutLayout) -> float:
    """Return the smallest second-level hit distance on shared boundaries."""

    minimum = float("inf")
    for first_index, first in enumerate(layout.regions):
        for second in layout.regions[first_index + 1:]:
            shared = _shared_boundary(first.polygon, second.polygon)
            if shared is None:
                continue
            first_points = _intersections_with_shared_boundary(
                first.polygon,
                first.split,
                shared,
            )
            second_points = _intersections_with_shared_boundary(
                second.polygon,
                second.split,
                shared,
            )
            for first_point in first_points:
                for second_point in second_points:
                    minimum = min(
                        minimum,
                        math.hypot(
                            first_point[0] - second_point[0],
                            first_point[1] - second_point[1],
                        ),
                    )
    return minimum


def build_cut_layout(
    width: int,
    height: int,
    centre_fraction: Point = (0.52, 0.48),
    seed: int = 20261028,
    min_boundary_stagger_px: float | None = None,
) -> CutLayout:
    """Build the first-level and second-level two-line cuts."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not (0.0 < centre_fraction[0] < 1.0 and 0.0 < centre_fraction[1] < 1.0):
        raise ValueError("Centre fraction must lie inside the image.")

    from shapely.geometry import box

    rng = random.Random(seed)
    canvas = box(0, 0, width, height)
    centre = (width * centre_fraction[0], height * centre_fraction[1])
    stagger_px = (
        float(min_boundary_stagger_px)
        if min_boundary_stagger_px is not None
        else min(width, height) * 0.10
    )

    first = choose_balanced_split(
        canvas,
        centre,
        preferred_angle_deg=rng.uniform(0, 180),
        min_area_fraction=0.12,
    )
    if first is None:
        raise RuntimeError("Could not create four first-level regions.")
    first_parts, first_spec = first

    regions: list[Region] = []
    pieces: list[Piece] = []

    for region_index, region_polygon in enumerate(_sort_top_left(first_parts), start=1):
        region_id = f"R{region_index:02d}"
        region_centre = region_polygon.centroid
        region_point = (float(region_centre.x), float(region_centre.y))
        preferred_angle = rng.uniform(0, 180)
        second = None
        for factor in (1.0, 0.85, 0.7, 0.55, 0.4):
            second = choose_balanced_split(
                region_polygon,
                region_point,
                preferred_angle_deg=preferred_angle,
                min_area_fraction=0.06,
                constraint=lambda candidate: _minimum_stagger_distance(
                    region_polygon,
                    candidate,
                    regions,
                ),
                constraint_min=stagger_px * factor,
            )
            if second is not None:
                break
        if second is None:
            raise RuntimeError(f"Could not split region {region_id} into four.")
        second_parts, second_spec = second
        region = Region(
            region_id=region_id,
            polygon=region_polygon,
            split=second_spec,
        )

        for child_polygon in _sort_top_left(second_parts):
            piece_id = f"P{len(pieces) + 1:02d}"
            centroid = child_polygon.centroid
            piece = Piece(
                piece_id=piece_id,
                parent_region_id=region_id,
                polygon=child_polygon,
                centroid=(float(centroid.x), float(centroid.y)),
            )
            pieces.append(piece)
            region.piece_ids.append(piece_id)
        regions.append(region)

    if len(pieces) != 16:
        raise RuntimeError(f"Expected 16 pieces, got {len(pieces)}.")

    return CutLayout(
        width=width,
        height=height,
        seed=seed,
        centre_fraction=centre_fraction,
        first_split=first_spec,
        regions=regions,
        pieces=pieces,
    )


def polygon_points(polygon: Polygon) -> list[Point]:
    """Return exterior coordinates without the repeated closing point."""

    return [(float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]]


def line_segments(polygon: Polygon) -> list[tuple[Point, Point]]:
    """Return polygon edges for drawing and de-duplication."""

    points = list(polygon.exterior.coords)
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _polygon_axes(polygon: Polygon) -> list[Point]:
    points = list(polygon.exterior.coords)[:-1]
    axes: list[Point] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > 1e-9:
            axes.append((-dy / length, dx / length))
    return axes


def _minimum_translation(
    first: Polygon,
    second: Polygon,
    target_gap: float,
) -> tuple[float, Point]:
    """Minimum translation separating two convex polygons by ``target_gap``."""

    first_points = list(first.exterior.coords)[:-1]
    second_points = list(second.exterior.coords)[:-1]
    best: tuple[float, Point] | None = None

    for nx, ny in _polygon_axes(first) + _polygon_axes(second):
        first_projection = [x * nx + y * ny for x, y in first_points]
        second_projection = [x * nx + y * ny for x, y in second_points]
        first_min, first_max = min(first_projection), max(first_projection)
        second_min, second_max = min(second_projection), max(second_projection)

        positive = first_max + target_gap - second_min
        negative = second_max - first_min + target_gap
        if positive <= negative:
            distance = positive
            normal = (nx, ny)
        else:
            distance = negative
            normal = (-nx, -ny)

        if best is None or distance < best[0]:
            best = (distance, normal)

    if best is None:
        raise RuntimeError("Could not find a separating axis.")
    return best


def _minimum_pair_distance(polygons: Sequence[Polygon]) -> float:
    minimum = float("inf")
    for first_index in range(len(polygons)):
        for second_index in range(first_index + 1, len(polygons)):
            minimum = min(
                minimum,
                polygons[first_index].distance(polygons[second_index]),
            )
    return minimum


def build_exploded_pieces(
    layout: CutLayout,
    target_gap_px: float,
    max_iterations: int = 500,
) -> list[ExplodedPiece]:
    """Push rigid pieces apart until every pair has a cut-friendly gap.

    The initial polygons tile the source image exactly, so a simple radial
    translation is not enough: two neighbouring pieces can have almost the
    same radial direction and remain nearly touching.  This relaxation uses a
    separating-axis translation for each too-close pair while keeping every
    piece rigid and preserving its orientation.
    """

    if target_gap_px <= 0:
        raise ValueError("target_gap_px must be positive.")

    polygons = [piece.polygon.buffer(0) for piece in layout.pieces]
    translations: list[Point] = [(0.0, 0.0) for _ in polygons]
    tolerance = max(1e-4, target_gap_px * 1e-4)

    for _ in range(max_iterations):
        displacement: list[Point] = [(0.0, 0.0) for _ in polygons]
        any_too_close = False

        for first_index in range(len(polygons)):
            for second_index in range(first_index + 1, len(polygons)):
                distance = polygons[first_index].distance(polygons[second_index])
                if distance >= target_gap_px - tolerance:
                    continue
                any_too_close = True
                separation, normal = _minimum_translation(
                    polygons[first_index],
                    polygons[second_index],
                    target_gap_px,
                )
                half_x = normal[0] * separation / 2.0
                half_y = normal[1] * separation / 2.0
                displacement[first_index] = (
                    displacement[first_index][0] - half_x,
                    displacement[first_index][1] - half_y,
                )
                displacement[second_index] = (
                    displacement[second_index][0] + half_x,
                    displacement[second_index][1] + half_y,
                )

        if not any_too_close:
            break

        # A damped update avoids oscillation when one piece has several
        # neighbours pushing it in different directions.
        damping = 0.55
        for index, (dx, dy) in enumerate(displacement):
            dx *= damping
            dy *= damping
            length = math.hypot(dx, dy)
            if length > target_gap_px:
                scale = target_gap_px / length
                dx *= scale
                dy *= scale
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                polygons[index] = translate(polygons[index], xoff=dx, yoff=dy)
                translations[index] = (
                    translations[index][0] + dx,
                    translations[index][1] + dy,
                )

    distance = _minimum_pair_distance(polygons)
    if distance < target_gap_px * 0.98:
        raise RuntimeError(
            "Could not separate all pieces to the requested cutting gap: "
            f"got {distance:.3f}px, expected {target_gap_px:.3f}px."
        )

    return [
        ExplodedPiece(
            piece_id=piece.piece_id,
            polygon=polygons[index],
            shift=translations[index],
        )
        for index, piece in enumerate(layout.pieces)
    ]

