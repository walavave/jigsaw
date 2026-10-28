import unittest
from pathlib import Path
import sys

from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from jigsaw_common.cut_layout import (  # noqa: E402
    build_cut_layout,
    build_exploded_pieces,
    minimum_boundary_stagger,
)


class CutLayoutTests(unittest.TestCase):
    def test_creates_sixteen_convex_pieces_without_area_loss(self) -> None:
        width, height = 1280, 720
        layout = build_cut_layout(width, height, centre_fraction=(0.52, 0.48), seed=1)

        self.assertEqual(len(layout.pieces), 16)
        total_area = sum(piece.polygon.area for piece in layout.pieces)
        self.assertAlmostEqual(total_area, width * height, places=3)
        for piece in layout.pieces:
            self.assertTrue(piece.polygon.is_valid)
            self.assertLess(piece.polygon.convex_hull.area - piece.polygon.area, 1e-4)

    def test_exploded_pieces_have_a_cutting_gap(self) -> None:
        layout = build_cut_layout(800, 800, seed=2)
        exploded = build_exploded_pieces(layout, target_gap_px=12)
        distances = [
            exploded[first].polygon.distance(exploded[second].polygon)
            for first in range(16)
            for second in range(first + 1, 16)
        ]
        self.assertGreaterEqual(min(distances), 11.9)

    def test_second_level_cuts_are_staggered_on_shared_boundaries(self) -> None:
        layout = build_cut_layout(
            1280,
            720,
            seed=1,
            min_boundary_stagger_px=72,
        )
        self.assertGreaterEqual(minimum_boundary_stagger(layout), 64.0)

    def test_works_for_a_tall_source_image(self) -> None:
        layout = build_cut_layout(300, 1000, seed=3)
        self.assertEqual(len(layout.pieces), 16)
        self.assertAlmostEqual(
            sum(piece.polygon.area for piece in layout.pieces),
            300 * 1000,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()

