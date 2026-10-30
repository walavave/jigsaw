"""Independent pixel fixtures and real Stage 2/3 hand-off regression tests."""

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage4_assembly"))
from assemble import (  # noqa: E402
    DEFAULT_PIECES,
    DEFAULT_PLACEMENTS,
    DEFAULT_REFERENCE,
    Fragment,
    assemble,
    color_correct,
    image_to_base,
    resolve_config,
    shared_edge_length,
    transform,
    warp_fragment,
)
from jigsaw_common.io import imread, imwrite  # noqa: E402


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "中文交接"
        self.root.mkdir()
        self.output = self.root / "out"
        self.pieces_path = self.root / "pieces.json"
        self.poses_path = self.root / "poses.json"
        self.reference_path = self.root / "reference.png"
        self.config = {"color_correction": False, "adjacency_tolerance_px": 0.5}
        # A known 20x8 RGB field split into two 10x8 rectangles. PNG pixels
        # include asymmetric padding; table coordinates have a 30 degree tilt.
        y, x = np.mgrid[:8, :20]
        self.expected = np.stack(
            [30 + 3 * x + 2 * y, 80 + 5 * y + 2 * x, 120 + 2 * x - y], axis=-1
        ).astype(np.uint8)
        imwrite(self.reference_path, self.expected)
        self.manifest = {
            "schema_version": "1.0",
            "session_id": "fixture",
            "coordinate_system": "rectified_px",
            "piece_scale": 1,
            "px_per_mm": [2, 2],
            "detected_piece_count": 2,
            "expected_piece_count": 2,
            "pieces": [],
        }
        self.result = {
            "schema": "polygon-assembly-result/v2",
            "coordinateSystem": "image",
            "unit": "rectified_px",
            "status": "feasible",
            "rectangle": {"width": 20, "height": 8},
            "placements": [],
        }
        base = np.array([[0, 0], [10, 0], [10, 8], [0, 8]], float)
        theta = math.radians(30)
        table_to_base = np.array(
            [
                [math.cos(theta), math.sin(theta), -30],
                [-math.sin(theta), math.cos(theta), -10],
                [0, 0, 1],
            ],
            float,
        )
        offset = np.array([[1, 0, 7], [0, 1, 5], [0, 0, 1]], float)
        for i in range(2):
            id_ = f"S{i + 1:02d}"
            rgba = np.zeros((19, 27, 4), np.uint8)
            rgba[5:13, 7:17, :3] = self.expected[:, i * 10 : (i + 1) * 10]
            rgba[5:13, 7:17, 3] = 255
            imwrite(self.root / f"{id_}.png", rgba)
            imwrite(self.root / f"{id_}_mask.png", rgba[:, :, 3])
            self.manifest["pieces"].append(
                {
                    "piece_id": id_,
                    "rgba_path": f"{id_}.png",
                    "mask_path": f"{id_}_mask.png",
                    "base_edge_corners_px": base.tolist(),
                    "image_corners_px": (base + [7, 5]).tolist(),
                    "piece_size_px": [27, 19],
                    "rectified_to_base_edge_matrix": table_to_base.tolist(),
                    "rectified_to_piece_matrix": (offset @ table_to_base).tolist(),
                    "center_px": transform(
                        np.linalg.inv(table_to_base), np.array([[5, 4]])
                    )[0].tolist(),
                    "corners_px": transform(
                        np.linalg.inv(table_to_base), base
                    ).tolist(),
                    "quality": 0.99,
                }
            )
            self.result["placements"].append(
                {"id": id_, "rotationDeg": 0, "origin": [i * 10, 0]}
            )
        self.flush()

    def flush(self):
        write(self.pieces_path, self.manifest)
        write(self.poses_path, self.result)

    def run_assembly(self, **kwargs):
        return assemble(
            self.pieces_path, self.poses_path, self.output, config=self.config, **kwargs
        )

    def test_exact_pixels_with_nonzero_png_offset_and_manual_table_rotation(self):
        report = self.run_assembly(reference_path=self.reference_path)
        np.testing.assert_array_equal(
            imread(self.output / "assembled.png"), self.expected
        )
        self.assertEqual(report["geometry"]["coverage_fraction"], 1)
        self.assertEqual(report["metrics"]["raw"]["ssim_gray"], 1)
        self.assertTrue(report["metrics"]["raw"]["identical"])
        self.assertIsNone(report["metrics"]["raw"]["psnr_db"])
        plan = json.loads((self.output / "placement_plan.json").read_text())
        self.assertEqual([p["scan_piece_id"] for p in plan["slot"]], ["S01", "S02"])
        self.assertAlmostEqual(plan["slot"][0]["rotation_deg"], -30)
        self.assertEqual(plan["slot"][1]["align_with_placed_pieces"], ["S01"])
        self.assertEqual(plan["rectangle_mm"], [10, 4])
        self.assertTrue((self.output / "steps/02.png").is_file())
        self.assertTrue((self.output / "piece_index.png").is_file())
        self.assertFalse(report["reference_used"]["pixels_for_gap_filling"])

    def test_negative_base_x_and_fallback_corner_offsets(self):
        base = np.array([[0, 0], [12, 0], [11, 6], [-2, 4]], float)
        piece = {"image_corners_px": (base + [9, 5]).tolist()}
        h = image_to_base(piece, base)
        np.testing.assert_allclose(
            transform(h, np.array([[9, 5], [7, 9]])), [[0, 0], [-2, 4]]
        )
        piece["image_corners_px"][2][0] += 1
        with self.assertRaisesRegex(ValueError, "crop translation"):
            image_to_base(piece, base)

    def test_no_black_halo_from_fractional_alpha_warp(self):
        rgba = np.zeros((6, 6, 4), np.uint8)
        rgba[2:4, 2:4] = [30, 120, 240, 255]
        h = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], float)
        f = Fragment(
            "S", {}, rgba, h, np.array([[0, 0], [6, 0], [6, 6], [0, 6]], float)
        )
        color, alpha, _ = warp_fragment(f, (7, 7), resolve_config(None))
        np.testing.assert_allclose(
            color[alpha > 0], np.tile([30, 120, 240], ((alpha > 0).sum(), 1)), atol=1e-4
        )

    def test_holes_remain_holes_even_with_reference(self):
        self.result["placements"][1]["origin"] = [12, 0]
        self.result["rectangle"]["width"] = 22
        self.flush()
        reference = np.full((8, 22, 3), 170, np.uint8)
        imwrite(self.reference_path, reference)
        self.run_assembly(reference_path=self.reference_path, orientation="0")
        assembled = imread(self.output / "assembled.png")
        self.assertTrue(np.all(assembled[:, 10:12] == 0))
        self.assertTrue(
            np.all(imread(self.output / "coverage_mask.png")[:, 10:12] == 0)
        )

    def test_assignments_protocol_reconstructs_the_same_pixels(self):
        self.result = {"schema_version": "1.0", "assignments": [], "unassigned": []}
        layout = {"schema_version": "1.0", "source_size_px": [20, 8], "pieces": []}
        for i in range(2):
            layout["pieces"].append(
                {
                    "piece_id": f"P{i + 1:02d}",
                    "polygon_px": [
                        [i * 10, 0],
                        [(i + 1) * 10, 0],
                        [(i + 1) * 10, 8],
                        [i * 10, 8],
                    ],
                }
            )
            self.result["assignments"].append(
                {
                    "scan_piece_id": f"S{i + 1:02d}",
                    "source_piece_id": f"P{i + 1:02d}",
                    "homography_piece_to_source": [
                        [1, 0, i * 10 - 7],
                        [0, 1, -5],
                        [0, 0, 1],
                    ],
                    "confidence": 0.95,
                    "status": "accepted",
                }
            )
        layout_path = self.root / "layout.json"
        write(layout_path, layout)
        self.flush()
        report = self.run_assembly(
            layout_path=layout_path, reference_path=self.reference_path
        )
        self.assertEqual(report["input_mode"], "reference_assignments")
        np.testing.assert_array_equal(
            imread(self.output / "assembled.png"), self.expected
        )
        plan = json.loads((self.output / "placement_plan.json").read_text())
        self.assertIsNone(plan["rectangle_mm"])

    def test_assignment_rectified_homography_is_composed_with_crop_inverse(self):
        layout_path = self.root / "layout.json"
        layout = {"schema_version": "1.0", "source_size_px": [20, 8], "pieces": []}
        assignments = []
        for i, p in enumerate(self.manifest["pieces"]):
            h = np.array([[1, 0, i * 10 - 7], [0, 1, -5], [0, 0, 1]]) @ np.array(
                p["rectified_to_piece_matrix"]
            )
            assignments.append(
                {
                    "scan_piece_id": p["piece_id"],
                    "source_piece_id": f"P{i}",
                    "H_scan_to_source": h.tolist(),
                    "transform_source_frame": "rectified_px",
                }
            )
            layout["pieces"].append(
                {
                    "piece_id": f"P{i}",
                    "polygon_px": [
                        [i * 10, 0],
                        [(i + 1) * 10, 0],
                        [(i + 1) * 10, 8],
                        [i * 10, 8],
                    ],
                }
            )
        self.result = {"schema_version": "1.0", "assignments": assignments}
        self.flush()
        write(layout_path, layout)
        self.run_assembly(layout_path=layout_path)
        np.testing.assert_array_equal(
            imread(self.output / "assembled.png"), self.expected
        )

    def test_incomplete_duplicate_failed_or_wrong_unit_results_are_rejected(self):
        original = copy.deepcopy(self.result)
        for mutate in (
            lambda r: r["placements"].pop(),
            lambda r: r["placements"].append(r["placements"][0]),
            lambda r: r.update(status="unknown"),
            lambda r: r.update(unit="mm"),
            lambda r: r.update(session_id="other"),
            lambda r: r["placements"][0].update(rotationDeg=float("nan")),
        ):
            self.result = copy.deepcopy(original)
            mutate(self.result)
            self.flush()
            with (
                self.subTest(result=self.result),
                self.assertRaises((ValueError, KeyError)),
            ):
                self.run_assembly()
            self.assertFalse(self.output.exists())

    def test_no_input_overwrite(self):
        before = self.pieces_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "contain input"):
            assemble(self.pieces_path, self.poses_path, self.root, config=self.config)
        self.assertEqual(before, self.pieces_path.read_bytes())

    def test_empty_alpha_is_rejected_before_writing(self):
        imwrite(self.root / "S01_mask.png", np.zeros((19, 27), np.uint8))
        with self.assertRaisesRegex(ValueError, "empty alpha"):
            self.run_assembly()
        self.assertFalse(self.output.exists())

    def test_manual_review_is_not_hidden_and_no_steps_option(self):
        self.manifest["manual_review"] = ["S02"]
        self.manifest["pieces"][1]["needs_manual_review"] = True
        self.config["write_steps"] = False
        self.flush()
        report = self.run_assembly()
        self.assertEqual(report["status"], "review")
        self.assertTrue(any("S02" in w for w in report["warnings"]))
        plan = json.loads((self.output / "placement_plan.json").read_text())
        self.assertTrue(plan["slot"][1]["needs_manual_review"])
        self.assertIsNone(plan["slot"][1]["step_image"])
        self.assertFalse((self.output / "steps").exists())

    def test_rerun_removes_obsolete_reference_and_step_products(self):
        self.run_assembly(reference_path=self.reference_path)
        self.assertTrue((self.output / "comparison.png").exists())
        self.config["write_steps"] = False
        self.run_assembly()
        self.assertFalse((self.output / "comparison.png").exists())
        self.assertFalse((self.output / "steps").exists())

    def test_contact_at_one_corner_does_not_count_as_shared_side(self):
        a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        b = a + [10, 10]
        self.assertEqual(shared_edge_length(a, b, 0.5), 0)
        b = a + [10, 0]
        self.assertAlmostEqual(shared_edge_length(a, b, 0.5), 10)

    def test_conservative_correction_reduces_brightness_error(self):
        y, x = np.mgrid[:20, :20]
        src = np.repeat((70 + x + y)[:, :, None], 3, axis=2).astype(np.float32)
        ref = (src + 15).astype(np.uint8)
        corrected, params = color_correct(
            src, ref, np.ones((20, 20), np.float32), resolve_config(None)
        )
        self.assertTrue(params["applied"])
        self.assertLess(np.mean((corrected - ref) ** 2), np.mean((src - ref) ** 2))
        self.assertLessEqual(max(abs(v) for v in params["mean_shift_Lab"]), 5.0001)
        self.assertTrue(all(0.9 <= v <= 1.125 for v in params["gain"]))

    def test_explicit_global_turn_also_rotates_plan_geometry(self):
        report = self.run_assembly(orientation="90")
        self.assertEqual(report["canvas_size_px"], [8, 20])
        plan = json.loads((self.output / "placement_plan.json").read_text())
        # Original first rectangle center (5,4) -> (8-4,5).
        slot = next(p for p in plan["slot"] if p["scan_piece_id"] == "S01")
        np.testing.assert_allclose(slot["target_center_px"], [4, 5])
        self.assertAlmostEqual(slot["rotation_deg"], 60)

    def test_bad_config_and_excessive_canvas_fail_before_writing(self):
        for config in (
            {"feather_px": float("nan")},
            {"blend_mode": "multiband"},
            {"color_strength": 2},
            {"max_canvas_pixels": 10},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                assemble(self.pieces_path, self.poses_path, self.output, config=config)
        self.assertFalse(self.output.exists())

    def test_cli_reports_invalid_input_exit_code(self):
        self.result["status"] = "infeasible"
        self.flush()
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "stage4_assembly/assemble.py"),
                "--pieces",
                str(self.pieces_path),
                "--placements",
                str(self.poses_path),
                "--output",
                str(self.output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 2)
        self.assertIn("Stage 4 failed", p.stderr)


class RealHandoffTests(unittest.TestCase):
    @unittest.skipUnless(
        DEFAULT_PIECES.is_file() and DEFAULT_PLACEMENTS.is_file(),
        "real hand-off not available",
    )
    def test_real_16_piece_geometry_textures_and_upstream_review(self):
        with tempfile.TemporaryDirectory() as directory:
            report = assemble(
                DEFAULT_PIECES,
                DEFAULT_PLACEMENTS,
                Path(directory),
                reference_path=DEFAULT_REFERENCE,
                config={"write_steps": False},
            )
            self.assertEqual(report["piece_count"], 16)
            self.assertEqual(report["board_rotation_clockwise_deg"], 180)
            self.assertAlmostEqual(
                report["geometry"]["coverage_fraction"], 0.9944657281801332, places=7
            )
            self.assertEqual(report["status"], "review")
            self.assertGreater(report["metrics"]["corrected"]["ssim_gray"], 0.7)
            self.assertGreater(
                report["metrics"]["corrected"]["psnr_db"],
                report["metrics"]["raw"]["psnr_db"],
            )
            plan = json.loads((Path(directory) / "placement_plan.json").read_text())
            self.assertEqual(len({s["scan_piece_id"] for s in plan["slot"]}), 16)
            self.assertTrue(
                all(s["align_with_placed_pieces"] for s in plan["slot"][1:])
            )
            self.assertTrue(all(s["source_piece_id"] is None for s in plan["slot"]))


if __name__ == "__main__":
    unittest.main()
