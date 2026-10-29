"""Tests for the generated backgrounds and for the shadow-rejection additions.

The module covers two units of production code:

* ``backgrounds/make_backgrounds.py`` - procedural floors (wood / marble /
  concrete), the ``*_mat`` mat-cloth variants, the alpha / un-premultiply /
  compositing helpers and the ``build_scene`` / ``main`` pipeline.
* ``stage2_piece_extraction/extract_pieces.py`` - the shadow-rejection helpers
  ``_local_correlation``, ``shadow_suspicion`` and the ``foreground_mask``
  options ``shadow_rejection`` / ``mask_erode_px`` / ``ignore_border_px``.

Everything runs on small synthetic canvases (160-400 px, 2-3 convex quads)
so the whole module stays well under a second.  All artefacts live in
``tempfile.TemporaryDirectory()``; nothing is written into the repository tree
and no real capture is required.  Images are read/written through
``shared/jigsaw_common/io.py`` because the repository path contains non-ASCII
characters (``cv2.imread``/``cv2.imwrite`` fail on such paths on Windows).

Coverage map:

1. Noise helpers: ``fbm`` determinism/shape, ``_normalise``, ``_blur``.
2. Floor generators: shape/dtype/finiteness/non-constancy/determinism for all
   three floors, ``generator_for`` suffix handling and ``scene_parameters``
   merging of the ``mat`` block.
3. Compositing helpers: ``_mat_alpha``, ``_polygon_mask`` (area monotonicity,
   blur feather, supersampled area vs shoelace), ``unmultiply_pieces`` and
   ``composite_scene`` (piece/background layering, determinism, modulation).
4. ``build_scene`` end-to-end: the four output files, ``scene.json`` contract,
   byte-identical determinism and repo-relative POSIX input paths.
5. Shadow rejection: high correlation for a multiplicative cast shadow, zero
   suspicion on textureless frames, mask rejection, erosion and border guard.
6. A ``main(argv)`` CLI smoke test on a 320x320 scene.
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "backgrounds"))
sys.path.insert(0, str(ROOT / "stage2_piece_extraction"))

import extract_pieces  # noqa: E402
import make_backgrounds  # noqa: E402
from jigsaw_common.io import imread, imwrite, read_json, write_json  # noqa: E402

REAL_PIECES_PNG = ROOT / "stage1_capture_calibration/output/rectified/pieces_rectified.png"
REAL_EMPTY_PNG = ROOT / "stage1_capture_calibration/output/rectified/empty_rectified.png"
REAL_PIECES_JSON = ROOT / "stage2_piece_extraction/output/pieces.json"
EXAMPLE_BACKGROUND_CONFIG = ROOT / "backgrounds/config/backgrounds.example.json"

# A tiny scene: 320x320 px at 4 px/mm is an 80x80 mm canvas, i.e. a wood floor
# with one or two planks.  Three convex quads keep the geometry realistic while
# staying cheap to rasterise.
SCENE_CANVAS = (320, 320)  # (height, width)
SCENE_PX_PER_MM = 4.0
SCENE_SEED = 20261029
SCENE_QUADS = (
    ((60, 60), (140, 72), (132, 150), (52, 138)),
    ((170, 90), (250, 100), (242, 180), (162, 170)),
    ((90, 200), (170, 210), (162, 290), (82, 280)),
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _shoelace_area(polygon) -> float:
    points = np.asarray(polygon, np.float64)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _scene_config(**overrides) -> dict:
    """Tiny ``make_backgrounds`` config: 320x320 wood floor, repo input paths."""

    config = dict(make_backgrounds.DEFAULT_CONFIG_VALUES)
    config.update(
        {
            "seed": SCENE_SEED,
            "canvas_px": [SCENE_CANVAS[1], SCENE_CANVAS[0]],
            "px_per_mm": SCENE_PX_PER_MM,
            "backgrounds": ["wood_floor"],
            "_pieces_png": str(REAL_PIECES_PNG),
            "_empty_png": str(REAL_EMPTY_PNG),
            "_pieces_json": str(REAL_PIECES_JSON),
        }
    )
    config.update(overrides)
    return config


def _scene_pieces() -> list:
    pieces = []
    for index, quad in enumerate(SCENE_QUADS, 1):
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        pieces.append(
            {
                "piece_id": f"S{index:02d}",
                "corners_px": [[float(x), float(y)] for x, y in quad],
                "center_px": [sum(xs) / len(xs), sum(ys) / len(ys)],
                "area_px": 5000.0,
            }
        )
    return pieces


def _synthetic_capture(shape, quads):
    """Synthetic "real capture": pieces over a dark bench, as uint8 BGR frames."""

    rng = np.random.default_rng(4242)
    empty = np.full((*shape, 3), 25.0, np.float32) + rng.normal(0.0, 2.0, (*shape, 3))
    empty = np.clip(empty, 0.0, 255.0)
    alpha = make_backgrounds._polygon_mask(shape, list(quads), supersample=4, blur_px=0.7)
    piece = np.full((*shape, 3), 190.0, np.float32) + rng.normal(0.0, 6.0, (*shape, 3))
    observed = piece * alpha[:, :, None] + empty * (1.0 - alpha[:, :, None])
    return (
        np.clip(observed, 0.0, 255.0).astype(np.uint8),
        empty.astype(np.uint8),
    )


def _composite_fixture() -> dict:
    """Background + observed capture + a known true piece texture for compositing."""

    shape = (200, 240)
    rng = np.random.default_rng(3)
    background = np.full((*shape, 3), 120.0, np.float32)
    background += rng.normal(0.0, 8.0, (*shape, 3)).astype(np.float32)

    quad = ((80, 70), (150, 80), (145, 130), (75, 122))
    params = dict(make_backgrounds.DEFAULT_CONFIG_VALUES["piece_composite"])
    alpha = make_backgrounds._polygon_mask(
        shape, [quad], supersample=4, blur_px=params["edge_blur_px"]
    )

    piece = np.empty((*shape, 3), np.float32)
    piece[:, :, 0] = 200.0 + 20.0 * np.sin(np.linspace(0.0, 6.0, shape[1]))[None, :]
    piece[:, :, 1] = 180.0 + 25.0 * np.cos(np.linspace(0.0, 5.0, shape[0]))[:, None]
    piece[:, :, 2] = 160.0

    empty = np.full((*shape, 3), 30.0, np.float32)
    empty += rng.normal(0.0, 4.0, (*shape, 3)).astype(np.float32)
    observed = piece * alpha[:, :, None] + empty * (1.0 - alpha[:, :, None])

    return {
        "shape": shape,
        "background": background,
        "observed": observed,
        "empty": empty,
        "piece": piece,
        "alpha": alpha,
        "quad": quad,
        "params": params,
    }


def _mask_config(**overrides) -> dict:
    """``extract_pieces`` config pinned to a deterministic absolute threshold."""

    config = dict(extract_pieces.DEFAULT_CONFIG_VALUES)
    config.update(
        {
            "threshold_mode": "absolute",
            "threshold_min": 12.0,
            "threshold_max": 60.0,
            "open_ksize": 1,
            "close_ksize": 1,
            "mask_erode_px": 0.0,
            "shadow_rejection": False,
            "shadow_correlation": 0.55,
        }
    )
    config.update(overrides)
    return config


def _synthetic_shadow_pair(size: int = 240, seed: int = 11):
    """Empty/floor texture plus a pieces frame with one multiplicative shadow.

    Outside the shadow the pieces frame shows an *uncorrelated* surface (a paper
    piece replaces the floor); inside the shadow it is the empty frame scaled by
    0.5, which is exactly what a cast shadow does to a textured floor.
    """

    rng = np.random.default_rng(seed)

    def textured():
        field = rng.normal(128.0, 40.0, (size, size)).astype(np.float32)
        field = cv2.GaussianBlur(field, (0, 0), 1.2)
        return np.clip(field, 0.0, 255.0).astype(np.uint8)

    empty_gray = textured()
    clean_gray = textured()

    empty = empty_gray.astype(np.float32)
    pieces = clean_gray.astype(np.float32)
    shadow = (slice(60, 160), slice(70, 190))
    pieces[shadow] = 0.5 * empty[shadow]
    pieces_gray = np.clip(np.round(pieces), 0.0, 255.0).astype(np.uint8)

    pieces_bgr = cv2.cvtColor(pieces_gray, cv2.COLOR_GRAY2BGR)
    empty_bgr = cv2.cvtColor(empty_gray, cv2.COLOR_GRAY2BGR)
    return pieces_bgr, empty_bgr, shadow


# --------------------------------------------------------------------------- #
# 1. noise helpers
# --------------------------------------------------------------------------- #
class BackgroundNoiseTests(unittest.TestCase):
    def test_fbm_has_requested_shape_and_finite_values(self) -> None:
        shape = (48, 64)
        field = make_backgrounds.fbm(shape, cell=16.0, rng=np.random.default_rng(5))
        self.assertEqual(field.shape, shape)
        self.assertEqual(field.dtype, np.float32)
        self.assertTrue(np.isfinite(field).all())
        self.assertGreater(float(field.max() - field.min()), 0.1)
        # The optional ``cell_y`` argument stretches the correlation length.
        stretched = make_backgrounds.fbm(
            shape, cell=16.0, rng=np.random.default_rng(5), cell_y=48.0
        )
        self.assertEqual(stretched.shape, shape)
        self.assertTrue(np.isfinite(stretched).all())

    def test_fbm_is_deterministic_per_seed(self) -> None:
        shape = (40, 56)
        first = make_backgrounds.fbm(shape, cell=8.0, rng=np.random.default_rng(7))
        second = make_backgrounds.fbm(shape, cell=8.0, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

        other = make_backgrounds.fbm(shape, cell=8.0, rng=np.random.default_rng(8))
        self.assertFalse(np.array_equal(first, other))

    def test_normalise_maps_to_unit_range_and_handles_constants(self) -> None:
        field = np.random.default_rng(1).random((20, 30)).astype(np.float32) * 50.0 + 7.0
        normalised = make_backgrounds._normalise(field)
        self.assertEqual(normalised.shape, field.shape)
        self.assertEqual(normalised.dtype, np.float32)
        self.assertAlmostEqual(float(normalised.min()), 0.0, places=6)
        self.assertAlmostEqual(float(normalised.max()), 1.0, places=6)

        constant = np.full((8, 9), 7.5, np.float32)
        flat = make_backgrounds._normalise(constant)
        self.assertEqual(flat.shape, constant.shape)
        self.assertTrue(np.all(flat == 0.0))

    def test_blur_identity_and_smoothing(self) -> None:
        speckle = np.random.default_rng(2).random((32, 32)).astype(np.float32)
        unchanged = make_backgrounds._blur(speckle, 0.0)
        self.assertTrue(np.array_equal(unchanged, speckle))
        self.assertEqual(unchanged.dtype, np.float32)

        smoothed = make_backgrounds._blur(speckle, 2.0)
        self.assertEqual(smoothed.shape, speckle.shape)
        self.assertEqual(smoothed.dtype, np.float32)
        self.assertTrue(np.isfinite(smoothed).all())
        self.assertLess(float(smoothed.std()), 0.5 * float(speckle.std()))


# --------------------------------------------------------------------------- #
# 2. floor generators and parameter resolution
# --------------------------------------------------------------------------- #
class FloorGeneratorTests(unittest.TestCase):
    def _assert_floor_generator(self, name: str) -> None:
        generator = make_backgrounds.GENERATORS[name]
        params = make_backgrounds.DEFAULT_CONFIG_VALUES[name]
        shape = (160, 192)  # height, width

        image = generator(shape, np.random.default_rng(11), params, SCENE_PX_PER_MM)
        self.assertEqual(image.shape, (shape[0], shape[1], 3))
        self.assertTrue(np.issubdtype(image.dtype, np.floating))
        self.assertTrue(np.isfinite(image).all())
        self.assertGreater(float(image.std()), 1.0)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 300.0)

        again = generator(shape, np.random.default_rng(11), params, SCENE_PX_PER_MM)
        np.testing.assert_array_equal(image, again)
        different = generator(shape, np.random.default_rng(12), params, SCENE_PX_PER_MM)
        self.assertFalse(np.array_equal(image, different))

    def test_wood_floor_generator(self) -> None:
        self._assert_floor_generator("wood_floor")

    def test_marble_floor_generator(self) -> None:
        self._assert_floor_generator("marble_floor")

    def test_concrete_floor_generator(self) -> None:
        self._assert_floor_generator("concrete_floor")

    def test_background_labels_cover_all_generator_variants(self) -> None:
        for name in make_backgrounds.GENERATORS:
            self.assertIn(name, make_backgrounds.BACKGROUND_LABELS)
            self.assertIn(name + make_backgrounds.MAT_SUFFIX, make_backgrounds.BACKGROUND_LABELS)

    def test_generator_for_resolves_floor_and_mat_variants(self) -> None:
        self.assertEqual(make_backgrounds.MAT_SUFFIX, "_mat")
        self.assertEqual(
            set(make_backgrounds.GENERATORS),
            {"wood_floor", "marble_floor", "concrete_floor"},
        )
        self.assertIs(make_backgrounds.generator_for("wood_floor"), make_backgrounds.make_wood_floor)
        self.assertIs(
            make_backgrounds.generator_for("wood_floor_mat"), make_backgrounds.make_wood_floor
        )
        self.assertIs(
            make_backgrounds.generator_for("marble_floor_mat"),
            make_backgrounds.make_marble_floor,
        )
        self.assertIs(
            make_backgrounds.generator_for("concrete_floor_mat"),
            make_backgrounds.make_concrete_floor,
        )

    def test_generator_for_rejects_unknown_names(self) -> None:
        for name in ("bamboo_floor", "bamboo_floor_mat", "marble"):
            with self.assertRaises(KeyError):
                make_backgrounds.generator_for(name)

    def test_scene_parameters_merges_mat_block(self) -> None:
        config = dict(make_backgrounds.DEFAULT_CONFIG_VALUES)
        config["marble_floor_mat"] = {"mat": {"margin_mm": 12.0, "base_rgb": [10.0, 10.0, 12.0]}}

        params = make_backgrounds.scene_parameters("marble_floor_mat", config)
        # Floor parameters are inherited from the base floor.
        self.assertEqual(params["base_rgb"], config["marble_floor"]["base_rgb"])
        self.assertEqual(params["vein_rgb"], config["marble_floor"]["vein_rgb"])
        # The mat block is the default block overridden by the per-scene values.
        self.assertIn("mat", params)
        self.assertEqual(params["mat"]["margin_mm"], 12.0)
        self.assertEqual(params["mat"]["base_rgb"], [10.0, 10.0, 12.0])
        self.assertEqual(
            params["mat"]["feather_mm"], config["mat_default"]["feather_mm"]
        )
        self.assertIn("texture_cell_px", params["mat"])
        # The source config must not be mutated.
        self.assertNotIn("mat", config["marble_floor"])

    def test_scene_parameters_returns_plain_floor_without_mat(self) -> None:
        config = dict(make_backgrounds.DEFAULT_CONFIG_VALUES)
        params = make_backgrounds.scene_parameters("wood_floor", config)
        self.assertEqual(params, config["wood_floor"])
        self.assertNotIn("mat", params)


# --------------------------------------------------------------------------- #
# 3. compositing helpers
# --------------------------------------------------------------------------- #
class CompositeHelperTests(unittest.TestCase):
    def test_mat_alpha_covers_piece_bounds_with_feather(self) -> None:
        shape = (400, 400)
        polygons = [
            [[150, 120], [220, 132], [210, 200], [140, 188]],
            [[230, 150], [300, 170], [290, 240], [225, 225]],
        ]
        mat = {"margin_mm": 25.0, "feather_mm": 1.5}  # 100 px margin, 6 px feather
        alpha = make_backgrounds._mat_alpha(shape, polygons, mat, SCENE_PX_PER_MM)

        self.assertEqual(alpha.shape, shape)
        self.assertEqual(alpha.dtype, np.float32)
        self.assertGreaterEqual(float(alpha.min()), 0.0)
        self.assertLessEqual(float(alpha.max()), 1.0 + 1e-6)
        # Deep inside the (expanded) bounding box the cloth fully covers the floor.
        self.assertGreater(float(alpha[180, 220]), 0.99)
        # Far outside the feathered box the alpha must be ~0.
        self.assertLess(float(alpha[5, 5]), 1e-3)
        intermediate = np.logical_and(alpha > 0.05, alpha < 0.95)
        self.assertGreater(int(intermediate.sum()), 0)

    def test_polygon_mask_area_is_monotonic(self) -> None:
        shape = (240, 240)
        small = make_backgrounds._polygon_mask(
            shape, [[[80, 80], [120, 80], [120, 120], [80, 120]]], supersample=4
        )
        large = make_backgrounds._polygon_mask(
            shape, [[[40, 40], [200, 40], [200, 200], [40, 200]]], supersample=4
        )
        self.assertEqual(small.shape, shape)
        self.assertEqual(small.dtype, np.float32)
        self.assertGreaterEqual(float(small.min()), 0.0)
        self.assertLessEqual(float(large.max()), 1.0)
        self.assertLess(float(small.sum()), float(large.sum()))

    def test_polygon_mask_supersampling_matches_analytic_area(self) -> None:
        quad = [[60, 50], [180, 70], [170, 160], [50, 140]]
        analytic = _shoelace_area(quad)
        mask = make_backgrounds._polygon_mask(
            (240, 240), [quad], supersample=4, blur_px=0.0
        )
        self.assertAlmostEqual(float(mask.sum()), analytic, delta=0.02 * analytic)

    def test_polygon_mask_blur_adds_intermediate_alpha(self) -> None:
        quad = [[60, 50], [180, 70], [170, 160], [50, 140]]
        plain = make_backgrounds._polygon_mask((240, 240), [quad], supersample=4, blur_px=0.0)
        blurred = make_backgrounds._polygon_mask((240, 240), [quad], supersample=4, blur_px=2.0)
        plain_feather = int(np.logical_and(plain > 0.01, plain < 0.99).sum())
        blurred_feather = int(np.logical_and(blurred > 0.01, blurred < 0.99).sum())
        self.assertGreater(blurred_feather, plain_feather + 100)
        self.assertGreaterEqual(float(blurred.min()), 0.0)
        self.assertLessEqual(float(blurred.max()), 1.0)

    def test_unmultiply_pieces_recovers_piece_colour(self) -> None:
        shape = (120, 160)
        quad = [[30, 20], [120, 30], [112, 90], [25, 78]]
        alpha = make_backgrounds._polygon_mask(shape, [quad], supersample=4, blur_px=0.0)

        rng = np.random.default_rng(0)
        piece = np.empty((*shape, 3), np.float32)
        piece[:, :, 0] = 180.0
        piece[:, :, 1] = 150.0
        piece[:, :, 2] = 120.0
        piece += rng.normal(0.0, 3.0, (*shape, 3)).astype(np.float32)
        empty = np.full((*shape, 3), 30.0, np.float32)
        empty += rng.normal(0.0, 2.0, (*shape, 3)).astype(np.float32)

        observed = piece * alpha[:, :, None] + empty * (1.0 - alpha[:, :, None])
        recovered = make_backgrounds.unmultiply_pieces(observed, empty, alpha)

        self.assertEqual(recovered.shape, (*shape, 3))
        self.assertEqual(recovered.dtype, np.float32)
        covered = alpha > 0.5
        self.assertGreater(int(covered.sum()), 100)
        np.testing.assert_allclose(recovered[covered], piece[covered], atol=1e-2)

    def test_unmultiply_pieces_is_finite_at_zero_alpha(self) -> None:
        shape = (48, 64)
        empty = np.full((*shape, 3), 25.0, np.float32)
        observed = empty.copy()

        # Zero coverage with observed == empty: the recovered piece is pure black,
        # i.e. the clamped denominator must not blow the result up.
        alpha = np.zeros(shape, np.float32)
        recovered = make_backgrounds.unmultiply_pieces(observed, empty, alpha)
        self.assertEqual(recovered.dtype, np.float32)
        self.assertTrue(np.isfinite(recovered).all())
        self.assertLessEqual(float(np.abs(recovered).max()), 1e-6)

        # A near-zero alpha pixel is still a division by (almost) zero: the result may
        # leak float32 cancellation (~1e-4 DN here), but it must stay finite and bounded.
        alpha[0, 0] = 1e-6
        recovered = make_backgrounds.unmultiply_pieces(observed, empty, alpha)
        self.assertEqual(recovered.dtype, np.float32)
        self.assertTrue(np.isfinite(recovered).all())
        self.assertLess(float(np.abs(recovered).max()), 1.0)
        self.assertLessEqual(float(recovered.max()), 255.0)

    def test_composite_scene_layers_pieces_over_background(self) -> None:
        fixture = _composite_fixture()
        scene, alpha = make_backgrounds.composite_scene(
            fixture["background"],
            fixture["observed"],
            fixture["empty"],
            [fixture["quad"]],
            np.random.default_rng(0),
            fixture["params"],
        )
        self.assertEqual(scene.shape, (*fixture["shape"], 3))
        self.assertEqual(scene.dtype, np.float32)
        self.assertEqual(alpha.shape, fixture["shape"])
        self.assertGreaterEqual(float(alpha.min()), 0.0)
        self.assertLessEqual(float(alpha.max()), 1.0)

        core = alpha > 0.999
        self.assertGreater(int(core.sum()), 100)
        # Where the piece fully covers the floor, the recovered piece colour is exact.
        np.testing.assert_allclose(scene[core], fixture["piece"][core], atol=0.5)
        # The piece texture is preserved (not flattened into one colour).
        self.assertGreater(float(scene[core][:, 0].std()), 3.0)
        # Far from the piece (and from its shadow) the generated floor shows through.
        np.testing.assert_allclose(scene[5, 5], fixture["background"][5, 5], atol=1e-3)

        repeat, _ = make_backgrounds.composite_scene(
            fixture["background"],
            fixture["observed"],
            fixture["empty"],
            [fixture["quad"]],
            np.random.default_rng(0),
            fixture["params"],
        )
        np.testing.assert_array_equal(scene, repeat)

    def test_composite_scene_applies_piece_modulation(self) -> None:
        fixture = _composite_fixture()
        scene, alpha = make_backgrounds.composite_scene(
            fixture["background"],
            fixture["observed"],
            fixture["empty"],
            [fixture["quad"]],
            np.random.default_rng(0),
            fixture["params"],
        )
        modulation = np.full(fixture["shape"], 0.5, np.float32)
        modulated, _ = make_backgrounds.composite_scene(
            fixture["background"],
            fixture["observed"],
            fixture["empty"],
            [fixture["quad"]],
            np.random.default_rng(0),
            fixture["params"],
            piece_modulation=modulation,
        )
        core = alpha > 0.999
        ratio = float(modulated[core].mean() / scene[core].mean())
        self.assertAlmostEqual(ratio, 0.5, delta=0.02)


# --------------------------------------------------------------------------- #
# 4. build_scene / scene.json / _portable
# --------------------------------------------------------------------------- #
class SceneBuilderTests(unittest.TestCase):
    def test_build_scene_writes_expected_files_and_metadata(self) -> None:
        observed, empty = _synthetic_capture(SCENE_CANVAS, SCENE_QUADS)
        pieces = _scene_pieces()
        config = _scene_config()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "scenes"
            scene, empty_frame, returned_pieces = make_backgrounds.build_scene(
                "wood_floor",
                config,
                pieces,
                output_dir,
                np.random.default_rng(SCENE_SEED),
                observed,
                empty,
            )
            scene_dir = output_dir / "wood_floor"
            for filename in ("empty.png", "pieces.png", "alpha.png", "scene.json"):
                self.assertTrue((scene_dir / filename).is_file(), filename)

            self.assertEqual(scene.shape, (*SCENE_CANVAS, 3))
            self.assertEqual(empty_frame.shape, (*SCENE_CANVAS, 3))
            self.assertEqual(scene.dtype, np.uint8)
            self.assertEqual(empty_frame.dtype, np.uint8)
            self.assertEqual(returned_pieces, pieces)

            empty_png = imread(scene_dir / "empty.png")
            pieces_png = imread(scene_dir / "pieces.png")
            alpha_png = imread(scene_dir / "alpha.png", flags=cv2.IMREAD_GRAYSCALE)
            self.assertEqual(empty_png.shape, (*SCENE_CANVAS, 3))
            self.assertEqual(pieces_png.shape, (*SCENE_CANVAS, 3))
            self.assertEqual(alpha_png.shape, SCENE_CANVAS)
            self.assertFalse(np.array_equal(empty_png, pieces_png))

            payload = read_json(scene_dir / "scene.json")
            self.assertEqual(payload["schema_version"], config["schema_version"])
            self.assertEqual(payload["generator_version"], make_backgrounds.GENERATOR_VERSION)
            self.assertEqual(payload["background"], "wood_floor")
            self.assertEqual(payload["label"], make_backgrounds.BACKGROUND_LABELS["wood_floor"])
            self.assertEqual(payload["seed"], SCENE_SEED)
            self.assertEqual(payload["canvas_px"], [SCENE_CANVAS[1], SCENE_CANVAS[0]])
            self.assertEqual(payload["px_per_mm"], SCENE_PX_PER_MM)
            self.assertEqual(
                set(payload["rendered_with"]),
                {"lighting", "piece_composite", "background_parameters"},
            )
            # Input paths must be repo-relative POSIX strings (not the raw Windows paths).
            self.assertEqual(
                payload["inputs"]["pieces_png"],
                REAL_PIECES_PNG.relative_to(ROOT).as_posix(),
            )
            self.assertEqual(
                payload["inputs"]["empty_png"],
                REAL_EMPTY_PNG.relative_to(ROOT).as_posix(),
            )
            self.assertEqual(
                payload["inputs"]["pieces_json"],
                REAL_PIECES_JSON.relative_to(ROOT).as_posix(),
            )
            for value in payload["inputs"].values():
                self.assertIsInstance(value, str)
                self.assertNotIn("\\", value)
                self.assertFalse(Path(value).is_absolute())

            self.assertEqual(len(payload["pieces"]), len(pieces))
            for stored, expected in zip(payload["pieces"], pieces):
                self.assertEqual(stored["piece_id"], expected["piece_id"])
                self.assertEqual(len(stored["corners_px"]), 4)
                self.assertEqual(len(stored["center_px"]), 2)
                self.assertAlmostEqual(float(stored["area_px"]), expected["area_px"])

    def test_build_scene_is_deterministic_for_fixed_seed(self) -> None:
        observed, empty = _synthetic_capture(SCENE_CANVAS, SCENE_QUADS)
        pieces = _scene_pieces()
        config = _scene_config()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_backgrounds.build_scene(
                "wood_floor",
                config,
                pieces,
                tmp / "first",
                np.random.default_rng(SCENE_SEED),
                observed,
                empty,
            )
            make_backgrounds.build_scene(
                "wood_floor",
                config,
                pieces,
                tmp / "second",
                np.random.default_rng(SCENE_SEED),
                observed,
                empty,
            )
            for filename in ("empty.png", "pieces.png", "alpha.png"):
                first = (tmp / "first" / "wood_floor" / filename).read_bytes()
                second = (tmp / "second" / "wood_floor" / filename).read_bytes()
                self.assertEqual(first, second, filename)

    def test_portable_paths_are_repo_relative_posix(self) -> None:
        inside = make_backgrounds._portable(EXAMPLE_BACKGROUND_CONFIG)
        self.assertEqual(inside, "backgrounds/config/backgrounds.example.json")
        self.assertNotIn("\\", inside)
        self.assertEqual(make_backgrounds._portable(ROOT / "tests"), "tests")

        outside = Path(tempfile.gettempdir()) / "jigsaw_portable_probe.png"
        self.assertEqual(make_backgrounds._portable(outside), str(outside.resolve()))


# --------------------------------------------------------------------------- #
# 5. shadow rejection (stage 2)
# --------------------------------------------------------------------------- #
class ShadowRejectionTests(unittest.TestCase):
    def test_local_correlation_contract(self) -> None:
        field = np.random.default_rng(3).random((40, 50)).astype(np.float32)
        field = cv2.GaussianBlur(field, (0, 0), 1.0)
        same = extract_pieces._local_correlation(field, field, 9)
        inverted = extract_pieces._local_correlation(field, -field, 9)
        self.assertEqual(same.shape, field.shape)
        self.assertTrue(np.isfinite(same).all())
        self.assertTrue(np.all(same > 0.99))
        self.assertTrue(np.all(inverted < -0.99))

    def test_shadow_suspicion_flags_cast_shadow_only(self) -> None:
        pieces_bgr, empty_bgr, shadow = _synthetic_shadow_pair()
        config = dict(extract_pieces.DEFAULT_CONFIG_VALUES)
        suspicion = extract_pieces.shadow_suspicion(pieces_bgr, empty_bgr, config)

        self.assertEqual(suspicion.shape, pieces_bgr.shape[:2])
        self.assertEqual(suspicion.dtype, np.float32)
        self.assertGreaterEqual(float(suspicion.min()), 0.0)
        self.assertLessEqual(float(suspicion.max()), 1.0)

        shadow_interior = suspicion[80:140, 90:170]
        untouched_texture = suspicion[10:50, 10:60]
        self.assertGreater(float(shadow_interior.mean()), 0.6)
        self.assertLess(float(untouched_texture.mean()), 0.3)

    def test_shadow_suspicion_ignores_textureless_empty_frame(self) -> None:
        pieces_bgr, _empty_bgr, _shadow = _synthetic_shadow_pair()
        black_empty = np.full_like(pieces_bgr, 12, dtype=np.uint8)
        config = dict(extract_pieces.DEFAULT_CONFIG_VALUES)
        suspicion = extract_pieces.shadow_suspicion(pieces_bgr, black_empty, config)
        self.assertEqual(suspicion.dtype, np.float32)
        self.assertEqual(suspicion.shape, pieces_bgr.shape[:2])
        self.assertEqual(float(suspicion.max()), 0.0)

    def test_foreground_mask_shadow_rejection_removes_cast_shadow(self) -> None:
        height = width = 240
        response = np.zeros((height, width), np.float32)
        response[60:160, 70:190] = 45.0  # cast shadow: bright enough to pass
        response[170:225, 70:190] = 45.0  # actual piece
        suspicion = np.zeros((height, width), np.float32)
        suspicion[60:160, 70:190] = 0.9

        enabled = _mask_config(shadow_rejection=True)
        disabled = _mask_config(shadow_rejection=False)
        mask_off, threshold_off = extract_pieces.foreground_mask(
            response, disabled, 0, shadow_suspicion=suspicion
        )
        mask_on, threshold_on = extract_pieces.foreground_mask(
            response, enabled, 0, shadow_suspicion=suspicion
        )
        self.assertEqual(threshold_off, 12.0)
        self.assertEqual(threshold_on, 12.0)

        self.assertEqual(mask_off.dtype, np.uint8)
        self.assertEqual(int(mask_off[100, 100]), 255)  # shadow kept when disabled
        self.assertEqual(int(mask_on[100, 100]), 0)  # shadow dropped when enabled
        self.assertEqual(int(mask_on[200, 130]), 255)  # piece kept
        self.assertEqual(int(mask_on[170:225, 70:190].min()), 255)

        # The suspicion map is ignored unless both the flag and the map are given.
        mask_without_map, _ = extract_pieces.foreground_mask(response, enabled, 0)
        self.assertEqual(int(mask_without_map[100, 100]), 255)

    def test_foreground_mask_erodes_mask_by_requested_pixels(self) -> None:
        response = np.zeros((240, 240), np.float32)
        response[80:160, 80:160] = 40.0  # 80x80 filled square
        mask, _threshold = extract_pieces.foreground_mask(
            response, _mask_config(mask_erode_px=2.0), 0
        )
        rows, cols = np.nonzero(mask)
        # A 5x5 elliptical kernel erodes an axis-aligned square by 2 px per side.
        self.assertEqual((int(cols.min()), int(cols.max())), (82, 157))
        self.assertEqual((int(rows.min()), int(rows.max())), (82, 157))
        self.assertEqual(int(mask.sum()) // 255, 76 * 76)

    def test_foreground_mask_ignores_border(self) -> None:
        response = np.full((240, 240), 255.0, np.float32)
        mask, _threshold = extract_pieces.foreground_mask(response, _mask_config(), 7)
        self.assertTrue(np.all(mask[:7, :] == 0))
        self.assertTrue(np.all(mask[-7:, :] == 0))
        self.assertTrue(np.all(mask[:, :7] == 0))
        self.assertTrue(np.all(mask[:, -7:] == 0))
        self.assertEqual(int(mask[7, 7]), 255)
        self.assertGreater(int(mask.sum()), 0)


# --------------------------------------------------------------------------- #
# 6. CLI smoke test
# --------------------------------------------------------------------------- #
class CliSmokeTests(unittest.TestCase):
    def test_main_generates_single_tiny_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            empty = np.full((*SCENE_CANVAS, 3), 25, np.uint8)
            observed, _ = _synthetic_capture(SCENE_CANVAS, SCENE_QUADS)
            imwrite(tmp / "pieces.png", observed)
            imwrite(tmp / "empty.png", empty)

            pieces = _scene_pieces()
            write_json(tmp / "pieces.json", {"pieces": pieces})
            write_json(
                tmp / "config.json",
                {
                    "seed": 99,
                    "canvas_px": [SCENE_CANVAS[1], SCENE_CANVAS[0]],
                    "px_per_mm": SCENE_PX_PER_MM,
                    "backgrounds": ["wood_floor"],
                },
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = make_backgrounds.main(
                    [
                        "--config",
                        str(tmp / "config.json"),
                        "--output",
                        str(tmp / "out"),
                        "--pieces-png",
                        str(tmp / "pieces.png"),
                        "--empty-png",
                        str(tmp / "empty.png"),
                        "--pieces-json",
                        str(tmp / "pieces.json"),
                    ]
                )
            self.assertEqual(exit_code, 0)

            output_dir = tmp / "out"
            self.assertTrue((output_dir / "preview.png").is_file())
            scene_dir = output_dir / "wood_floor"
            for filename in ("empty.png", "pieces.png", "alpha.png", "scene.json"):
                self.assertTrue((scene_dir / filename).is_file(), filename)

            payload = read_json(scene_dir / "scene.json")
            self.assertEqual(payload["background"], "wood_floor")
            self.assertEqual(payload["seed"], 99)
            self.assertEqual(payload["canvas_px"], [SCENE_CANVAS[1], SCENE_CANVAS[0]])
            self.assertEqual(len(payload["pieces"]), len(pieces))


if __name__ == "__main__":
    unittest.main()
