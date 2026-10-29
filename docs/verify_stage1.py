#!/usr/bin/env python3
"""Lead-side verification: compare regenerated Stage 1 outputs with the committed baseline.

Compares JSON numerically and images per-pixel. Read-only: writes nothing except stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage1_capture_calibration" / "output"
# Optional CLI override: python docs/verify_stage1.py <baseline_output_dir>
# The baseline is a snapshot of the *committed* Stage 1 tree, e.g.
#   git archive --format=tar -o tmp/baseline.tar HEAD stage1_capture_calibration/output
#   tar -xf tmp/baseline.tar -C tmp/baseline_extract
#   python docs/verify_stage1.py tmp/baseline_extract/stage1_capture_calibration/output
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp" / "baseline_outputs"

results: list[tuple[str, str, str]] = []


def record(check: str, status: str, detail: str) -> None:
    results.append((check, status, detail))
    print(f"[{status}] {check}: {detail}")


def load(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def imread_unicode(path: Path):
    """cv2.imread cannot open non-ASCII Windows paths; decode bytes instead."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def max_delta(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------- layout JSON
new_layout = load(OUT / "cut_pdf" / "jigsaw_layout.json")
base_layout = load(BASE / "cut_pdf" / "jigsaw_layout.json")

n_pieces = len(new_layout["pieces"])
n_regions = len(new_layout["regions"])
record("layout piece count", "PASS" if n_pieces == 16 else "FAIL", f"{n_pieces} pieces")
record("layout region count", "PASS" if n_regions == 4 else "FAIL", f"{n_regions} regions")

new_poly = np.array([p["polygon_px"] for p in new_layout["pieces"]])
base_poly = np.array([p["polygon_px"] for p in base_layout["pieces"]])
delta = max_delta(new_poly, base_poly)
record("layout polygon reproduction", "PASS" if delta < 1e-6 else "FAIL", f"max abs delta = {delta:.3e} px")

ids_ok = [p["piece_id"] for p in new_layout["pieces"]] == [f"P{i:02d}" for i in range(1, 17)]
record("layout piece ids", "PASS" if ids_ok else "FAIL", "P01..P16 in order" if ids_ok else str([p["piece_id"] for p in new_layout["pieces"]]))

# Area conservation: piece areas must sum to the reference image area.
def polygon_area(points) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

width, height = new_layout["source_size_px"]
area_sum = sum(polygon_area(p["polygon_px"]) for p in new_layout["pieces"])
ratio = area_sum / (width * height)
record(
    "layout area conservation",
    "PASS" if abs(ratio - 1.0) < 1e-3 else "FAIL",
    f"sum(piece area)/image area = {ratio:.9f} (image {width}x{height})",
)

# Parent/child consistency: every piece has a valid parent region.
region_ids = {r["region_id"] for r in new_layout["regions"]}
parents = {p["parent_region_id"] for p in new_layout["pieces"]}
record("layout parent regions", "PASS" if parents <= region_ids else "FAIL", f"parents={sorted(parents)}")

# --------------------------------------------------------- intrinsics JSON
new_intr = load(OUT / "calibration" / "intrinsics.json")
base_intr = load(BASE / "calibration" / "intrinsics.json")
record("intrinsics accepted views",
       "PASS" if len(new_intr["accepted_images"]) == 25 else "FAIL",
       f"{len(new_intr['accepted_images'])} accepted")
rms = new_intr["rms_reprojection_px"]
record("intrinsics RMS", "PASS" if rms < 1.0 else "FAIL", f"{rms:.4f} px (target < 1.0)")
K_delta = max_delta(new_intr["camera_matrix"], base_intr["camera_matrix"])
K_rel = K_delta / abs(base_intr["camera_matrix"][0][0])
record("intrinsics camera matrix", "PASS" if K_rel < 0.01 else "WARN", f"max |dK| = {K_delta:.4f} ({K_rel*100:.3f}% of fx)")
d_new = np.asarray(new_intr["distortion_coefficients"])
d_base = np.asarray(base_intr["distortion_coefficients"])
record("intrinsics distortion", "PASS" if d_new.shape == d_base.shape else "FAIL",
       f"new={np.round(d_new, 4).tolist()} base={np.round(d_base, 4).tolist()}")
record("intrinsics image size", "PASS" if new_intr["image_size_px"] == base_intr["image_size_px"] else "WARN",
       f"{new_intr['image_size_px']} vs baseline {base_intr['image_size_px']}")
new_norm = new_intr.get("orientation_normalization", {})
base_norm = base_intr.get("orientation_normalization", {})
norm_same = {Path(k).name: v for k, v in new_norm.items()} == {Path(k).name: v for k, v in base_norm.items()}
record("intrinsics EXIF normalization", "PASS" if norm_same else "WARN",
       f"new={new_norm} baseline={base_norm} (path separators differ, basenames compared)")

# ------------------------------------------------------ rectification JSON
new_rect = load(OUT / "rectified" / "rectification.json")
base_rect = load(BASE / "rectified" / "rectification.json")
record("rectified canvas", "PASS" if new_rect["canvas_px"] == [2640, 2640] else "FAIL", f"{new_rect['canvas_px']}")
record("rectified inputs",
       "PASS" if Path(new_rect["input_pieces"]).name == "pieces1.jpg" and Path(new_rect["input_plane"]).name == "plane3.jpg" else "WARN",
       f"pieces={Path(new_rect['input_pieces']).name} empty={Path(new_rect['input_empty']).name} plane={Path(new_rect['input_plane']).name}")
h_rms = new_rect["rms_reprojection_px"]
record("homography RMS", "PASS" if h_rms < 2.0 else "FAIL", f"{h_rms:.4f} px (target < 2.0)")
H_new = np.asarray(new_rect["homography_raw_undistorted_to_rectified"], dtype=np.float64)
H_base = np.asarray(base_rect["homography_raw_undistorted_to_rectified"], dtype=np.float64)
H_delta = max_delta(H_new, H_base)
corner_src = np.asarray(new_rect["source_checkerboard_corners_px"], dtype=np.float64)
pts = np.hstack([corner_src, np.ones((corner_src.shape[0], 1))])
map_new = (H_new @ pts.T).T
map_new = map_new[:, :2] / map_new[:, 2:3]
map_base = (H_base @ pts.T).T
map_base = map_base[:, :2] / map_base[:, 2:3]
map_err = np.linalg.norm(map_new - map_base, axis=1)
record("homography reproduction (direct entries)", "PASS" if H_delta < 10.0 else "WARN", f"max |dH| = {H_delta:.4f} (raw entries scale ~5e3, poorly conditioned)")
record("homography mapping error", "PASS" if map_err.max() < 3.0 else "WARN",
       f"48 checkerboard corners -> canvas: mean {map_err.mean():.4f} px, max {map_err.max():.4f} px")
corner_delta = max_delta(new_rect["source_checkerboard_corners_px"], base_rect["source_checkerboard_corners_px"])
record("checkerboard corner detection", "PASS" if corner_delta < 3.0 else "WARN", f"max corner delta = {corner_delta:.3f} px")

# -------------------------------------------------------------- images
def compare_image(rel: str, label: str) -> None:
    p_new = OUT / rel
    p_base = BASE / rel
    if not p_new.exists() or not p_base.exists():
        record(label, "FAIL", f"missing file (new={p_new.exists()}, base={p_base.exists()})")
        return
    a = imread_unicode(p_new)
    b = imread_unicode(p_base)
    if a is None or b is None:
        record(label, "FAIL", "imdecode returned None")
        return
    if a.shape != b.shape:
        record(label, "WARN", f"shape mismatch new={a.shape} base={b.shape}")
        return
    diff = cv2.absdiff(a, b).astype(np.float64)
    mad = float(diff.mean())
    frac8 = float((diff.max(axis=-1) if diff.ndim == 3 else diff > 8).mean())
    status = "PASS" if mad < 1.0 else ("WARN" if mad < 6.0 else "FAIL")
    record(label, status, f"shape={a.shape} mean|diff|={mad:.4f} frac(>8)={frac8*100:.3f}%")


compare_image("reference/reference.png", "reference.png")
for i in range(1, 17):
    compare_image(f"reference/source_pieces/P{i:02d}_rgba.png", f"source P{i:02d} rgba")
    compare_image(f"reference/source_pieces/P{i:02d}_mask.png", f"source P{i:02d} mask")
compare_image("rectified/pieces_rectified.png", "pieces_rectified.png")
compare_image("rectified/empty_rectified.png", "empty_rectified.png")
compare_image("rectified/checkerboard_detection_debug.png", "checkerboard_detection_debug.png")

# PDFs: compare page counts and rough size.
def pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    return data.count(b"/Type /Page") - data.count(b"/Type /Pages")


for rel, label in [
    ("cut_pdf/jigsaw_16_pieces.pdf", "jigsaw_16_pieces.pdf"),
    ("calibration_targets/checkerboard_6x8_25mm.pdf", "checkerboard_6x8_25mm.pdf"),
]:
    p_new, p_base = OUT / rel, BASE / rel
    if not p_new.exists():
        record(label, "FAIL", "missing regenerated PDF")
        continue
    pages_new, pages_base = pdf_pages(p_new), pdf_pages(p_base)
    record(label, "PASS" if pages_new == pages_base else "WARN",
           f"pages new={pages_new} base={pages_base}, size {p_new.stat().st_size} vs {p_base.stat().st_size} bytes")

# -------------------------------------------------------------- summary
print("\n" + "=" * 78)
fails = [r for r in results if r[1] == "FAIL"]
warns = [r for r in results if r[1] == "WARN"]
print(f"SUMMARY: {len(results)} checks, {len(fails)} FAIL, {len(warns)} WARN")
for check, status, detail in results:
    if status != "PASS":
        print(f"  {status}: {check} - {detail}")
print("=" * 78)
sys.exit(1 if fails else 0)
