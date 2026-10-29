#!/usr/bin/env python3
"""Independent verifier for Stage 2 (piece extraction) outputs -- task-4.

Read-only with respect to the repository: it consumes the committed Stage 1
outputs and the Stage 2 worker's products, and writes only:
  tmp/verify_stage2_result.json   (machine-readable results)
  tmp/verify_stage2_overlay.png   (diagnostic overlay)

PSEUDO GROUND TRUTH
-------------------
Every physical piece is assumed to be a rigid similarity copy (rotation +
uniform scale + translation, NO reflection) of exactly one source polygon in
stage1_capture_calibration/output/cut_pdf/jigsaw_layout.json (`polygon_px`,
source-image coordinates, 1280x720).  All 16 source polygons are convex
quadrilaterals, so a correct extraction must also return 16 convex
quadrilaterals.

SHAPE-ONLY MATCHING
-------------------
For every (detected, source) pair with equal vertex count we try all n cyclic
vertex correspondences and solve the optimal similarity transform in closed
form (least squares on centroid-centred points; a = s*cos(t), b = s*sin(t) is
linear, so the solution is exact and reflection-free by construction).  The
pair cost is the resulting max corner residual measured in rectified-canvas
pixels (position, rotation and scale are all free -> shape only).  A 16x16
Hungarian assignment then yields the unique 1:1 mapping.

PRINTED SCALE
-------------
The physical pieces were printed from
stage1_capture_calibration/output/cut_pdf/jigsaw_16_pieces.pdf.  The printed
millimetres-per-source-pixel is recovered INDEPENDENTLY from that committed PDF
(page 1 draws the 1280x720 reference image inside a `w 0 0 h x y cm` matrix, so
mm/srcpx = w * 25.4/72 / 1280) and cross-checked against the value echoed in
docs/REPRODUCTION.md.  Then

  expected_total_canvas_px = sum(layout polygon area) * (mm/srcpx * px_per_mm)^2

and this is compared with the observed total of the 16 detected polygons.  The
global check is therefore independent of the Stage 2 output (the fallback --
derive the scale from the observed total -- is only used if both the PDF and
docs are unavailable, and it is then reported as circular/WARN).

PIECE FRAME (v1.4.0 base-edge contract)
---------------------------------------
Pieces are no longer normalised into a shared 512x512 canvas.  Every piece is
cropped in its OWN base-edge frame at 1:1 scale: the longest side becomes the
base, mapped to P0=(0,0) and P1=(baseLength,0), while the remaining corners lie
below (y > 0) and the quad is clockwise in the y-down image frame; the crop is
that frame translated by (margin, margin) and rounded up to whole pixels, with
`piece_scale = 1`.  This script re-derives the frame from `corners_px` alone
(see derive_base_edge_frame) and cross-checks it against the rounded preview
fields and against the minimal Stage 3 hand-off file pieces_base_edge.json --
it never imports Stage 2 code, so the derivation is independent.

Determinism of the extractor is the Lead's job: this script never runs it.
"""
from __future__ import annotations

import base64
import json
import math
import re
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

# --------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
S1_OUT = ROOT / "stage1_capture_calibration" / "output"
S2_OUT = ROOT / "stage2_piece_extraction" / "output"
S2_CFG = ROOT / "stage2_piece_extraction" / "config"
TMP = ROOT / "tmp"
LAYOUT_JSON = S1_OUT / "cut_pdf" / "jigsaw_layout.json"
PDF_PATH = S1_OUT / "cut_pdf" / "jigsaw_16_pieces.pdf"
RECT_JSON = S1_OUT / "rectified" / "rectification.json"
PIECES_JSON = S2_OUT / "pieces.json"
BASE_JSON = S2_OUT / "pieces_base_edge.json"     # minimal Stage 3 hand-off (v1.4.0)
BASELINE_JSON = TMP / "verify_stage2_result_v130.json"   # previous verified run
RESULT_JSON = TMP / "verify_stage2_result.json"
OVERLAY_PNG = TMP / "verify_stage2_overlay.png"
REPRO_DOC = ROOT / "docs" / "REPRODUCTION.md"

# Acceptance thresholds (README 4.4 + task-4).
CORNER_PASS_PX = 2.0        # README target < 2 px
CORNER_WARN_PX = 3.0        # task-4 allows 2-3 px as WARN
IOU_PASS = 0.98             # README mask IoU target
IOU_WARN = 0.96
AREA_TOTAL_PASS = 0.03      # global area conservation, relative
AREA_TOTAL_WARN = 0.06
SCALE_SPREAD_PASS = 0.01    # relative std of fitted similarity scale
SCALE_SPREAD_WARN = 0.02
PRINT_SCALE_PASS = 0.02     # fitted median scale vs PDF-derived expected
PRINT_SCALE_WARN = 0.05

results: list[tuple[str, str, str]] = []
details: dict = {}


def record(check: str, status: str, detail: str) -> None:
    results.append((check, status, detail))
    print(f"[{status}] {check}: {detail}")


# ------------------------------------------------------------------- helpers
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def imread_u(path: Path, flags: int = cv2.IMREAD_UNCHANGED):
    """cv2.imread cannot open non-ASCII Windows paths; decode bytes instead."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def polygon_signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_area(pts: np.ndarray) -> float:
    return abs(polygon_signed_area(pts))


def canonical_orientation(pts: np.ndarray) -> np.ndarray:
    """Return points in a fixed (positive signed area) traversal direction."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts[::-1] if polygon_signed_area(pts) < 0 else pts


def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return pts.mean(axis=0)
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6.0 * a)


def point_in_convex(pts: np.ndarray, p: np.ndarray, tol: float = 1e-6) -> bool:
    """Convex polygon, either winding; p inside if all cross products share sign."""
    n = len(pts)
    crosses = []
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        crosses.append((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
    crosses = np.asarray(crosses)
    return bool(np.all(crosses >= -tol) or np.all(crosses <= tol))


def convexity(pts: np.ndarray) -> dict:
    """Strict convexity diagnostics: cross-product signs and interior angles."""
    n = len(pts)
    if n < 3:
        return {"convex": False, "min_cross": 0.0, "max_cross": 0.0,
                "min_angle_deg": 0.0, "max_angle_deg": 0.0}
    crosses = []
    angles = []
    for i in range(n):
        a = pts[i - 1] - pts[i]
        b = pts[(i + 1) % n] - pts[i]
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))
        c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    crosses = np.asarray(crosses)
    same_sign = bool(np.all(crosses > 0) or np.all(crosses < 0))
    angles = np.asarray(angles)
    return {
        "convex": bool(same_sign and angles.max() < 180.0),
        "min_cross": float(crosses.min()),
        "max_cross": float(crosses.max()),
        "min_angle_deg": float(angles.min()),
        "max_angle_deg": float(angles.max()),
    }


def adjacent_edge_lengths(pts: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def similarity_lsq(src: np.ndarray, dst: np.ndarray) -> dict:
    """Exact least-squares similarity dst ~= s*R(theta)*src + t (no reflection).

    dst = [[a,-b],[b,a]] src + t with a = s cos(theta), b = s sin(theta).
    Solving on centroid-centred points is exact for the LS objective.
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    # unknowns a,b: Yx = a Xx - b Xy ; Yy = b Xx + a Xy
    A = np.zeros((2 * len(src), 2))
    rhs = np.empty(2 * len(src))
    A[0::2, 0] = X[:, 0]
    A[0::2, 1] = -X[:, 1]
    A[1::2, 0] = X[:, 1]
    A[1::2, 1] = X[:, 0]
    rhs[0::2] = Y[:, 0]
    rhs[1::2] = Y[:, 1]
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    t = mu_d - np.array([a * mu_s[0] - b * mu_s[1], b * mu_s[0] + a * mu_s[1]])
    pred = np.empty_like(dst)
    pred[:, 0] = a * src[:, 0] - b * src[:, 1] + t[0]
    pred[:, 1] = b * src[:, 0] + a * src[:, 1] + t[1]
    res = np.linalg.norm(pred - dst, axis=1)
    scale = math.hypot(a, b)
    return {
        "a": a, "b": b, "tx": float(t[0]), "ty": float(t[1]),
        "scale": float(scale),
        "rot_deg": float(math.degrees(math.atan2(b, a))),
        "max_err": float(res.max()),
        "rms_err": float(np.sqrt(np.mean(res ** 2))),
        "pred": pred,
        "residuals": res,
    }


def best_correspondence(src: np.ndarray, dst: np.ndarray) -> dict:
    """Try all cyclic vertex correspondences of equal-length polygons."""
    src = canonical_orientation(src)
    dst = canonical_orientation(dst)
    n = len(src)
    best = None
    for k in range(n):
        d2 = np.roll(dst, -k, axis=0)  # d2[i] = dst[(i+k) % n]
        fit = similarity_lsq(src, d2)
        fit["shift"] = k
        # residual vectors in dst order (same coordinates, just rolled)
        if best is None or fit["max_err"] < best["max_err"] - 1e-12:
            best = fit
    return best


def match_shapes(det_polys: list[np.ndarray], src_polys: list[np.ndarray]):
    """Hungarian 1:1 assignment using shape-only similarity residual as cost."""
    nd, ns = len(det_polys), len(src_polys)
    BIG = 1e9
    cost = np.full((nd, ns), BIG)
    for i, d in enumerate(det_polys):
        if d is None or len(d) < 3:
            continue
        for j, s in enumerate(src_polys):
            if len(d) != len(s):
                continue
            fit = best_correspondence(s, d)
            cost[i, j] = fit["max_err"]
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost)
    assignment = np.full(nd, -1, dtype=int)
    for r, c in zip(rows, cols):
        assignment[r] = c
    return assignment, cost


def raster_iou(poly_a: np.ndarray, poly_b: np.ndarray, shift_bits: int = 4) -> float:
    """IoU of two polygons rasterised into a shared crop (sub-pixel via shift)."""
    allp = np.vstack([poly_a, poly_b])
    x0, y0 = np.floor(allp.min(axis=0)) - 2
    x1, y1 = np.ceil(allp.max(axis=0)) + 2
    w = int(x1 - x0) + 1
    h = int(y1 - y0) + 1
    scale = 1 << shift_bits
    origin = np.array([x0, y0])

    def fill(pts):
        m = np.zeros((h, w), np.uint8)
        p = np.round((np.asarray(pts, float) - origin) * scale).astype(np.int32)
        cv2.fillPoly(m, [p], 255, shift=shift_bits)
        return m

    ma, mb = fill(poly_a), fill(poly_b)
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return inter / union if union else 0.0


def raster_polygon(poly: np.ndarray, hw, shift_bits: int = 4) -> np.ndarray:
    """Rasterise a polygon (positive coords) into an h x w single-channel mask."""
    h, w = int(hw[0]), int(hw[1])
    m = np.zeros((h, w), np.uint8)
    pts = np.round(np.asarray(poly, dtype=np.float64) * (1 << shift_bits)).astype(np.int32)
    cv2.fillPoly(m, [pts], 255, shift=shift_bits)
    return m


def mask_iou(poly: np.ndarray, mask: np.ndarray) -> float:
    m = raster_polygon(poly, mask.shape)
    inter = int(np.count_nonzero((m > 0) & (mask > 0)))
    union = int(np.count_nonzero((m > 0) | (mask > 0)))
    return inter / union if union else 0.0


# --------------------------------------------- independent photo measurement
def _line_through(pts: np.ndarray):
    """Total-least-squares line: point on line c, unit normal n with n.p = c."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0]
    n = np.array([-d[1], d[0]], dtype=np.float64)
    n /= np.linalg.norm(n)
    return n, float(n @ c)


def _quad_from_blob(blob: np.ndarray, quad0: np.ndarray,
                    dist_tol: float = 8.0, trim: float = 2.5):
    """Refit the four sides of quad0 from a blob contour, then intersect them.

    Returns sub-pixel corners (independent of the extractor's own corner fit).
    """
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cpts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    lines = []
    for k in range(len(quad0)):
        a = quad0[k]
        b = quad0[(k + 1) % len(quad0)]
        d = b - a
        L2 = float(d @ d)
        if L2 <= 1e-9:
            return None
        rel = cpts - a
        t = (rel @ d) / L2
        perp = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / math.sqrt(L2)
        sel = (perp < dist_tol) & (t > 0.08) & (t < 0.92)
        pts = cpts[sel]
        if len(pts) < 25:
            lines.append(None)
            continue
        n, c = _line_through(pts)
        for _ in range(5):
            r = np.abs(pts @ n - c)
            keep = r < trim
            if keep.sum() < 20 or keep.all():
                break
            pts = pts[keep]
            n, c = _line_through(pts)
        lines.append((n, c))
    out = []
    for k in range(len(lines)):
        l1, l2 = lines[k - 1], lines[k]
        if l1 is None or l2 is None:
            return None
        out.append(np.linalg.solve(np.stack([l1[0], l2[0]]), np.array([l1[1], l2[1]])))
    return np.asarray(out, dtype=np.float64)


def _iou_raster(quad, blob_crop, origin, hw):
    canvas = np.zeros(hw, np.uint8)
    pts = np.round((np.asarray(quad, float) - origin)).astype(np.int32)
    cv2.fillPoly(canvas, [pts], 1)
    inter = int(np.count_nonzero(canvas & blob_crop))
    union = int(np.count_nonzero(canvas | blob_crop))
    return inter / union if union else 0.0


def iou_refine_quad(blob: np.ndarray, quad0: np.ndarray):
    """Second, independent photo estimator: the quad that maximises raster IoU
    against the foreground blob (Nelder-Mead on 8 vertex coordinates).

    Unlike the TLS refit this makes no seed-side assumption; it is however
    pulled inwards when the blob is inflated by a shadow halo, so it is used
    only as a bias cross-check on the TLS refit.
    """
    q = np.asarray(quad0, float)
    x0 = int(max(0, q[:, 0].min() - 40))
    y0 = int(max(0, q[:, 1].min() - 40))
    x1 = int(min(blob.shape[1], q[:, 0].max() + 41))
    y1 = int(min(blob.shape[0], q[:, 1].max() + 41))
    crop = blob[y0:y1, x0:x1]
    if crop.size == 0 or not crop.any():
        return None
    hw = crop.shape
    f = lambda v: -_iou_raster(v.reshape(4, 2), crop, (x0, y0), hw)
    res = minimize(f, q.ravel(), method="Nelder-Mead",
                   options={"maxiter": 3000, "xatol": 1e-3, "fatol": 1e-6})
    return res.x.reshape(4, 2)


def photo_reference_quads(det_polys, ids, thr: float = 30.0):
    """Measure every piece's quad from the rectified PHOTO (independent of Stage 2).

    Lab background difference vs empty_rectified -> foreground -> largest blobs ->
    per-blob sub-pixel side-line refit seeded by the detected polygon.  This gives
    a photo-side reference that lets a layout-vs-detected corner error be
    attributed to the extractor (json != photo) or to the capture (photo != layout).
    """
    base = imread_u(S1_OUT / "rectified" / "pieces_rectified.png", cv2.IMREAD_COLOR)
    empty = imread_u(S1_OUT / "rectified" / "empty_rectified.png", cv2.IMREAD_COLOR)
    if base is None or empty is None or base.shape[:2] != empty.shape[:2]:
        return {}, {"error": "rectified images unavailable"}
    la = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(empty, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.linalg.norm(la - lb, axis=2)
    fg = (diff > thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labmap, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 20000]
    quads, quads_iou, used = {}, {}, set()
    info = {"n_components": len(comps), "thr": thr, "unmatched": []}
    for pid, poly in zip(ids, det_polys):
        if poly is None:
            info["unmatched"].append(f"{pid}: no detected polygon")
            continue
        c0 = np.asarray(poly, float).mean(axis=0)
        avail = [c for c in comps if c not in used]
        if not avail:
            info["unmatched"].append(f"{pid}: no unmatched blob")
            continue
        ci = min(avail, key=lambda c: float(np.linalg.norm(cents[c] - c0)))
        dist = float(np.linalg.norm(cents[ci] - c0))
        if dist > 120.0:
            info["unmatched"].append(f"{pid}: nearest blob {dist:.1f} px away")
            continue
        blob = (labmap == ci).astype(np.uint8)
        q = _quad_from_blob(blob, np.asarray(poly, float))
        if q is None:
            info["unmatched"].append(f"{pid}: side-line refit failed")
            continue
        used.add(ci)
        quads[pid] = q
        q2 = iou_refine_quad(blob, np.asarray(poly, float))
        if q2 is not None:
            quads_iou[pid] = q2
    info["assigned"] = len(quads)
    info["blobs_used"] = len(used)
    info["unused_blobs"] = len(comps) - len(used)
    info["quads_iou"] = quads_iou
    return quads, info


# ------------------------------------------------------------------ self-test
def synthetic_self_test(src_polys: list[np.ndarray], seed: int = 20261029) -> dict:
    """Project layout polygons with random similarity transforms and recover them.

    Clean case must give identity assignment and < 0.5 px corner error; a noisy
    case (sigma = 0.25 px) is used only to show the matcher degrades gracefully.
    """
    rng = np.random.default_rng(seed)
    canvas_w = canvas_h = 2640
    det_polys = []
    truth = []
    for k, p in enumerate(src_polys):
        theta = rng.uniform(-math.pi, math.pi)
        s = rng.uniform(0.9, 2.2)
        a, b = s * math.cos(theta), s * math.sin(theta)
        R = np.array([[a, -b], [b, a]])
        q = p @ R.T
        centre = rng.uniform(500.0, canvas_w - 500.0, size=2)
        q = q - q.mean(axis=0) + centre
        det_polys.append(q)
        truth.append(k)

    def run(polys, label):
        assignment, cost = match_shapes(polys, src_polys)
        ok = bool(np.all(assignment == np.arange(len(src_polys))))
        errs = []
        for i in range(len(src_polys)):
            j = int(assignment[i])
            if j < 0 or not np.isfinite(cost[i, j]) or cost[i, j] > 1e8:
                errs.append(float("inf"))
                continue
            fit = best_correspondence(src_polys[j], polys[i])
            errs.append(fit["max_err"])
        return {"label": label, "identity_assignment": ok,
                "max_err_px": float(np.max(errs)), "errors": errs,
                "assignment": [int(x) for x in assignment],
                "diag": [float(cost[i, i]) for i in range(len(src_polys))]}

    clean = run(det_polys, "clean similarity")
    # Also feed the same geometry with shuffled start vertex / reversed order:
    shuffled = []
    for i, q in enumerate(det_polys):
        qq = np.roll(q, (i * 3) % 4, axis=0)
        if i % 2 == 0:
            qq = qq[::-1]
        shuffled.append(qq)
    shuffled_case = run(shuffled, "shuffled start / reversed order")
    noisy = [q + rng.normal(0.0, 0.25, q.shape) for q in det_polys]
    noisy_case = run(noisy, "sigma=0.25 px noise")

    details["self_test"] = {"clean": clean, "shuffled": shuffled_case, "noisy": noisy_case}
    record("self-test shape matcher (clean)",
           "PASS" if clean["identity_assignment"] and clean["max_err_px"] < 0.5 else "FAIL",
           f"identity assignment={clean['identity_assignment']}, max corner err={clean['max_err_px']:.3e} px (target < 0.5)")
    record("self-test shape matcher (reordered vertices)",
           "PASS" if shuffled_case["identity_assignment"] and shuffled_case["max_err_px"] < 0.5 else "FAIL",
           f"identity assignment={shuffled_case['identity_assignment']}, max corner err={shuffled_case['max_err_px']:.3e} px")
    record("self-test shape matcher (0.25 px noise)",
           "PASS" if noisy_case["identity_assignment"] and noisy_case["max_err_px"] < 1.5 else "WARN",
           f"identity assignment={noisy_case['identity_assignment']}, max corner err={noisy_case['max_err_px']:.3f} px")
    return clean


# ------------------------------------------------------- printed-scale source
def printed_scale_from_pdf(pdf_path: Path, source_size: tuple[int, int]):
    """Recover mm per source pixel from the committed cut PDF.

    reportlab draws the 1280x720 reference image into the page with
    `w 0 0 h x y cm /FormXob.<hash> Do`; w points correspond to source width px.
    """
    if not pdf_path.exists():
        return None, "missing"
    data = pdf_path.read_bytes()
    decoded = []
    for m in re.finditer(rb"stream(\r\n|\n|\r)", data):
        start = m.end()
        end = data.find(b"endstream", start)
        raw = data[start:end]
        for decoder in (
            lambda r: zlib.decompress(r),
            lambda r: zlib.decompress(base64.a85decode(r.strip(), adobe=True)),
        ):
            try:
                decoded.append(decoder(raw))
                break
            except Exception:  # noqa: BLE001 - try the next filter chain
                continue
    pat = re.compile(
        rb"([0-9]+(?:\.[0-9]+)?)\s+0\s+0\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+cm\s*/FormXob\.[0-9a-fA-F]+\s+Do"
    )
    expected_ratio = source_size[0] / source_size[1]
    for d in decoded:
        for m in pat.finditer(d):
            w_pt = float(m.group(1))
            h_pt = float(m.group(2))
            if w_pt <= 0 or h_pt <= 0:
                continue
            if abs(w_pt / h_pt - expected_ratio) > 0.02:
                continue
            mm_per_px = w_pt * 25.4 / 72.0 / source_size[0]
            return mm_per_px, f"PDF page-1 image matrix w={w_pt:.4f}pt h={h_pt:.4f}pt"
    return None, "no image-placement matrix found"


def printed_scale_from_docs(doc_path: Path):
    if not doc_path.exists():
        return None
    txt = doc_path.read_text(encoding="utf-8")
    m = re.search(r"Scale\s+([0-9]+(?:\.[0-9]+)?)\s*mm/px", txt)
    return float(m.group(1)) if m else None


def derive_printed_scale(layout, source_size, px_per_mm):
    """Use PDF first, docs second, observed-total fallback last."""
    mm_pdf, why = printed_scale_from_pdf(PDF_PATH, source_size)
    mm_doc = printed_scale_from_docs(REPRO_DOC)
    if mm_pdf is not None and mm_doc is not None:
        rel = abs(mm_pdf / mm_doc - 1.0)
        record("printed scale cross-check (PDF vs docs)",
               "PASS" if rel < 0.005 else "WARN",
               f"PDF-derived {mm_pdf:.6f} mm/srcpx vs docs/REPRODUCTION.md {mm_doc:.6f} mm/srcpx (rel {rel*100:.3f}%)")
    elif mm_pdf is not None:
        record("printed scale cross-check (PDF vs docs)", "WARN",
               f"PDF-derived {mm_pdf:.6f} mm/srcpx; docs value unavailable")
    if mm_pdf is not None:
        return mm_pdf, f"recovered independently from committed cut PDF ({why})", False
    if mm_doc is not None:
        record("printed scale source", "WARN",
               f"PDF parse failed ({why}); using docs/REPRODUCTION.md value {mm_doc:.6f} mm/srcpx")
        return mm_doc, "docs/REPRODUCTION.md echoed value", False
    # Last resort: derive from the observed detected total (documented as circular).
    return None, "unavailable", True


# ---------------------------------------------------------------------- main
def main() -> int:
    print("=" * 78)
    print("Stage 2 independent verifier (shape-based pseudo ground truth)")
    print(f"repo root: {ROOT}")
    print("=" * 78)

    # Stage 1 inputs are committed and required.
    missing = [str(p.relative_to(ROOT)) for p in (LAYOUT_JSON, RECT_JSON, S1_OUT / "rectified" / "pieces_rectified.png")
               if not p.exists()]
    if missing:
        for m in missing:
            record("Stage 1 input present", "FAIL", m)
        return 1

    layout = load_json(LAYOUT_JSON)
    rect = load_json(RECT_JSON)
    canvas = [int(v) for v in rect["canvas_px"]]
    px_per_mm = [float(v) for v in rect["px_per_mm"]]
    canvas_w, canvas_h = canvas
    source_size = tuple(int(v) for v in layout["source_size_px"])
    src_polys = [np.asarray(p["polygon_px"], dtype=np.float64) for p in layout["pieces"]]
    src_ids = [p["piece_id"] for p in layout["pieces"]]
    details["stage1"] = {"canvas_px": canvas, "px_per_mm": px_per_mm,
                         "source_size_px": list(source_size), "layout_ids": src_ids}

    # 1) matcher self-test must pass before any real-data conclusion is drawn.
    synthetic_self_test(src_polys)

    # ---------------------------------------------------------- JSON contract
    if not PIECES_JSON.exists():
        record("Stage 2 pieces.json present", "WARN",
               f"{PIECES_JSON.relative_to(ROOT)} does not exist yet - waiting for the Stage 2 worker")
        print("\nSTAGE2_OUTPUT_NOT_READY: verifier self-test completed; re-run when pieces.json exists.")
        _write_result(verdict="not_ready")
        return 2
    top = load_json(PIECES_JSON)
    details["pieces_json"] = {"path": str(PIECES_JSON), "top_level_keys": sorted(top.keys())}

    # Previous verified run (v1.3.0 baseline): used only as a determinism proxy --
    # the v1.4.0 change is the hand-off frame, the detected corners must not move.
    baseline_rows: dict = {}
    for cand in (BASELINE_JSON, RESULT_JSON):
        if not cand.exists():
            continue
        try:
            prev = load_json(cand)
            rows = {r.get("piece_id"): r for r in prev.get("details", {}).get("pieces", [])}
        except Exception:  # noqa: BLE001
            rows = {}
        if rows:
            baseline_rows = rows
            details["baseline"] = {"path": str(cand), "n_pieces": len(rows),
                                   "verdict": prev.get("verdict")}
            break

    record("JSON schema_version", "PASS" if "schema_version" in top else "FAIL",
           f"schema_version={top.get('schema_version')!r}")
    record("JSON coordinate_system",
           "PASS" if top.get("coordinate_system") == "rectified_px" else "FAIL",
           f"coordinate_system={top.get('coordinate_system')!r} (expected 'rectified_px')")

    # v1.4.0 hand-off: per-piece base-edge frame at 1:1 scale, NOT a shared
    # normalised canvas, so the old top-level piece_size_px must be gone.
    algo = str(top.get("algorithm_version", ""))
    record("JSON algorithm_version is the base-edge contract",
           "PASS" if algo == "stage2-extract-1.4.0" else "WARN",
           f"algorithm_version={algo!r} (expected 'stage2-extract-1.4.0')")
    pframe = top.get("piece_frame")
    frame_declared = (isinstance(pframe, str) and "base" in pframe.lower()
                      and "longest" in pframe.lower() and "scale" in pframe.lower())
    record("JSON piece_frame declares the base-edge frame",
           "PASS" if frame_declared else "FAIL",
           f"piece_frame declares the longest-side base, y-down clockwise order and "
           f"unit scale: {pframe!r}" if frame_declared else
           f"piece_frame={pframe!r} (expected a base-edge description: longest side = base, "
           f"clockwise in y-down, 1:1 scale)")
    pscale = top.get("piece_scale")
    record("JSON piece_scale == 1.0 (no per-piece rescaling)",
           "PASS" if isinstance(pscale, (int, float)) and abs(float(pscale) - 1.0) <= 1e-9 else "FAIL",
           f"piece_scale={pscale!r}")
    margin_top = top.get("piece_margin_px")
    record("JSON piece_margin_px is a positive number",
           "PASS" if isinstance(margin_top, (int, float)) and float(margin_top) > 0 else "FAIL",
           f"piece_margin_px={margin_top!r}")
    deprecated_top = [k for k in ("piece_size_px", "piece_output_size_px") if k in top]
    record("deprecated shared-canvas keys removed from the top level",
           "PASS" if not deprecated_top else "WARN",
           "no top-level piece_size_px / piece_output_size_px (each piece carries its own crop size)"
           if not deprecated_top else
           f"still present: {deprecated_top} (the v1.4.0 contract says the shared canvas is gone)")
    margin_px = float(margin_top) if isinstance(margin_top, (int, float)) else 20.0

    # Optional/valuable top-level fields per README 2.3 / task-3.
    for key, label in [("session_id", "JSON session_id"), ("canvas_px", "JSON canvas_px"),
                       ("px_per_mm", "JSON px_per_mm")]:
        present = key in top
        if key == "canvas_px" and present:
            ok = list(top[key]) == canvas
            record(label, "PASS" if ok else "FAIL", f"{top[key]!r} vs rectification.json {canvas}")
        elif key == "px_per_mm" and present:
            ok = np.allclose(np.asarray(top[key], float), np.asarray(px_per_mm, float), atol=1e-6)
            record(label, "PASS" if ok else "FAIL", f"{top[key]!r} vs rectification.json {px_per_mm}")
        else:
            record(label, "PASS" if present else "WARN",
                   f"{top.get(key)!r}" if present else "missing (README 2.3 recommends it)")

    pieces = top.get("pieces")
    if not isinstance(pieces, list):
        record("JSON pieces list", "FAIL", f"pieces is {type(pieces).__name__}, expected list")
        _write_result(verdict="fail")
        return 1
    n_pieces = len(pieces)
    record("piece count", "PASS" if n_pieces == 16 else "FAIL", f"{n_pieces} pieces detected (expected 16)")
    ids = [str(p.get("piece_id", "")) for p in pieces]
    expected_ids = [f"S{i:02d}" for i in range(1, 17)]
    record("piece ids unique", "PASS" if len(set(ids)) == len(ids) else "FAIL",
           f"{len(set(ids))} unique ids")
    record("piece ids S01..S16", "PASS" if ids == expected_ids else "FAIL",
           f"ids={ids[:20]}")

    required = ["piece_id", "rgba_path", "mask_path", "corners_px", "center_px", "area_px", "quality"]
    missing_fields = {k: [ids[i] for i, p in enumerate(pieces) if k not in p] for k in required}
    missing_fields = {k: v for k, v in missing_fields.items() if v}
    record("required piece fields", "PASS" if not missing_fields else "FAIL",
           "all present" if not missing_fields else f"missing: {missing_fields}")

    # Scan-id order: sorted by center y then x (README 2.1).  center_px is [x, y].
    try:
        centres = np.array([np.asarray(p["center_px"], float).ravel()[:2] for p in pieces], dtype=float)
        ys, xs = centres[:, 1], centres[:, 0]          # y down, x right
        order_ok = True
        for i in range(len(pieces) - 1):
            if not (ys[i + 1] > ys[i] + 1e-9
                    or (abs(ys[i + 1] - ys[i]) <= 1e-9 and xs[i + 1] >= xs[i] - 1e-9)):
                order_ok = False
                break
        y_gaps = np.diff(np.sort(ys))
        record("scan ids sorted by center y then x", "PASS" if order_ok else "FAIL",
               f"S01..S16 ordered by (center_y, center_x); center_y {ys.min():.2f}..{ys.max():.2f}, "
               f"min consecutive y gap {y_gaps.min():.2f} px")
    except Exception as exc:  # noqa: BLE001
        record("scan ids sorted by center y then x", "FAIL", f"cannot evaluate centers: {exc}")

    # ------------------------------------------------------- file path checks
    def resolve(value):
        if value is None:
            return None
        p = Path(str(value))
        if p.is_absolute():
            return p if p.exists() else None
        for base in (PIECES_JSON.parent, ROOT, S2_OUT):
            q = base / p
            if q.exists():
                return q
        return None

    missing_files = []
    filenames_ok = True
    for i, piece in enumerate(pieces):
        pid = ids[i] if i < len(ids) else f"#{i}"
        for key in ("rgba_path", "mask_path"):
            rp = resolve(piece.get(key))
            if rp is None:
                missing_files.append(f"{pid}:{key}")
            elif not Path(str(piece.get(key))).name.startswith(pid):
                filenames_ok = False
        dbg = S2_OUT / "pieces" / f"{pid}_debug.png"
        if not dbg.exists():
            missing_files.append(f"{pid}:debug")
    record("piece output files exist", "PASS" if not missing_files else "FAIL",
           "rgba/mask/debug present for all pieces" if not missing_files else f"missing: {missing_files[:12]}")
    record("file names carry piece id", "PASS" if filenames_ok else "WARN",
           "rgba/mask basenames start with piece_id" if filenames_ok else "some basenames do not start with piece_id")

    # ------------------------------------------------- per-piece geometry checks
    geom_rows = []
    bad_geom = {k: [] for k in (
        "vertex_count", "degenerate_edge", "convex", "bounds", "area_px",
        "centroid_inside", "quality")}
    winds = []
    detected_polys: list[np.ndarray] = []
    detected_ok: list[bool] = []
    arrs = []
    for i, piece in enumerate(pieces):
        pid = ids[i] if i < len(ids) else f"#{i}"
        corners = piece.get("corners_px")
        arr = None
        try:
            arr = np.asarray(corners, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
                arr = None
        except Exception:  # noqa: BLE001
            arr = None
        arrs.append(arr)
        detected_polys.append(canonical_orientation(arr) if arr is not None else None)
        detected_ok.append(arr is not None)
        row = {"piece_id": pid, "vertex_count": None, "area_px": None, "iou": None,
               "source_id": None, "corner_max_err_px": None, "corner_rms_err_px": None,
               "scale": None, "rot_deg": None, "quality": piece.get("quality")}
        geom_rows.append(row)
        if arr is None:
            bad_geom["vertex_count"].append(pid)
            continue
        row["vertex_count"] = len(arr)
        row["center_px"] = [float(v) for v in np.asarray(piece.get("center_px", [np.nan, np.nan]), float).ravel()[:2]]
        if not np.all(np.isfinite(arr)):
            bad_geom["vertex_count"].append(pid)
            continue
        if len(arr) < 3 or len(arr) > 6:
            bad_geom["vertex_count"].append(pid)
        edges = adjacent_edge_lengths(arr)
        if float(edges.min()) < 20.0:
            bad_geom["degenerate_edge"].append(f"{pid}({edges.min():.2f}px)")
        conv = convexity(arr)
        row["convex"] = conv
        if not conv["convex"]:
            bad_geom["convex"].append(pid)
        area = polygon_area(arr)
        row["area_px"] = area
        if (arr[:, 0].min() < -1e-6 or arr[:, 1].min() < -1e-6
                or arr[:, 0].max() > canvas_w + 1e-6 or arr[:, 1].max() > canvas_h + 1e-6):
            bad_geom["bounds"].append(pid)
        claimed = piece.get("area_px")
        if claimed is None or area <= 0 or abs(float(claimed) / area - 1.0) > 0.005:
            bad_geom["area_px"].append(f"{pid}({claimed} vs {area:.1f})")
        cy, cx = np.asarray(piece.get("center_px", [np.nan, np.nan]), float).ravel()[:2]
        if not np.isfinite([cy, cx]).all() or not point_in_convex(arr, np.array([cy, cx])):
            bad_geom["centroid_inside"].append(pid)
        q = piece.get("quality")
        if q is None or not (np.isfinite(float(q)) and 0.0 <= float(q) <= 1.0):
            bad_geom["quality"].append(f"{pid}({q})")
        winds.append(polygon_signed_area(arr))

    record("vertex count in 3..6 (all source polygons are quads)",
           "PASS" if not bad_geom["vertex_count"] else "FAIL",
           "all 3..6" if not bad_geom["vertex_count"] else f"bad: {bad_geom['vertex_count']}")
    record("corner ordering: no duplicate/adjacent-identical vertices",
           "PASS" if not bad_geom["degenerate_edge"] else "WARN",
           f"min adjacent edge >= 20 px" if not bad_geom["degenerate_edge"] else f"tiny edges: {bad_geom['degenerate_edge']}")
    record("corner order: strictly convex polygons",
           "PASS" if not bad_geom["convex"] else "FAIL",
           "all convex with interior angles < 180 deg" if not bad_geom["convex"] else f"non-convex: {bad_geom['convex']}")
    record("polygons inside rectified canvas bounds",
           "PASS" if not bad_geom["bounds"] else "FAIL",
           f"all inside {canvas_w}x{canvas_h}" if not bad_geom["bounds"] else f"out of bounds: {bad_geom['bounds']}")
    if detected_polys:
        margins = []
        for i, poly in enumerate(detected_polys):
            if poly is None:
                continue
            m = min(float(np.min(poly[:, 0])), float(np.min(poly[:, 1])),
                    canvas_w - float(np.max(poly[:, 0])), canvas_h - float(np.max(poly[:, 1])))
            margins.append((ids[i] if i < len(ids) else f"#{i}", m))
        if margins:
            tight = [f"{pid}:{m:.1f}px" for pid, m in margins if m < 20.0]
            min_m = min(m for _, m in margins)
            record("no piece touches the canvas border (>= 20 px margin)",
                   "PASS" if not tight else "WARN",
                   f"min margin {min_m:.1f} px (pieces well inside the 2640x2640 canvas)" if not tight
                   else f"narrow margins: {tight}")
    record("area_px matches corner polygon area (0.5%)",
           "PASS" if not bad_geom["area_px"] else "FAIL",
           "all consistent" if not bad_geom["area_px"] else f"mismatch: {bad_geom['area_px'][:8]}")
    record("center_px inside polygon",
           "PASS" if not bad_geom["centroid_inside"] else "FAIL",
           "all centers inside their polygon" if not bad_geom["centroid_inside"] else f"outside: {bad_geom['centroid_inside']}")
    record("quality in [0,1]",
           "PASS" if not bad_geom["quality"] else "FAIL",
           "all valid" if not bad_geom["quality"] else f"invalid: {bad_geom['quality']}")
    if winds:
        sign_consistent = bool(np.all(np.asarray(winds) > 0) or np.all(np.asarray(winds) < 0))
        record("corner winding consistent across pieces", "PASS" if sign_consistent else "WARN",
               f"signed areas {'all same sign' if sign_consistent else 'mixed sign'} (n={len(winds)})")

    # Fixed start-corner rule (README 4.2 step 3: corner order fixed and documented).
    # Documented rule in pieces.json: "first corner is top-most then left-most".
    rule_violations = []
    for i, arr in enumerate(arrs):
        if arr is None:
            continue
        pid = ids[i] if i < len(ids) else f"#{i}"
        first = int(np.lexsort((arr[:, 0], arr[:, 1]))[0])     # min y, then min x
        if first != 0:
            rule_violations.append(f"{pid}(first corner is not top-left-most; rule starts at index {first})")
    documented = str(top.get("corner_order", "")).lower()
    rule_name = ("first corner = top-most then left-most"
                 if "top" in documented and "left" in documented else "start-corner index 0")
    record("start-corner rule consistent (documented: first = top-most then left-most)",
           "PASS" if not rule_violations else "FAIL",
           f"{rule_name}: holds for all {len([a for a in arrs if a is not None])} pieces"
           if not rule_violations else f"violations: {rule_violations[:8]}")

    # ------------------------------------------- independent base-edge frames
    # v1.4.0 hand-off: no shared normalised canvas any more.  Each piece is
    # cropped in its own base-edge frame at 1:1 scale (longest side -> P0/P1 on
    # the x axis, remaining corners below, clockwise in the y-down image frame).
    # Derive that frame here from corners_px ALONE -- nothing below relies on the
    # extractor's own frame bookkeeping, so the checks are independent.
    frames: list[dict | None] = []
    frame_fail = []
    for i, arr in enumerate(arrs):
        pid = ids[i] if i < len(ids) else f"#{i}"
        if arr is None:
            frames.append(None)
            continue
        try:
            frames.append(derive_base_edge_frame(arr, margin_px))
        except Exception as exc:  # noqa: BLE001
            frame_fail.append(f"{pid}: {exc}")
            frames.append(None)
    frames_ok = [f is not None for f in frames]
    n_frames = int(sum(frames_ok))
    record("base-edge frame derivable from corners_px alone",
           "PASS" if n_frames == len(pieces) and not frame_fail else "FAIL",
           f"{n_frames}/{len(pieces)} pieces: longest side -> P0=(0,0), P1=(baseLength,0), "
           f"corners below (y>0), crop = frame + margin, 1:1 scale"
           if n_frames == len(pieces) and not frame_fail else f"failed: {frame_fail[:8]}")

    # Determinism proxy for the Lead: v1.2.1/v1.3.0/v1.4.0 must keep the same
    # piece geometry (the hand-off frame changed, the detected corners did not).
    if baseline_rows:
        base_ids = set(baseline_rows)
        center_d, area_rel, missing_base = [], [], []
        for i, row in enumerate(geom_rows):
            pid = row["piece_id"]
            prev = baseline_rows.get(pid)
            if prev is None:
                missing_base.append(pid)
                continue
            if prev.get("center_px") is not None and row.get("center_px") is not None:
                center_d.append(float(np.abs(np.asarray(prev["center_px"], float)
                                             - np.asarray(row["center_px"], float)).max()))
            if prev.get("area_px") and row.get("area_px"):
                area_rel.append(abs(float(row["area_px"]) / float(prev["area_px"]) - 1.0))
        if not missing_base and center_d and area_rel:
            worst_center = max(center_d)
            worst_area = max(area_rel)
            status = "PASS" if worst_center <= 1e-3 and worst_area <= 1e-6 else (
                "WARN" if worst_center <= 0.01 and worst_area <= 1e-4 else "FAIL")
            record("detected geometry unchanged vs previous verified run", status,
                   f"center_px max delta {worst_center:.2e} px, |area ratio - 1| max {worst_area:.2e} "
                   f"vs {Path(details.get('baseline', {}).get('path', 'previous result')).name} "
                   f"(the v1.4.0 change is the hand-off frame only, not the geometry)")
        else:
            record("detected geometry unchanged vs previous verified run", "WARN",
                   f"baseline incomplete (missing ids {missing_base[:8]})")
    else:
        record("detected geometry unchanged vs previous verified run", "WARN",
               "no previous verify_stage2_result_v130.json / result archive - determinism proxy not evaluable")

    # quality vs config threshold / manual_review marking
    min_quality = None
    cfg_paths = [S2_CFG / "extraction.json"] + sorted(S2_CFG.glob("*.json"))
    for cp in cfg_paths:
        if cp.exists():
            try:
                cfg = load_json(cp)
                if isinstance(cfg, dict) and "min_quality" in cfg:
                    min_quality = float(cfg["min_quality"])
                    break
            except Exception:  # noqa: BLE001
                pass
    qualities = [float(p.get("quality", np.nan)) for p in pieces if p.get("quality") is not None]
    if qualities:
        worst_q = float(np.nanmin(qualities))
        review = set()
        for key in ("manual_review", "manual_review_piece_ids"):
            val = top.get(key)
            if isinstance(val, list):
                review.update(str(v) for v in val)
        if min_quality is not None:
            below = [ids[i] for i, q in enumerate(qualities) if q < min_quality]
            unmarked = [pid for pid in below if pid not in review]
            record("quality scores vs config min_quality",
                   "PASS" if not below else ("WARN" if not unmarked else "WARN"),
                   f"min_quality={min_quality}, all >= threshold" if not below
                   else f"below threshold: {below}; manual_review={sorted(review)}")
        else:
            record("quality scores present", "PASS", f"min={worst_q:.4f}, mean={float(np.nanmean(qualities)):.4f}")

    # --------------------------------------------------------- image/mask checks
    mask_bad = {k: [] for k in ("size", "binary", "empty", "alpha", "components",
                                "local_area", "local_iou", "local_bounds", "bleed")}
    local_polys: list[np.ndarray | None] = []
    local_info: list[str] = []
    mask_areas = []
    local_ious = []
    bleed_stats = []
    for i, piece in enumerate(pieces):
        pid = ids[i] if i < len(ids) else f"#{i}"
        rgba_p = resolve(piece.get("rgba_path"))
        mask_p = resolve(piece.get("mask_path"))
        rgba = imread_u(rgba_p) if rgba_p is not None else None
        mask = imread_u(mask_p, cv2.IMREAD_GRAYSCALE) if mask_p is not None else None
        if rgba is None or mask is None:
            mask_bad["size"].append(f"{pid}:unreadable")
            local_polys.append(None)
            local_info.append("unreadable")
            continue
        if rgba.ndim == 2:
            rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGRA)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            mask_bad["size"].append(f"{pid}:rgba channels {None if rgba.ndim != 3 else rgba.shape[2]}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if rgba.shape[:2] != mask.shape[:2]:
            mask_bad["size"].append(f"{pid}:{rgba.shape[:2]} vs {mask.shape[:2]}")
        # v1.4.0: the crop is the piece's own base-edge frame, so its size must be
        # the derived ceil(bbox + 2*margin) and must match the recorded crop size.
        if frames_ok[i]:
            exp_hw = (frames[i]["size"][1], frames[i]["size"][0])
            if rgba.shape[:2] != exp_hw:
                mask_bad["size"].append(f"{pid}:{rgba.shape[:2]} vs derived base-edge crop {exp_hw}")
        rep_size = piece.get("piece_size_px")
        if isinstance(rep_size, (list, tuple)) and len(rep_size) == 2:
            if (int(rep_size[1]), int(rep_size[0])) != rgba.shape[:2]:
                mask_bad["size"].append(f"{pid}:{rgba.shape[:2]} vs piece_size_px {list(rep_size)}")
        vals = np.unique(mask)
        if not np.all(np.isin(vals, (0, 255))):
            if np.all(np.isin(vals, (0, 1))):
                mask_bad["binary"].append(f"{pid}:0/1 not 0/255")
            else:
                mask_bad["binary"].append(f"{pid}:{vals[:6].tolist()}...")
        mask_area = int(np.count_nonzero(mask))
        mask_areas.append(mask_area)
        if mask_area == 0:
            mask_bad["empty"].append(pid)
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        if n_cc > 1:
            cc_areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]
            if cc_areas.size > 1 and cc_areas[1] > 0.01 * mask_area:
                mask_bad["components"].append(f"{pid}:{n_cc - 1} comps, 2nd={int(cc_areas[1])}px")
        # rgba alpha vs mask
        alpha = rgba[:, :, 3]
        expected = np.where(mask > 0, 255, 0).astype(np.int16)
        adiff = np.abs(alpha.astype(np.int16) - expected)
        frac_bad = float(np.count_nonzero(adiff > 1)) / float(adiff.size)
        if adiff.max() > 1 and frac_bad > 0.001:
            mask_bad["alpha"].append(f"{pid}: max|a-m|={int(adiff.max())}, {frac_bad*100:.2f}% pixels")
        # local polygon in the piece's own base-edge crop frame (independently
        # derived above from corners_px + margin, never taken from the JSON)
        lab = frames[i]["image"] if frames_ok[i] else None
        info = "derived:base_edge" if lab is not None else "unrecoverable"
        local_polys.append(lab)
        local_info.append(info)
        if lab is not None:
            if (lab[:, 0].min() < -2 or lab[:, 1].min() < -2
                    or lab[:, 0].max() > rgba.shape[1] + 2 or lab[:, 1].max() > rgba.shape[0] + 2):
                mask_bad["local_bounds"].append(pid)
            # Compare the mask with the RASTERISED local polygon (not its shoelace
            # area): cv2.fillPoly includes the boundary pixel band, which makes the
            # pixel count ~0.5% larger than the analytic area even for a perfect
            # rasterisation, so shoelace-based area ratios are not a defect signal.
            rp = raster_polygon(lab, mask.shape)
            n_raster = int(np.count_nonzero(rp))
            if n_raster > 0 and abs(mask_area / n_raster - 1.0) > 0.005:
                mask_bad["local_area"].append(f"{pid}: mask/raster={mask_area / n_raster:.4f} "
                                              f"(shoelace {polygon_area(lab):.1f}, raster {n_raster})")
            iou = mask_iou(lab, mask)
            local_ious.append((pid, iou))
            if iou < IOU_WARN:
                mask_bad["local_iou"].append(f"{pid}:{iou:.4f}")
        # bench bleed: everything outside the mask must be fully transparent/black
        outside = mask == 0
        if outside.any():
            rgb = rgba[:, :, :3]
            rgb_out_max = int(rgb[outside].max()) if rgb.ndim == 3 else int(rgb[outside].max())
            alpha_out_max = int(rgba[:, :, 3][outside].max())
            inside = ~outside
            rgb_in_mean = float(rgb[inside].mean()) if inside.any() else 0.0
            if rgb_out_max > 2 or alpha_out_max > 1:
                mask_bad["bleed"].append(f"{pid}: rgb_max={rgb_out_max}, alpha_max={alpha_out_max}")
            bleed_stats.append((pid, rgb_out_max, alpha_out_max, rgb_in_mean))
    n_local = sum(1 for p in local_polys if p is not None)
    local_src = {}
    for inf in local_info:
        tag = inf.split(" (")[0].split(":")[0]
        local_src[tag] = local_src.get(tag, 0) + 1

    record("rgba/mask image sizes consistent with piece_size_px",
           "PASS" if not mask_bad["size"] else "FAIL",
           "all consistent" if not mask_bad["size"] else f"{mask_bad['size'][:8]}")
    record("mask binary (0/255)",
           "PASS" if not mask_bad["binary"] else "FAIL",
           "all binary 0/255" if not mask_bad["binary"] else f"{mask_bad['binary'][:8]}")
    record("mask non-empty",
           "PASS" if not mask_bad["empty"] else "FAIL",
           "all non-empty" if not mask_bad["empty"] else f"empty: {mask_bad['empty']}")
    record("mask single connected component",
           "PASS" if not mask_bad["components"] else "WARN",
           "one component per piece" if not mask_bad["components"] else f"{mask_bad['components'][:8]}")
    record("rgba alpha channel consistent with mask",
           "PASS" if not mask_bad["alpha"] else "FAIL",
           "alpha == mask (max diff <= 1)" if not mask_bad["alpha"] else f"{mask_bad['alpha'][:8]}")
    record("per-piece local polygon recoverable",
           "PASS" if n_local == len(pieces) and len(pieces) else ("WARN" if n_local else "WARN"),
           f"recovered {n_local}/{len(pieces)} local polygons; sources={local_src}")
    record("mask area vs rasterised local polygon (0.5%)",
           ("PASS" if not mask_bad["local_area"] else "WARN") if n_local else "WARN",
           ("mask pixel count matches its own polygon raster within 0.5%" if not mask_bad["local_area"]
            else f"mask differs from its own polygon raster: {mask_bad['local_area'][:8]}") if n_local
           else "no local polygons recovered - check not evaluable")
    record("mask shape vs local polygon shape (IoU)",
           (("PASS" if not mask_bad["local_iou"] else
             ("WARN" if all(float(x.split(':')[1]) >= 0.90 for x in mask_bad["local_iou"]) else "FAIL"))
            if n_local else "WARN"),
           ("IoU >= 0.96 for all" if not mask_bad["local_iou"]
            else f"low IoU: {mask_bad['local_iou'][:8]}") if n_local
           else "no local polygons recovered - check not evaluable")
    record("local polygon inside piece canvas",
           "PASS" if not mask_bad["local_bounds"] else "FAIL",
           "all local polygons inside the piece canvas" if not mask_bad["local_bounds"] else f"{mask_bad['local_bounds'][:8]}")
    # Transparency hygiene: no bench/background pixels leak into the RGBA crops.
    if bleed_stats:
        worst_rgb = max(s[1] for s in bleed_stats)
        worst_alpha = max(s[2] for s in bleed_stats)
        record("no bench bleed outside mask (rgba transparent/black outside)",
               "PASS" if not mask_bad["bleed"] else "FAIL",
               f"outside-mask rgb max {worst_rgb}, alpha max {worst_alpha} for all 16 pieces"
               if not mask_bad["bleed"] else f"bleed: {mask_bad['bleed'][:8]}")
    # NOTE (circularity): the saved mask is a pixel-exact rasterisation of the
    # reported polygon (IoU ~1.00000), so mask-vs-polygon agreement cannot be
    # evidence of segmentation quality; the extractor's own segmentation metric
    # lives in quality_breakdown.mask_iou.  Flagged when all pieces look like this.
    if local_ious and min(v for _, v in local_ious) > 0.995:
        qb = []
        for piece in pieces:
            qv = piece.get("quality_breakdown")
            if isinstance(qv, dict) and "mask_iou" in qv:
                qb.append(float(qv["mask_iou"]))
        record("mask-vs-polygon agreement is independent (not circular)",
               "WARN",
               f"saved mask equals the rasterised reported polygon (IoU min {min(v for _, v in local_ious):.5f}); "
               f"true segmentation agreement is only quality_breakdown.mask_iou "
               f"{min(qb):.5f}..{max(qb):.5f}" if qb else
               f"saved mask equals the rasterised reported polygon (IoU min {min(v for _, v in local_ious):.5f})")
        # README 4.4 accepts on "mask IoU vs manual annotation > 0.98".  No manual
        # annotation is committed for these captures, so only the extractor's own
        # self-reported segmentation IoU exists - report it, do not certify it.
        if qb:
            record("README 4.4 mask IoU vs manual annotation (> 0.98)", "WARN",
                   f"not evaluable: no manual piece annotation in the repo; "
                   f"extractor self-reported quality_breakdown.mask_iou "
                   f"min {min(qb):.5f} median {float(np.median(qb)):.5f} "
                   f"(needs an independent annotation to certify)")

    # ------------------------------------------------------ base-edge frame contract
    # v1.4.0 hand-off: the extractor records, per piece, the base-edge frame it
    # cropped at 1:1 scale.  Everything below is checked against the frame this
    # script derived from corners_px alone (frames[] above) and against the
    # minimal Stage 3 hand-off file pieces_base_edge.json.
    fb = {"derive": [], "quad": [], "matrix": [], "rot": [], "preview": [],
          "crop": [], "margin": [], "deprecated": []}
    frame_rows = []
    for i, piece in enumerate(pieces):
        pid = ids[i] if i < len(ids) else f"#{i}"
        P = arrs[i]
        f = frames[i]
        if P is None or f is None:
            continue
        n = len(P)
        L = f["base_side_index"]
        base, image, size = f["base"], f["image"], f["size"]
        # (a) the recorded bookkeeping must describe the derived frame
        rep_idx = piece.get("base_side_index")
        if rep_idx is None or int(rep_idx) != L:
            fb["derive"].append(f"{pid}: base_side_index {rep_idx!r} vs derived {L}")
        rep_len = piece.get("base_length_px")
        if rep_len is None or abs(float(rep_len) / f["base_length"] - 1.0) > 5e-4:
            fb["derive"].append(f"{pid}: base_length_px {rep_len!r} vs derived {f['base_length']:.6f}")
        if n > 1 and f["second_longest"] > f["base_length"] * (1.0 - 1e-9):
            fb["derive"].append(f"{pid}: base side is not uniquely the longest "
                                f"({f['base_length']:.3f} vs {f['second_longest']:.3f} px)")
        be = piece.get("base_edge")
        if not isinstance(be, dict):
            fb["derive"].append(f"{pid}: base_edge record missing")
        else:
            if be.get("id") != pid:
                fb["derive"].append(f"{pid}: base_edge.id {be.get('id')!r}")
            rec_len = be.get("baseLength")
            if rec_len is None or abs(float(rec_len) - f["base_length"]) > 2e-3:
                fb["derive"].append(f"{pid}: base_edge.baseLength {rec_len!r} vs derived {f['base_length']:.6f}")
            for k, key in ((2, "p2"), (3, "p3")):
                rec = be.get(key)
                if rec is None:
                    fb["derive"].append(f"{pid}: base_edge.{key} missing")
                    continue
                err = float(np.abs(np.asarray(rec, float).ravel()[:2] - base[k]).max())
                if err > 2e-3:
                    fb["derive"].append(f"{pid}: base_edge.{key} off {err:.2e} px")
        # (b) the derived base frame itself obeys the documented axes
        if float(np.abs(base[0]).max()) > 1e-9:
            fb["quad"].append(f"{pid}: P0={base[0].tolist()} not (0,0)")
        if abs(float(base[1][1])) > 1e-6 or abs(float(base[1][0]) - f["base_length"]) > 1e-6:
            fb["quad"].append(f"{pid}: P1={base[1].tolist()} not (baseLength,0)")
        below = [k for k in range(2, n) if float(base[k][1]) <= 0.0]
        if below:
            fb["quad"].append(f"{pid}: corners {below} not below the base")
        sa = float(polygon_signed_area(base))
        if sa <= 0.0:
            fb["quad"].append(f"{pid}: signed area {sa:+.1f} (expected > 0, clockwise in y-down)")
        if not convexity(base)["convex"]:
            fb["quad"].append(f"{pid}: base-frame quad not strictly convex")
        if abs(sa / float(polygon_signed_area(P)) - 1.0) > 1e-9:
            fb["quad"].append(f"{pid}: base-frame area != canvas area")
        if float(np.min(f["edge_lengths"])) < 20.0:
            fb["quad"].append(f"{pid}: min side {float(np.min(f['edge_lengths'])):.1f} px")
        # (c) recorded matrices: unit rotation applied to the ROLLED corner list
        M = piece.get("rectified_to_base_edge_matrix")
        Mp = piece.get("rectified_to_piece_matrix")
        if M is None or Mp is None:
            fb["matrix"].append(f"{pid}: matrix missing")
        else:
            M = np.asarray(M, np.float64)
            Mp = np.asarray(Mp, np.float64)
            if (M.shape != (3, 3) or Mp.shape != (3, 3)
                    or not np.isfinite(M).all() or not np.isfinite(Mp).all()):
                fb["matrix"].append(f"{pid}: matrix shape/values {M.shape}/{Mp.shape}")
            else:
                A = M[:2, :2]
                unit_err = float(np.abs(A.T @ A - np.eye(2)).max())
                det = float(np.linalg.det(A))
                if unit_err > 1e-6 or det <= 0.0:
                    fb["matrix"].append(f"{pid}: |A^T A - I|={unit_err:.2e}, det={det:+.6f} "
                                        f"(piece_scale is 1.0, so the linear part must be a rotation)")
                Prow = np.roll(P, -L, axis=0)
                hom = M @ np.c_[Prow, np.ones(n)].T
                pred = (hom[:2] / hom[2]).T
                err = float(np.abs(pred - base).max())
                if err > 2e-3:
                    fb["matrix"].append(f"{pid}: base matrix round-trip off {err:.2e} px "
                                        f"(corners_px must be rolled by -base_side_index)")
                homp = Mp @ np.c_[Prow, np.ones(n)].T
                predp = (homp[:2] / homp[2]).T
                errp = float(np.abs(predp - image).max())
                if errp > 2e-3:
                    fb["matrix"].append(f"{pid}: crop matrix round-trip off {errp:.2e} px")
                if float(np.abs(A - Mp[:2, :2]).max()) > 1e-8:
                    fb["matrix"].append(f"{pid}: base/crop matrices differ in the linear part")
                t_off = Mp[:2, 2] - M[:2, 2]
                if float(np.abs(t_off - margin_px).max()) > 1e-6:
                    fb["matrix"].append(f"{pid}: crop translation offset {t_off.tolist()} != margin {margin_px:g}")
                rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
                rj = piece.get("rotation_deg")
                if rj is None or abs((rot - float(rj) + 180.0) % 360.0 - 180.0) > 1e-3:
                    fb["rot"].append(f"{pid}: rotation_deg {rj!r} vs matrix angle {rot:.6f}")
        # (d) rounded preview fields must match the full-precision frame
        bc = piece.get("base_edge_corners_px")
        ic = piece.get("image_corners_px")
        if bc is None or ic is None:
            fb["preview"].append(f"{pid}: preview corners missing")
        else:
            bc = np.asarray(bc, np.float64)
            ic = np.asarray(ic, np.float64)
            if bc.shape != base.shape or ic.shape != image.shape:
                fb["preview"].append(f"{pid}: preview shapes {bc.shape}/{ic.shape}")
            else:
                eb = float(np.abs(bc - base).max())
                ei = float(np.abs(ic - image).max())
                if eb > 2e-3 or ei > 2e-3:
                    fb["preview"].append(f"{pid}: base preview off {eb:.2e} px, image preview off {ei:.2e} px")
        # (e) crop size, margin band and horizontal base edge
        rep_size = piece.get("piece_size_px")
        if not (isinstance(rep_size, (list, tuple)) and len(rep_size) == 2
                and [int(v) for v in rep_size] == size):
            fb["crop"].append(f"{pid}: piece_size_px {rep_size!r} vs derived {size}")
        if abs(float(image[0][1]) - margin_px) > 1e-6 or abs(float(image[1][1]) - margin_px) > 1e-6:
            fb["margin"].append(f"{pid}: base edge not horizontal at y={margin_px:g} "
                                f"(P0.y {image[0][1]:.6f}, P1.y {image[1][1]:.6f})")
        if float(image[:, 0].min()) < margin_px - 2e-3 or float(image[:, 1].min()) < margin_px - 2e-3:
            fb["margin"].append(f"{pid}: piece not inset by margin (min {image.min(axis=0).tolist()})")
        if (float(image[:, 0].max()) > size[0] - margin_px + 1e-6
                or float(image[:, 1].max()) > size[1] - margin_px + 1e-6):
            fb["margin"].append(f"{pid}: piece exceeds the margin band ({image.max(axis=0).tolist()} in {size})")
        # (f) the shared normalised canvas is gone in v1.4.0
        for stale in ("normalized_corners_px", "piece_output_size_px"):
            if stale in piece:
                fb["deprecated"].append(f"{pid}:{stale}")
        frame_rows.append({"piece_id": pid, "base_side_index": L,
                           "base_length_px": f["base_length"], "crop_size_px": size})
    if frame_rows:
        n_fr = len(frame_rows)
        record("base-edge frame record consistent with the derived frame",
               "PASS" if not fb["derive"] else "FAIL",
               f"base_side_index / base_length_px / base_edge{{baseLength,p2,p3}} match the frame "
               f"derived from corners_px (<= 2e-3 px) for all {n_fr} pieces"
               if not fb["derive"] else f"{fb['derive'][:8]}")
        record("base-edge frame axes: P0=(0,0), P1=(baseLength,0), corners below, clockwise",
               "PASS" if not fb["quad"] else "FAIL",
               f"all {n_fr} pieces: positive signed area in the y-down base frame, strictly convex, "
               f"corners 2..3 below the base, min side >= 20 px"
               if not fb["quad"] else f"{fb['quad'][:8]}")
        record("unit-scale rotation matrix round-trips corners_px (rolled by base_side_index)",
               "PASS" if not fb["matrix"] else "FAIL",
               f"|A^T A - I| <= 1e-6, det > 0 (piece_scale 1.0), M @ roll(corners_px, -base_side_index) "
               f"== base_edge_corners_px <= 2e-3 px and crop matrix translation = base + margin, "
               f"for all {n_fr} pieces" if not fb["matrix"] else f"{fb['matrix'][:8]}")
        record("rotation_deg describes the base-edge frame",
               "PASS" if not fb["rot"] else "FAIL",
               f"rotation_deg == atan2(A10, A00) for all {n_fr} pieces"
               if not fb["rot"] else f"{fb['rot'][:8]}")
        record("rounded preview fields match the full-precision frame (3 dp)",
               "PASS" if not fb["preview"] else "FAIL",
               f"base_edge_corners_px / image_corners_px agree with the derived frame within 2e-3 px"
               if not fb["preview"] else f"{fb['preview'][:8]}")
        record("piece_size_px == ceil(base bbox + 2 * margin)",
               "PASS" if not fb["crop"] else "FAIL",
               f"crop sizes {sorted({tuple(r['crop_size_px']) for r in frame_rows})} for all {n_fr} pieces"
               if not fb["crop"] else f"{fb['crop'][:8]}")
        record("crop obeys the margin band and keeps the base edge horizontal",
               "PASS" if not fb["margin"] else "FAIL",
               f"P0/P1 sit at y={margin_px:g}, every corner is inset by >= {margin_px:g} px and the "
               f"piece fits inside its crop (base frame at 1:1 scale)"
               if not fb["margin"] else f"{fb['margin'][:8]}")
        record("normalised-canvas fields removed from the piece records",
               "PASS" if not fb["deprecated"] else "WARN",
               "no normalized_corners_px / piece_output_size_px in any piece record"
               if not fb["deprecated"] else f"still present: {fb['deprecated'][:8]}")
    else:
        record("base-edge frame record consistent with the derived frame", "WARN",
               "no pieces with derivable corners - check not evaluable")
    details["base_edge_frame"] = {
        "margin_px": margin_px,
        "n_pieces": len(frame_rows),
        "crop_sizes": {r["piece_id"]: r["crop_size_px"] for r in frame_rows},
        "base_side_index": {r["piece_id"]: r["base_side_index"] for r in frame_rows},
        "base_length_px": {r["piece_id"]: r["base_length_px"] for r in frame_rows},
        "derive_failures": fb["derive"],
        "quad_failures": fb["quad"],
        "matrix_failures": fb["matrix"],
        "margin_failures": fb["margin"],
    }

    # --------------------------------- pieces_base_edge.json (Stage 3 hand-off)
    print("\n--- minimal Stage 3 hand-off: pieces_base_edge.json ---")
    bjson = None
    if not BASE_JSON.exists():
        record("pieces_base_edge.json present", "FAIL",
               f"{BASE_JSON.relative_to(ROOT)} is missing (it is the Stage 3 hand-off)")
    else:
        bjson = load_json(BASE_JSON)
        details["pieces_base_edge_json"] = {"path": str(BASE_JSON),
                                            "keys": sorted(bjson.keys()),
                                            "bytes": BASE_JSON.stat().st_size}
        required_b = ("schema_version", "algorithm_version", "coordinate_system", "unit_scale",
                      "px_per_mm", "session_id", "piece_count", "pieces")
        missing_b = [k for k in required_b if k not in bjson]
        record("pieces_base_edge.json schema", "PASS" if not missing_b else "FAIL",
               "required keys present" if not missing_b else f"missing: {missing_b}")
        stmt_bad = []
        if bjson.get("coordinate_system") != top.get("coordinate_system"):
            stmt_bad.append(f"coordinate_system {bjson.get('coordinate_system')!r}")
        if bjson.get("algorithm_version") != top.get("algorithm_version"):
            stmt_bad.append(f"algorithm_version {bjson.get('algorithm_version')!r}")
        if bjson.get("session_id") != top.get("session_id"):
            stmt_bad.append(f"session_id {bjson.get('session_id')!r} vs {top.get('session_id')!r}")
        us = bjson.get("unit_scale")
        if not (isinstance(us, (int, float)) and abs(float(us) - 1.0) <= 1e-9):
            stmt_bad.append(f"unit_scale {us!r} (expected 1.0)")
        if not np.allclose(np.asarray(bjson.get("px_per_mm", []), float),
                           np.asarray(px_per_mm, float), atol=1e-9):
            stmt_bad.append(f"px_per_mm {bjson.get('px_per_mm')!r} vs {px_per_mm}")
        record("pieces_base_edge.json metadata agrees with pieces.json",
               "PASS" if not stmt_bad else "FAIL",
               "coordinate_system / algorithm_version / session_id / unit_scale=1.0 / px_per_mm all agree"
               if not stmt_bad else f"{stmt_bad}")
        bp = bjson.get("pieces")
        n_bp = len(bp) if isinstance(bp, list) else 0
        record("pieces_base_edge.json piece count",
               "PASS" if isinstance(bp, list) and n_bp == len(pieces) else "FAIL",
               f"{n_bp} records vs {len(pieces)} pieces")
        bmap = {}
        if isinstance(bp, list):
            bmap = {str(r.get("id")): r for r in bp if isinstance(r, dict)}
        bbad = []
        max_prec = 0.0
        for i, piece in enumerate(pieces):
            pid = ids[i] if i < len(ids) else f"#{i}"
            rec = bmap.get(pid)
            if rec is None:
                bbad.append(f"{pid}: missing record")
                continue
            pj_be = piece.get("base_edge") if isinstance(piece.get("base_edge"), dict) else {}
            for key in ("baseLength", "p2", "p3"):
                if key not in rec:
                    bbad.append(f"{pid}: missing {key}")
                    continue
                pv = pj_be.get(key)
                if pv is None:
                    bbad.append(f"{pid}: pieces.json base_edge.{key} missing")
                    continue
                d = float(np.abs(np.asarray(rec[key], float) - np.asarray(pv, float)).max())
                max_prec = max(max_prec, d)
                if d > 1e-12:
                    bbad.append(f"{pid}: {key} differs by {d:.2e} px")
        extra = sorted(set(bmap) - set(ids))
        record("pieces_base_edge.json == pieces.json base_edge (full precision)",
               "PASS" if not bbad and not extra else "FAIL",
               f"{len(pieces)}/{len(pieces)} id/baseLength/p2/p3 bitwise equal (max diff {max_prec:.2e} px)"
               if not bbad and not extra else f"{bbad[:8]} extra_ids={extra[:8]}")
        if isinstance(bp, list):
            border_ok = [str(r.get("id")) for r in bp if isinstance(r, dict)] == ids
            record("pieces_base_edge.json keeps the scan order", "PASS" if border_ok else "WARN",
                   "records in the same S01..S16 scan order as pieces.json" if border_ok
                   else "record order differs from pieces.json")

    # pairwise overlap of detected polygons in the rectified canvas
    valid = [p for p in detected_polys if p is not None]
    if len(valid) == 16:
        label_map = np.zeros((canvas_h, canvas_w), np.int16)
        for k, poly in enumerate(valid, start=1):
            m = np.zeros((canvas_h, canvas_w), np.uint8)
            pts = np.round(poly * 16).astype(np.int32)
            cv2.fillPoly(m, [pts], 1, shift=4)
            label_map += m.astype(np.int16)
        overlap_px = int(np.count_nonzero(label_map > 1))
        record("detected polygons do not overlap",
               "PASS" if overlap_px < 50 else "FAIL",
               f"{overlap_px} overlapping px among 16 polygons")

    # ------------------------------------- pseudo ground truth shape matching
    print("\n--- pseudo ground truth (shape-only similarity matching) ---")
    assignment, cost = match_shapes(detected_polys, src_polys)
    fit_rows = []
    match_ok = True
    used_sources = []
    matched = 0
    corner_for_stats = []
    for i, pid in enumerate(ids if len(ids) == len(pieces) else [f"#{k}" for k in range(len(pieces))]):
        arr = arrs[i]
        row = geom_rows[i]
        if arr is None or not np.isfinite(arr).all():
            match_ok = False
            continue
        j = int(assignment[i]) if i < len(assignment) else -1
        if j < 0 or cost[i, j] > 1e8:
            match_ok = False
            row["source_id"] = None
            continue
        fit = best_correspondence(src_polys[j], detected_polys[i])
        iou = raster_iou(detected_polys[i], fit["pred"])
        row.update({
            "source_id": src_ids[j],
            "corner_max_err_px": fit["max_err"],
            "corner_rms_err_px": fit["rms_err"],
            "scale": fit["scale"],
            "rot_deg": fit["rot_deg"],
            "iou": iou,
            "vertex_count": len(arr),
        })
        used_sources.append(j)
        matched += 1
        corner_for_stats.append(fit["max_err"])
        fit_rows.append(row)
        print(f"  {pid} -> {src_ids[j]}  nv={len(arr)}  max_err={fit['max_err']:6.3f}px  "
              f"rms={fit['rms_err']:5.3f}px  scale={fit['scale']:.4f}  rot={fit['rot_deg']:+7.2f}deg  IoU={iou:.5f}")

    record("shape-only 1:1 assignment (Hungarian, all 16)",
           "PASS" if match_ok and matched == 16 and len(set(used_sources)) == 16 else "FAIL",
           f"matched {matched}/16 pieces to {len(set(used_sources))} unique source polygons")
    if corner_for_stats:
        errs = np.asarray(corner_for_stats, dtype=float)
        worst_i = int(np.nanargmax(errs))
        worst_row = fit_rows[worst_i] if fit_rows else {}
        stats = {"mean": float(errs.mean()), "median": float(np.median(errs)),
                 "max": float(errs.max()), "min": float(errs.min()),
                 "worst_piece": worst_row.get("piece_id"), "worst_source": worst_row.get("source_id")}
        details["corner_error_px"] = stats
        record("pseudo-GT corner error (max similarity residual)",
               "PASS" if stats["max"] < CORNER_PASS_PX else ("WARN" if stats["max"] < CORNER_WARN_PX else "FAIL"),
               f"mean {stats['mean']:.3f}, median {stats['median']:.3f}, max {stats['max']:.3f} px "
               f"(worst {stats['worst_piece']}->{stats['worst_source']}; target < {CORNER_PASS_PX} px)")
    else:
        record("pseudo-GT corner error (max similarity residual)", "FAIL", "no matched pieces")

    # ------------------------------------- attribution: independent photo measurement
    print("\n--- attribution: independent measurement from the rectified photo ---")
    photo_quads, photo_info = photo_reference_quads(detected_polys, ids)
    photo_iou_quads = photo_info.pop("quads_iou", {})
    details["photo_reference"] = photo_info
    record("independent photo segmentation: one blob per piece",
           "PASS" if (photo_info.get("assigned") == len(pieces) and len(pieces)
                      and not photo_info.get("unmatched")
                      and photo_info.get("unused_blobs") == 0) else "WARN",
           f"{photo_info.get('n_components')} foreground blobs (Lab diff > {photo_info.get('thr')}, "
           f"8-connectivity); {photo_info.get('assigned')}/{len(pieces)} pieces measured, "
           f"{photo_info.get('unused_blobs')} unexplained blob(s)"
           + (f"; {photo_info['unmatched'][:4]}" if photo_info.get("unmatched") else "")
           + " => no false detection of shadow/board/dust")
    det_by_id = {pid: detected_polys[i] for i, pid in enumerate(ids) if detected_polys[i] is not None}
    row_by_id = {r["piece_id"]: r for r in fit_rows}
    fid, fid_iou, photo_vs_src, meas_disagree = [], [], [], []
    for pid, q in photo_quads.items():
        det = det_by_id.get(pid)
        srcid = (row_by_id.get(pid) or {}).get("source_id")
        if det is not None:
            fits = best_correspondence(q, det)
            row_by_id.setdefault(pid, {})["photo_corner_err_px"] = fits["max_err"]
            fid.append((pid, fits["max_err"]))
        q2 = photo_iou_quads.get(pid)
        if q2 is not None:
            if det is not None:
                e2 = best_correspondence(q2, det)["max_err"]
                row_by_id.setdefault(pid, {})["photo_iou_corner_err_px"] = e2
                fid_iou.append((pid, e2))
            meas_disagree.append((pid, best_correspondence(q, q2)["max_err"]))
        if srcid is not None:
            sj = src_ids.index(srcid)
            pv = best_correspondence(src_polys[sj], q)["max_err"]
            row_by_id.setdefault(pid, {})["photo_vs_source_px"] = pv
            photo_vs_src.append((pid, pv))
    if fid:
        errs = np.asarray([v for _, v in fid], dtype=float)
        fstats = {"mean": float(errs.mean()), "median": float(np.median(errs)),
                  "max": float(errs.max()),
                  "worst_piece": fid[int(np.argmax(errs))][0]}
        details["extractor_fidelity_px"] = fstats
        status = "PASS" if (fstats["median"] <= 1.5 and fstats["max"] <= 3.0) else (
            "WARN" if fstats["max"] <= 8.0 else "FAIL")
        extra = ""
        if fid_iou:
            e2 = np.asarray([v for _, v in fid_iou], dtype=float)
            fstats["iou_optimal_estimator"] = {
                "mean": float(e2.mean()), "median": float(np.median(e2)),
                "max": float(e2.max()), "worst_piece": fid_iou[int(np.argmax(e2))][0]}
            extra = (f"; second photo estimator (IoU-optimal quad): mean {e2.mean():.3f}, "
                     f"median {np.median(e2):.3f}, max {e2.max():.3f} px")
        record("extractor corner fidelity vs independently measured photo quad",
               status,
               f"max corner offset between pieces.json quad and photo edge fit: mean {fstats['mean']:.3f}, "
               f"median {fstats['median']:.3f}, max {fstats['max']:.3f} px (worst {fstats['worst_piece']}){extra}; "
               f"this isolates Stage 2 from capture error (photo-quad noise <= ~1.2 px)")
    if meas_disagree:
        md = np.asarray([v for _, v in meas_disagree], dtype=float)
        mstats = {"mean": float(md.mean()), "median": float(np.median(md)),
                  "max": float(md.max()), "worst_piece": meas_disagree[int(np.argmax(md))][0],
                  "pieces_over_3px": [pid for pid, v in meas_disagree if v > 3.0]}
        details["photo_estimator_disagreement_px"] = mstats
        record("photo measurement consistency (edge refit vs IoU-optimal quad)",
               "PASS" if mstats["max"] <= 3.0 else ("WARN" if mstats["max"] <= 8.0 else "FAIL"),
               f"two independent estimators of the same photo quad differ by median {mstats['median']:.3f}, "
               f"max {mstats['max']:.3f} px (worst {mstats['worst_piece']})"
               + (f"; > 3 px for {mstats['pieces_over_3px']} (shadow halo inflates that blob, "
                  f"so the photo-side error for those pieces is only bounded, not pinned)"
                  if mstats["pieces_over_3px"] else ""))
    if photo_vs_src:
        pv = np.asarray([v for _, v in photo_vs_src], dtype=float)
        pstats = {"mean": float(pv.mean()), "median": float(np.median(pv)),
                  "max": float(pv.max()),
                  "worst_piece": photo_vs_src[int(np.argmax(pv))][0],
                  "pieces_under_2px": int((pv <= CORNER_PASS_PX).sum())}
        details["photo_vs_layout_px"] = pstats
        record("pseudo-GT premise: photo quad vs layout polygon",
               "PASS" if pstats["max"] <= CORNER_PASS_PX else "WARN",
               f"same statistic measured on the PHOTO itself: mean {pstats['mean']:.3f}, "
               f"median {pstats['median']:.3f}, max {pstats['max']:.3f} px "
               f"({pstats['pieces_under_2px']}/16 pieces within {CORNER_PASS_PX} px) "
               f"=> the layout polygon is only an approximate ground truth for the "
               f"hand-cut physical pieces, so the <{CORNER_PASS_PX} px target above is not "
               f"attainable by any extractor on this data")
        if isinstance(details.get("corner_error_px"), dict):
            details["corner_error_px"]["attribution"] = {
                "photo_vs_layout_px": pstats,
                "extractor_vs_photo_px": details.get("extractor_fidelity_px"),
                "photo_estimator_disagreement_px": details.get("photo_estimator_disagreement_px"),
                "conclusion": ("pseudo-GT corner error is dominated by physical/capture deviation "
                               "(hand-cut pieces, layout polygons are ideal); Stage 2 fidelity "
                               "against the actual photo is given by extractor_fidelity_px"),
            }

    ious = [r["iou"] for r in fit_rows if r["iou"] is not None]
    if ious:
        ious = np.asarray(ious, dtype=float)
        iou_stats = {"mean": float(ious.mean()), "median": float(np.median(ious)),
                     "min": float(ious.min()), "max": float(ious.max())}
        details["polygon_iou"] = iou_stats
        record("pseudo-GT polygon IoU (rasterised)",
               "PASS" if iou_stats["min"] >= IOU_PASS else ("WARN" if iou_stats["min"] >= IOU_WARN else "FAIL"),
               f"mean {iou_stats['mean']:.5f}, median {iou_stats['median']:.5f}, min {iou_stats['min']:.5f} "
               f"(target > {IOU_PASS})")

    scales = np.asarray([r["scale"] for r in fit_rows if r["scale"]], dtype=float)
    printed_scale, printed_why, circular = derive_printed_scale(layout, source_size, px_per_mm)
    details["printed_scale"] = {"mm_per_source_px": printed_scale, "source": printed_why}
    if scales.size:
        spread = float(scales.std() / scales.mean()) if scales.mean() else float("inf")
        details["fitted_scale"] = {"mean": float(scales.mean()), "std": float(scales.std()),
                                   "rel_spread": spread, "min": float(scales.min()),
                                   "max": float(scales.max())}
        record("fitted similarity scale consistent across pieces",
               "PASS" if spread < SCALE_SPREAD_PASS else ("WARN" if spread < SCALE_SPREAD_WARN else "FAIL"),
               f"mean {scales.mean():.5f} px/srcpx, rel std {spread*100:.3f}% (target < 1%)")
        if printed_scale is not None:
            expected_scale = printed_scale * float(px_per_mm[0])
            dev = abs(scales.mean() / expected_scale - 1.0)
            details["fitted_scale"]["expected_from_pdf"] = expected_scale
            details["fitted_scale"]["rel_dev_vs_pdf"] = dev
            record("fitted scale vs printed-PDF scale",
                   "PASS" if dev < PRINT_SCALE_PASS else ("WARN" if dev < PRINT_SCALE_WARN else "FAIL"),
                   f"fitted {scales.mean():.5f} px/srcpx vs expected {expected_scale:.5f} px/srcpx "
                   f"= {printed_scale:.6f} mm/srcpx * {float(px_per_mm[0]):.1f} px/mm (dev {dev*100:.2f}%)")

    # --------------------------------------------------- global area conservation
    print("\n--- global geometry ---")
    layout_total = sum(polygon_area(p) for p in src_polys)
    src_area_ratio = layout_total / float(source_size[0] * source_size[1])
    record("layout area conservation baseline",
           "PASS" if abs(src_area_ratio - 1.0) < 1e-3 else "WARN",
           f"sum(layout)/source image area = {src_area_ratio:.9f}")
    observed_total = float(sum(polygon_area(p) for p in detected_polys if p is not None))
    details["global_area"] = {"layout_total_px2": layout_total, "observed_total_px2": observed_total}
    if printed_scale is not None:
        expected_total = layout_total * (printed_scale * float(px_per_mm[0])) ** 2
        ratio = observed_total / expected_total
        implied_mm_per_px = math.sqrt(observed_total / layout_total) / float(px_per_mm[0]) if layout_total > 0 else float("nan")
        details["global_area"].update({
            "expected_total_px2": expected_total, "ratio": ratio,
            "implied_printed_mm_per_source_px": implied_mm_per_px,
            "derivation": printed_why,
        })
        record("global area conservation vs printed scale",
               "PASS" if abs(ratio - 1.0) <= AREA_TOTAL_PASS else ("WARN" if abs(ratio - 1.0) <= AREA_TOTAL_WARN else "FAIL"),
               f"observed {observed_total:,.0f} px^2 vs expected {expected_total:,.0f} px^2 = ratio {ratio:.5f} "
               f"(implied print scale {implied_mm_per_px:.6f} mm/srcpx vs {printed_scale:.6f}; target within {AREA_TOTAL_PASS*100:.0f}%)")
    else:
        # Documented fallback: derive the print scale from the observed total.
        implied = math.sqrt(observed_total / layout_total) / float(px_per_mm[0])
        expected_total = observed_total
        printed_scale = implied
        details["global_area"]["fallback_implied_mm_per_source_px"] = implied
        record("global area conservation vs printed scale", "WARN",
               f"printed scale unavailable; derived from observed total = {implied:.6f} mm/srcpx (circular, reported as WARN)")

    # overall target per-piece area vs assigned source area (dimensionless check)
    area_ratios = []
    for row in fit_rows:
        if row["source_id"] is None or row["area_px"] is None:
            continue
        sj = src_ids.index(row["source_id"])
        area_ratios.append(row["area_px"] / polygon_area(src_polys[sj]))
    if area_ratios:
        ar = np.asarray(area_ratios, dtype=float)
        details["per_piece_area_ratio"] = {"mean": float(ar.mean()), "std": float(ar.std()),
                                           "min": float(ar.min()), "max": float(ar.max())}
        expected_ar = (printed_scale * float(px_per_mm[0])) ** 2 if printed_scale else float(ar.mean())
        dev = abs(ar.mean() / expected_ar - 1.0)
        record("per-piece area ratio vs matched source (scale consistency)",
               "PASS" if dev < 0.03 else ("WARN" if dev < 0.06 else "FAIL"),
               f"mean detected/source area = {ar.mean():.5f} (expected {expected_ar:.5f} from print scale), "
               f"rel std {ar.std()/ar.mean()*100:.3f}%, min {ar.min():.4f}, max {ar.max():.4f}")

    # ---------------------------------------------------------------- overlay
    try:
        base = imread_u(S1_OUT / "rectified" / "pieces_rectified.png", cv2.IMREAD_COLOR)
        if base is not None:
            ov = base.copy()
            for i, row in enumerate(geom_rows):
                poly = detected_polys[i]
                if poly is None:
                    continue
                cv2.polylines(ov, [np.round(poly).astype(np.int32)], True, (0, 255, 0), 2)
                pq = photo_quads.get(row["piece_id"])
                if pq is not None:
                    cv2.polylines(ov, [np.round(pq).astype(np.int32)], True, (255, 0, 255), 1)
                if row.get("source_id"):
                    j = src_ids.index(row["source_id"])
                    if row.get("corner_max_err_px") is not None:
                        fit = best_correspondence(src_polys[j], detected_polys[i])
                        cv2.polylines(ov, [np.round(fit["pred"]).astype(np.int32)], True, (0, 0, 255), 1)
                c = np.round(np.asarray(row.get("center_px", [0, 0]), float)).astype(int)
                cv2.putText(ov, str(row["piece_id"]), tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 0), 2, cv2.LINE_AA)
            TMP.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".png", ov)
            if ok:
                buf.tofile(str(OVERLAY_PNG))
                record("diagnostic overlay written", "PASS", str(OVERLAY_PNG.relative_to(ROOT)))
    except Exception as exc:  # noqa: BLE001
        record("diagnostic overlay written", "WARN", f"{exc}")

    # ------------------------------------------------- per-piece machine table
    # Merge geometry, pseudo-GT fit and photo-based fidelity into one row per
    # piece; written to tmp/verify_stage2_result.json under details["pieces"].
    pieces_table = []
    for idx, row in enumerate(geom_rows):
        pid = row["piece_id"]
        fitr = next((r for r in fit_rows if r["piece_id"] == pid), {})
        pieces_table.append({
            "piece_id": pid,
            "source_id": fitr.get("source_id"),
            "center_px": fitr.get("center_px", row.get("center_px")),
            "corner_count": row.get("vertex_count"),
            "area_px": row.get("area_px"),
            "quality": row.get("quality"),
            "corner_rms_px": fitr.get("corner_rms_err_px"),
            "corner_max_px": fitr.get("corner_max_err_px"),
            "polygon_iou": fitr.get("iou"),
            "fitted_scale": fitr.get("scale"),
            "photo_vs_layout_px": fitr.get("photo_vs_source_px"),
            "extractor_vs_photo_px": fitr.get("photo_corner_err_px"),
            "extractor_vs_photo_iou_estimator_px": fitr.get("photo_iou_corner_err_px"),
            "mask_area_px": mask_areas[idx] if idx < len(mask_areas) else None,
            "crop_frame": "derived:base_edge" if idx < len(frames_ok) and frames_ok[idx] else None,
            "base_side_index": frames[idx]["base_side_index"] if idx < len(frames_ok) and frames_ok[idx] else None,
            "base_length_px": frames[idx]["base_length"] if idx < len(frames_ok) and frames_ok[idx] else None,
            "crop_size_px": frames[idx]["size"] if idx < len(frames_ok) and frames_ok[idx] else None,
        })
    details["pieces"] = pieces_table

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    fails = [r for r in results if r[1] == "FAIL"]
    warns = [r for r in results if r[1] == "WARN"]
    verdict = "PASS" if not fails and not warns else ("PASS_WITH_WARNINGS" if not fails else "FAIL")
    print(f"SUMMARY: {len(results)} checks, {len(fails)} FAIL, {len(warns)} WARN")
    for check, status, detail in results:
        if status != "PASS":
            print(f"  {status}: {check} - {detail}")
    print("=" * 78)
    details["summary"] = {"checks": len(results), "fail": len(fails), "warn": len(warns),
                          "verdict": verdict,
                          "fails": [[c, d] for c, s, d in results if s == "FAIL"],
                          "warns": [[c, d] for c, s, d in results if s == "WARN"]}
    _write_result(verdict=verdict)
    print(f"verdict: {verdict}")
    return 1 if fails else 0


def crop_size_from_base(base: np.ndarray, margin: float) -> list[int]:
    """Whole-pixel crop size for a piece in its own base-edge frame.

    base is the polygon in the base-edge frame (P0=(0,0), P1=(baseLength,0),
    remaining corners below); the crop is that frame translated by `margin`
    and rounded up, so the piece is fully inside with at least one margin band.
    """
    b = np.asarray(base, dtype=np.float64)
    w = int(np.ceil(float(b[:, 0].max() + 2.0 * margin)))
    h = int(np.ceil(float(b[:, 1].max() + 2.0 * margin)))
    return [max(1, w), max(1, h)]


def derive_base_edge_frame(corner_px, margin: float) -> dict:
    """Independently derive the v1.4.0 base-edge frame from the canvas corners.

    The frame is a pure function of `corners_px`, so the extractor's own
    bookkeeping (base_side_index, base_length_px, base_edge, matrices, crop
    size) can be checked against it without importing Stage 2 code:

      * the longest side is the base: p0 = P[L], p1 = P[(L+1) % n];
      * p0 -> (0, 0) and p1 -> (baseLength, 0) under a rigid rotation
        R = [[u0, u1], [-u1, u0]] with u = (p1 - p0) / baseLength, which keeps
        the polygon handedness, so the remaining corners sit below (y > 0);
      * the crop is that frame shifted by (margin, margin) at 1:1 scale.

    Returns the base-frame polygon (`base`), the crop-frame polygon (`image`),
    the rotation (`rotation`), the crop size (`size`), the base side index and
    the original edge lengths.
    """
    P = np.asarray(corner_px, dtype=np.float64)
    n = len(P)
    edges = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    L = int(np.argmax(edges))
    p0 = P[L]
    p1 = P[(L + 1) % n]
    d = p1 - p0
    base_length = float(np.linalg.norm(d))
    if base_length <= 0.0:
        raise ValueError("degenerate base edge")
    u = d / base_length
    R = np.array([[u[0], u[1]], [-u[1], u[0]]], dtype=np.float64)
    order = [(L + k) % n for k in range(n)]
    base = np.array([R @ (P[j] - p0) for j in order], dtype=np.float64)
    margin_arr = np.array([margin, margin], dtype=np.float64)
    image = base + margin_arr
    return {
        "base_side_index": L,
        "base_length": base_length,
        "rotation": R,
        "base": base,
        "image": image,
        "size": crop_size_from_base(base, margin),
        "edge_lengths": edges,
        "second_longest": float(np.sort(edges)[-2]) if n > 1 else 0.0,
    }


def _write_result(verdict: str) -> None:
    try:
        TMP.mkdir(parents=True, exist_ok=True)
        payload = {"verdict": verdict, "checks": [
            {"check": c, "status": s, "detail": d} for c, s, d in results], "details": details}
        with open(RESULT_JSON, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)
        print(f"results written to {RESULT_JSON.relative_to(ROOT)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not write {RESULT_JSON}: {exc}")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
