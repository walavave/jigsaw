"""在生成的背景场景上运行第二阶段提取，并按“已知真值”评估精度。

每个 backgrounds/output/<name>/ 场景包含：
    empty.png   只有底面
    pieces.png  底面上的 16 片真实纸片
    scene.json  真值：每片的 corners_px / center_px / area_px（与采集时完全一致）

本脚本对每个场景调用 stage2 的 extract_pieces()，把结果与真值比较：
    · 检出数量、是否等于期望值
    · 角点误差（对每个碎片做循环移位取最优后计算，单位 px）
    · 多边形 IoU、面积比
    · 质量分与人工复核项

角点误差是“合成时用的几何”与“提取出的几何”之差，因此同时反映了分割与亚像素定位误差，
不含手剪造成的物理偏差（那部分在第一部分的报告里单独讨论）。

用法
----
    python backgrounds/evaluate_backgrounds.py
    python backgrounds/evaluate_backgrounds.py --scenes wood_floor --output-dir backgrounds/output
    python backgrounds/evaluate_backgrounds.py --config backgrounds/config/extraction_soft_shadow.example.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "stage2_piece_extraction"))

from jigsaw_common.io import imread, imwrite, read_json, write_json  # noqa: E402
from extract_pieces import (  # noqa: E402
    _resolve_config,
    extract_pieces,
    polygon_iou,
    rasterize_polygon,
    scan_order_indices,
)

DEFAULT_SCENES = ROOT / "backgrounds" / "output"
DEFAULT_RECTIFICATION = ROOT / "stage1_capture_calibration" / "output" / "rectified" / "rectification.json"
DEFAULT_CONFIG = ROOT / "stage2_piece_extraction" / "config" / "extraction.example.json"


# --------------------------------------------------------------------------- #
# 几何比较
# --------------------------------------------------------------------------- #
def _point_segment_distances(points, polygon):
    """每个点到多边形各边的距离，返回 shape (n_points, n_edges)。"""
    points = np.asarray(points, np.float64)[:, None, :]
    start = np.asarray(polygon, np.float64)[None, :, :]
    end = np.roll(np.asarray(polygon, np.float64), -1, axis=0)[None, :, :]
    segment = end - start
    length_sq = np.maximum(np.sum(segment ** 2, axis=2), 1e-9)
    projection = np.clip(np.sum((points - start) * segment, axis=2) / length_sq, 0.0, 1.0)
    closest = start + projection[:, :, None] * segment
    return np.linalg.norm(points - closest, axis=2)


def boundary_statistics(ground_truth, estimate):
    """对称的“顶点到对方边界”距离统计，适用于顶点数不同的情况。"""
    truth = np.asarray(ground_truth, np.float64)
    candidate = np.asarray(estimate, np.float64)
    truth_to_estimate = _point_segment_distances(truth, candidate).min(axis=1)
    estimate_to_truth = _point_segment_distances(candidate, truth).min(axis=1)
    combined = np.concatenate([truth_to_estimate, estimate_to_truth])
    return {
        "mean": float(combined.mean()),
        "median": float(np.median(combined)),
        "max": float(combined.max()),
        "hausdorff": float(max(truth_to_estimate.max(), estimate_to_truth.max())),
    }


def compare_scene(ground_truth_pieces, payload, canvas_shape):
    """对单个场景的真值与提取结果做配对与误差统计。"""
    extracted = payload["pieces"]
    truth = [dict(piece) for piece in ground_truth_pieces]
    truth_order = scan_order_indices([piece["center_px"] for piece in truth])
    truth = [truth[index] for index in truth_order]

    # 一对一指派：按中心点距离贪心匹配（真值几何与提取几何本应重合）
    used = set()
    matches = []
    for gt in truth:
        best_index, best_distance = None, math.inf
        for index, piece in enumerate(extracted):
            if index in used:
                continue
            distance = math.dist(gt["center_px"], piece["center_px"])
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index is None:
            continue
        used.add(best_index)
        matches.append((gt, extracted[best_index], best_distance))

    corner_errors = []
    hausdorff = []
    ious = []
    area_ratios = []
    vertex_counts = []
    per_piece = []
    for gt, piece, centre_distance in matches:
        boundary = boundary_statistics(gt["corners_px"], piece["corners_px"])
        corner_errors.extend([boundary["mean"]] * len(gt["corners_px"]))
        hausdorff.append(boundary["hausdorff"])
        vertex_counts.append(len(piece["corners_px"]))

        gt_mask = rasterize_polygon(np.asarray(gt["corners_px"], np.float64), canvas_shape) > 0
        est_mask = rasterize_polygon(np.asarray(piece["corners_px"], np.float64), canvas_shape) > 0
        iou = float(np.logical_and(gt_mask, est_mask).sum() / max(np.logical_or(gt_mask, est_mask).sum(), 1))
        ratio = float(piece["area_px"] / gt["area_px"])
        ious.append(iou)
        area_ratios.append(ratio)
        per_piece.append({
            "piece_id": piece["piece_id"],
            "corner_count": len(piece["corners_px"]),
            "matched_truth_center": [round(float(v), 3) for v in gt["center_px"]],
            "center_distance_px": round(centre_distance, 3),
            "corner_error_mean_px": round(boundary["mean"], 3),
            "corner_error_max_px": round(boundary["max"], 3),
            "hausdorff_px": round(boundary["hausdorff"], 3),
            "iou": round(iou, 5),
            "area_ratio": round(ratio, 5),
            "quality": piece["quality"],
            "agreement_shift_px": piece["quality_breakdown"].get("agreement_shift_px"),
        })

    corner_errors = np.asarray(corner_errors) if corner_errors else np.zeros(1)
    hausdorff = np.asarray(hausdorff) if hausdorff else np.zeros(1)
    summary = {
        "expected_pieces": len(ground_truth_pieces),
        "detected_pieces": len(extracted),
        "matched_pieces": len(matches),
        "corner_error_mean_px": round(float(corner_errors.mean()), 3),
        "corner_error_median_px": round(float(np.median(corner_errors)), 3),
        "corner_error_p90_px": round(float(np.percentile(corner_errors, 90)), 3),
        "corner_error_max_px": round(float(corner_errors.max()), 3),
        "hausdorff_mean_px": round(float(hausdorff.mean()), 3),
        "hausdorff_max_px": round(float(hausdorff.max()), 3),
        "corner_count_histogram": {
            str(count): vertex_counts.count(count) for count in sorted(set(vertex_counts))
        },
        "iou_mean": round(float(np.mean(ious)), 5) if ious else 0.0,
        "iou_min": round(float(np.min(ious)), 5) if ious else 0.0,
        "area_ratio_mean": round(float(np.mean(area_ratios)), 5) if area_ratios else 0.0,
        "area_ratio_min": round(float(np.min(area_ratios)), 5) if area_ratios else 0.0,
        "area_ratio_max": round(float(np.max(area_ratios)), 5) if area_ratios else 0.0,
        "quality_min": payload["summary"]["min_quality"],
        "quality_median": payload["summary"]["median_quality"],
        "manual_review": payload["manual_review"],
        "segmentation": payload["segmentation"],
        "algorithm_version": payload["algorithm_version"],
        "config": {key: payload["config"][key] for key in (
            "threshold_mode", "threshold_min", "threshold_max", "ignore_border_mm",
            "open_ksize", "close_ksize", "edge_localizer", "max_side_shift_px",
            "min_quality", "mask_erode_px",
        ) if key in payload["config"]},
    }
    return summary, per_piece


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def evaluate_scene(scene_dir, rectification_path, config, rerun):
    scene_dir = Path(scene_dir)
    scene = read_json(scene_dir / "scene.json")
    stage2_dir = scene_dir / "stage2"
    pieces_json = stage2_dir / "pieces.json"

    if rerun or not pieces_json.exists():
        extract_pieces(
            pieces_path=scene_dir / "pieces.png",
            empty_path=scene_dir / "empty.png",
            rectification_path=rectification_path,
            layout_path=None,
            config=config,
            output_dir=stage2_dir,
            write_outputs=True,
        )

    payload = read_json(pieces_json)
    canvas_shape = imread(scene_dir / "pieces.png").shape[:2]
    summary, per_piece = compare_scene(scene["pieces"], payload, canvas_shape)
    summary["background"] = scene["background"]
    summary["label"] = scene.get("label", scene["background"])
    return summary, per_piece


def format_table(results):
    header = (f"{'background':<16}{'pieces':>8}{'corner mean':>13}{'median':>9}{'p90':>8}"
              f"{'haus max':>10}{'IoU mean':>10}{'area':>8}{'q min':>8}{'vtx':>10}  review")
    lines = [header, "-" * len(header)]
    for item in results:
        summary = item["summary"]
        pieces = f"{summary['detected_pieces']}/{summary['expected_pieces']}"
        vertices = "/".join(f"{key}:{value}" for key, value in summary["corner_count_histogram"].items())
        lines.append(
            f"{summary['background']:<16}{pieces:>8}"
            f"{summary['corner_error_mean_px']:>13.3f}"
            f"{summary['corner_error_median_px']:>9.3f}"
            f"{summary['corner_error_p90_px']:>8.3f}"
            f"{summary['hausdorff_max_px']:>10.3f}"
            f"{summary['iou_mean']:>10.5f}"
            f"{summary['area_ratio_mean']:>8.4f}"
            f"{summary['quality_min']:>8.3f}"
            f"{vertices:>10}  {','.join(summary['manual_review']) or '-'}"
        )
    return "\n".join(lines)


def draw_overlay(scene_dir, truth_pieces, payload, output_path, scale=0.42):
    """在场景图上叠加真值（品红）与提取结果（绿色），缩放到便于查看的尺寸。"""

    scene = imread(Path(scene_dir) / "pieces.png")
    truth_layer = scene.copy()
    detected_layer = scene.copy()
    for piece in truth_pieces:
        points = np.round(np.asarray(piece["corners_px"], np.float64)).astype(np.int32)
        cv2.polylines(truth_layer, [points], True, (255, 0, 255), 5, cv2.LINE_AA)
    for piece in payload["pieces"]:
        points = np.round(np.asarray(piece["corners_px"], np.float64)).astype(np.int32)
        cv2.polylines(detected_layer, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
        label_point = tuple(np.round(np.asarray(piece["center_px"], np.float64)).astype(int))
        cv2.putText(detected_layer, piece["piece_id"], (label_point[0] - 40, label_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
    canvas = cv2.addWeighted(truth_layer, 0.55, detected_layer, 0.75, 0.0)
    height, width = canvas.shape[:2]
    resized = cv2.resize(canvas, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    imwrite(Path(output_path), resized)
    return resized


def main(argv=None):
    parser = argparse.ArgumentParser(description="在生成的背景场景上评估第二阶段提取精度")
    parser.add_argument("--scenes", nargs="*", default=None, help="只评估指定背景目录名")
    parser.add_argument("--output-dir", default=str(DEFAULT_SCENES))
    parser.add_argument("--rectification", default=str(DEFAULT_RECTIFICATION))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--rerun", action="store_true", help="强制重新跑提取（默认复用已有结果）")
    parser.add_argument("--summary", default=None, help="汇总 JSON 路径")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    scene_dirs = sorted(path.parent for path in output_dir.glob("*/scene.json"))
    if args.scenes:
        scene_dirs = [path for path in scene_dirs if path.name in set(args.scenes)]
    if not scene_dirs:
        raise SystemExit(f"在 {output_dir} 下没有找到场景，请先运行 backgrounds/make_backgrounds.py")

    config = _resolve_config(Path(args.config))
    results = []
    for scene_dir in scene_dirs:
        summary, per_piece = evaluate_scene(scene_dir, Path(args.rectification), config, args.rerun)
        results.append({"summary": summary, "pieces": per_piece})
        scene = read_json(scene_dir / "scene.json")
        payload = read_json(scene_dir / "stage2" / "pieces.json")
        draw_overlay(scene_dir, scene["pieces"], payload, scene_dir / "overlay.png")
        print(f"[{summary['background']}] {summary['detected_pieces']}/{summary['expected_pieces']} "
              f"corner mean {summary['corner_error_mean_px']:.3f} px, max {summary['corner_error_max_px']:.3f} px")

    print()
    print(format_table(results))

    summary_path = Path(args.summary) if args.summary else output_dir / "evaluation.json"
    write_json(summary_path, {"scenes": results})
    print(f"\nsummary -> {summary_path}")

    failures = [item["summary"] for item in results
                if item["summary"]["detected_pieces"] != item["summary"]["expected_pieces"]]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
