"""生成 2-3 种“底层背景”（木地板 / 大理石地面 / 水泥地面）并合成到真实碎片场景。

用途
----
第二阶段（stage2_piece_extraction）的分割依赖“空桌面图”与“带碎片图”的 Lab 差分。
真实采集只拍了黑绒布桌面，这里在**不重拍照片**的前提下，把已经采集并整流好的 16 片
真实纸片重新合成到程序生成的其他底面上，得到可复现的鲁棒性评测场景：

    backgrounds/output/<name>/empty.png    只有底面（相当于重新拍的“空桌面”）
    backgrounds/output/<name>/pieces.png  底面上的 16 片真实纸片
    backgrounds/output/<name>/scene.json  场景元数据 + 碎片真值多边形（S01..S16）

为什么不是“随便贴图”
--------------------
1. 碎片像素来自真实照片 `stage1_capture_calibration/output/rectified/pieces_rectified.png`，
   用同一视角的 `empty_rectified.png` 做去底色（un-premultiply）还原纸片本身的颜色，
   因此不会把黑桌面边缘带进新场景，纸片纹理/印刷内容/反光都保留；
2. 碎片几何位置、角点、面积与 pieces.json 完全一致，因此可以当作**真值**来量化分割与
   角点定位误差；
3. 两帧（empty / pieces）使用相同几何但**不同的照明场与传感器噪声**，模拟“相机不动、
   拍两张图时灯光略有变化”，避免差分变成平凡零。

用法
----
    python backgrounds/make_backgrounds.py                 # 生成全部背景
    python backgrounds/make_backgrounds.py --only wood_floor
    python backgrounds/make_backgrounds.py --output backgrounds/output --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from jigsaw_common.io import imread, imwrite, read_json, write_json  # noqa: E402

GENERATOR_VERSION = "backgrounds-1.1.0"
DEFAULT_CONFIG = ROOT / "backgrounds" / "config" / "backgrounds.example.json"
DEFAULT_OUTPUT = ROOT / "backgrounds" / "output"
DEFAULT_PIECES_PNG = ROOT / "stage1_capture_calibration" / "output" / "rectified" / "pieces_rectified.png"
DEFAULT_EMPTY_PNG = ROOT / "stage1_capture_calibration" / "output" / "rectified" / "empty_rectified.png"
DEFAULT_PIECES_JSON = ROOT / "stage2_piece_extraction" / "output" / "pieces.json"
DEFAULT_RECTIFICATION = ROOT / "stage1_capture_calibration" / "output" / "rectified" / "rectification.json"

DEFAULT_CONFIG_VALUES = {
    "schema_version": "1.0",
    "seed": 20261029,
    "canvas_px": [2640, 2640],
    "px_per_mm": 8.0,
    "backgrounds": ["wood_floor", "marble_floor_mat", "concrete_floor_mat"],
    "mat_default": {
        "margin_mm": 25.0,
        "feather_mm": 1.5,
        "base_rgb": [26.0, 26.0, 28.0],
        "texture_strength": 0.10,
        "texture_cell_px": [5.0, 14.0],
    },
    "piece_composite": {
        "edge_blur_px": 0.7,
        "inset_px": 0.0,
        "shadow_offset_px": [16.0, 22.0],
        "soft_sigma_px": 16.0,
        "soft_strength": 0.55,
        "contact_sigma_px": 4.0,
        "contact_strength": 0.50,
        "shadow_tint": [0.94, 0.95, 1.00],
    },
    "lighting": {
        "vignette": 0.10,
        "gradient": 0.14,
        "vignette_jitter": 0.35,
        "gradient_angle_jitter_deg": 22.0,
        "noise_sigma": 1.6,
        "frame_gain_jitter": 0.025,
    },
    "wood_floor": {
        "plank_width_mm": [110.0, 165.0],
        "plank_gap_mm": 2.4,
        "plank_length_mm": [420.0, 900.0],
        "base_rgb": [171.0, 118.0, 72.0],
        "hue_jitter": 0.06,
        "value_range": [0.80, 1.06],
        "grain_strength": 0.30,
        "line_strength": 0.22,
        "knots_per_plank": 1,
        "varnish": 0.10,
    },
    "marble_floor": {
        "base_rgb": [233.0, 231.0, 226.0],
        "vein_rgb": [132.0, 134.0, 141.0],
        "vein_density": 1.0,
        "vein_width": 0.020,
        "mottle": 0.055,
        "polish": 0.06,
    },
    "concrete_floor": {
        "base_rgb": [104.0, 103.0, 99.0],
        "mottle": 0.10,
        "speck_density": 0.0016,
        "speck_darkness": 0.30,
        "stain_strength": 0.14,
        "fissure_count": 1,
    },
}

BACKGROUND_LABELS = {
    "wood_floor": "木地板（暖色木纹）",
    "marble_floor": "大理石地面（浅色石纹）",
    "concrete_floor": "水泥地面（中性灰）",
    "wood_floor_mat": "木地板 + 黑色垫布",
    "marble_floor_mat": "大理石地面 + 黑色垫布",
    "concrete_floor_mat": "水泥地面 + 黑色垫布",
}


# --------------------------------------------------------------------------- #
# 噪声工具
# --------------------------------------------------------------------------- #
def _value_noise_xy(shape, cell_x, cell_y, rng):
    """双三次插值的 value noise，cell_x/cell_y 分别控制 x/y 方向的相关长度（像素）。"""
    h, w = shape
    cell_x = max(2.0, float(cell_x))
    cell_y = max(2.0, float(cell_y))
    cols = int(np.ceil(w / cell_x)) + 2
    rows = int(np.ceil(h / cell_y)) + 2
    grid = rng.random((rows, cols)).astype(np.float32)
    noise = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return noise


def fbm(shape, cell, rng, octaves=4, gain=0.5, cell_y=None):
    """分形噪声，返回 float32（数值大致在 [0,1]，略有越界）。"""
    total = np.zeros(shape, np.float32)
    amplitude = 1.0
    norm = 0.0
    for octave in range(octaves):
        scale = max(2.0, cell / (2.0 ** octave))
        scale_y = max(2.0, (cell_y or cell) / (2.0 ** octave))
        total += amplitude * _value_noise_xy(shape, scale, scale_y, rng)
        norm += amplitude
        amplitude *= gain
    return total / norm


def _normalise(field):
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-6:
        return np.zeros_like(field)
    return (field - lo) / (hi - lo)


def _blur(field, sigma):
    if sigma <= 0:
        return field
    return cv2.GaussianBlur(field, (0, 0), float(sigma))


# --------------------------------------------------------------------------- #
# 底面生成
# --------------------------------------------------------------------------- #
def make_wood_floor(shape, rng, params, px_per_mm):
    """木地板：错缝木板 + 顺纹拉伸的纹理 + 树结 + 清漆光泽。

    注意：config 里的颜色按 RGB 书写，内部图像统一用 OpenCV 的 BGR，故取一次反序。
    """
    height, width = shape
    base_rgb = np.array(params["base_rgb"], np.float32)[::-1].copy()

    width_mm = width / px_per_mm
    mean_plank_mm = 0.5 * (params["plank_width_mm"][0] + params["plank_width_mm"][1])
    plank_count = max(2, int(round(width_mm / mean_plank_mm)))
    edges = [0.0]
    for index in range(plank_count - 1):
        lo, hi = params["plank_width_mm"]
        edges.append(edges[-1] + float(rng.uniform(lo, hi)))
    edges.append(width_mm)
    edges = np.array(edges, np.float32) * px_per_mm

    image = np.zeros((height, width, 3), np.float32)
    gap_px = params["plank_gap_mm"] * px_per_mm
    gap_colour = np.array([72.0, 52.0, 34.0], np.float32)[::-1].copy()

    for index in range(plank_count):
        x0 = int(np.clip(np.floor(edges[index]), 0, width - 1))
        x1 = int(np.clip(np.ceil(edges[index + 1]), 1, width))
        if x1 - x0 < 4:
            continue
        plank_w = x1 - x0
        value = float(rng.uniform(*params["value_range"]))
        hue = float(rng.uniform(-1.0, 1.0)) * params["hue_jitter"]
        plank_rgb = base_rgb * value * np.array([1.0 + hue, 1.0, 1.0 - hue], np.float32)

        # 顺纹纹理：木板竖铺，纹理沿 y 方向拉长（y 相关长度长、x 方向短）
        grain = fbm((height, plank_w), cell=4.0, rng=rng, octaves=4,
                    gain=0.55, cell_y=plank_w * 1.6)
        fine = fbm((height, plank_w), cell=2.0, rng=rng, octaves=3,
                   gain=0.6, cell_y=plank_w * 0.55)
        texture = 0.5 * _normalise(grain) + 0.5 * _normalise(fine)

        # 年轮细线：把高频条纹取绝对值，形成一圈圈深色线
        lines = np.abs(2.0 * _normalise(grain) - 1.0)
        lines = _normalise(1.0 - lines)

        shade = 1.0 + params["grain_strength"] * (texture - 0.5) * 2.0
        shade -= params["line_strength"] * np.clip(lines - 0.72, 0.0, 1.0) / 0.28

        # 树结（沿纹理方向拉长的椭圆，带年轮环）
        for _ in range(int(params["knots_per_plank"])):
            if rng.random() < 0.45:
                continue
            cx = float(rng.uniform(0.15, 0.85)) * plank_w
            cy = float(rng.uniform(0.1, 0.9)) * height
            rx = float(rng.uniform(10.0, 20.0))
            ry = rx * float(rng.uniform(1.8, 3.2))
            yy, xx = np.mgrid[0:height, 0:plank_w].astype(np.float32)
            radius = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
            knot = np.exp(-((radius / 0.55) ** 2))
            rings = 0.55 * np.exp(-((radius / 0.85) ** 2)) * np.abs(np.cos(radius * 7.0))
            shade *= 1.0 - 0.55 * knot * 0.5 - 0.35 * rings

        # 板缝：靠近板边压暗，模拟倒角 + 缝隙
        local_x = np.arange(plank_w, dtype=np.float32)
        edge_fade = np.ones(plank_w, np.float32)
        if gap_px > 1.0:
            ramp = np.clip(local_x / (gap_px * 1.6), 0.0, 1.0)
            ramp_r = np.clip((plank_w - 1 - local_x) / (gap_px * 1.6), 0.0, 1.0)
            edge_fade = np.minimum(ramp, ramp_r)
            edge_fade = 0.55 + 0.45 * edge_fade

        plank = plank_rgb[None, None, :] * shade[:, :, None] * edge_fade[None, :, None]
        # 清漆：板中间略亮（宽幅高光）
        polish = 0.5 - 0.5 * np.cos(local_x / plank_w * np.pi * 2.0)
        plank *= (1.0 + params["varnish"] * polish)[None, :, None]
        image[:, x0:x1] = plank

        # 板端接缝（横向短接缝，错缝布置）
        for _ in range(int(rng.integers(0, 2))):
            seam_y = int(rng.uniform(0.15, 0.85) * height)
            thickness = max(2, int(round(gap_px * 0.7)))
            y0 = max(0, seam_y - thickness // 2)
            y1 = min(height, y0 + thickness)
            image[y0:y1, x0:x1] = 0.70 * image[y0:y1, x0:x1] + 0.30 * gap_colour[None, None, :]

        # 板缝暗线（板与板之间的窄缝）
        if gap_px > 1.0:
            seam_half = max(1, int(round(gap_px * 0.5)))
            for offset in range(seam_half):
                for x in (x0 - offset, x1 + offset):
                    if 0 <= x < width:
                        image[:, x] = 0.65 * image[:, x] + 0.35 * gap_colour

    # 大面积明暗（木纹整体色调起伏）
    large = fbm(shape, cell=width * 0.6, rng=rng, octaves=3, gain=0.5, cell_y=height * 0.6)
    image *= (0.94 + 0.12 * _normalise(large))[:, :, None]
    return image


def make_marble_floor(shape, rng, params, px_per_mm):
    """大理石地面：浅色石底 + 弯曲细脉 + 云雾状斑驳 + 轻微抛光反光。"""
    height, width = shape
    base = np.array(params["base_rgb"], np.float32)[::-1].copy()
    vein_rgb = np.array(params["vein_rgb"], np.float32)[::-1].copy()

    # 大尺度云雾（石底的不均匀）
    cloud = _normalise(fbm(shape, cell=width * 0.55, rng=rng, octaves=4, gain=0.55,
                           cell_y=height * 0.55))
    stone = base[None, None, :] * (0.97 + params["mottle"] * (cloud - 0.5) * 2.0)[:, :, None]

    # 脉络：把坐标用湍流扰动后取正弦，脊线处压暗
    turbulence = fbm(shape, cell=width * 0.22, rng=rng, octaves=5, gain=0.55,
                     cell_y=height * 0.22)
    turbulence = _normalise(_blur(turbulence, 6.0))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    warp = 620.0 * (turbulence - 0.5)
    primary = np.sin(((xx * 0.82 + yy * 0.38) + warp) * params["vein_density"] * 0.0034)
    veins = (1.0 - np.abs(primary)) ** 9
    secondary = np.sin(((xx * 0.30 - yy * 0.95) + 0.6 * warp) * params["vein_density"] * 0.0075)
    veins += 0.45 * (1.0 - np.abs(secondary)) ** 14
    veins = np.clip(veins, 0.0, 1.0)
    veins = _blur(veins, 1.6)

    # 细若发丝的次级脉络
    hair = np.abs(np.sin((xx * 0.6 + yy * 0.6) * 0.011 + 1.7 * turbulence))
    hair = np.clip((hair - 0.985) / 0.015, 0.0, 1.0)
    veins = np.clip(veins + 0.35 * hair, 0.0, 1.0)

    width_map = np.clip(veins * params["vein_width"] * 12.0, 0.0, 1.0)
    stone = stone * (1.0 - width_map[:, :, None]) + vein_rgb[None, None, :] * width_map[:, :, None]

    # 抛光：低频镜面高光
    polish = _normalise(fbm(shape, cell=width * 0.9, rng=rng, octaves=2, gain=0.5,
                           cell_y=height * 0.9))
    stone *= (1.0 + params["polish"] * (polish - 0.5))[:, :, None]
    return stone


def make_concrete_floor(shape, rng, params, px_per_mm):
    """水泥地面：中性灰 + 细砂粒 + 骨料暗点 + 污渍 + 细裂缝。"""
    height, width = shape
    base = np.array(params["base_rgb"], np.float32)[::-1].copy()

    mottle = _normalise(fbm(shape, cell=width * 0.30, rng=rng, octaves=5, gain=0.55))
    grit = _normalise(fbm(shape, cell=6.0, rng=rng, octaves=2, gain=0.5))
    image = base[None, None, :] * (1.0 + params["mottle"] * (mottle - 0.5) * 2.0
                                   + 0.16 * (grit - 0.5))[:, :, None]

    # 骨料暗点
    density = params["speck_density"]
    count = int(density * height * width)
    ys = rng.integers(1, height - 1, size=count)
    xs = rng.integers(1, width - 1, size=count)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = 1.0 if (dy == 0 and dx == 0) else 0.45
            yy = np.clip(ys + dy, 0, height - 1)
            xx = np.clip(xs + dx, 0, width - 1)
            image[yy, xx] *= 1.0 - params["speck_darkness"] * w

    # 污渍（椭圆状淡色变深）
    stains = np.clip(_normalise(fbm(shape, cell=width * 0.30, rng=rng, octaves=3)) - 0.62, 0, 1) / 0.38
    image *= (1.0 - params["stain_strength"] * stains)[:, :, None]

    # 细裂缝
    for _ in range(int(params["fissure_count"])):
        current = np.array([rng.uniform(0, width), rng.uniform(0, height)], np.float32)
        angle = rng.uniform(0, np.pi * 2)
        points = [current.copy()]
        for _step in range(int(rng.integers(18, 40))):
            angle += rng.normal(0.0, 0.35)
            current = current + np.array([np.cos(angle), np.sin(angle)], np.float32) * rng.uniform(12, 30)
            if not (0 <= current[0] < width and 0 <= current[1] < height):
                break
            points.append(current.copy())
        if len(points) > 3:
            crack = np.zeros((height, width), np.float32)
            cv2.polylines(crack, [np.round(np.array(points)).astype(np.int32)], False, 1.0,
                          thickness=int(rng.integers(1, 3)), lineType=cv2.LINE_AA)
            crack = _blur(crack, 0.8)
            image *= (1.0 - 0.30 * crack)[:, :, None]
    return image


GENERATORS = {
    "wood_floor": make_wood_floor,
    "marble_floor": make_marble_floor,
    "concrete_floor": make_concrete_floor,
}
MAT_SUFFIX = "_mat"


def generator_for(name):
    """``*_mat`` 变体复用同一种底面，只是多铺一块垫布。"""
    base = name[: -len(MAT_SUFFIX)] if name.endswith(MAT_SUFFIX) else name
    if base not in GENERATORS:
        raise KeyError(name)
    return GENERATORS[base]


def scene_parameters(name, config):
    """把 ``<floor>_mat`` 解析成“底面参数 + 垫布参数”。"""
    if not name.endswith(MAT_SUFFIX):
        return dict(config[name])
    base = name[: -len(MAT_SUFFIX)]
    params = dict(config[base])
    mat = dict(config.get("mat_default") or {})
    mat.update((config.get(name) or {}).get("mat") or {})
    params["mat"] = mat
    return params


# --------------------------------------------------------------------------- #
# 照明 / 噪声 / 合成
# --------------------------------------------------------------------------- #
def lighting_field(shape, rng, params):
    """返回两帧共用的基础照明场，再各自叠加抖动（见 apply_lighting）。"""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    theta = np.deg2rad(float(rng.uniform(0.0, 360.0)))
    gradient = np.cos(theta) * (xx / width - 0.5) + np.sin(theta) * (yy / height - 0.5)
    radius = np.sqrt((xx / width - 0.5) ** 2 + (yy / height - 0.5) ** 2) / 0.7071
    return {
        "gradient": gradient,
        "vignette": np.clip(radius, 0.0, 1.0) ** 2,
        "angle": theta,
    }


def apply_lighting(image, field, rng, params, gain=None, return_modulation=False):
    """按照明场调制亮度，并加一点增益抖动（两次调用参数不同 → 两帧灯光微变）。"""
    jitter = float(rng.uniform(-1.0, 1.0))
    vignette = params["vignette"] * (1.0 + jitter * params["vignette_jitter"])
    angle_jitter = np.deg2rad(float(rng.uniform(-1.0, 1.0)) * params["gradient_angle_jitter_deg"])
    height, width = image.shape[:2]
    x = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :]
    y = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None]
    theta = field["angle"] + angle_jitter
    gradient = np.cos(theta) * x + np.sin(theta) * y
    gain = float(gain if gain is not None else 1.0 + params["frame_gain_jitter"] * jitter)
    modulation = (1.0 + params["gradient"] * gradient - vignette * field["vignette"]) * gain
    lit = image * modulation[:, :, None]
    if return_modulation:
        return lit, modulation
    return lit


def add_sensor_noise(image, rng, params):
    sigma = params["noise_sigma"]
    if sigma <= 0:
        return image
    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return image + noise


def _polygon_mask(shape, polygons, supersample=4, blur_px=0.0):
    height, width = shape
    canvas = np.zeros((height * supersample, width * supersample), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [np.round(np.asarray(polygon, np.float32) * supersample).astype(np.int32)], 255)
    alpha = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if blur_px > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_px)
    return np.clip(alpha, 0.0, 1.0)


def unmultiply_pieces(pieces_rgb, empty_rgb, alpha):
    """把“纸片叠在黑底面上”的观测还原成纸片自身的颜色（去底面污染）。

    观测: observed = alpha * piece + (1 - alpha) * background
    =>    piece   = (observed - (1 - alpha) * background) / alpha
    """
    safe_alpha = np.maximum(alpha, 0.20)[:, :, None]
    background = empty_rgb.astype(np.float32)
    piece = (pieces_rgb.astype(np.float32) - (1.0 - alpha[:, :, None]) * background) / safe_alpha
    return np.clip(piece, 0.0, 255.0)


def composite_scene(background, pieces_rgb, empty_rgb, polygons, rng, params,
                    piece_modulation=None):
    """把真实纸片合成到生成的底面上，返回 (带碎片的场景, alpha)。

    piece_modulation: 与场景同尺寸的亮度场，用于让纸片跟随该帧的照明（均值归一化后使用）。
    """
    shape = background.shape[:2]
    alpha = _polygon_mask(shape, polygons, supersample=4, blur_px=params["edge_blur_px"])
    piece_rgb = unmultiply_pieces(pieces_rgb, empty_rgb, alpha)
    if piece_modulation is not None:
        piece_rgb = piece_rgb * piece_modulation[:, :, None]

    # 合成阴影：软阴影（偏移）+ 接触阴影（贴边），只作用于新底面
    union = _polygon_mask(shape, polygons, supersample=2, blur_px=0.0)
    offset_x, offset_y = params["shadow_offset_px"]
    matrix = np.array([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y]], np.float32)
    shifted = cv2.warpAffine(union, matrix, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                             borderValue=0.0)
    soft = _blur(shifted, params["soft_sigma_px"]) * params["soft_strength"]
    contact = _blur(union, params["contact_sigma_px"]) * params["contact_strength"]
    occlusion = np.clip(0.55 * soft + contact, 0.0, 0.95)

    tint = np.array(params["shadow_tint"], np.float32)[None, None, :]
    shadowed_background = background * (1.0 - occlusion[:, :, None] * tint)

    scene = piece_rgb * alpha[:, :, None] + shadowed_background * (1.0 - alpha[:, :, None])
    return scene, alpha


# --------------------------------------------------------------------------- #
# 场景构建
# --------------------------------------------------------------------------- #
def load_pieces(pieces_json_path):
    payload = read_json(Path(pieces_json_path))
    pieces = payload["pieces"]
    return [
        {
            "piece_id": piece["piece_id"],
            "corners_px": [[float(x), float(y)] for x, y in piece["corners_px"]],
            "center_px": [float(v) for v in piece["center_px"]],
            "area_px": float(piece["area_px"]),
        }
        for piece in pieces
    ]


def _portable(path):
    """仓库内的路径写成仓库相 POSIX 路径，仓库外保留绝对路径。"""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _mat_alpha(shape, polygons, mat, px_per_mm):
    """垫布覆盖区域（铺在碎片下方的整块矩形绒布，边缘羽化）的 0..1 alpha。"""
    height, width = shape
    points = np.vstack([np.asarray(polygon, np.float64) for polygon in polygons])
    margin = float(mat.get("margin_mm", 20.0)) * px_per_mm
    x0, y0 = points.min(axis=0) - margin
    x1, y1 = points.max(axis=0) + margin
    x0, y0 = int(np.clip(np.floor(x0), 0, width - 1)), int(np.clip(np.floor(y0), 0, height - 1))
    x1, y1 = int(np.clip(np.ceil(x1), 1, width)), int(np.clip(np.ceil(y1), 1, height))
    mask = np.zeros((height, width), np.float32)
    mask[y0:y1, x0:x1] = 1.0
    feather = max(float(mat.get("feather_mm", 2.0)) * px_per_mm, 0.5)
    return _blur(mask, feather)


def _add_mat(background, polygons, mat, rng, px_per_mm):
    """在底面之上铺一块深色绒布垫布（真实采集时桌面上铺的就是黑绒布）。

    垫布像素来自碎片的真值多边形外扩，所以两帧（empty / pieces）都带同样的垫布；
    深色垫布让纸片与底面保持高对比，而落在垫布上的投影只改变很小的绝对亮度。
    """
    alpha = _mat_alpha(background.shape[:2], polygons, mat, px_per_mm)[:, :, None]
    cell_x, cell_y = mat.get("texture_cell_px", [5.0, 14.0])
    weave = fbm(background.shape[:2], cell=float(cell_x), rng=rng, cell_y=float(cell_y))
    base = np.array(mat["base_rgb"], np.float32)[::-1]
    strength = float(mat.get("texture_strength", 0.10))
    mat_rgb = base[None, None, :] * (1.0 + strength * (weave[:, :, None] - 0.5) * 2.0)
    return background * (1.0 - alpha) + mat_rgb * alpha


def build_scene(name, config, pieces, output_dir, rng, pieces_rgb, empty_rgb):
    params = scene_parameters(name, config)
    lighting = config["lighting"]
    shape = (int(config["canvas_px"][1]), int(config["canvas_px"][0]))
    px_per_mm = float(config["px_per_mm"])
    polygons = [piece["corners_px"] for piece in pieces]

    background = generator_for(name)(shape, rng, params, px_per_mm)
    mat = params.get("mat")
    if mat:
        background = _add_mat(background, polygons, mat, rng, px_per_mm)
    field = lighting_field(shape, rng, lighting)

    # 两帧：几何完全一致，照明各自抖动一次（模拟灯光微变），最后各自加独立噪声
    empty_frame, _ = apply_lighting(background, field, rng, lighting, return_modulation=True)
    pieces_background, modulation = apply_lighting(background, field, rng, lighting,
                                                   return_modulation=True)
    piece_modulation = modulation / max(float(modulation.mean()), 1e-6)

    scene, alpha = composite_scene(pieces_background, pieces_rgb, empty_rgb, polygons, rng,
                                   config["piece_composite"], piece_modulation=piece_modulation)

    empty_frame = add_sensor_noise(empty_frame, rng, lighting)
    scene = add_sensor_noise(scene, rng, lighting)
    empty_frame = np.clip(empty_frame, 0, 255).astype(np.uint8)
    scene = np.clip(scene, 0, 255).astype(np.uint8)

    scene_dir = Path(output_dir) / name
    imwrite(scene_dir / "empty.png", empty_frame)
    imwrite(scene_dir / "pieces.png", scene)
    imwrite(scene_dir / "alpha.png", np.round(alpha * 255.0).astype(np.uint8))

    payload = {
        "schema_version": config["schema_version"],
        "generator_version": GENERATOR_VERSION,
        "background": name,
        "label": BACKGROUND_LABELS.get(name, name),
        "seed": config["seed"],
        "canvas_px": [shape[1], shape[0]],
        "px_per_mm": px_per_mm,
        "inputs": {
            "pieces_png": _portable(config["_pieces_png"]),
            "empty_png": _portable(config["_empty_png"]),
            "pieces_json": _portable(config["_pieces_json"]),
        },
        "rendered_with": {
            "lighting": lighting,
            "piece_composite": config["piece_composite"],
            "background_parameters": params,
        },
        "pieces": pieces,
    }
    write_json(scene_dir / "scene.json", payload)
    return scene, empty_frame, pieces


def preview_sheet(scenes, output_path, tile=440):
    """把各背景的场景图拼成一张预览图。"""
    tiles = []
    for name, scene in scenes:
        height, width = scene.shape[:2]
        scale = tile / float(max(height, width))
        small = cv2.resize(scene, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA)
        canvas = np.full((tile + 30, tile, 3), 245, np.uint8)
        y = 30 + (tile - small.shape[0]) // 2
        x = (tile - small.shape[1]) // 2
        canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
        cv2.putText(canvas, name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2,
                    cv2.LINE_AA)
        tiles.append(canvas)
    sheet = np.hstack(tiles)
    imwrite(Path(output_path), sheet)
    return sheet


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成拼图底面背景与鲁棒性场景")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pieces-png", default=str(DEFAULT_PIECES_PNG))
    parser.add_argument("--empty-png", default=str(DEFAULT_EMPTY_PNG))
    parser.add_argument("--pieces-json", default=str(DEFAULT_PIECES_JSON))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--preview", default=None)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if config_path.exists():
        config = dict(DEFAULT_CONFIG_VALUES)
        config.update(read_json(config_path))
    else:
        config = dict(DEFAULT_CONFIG_VALUES)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    config["_pieces_png"] = Path(args.pieces_png)
    config["_empty_png"] = Path(args.empty_png)
    config["_pieces_json"] = Path(args.pieces_json)

    names = list(config["backgrounds"])
    if args.only:
        known = set(GENERATORS) | {base + MAT_SUFFIX for base in GENERATORS}
        unknown = [name for name in args.only if name not in known]
        if unknown:
            raise SystemExit(f"未知背景: {unknown}；可用: {sorted(known)}")
        names = list(args.only)

    pieces = load_pieces(args.pieces_json)
    pieces_rgb = imread(Path(args.pieces_png))
    empty_rgb = imread(Path(args.empty_png))
    if pieces_rgb.shape != empty_rgb.shape:
        raise SystemExit("pieces.png 与 empty.png 尺寸不一致")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for index, name in enumerate(names):
        rng = np.random.default_rng(int(config["seed"]) + index * 1013)
        scene, empty, _pieces = build_scene(name, config, pieces, output_dir, rng,
                                            pieces_rgb, empty_rgb)
        rendered.append((name, scene))
        print(f"[{name}] {BACKGROUND_LABELS.get(name, name)} -> {output_dir / name}")

    preview_path = Path(args.preview) if args.preview else output_dir / "preview.png"
    preview_sheet(rendered, preview_path)
    print(f"preview -> {preview_path}")
    print(f"scenes: {len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
