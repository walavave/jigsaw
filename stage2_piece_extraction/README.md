# Stage 2 - Piece extraction

输入来自阶段一：

- `stage1_capture_calibration/output/rectified/pieces_rectified.png`
- `stage1_capture_calibration/output/rectified/empty_rectified.png`
- `stage1_capture_calibration/output/rectified/rectification.json`
- `stage1_capture_calibration/output/cut_pdf/jigsaw_layout.json`

任务：

1. 用空桌面图做背景差分。
2. 分割 16 个直边凸多边形碎片。
3. 拟合多边形角点并做局部透视归一化。
4. 分配扫描编号 `S01..S16`，与源编号 `P01..P16` 无关。

输出：

- `output/pieces.json`
- `output/pieces/S01_rgba.png`（RGBA，透明背景）
- `output/pieces/S01_mask.png`（0/255 二值掩码，与 rgba 的 alpha 一致）
- `output/pieces/S01_debug.png`（棋盘格底 + 归一化轮廓 + 角点编号，便于人工复核）

当前状态：已实现（`extract_pieces.py`，算法版本 `stage2-extract-1.2.1`）。

## 运行

```bash
# 默认输入/输出（仓库内已提交的阶段一结果）
python stage2_piece_extraction/extract_pieces.py

# 指定输入输出
python stage2_piece_extraction/extract_pieces.py \
  --pieces stage1_capture_calibration/output/rectified/pieces_rectified.png \
  --empty  stage1_capture_calibration/output/rectified/empty_rectified.png \
  --rectification stage1_capture_calibration/output/rectified/rectification.json \
  --layout stage1_capture_calibration/output/cut_pdf/jigsaw_layout.json \
  --config stage2_piece_extraction/config/extraction.example.json \
  --output stage2_piece_extraction/output
```

退出码：`0` = 检测数量符合预期（`manual_review` 非空只是告警，结果照常写出）；`1` = 检测数量与 `expected_piece_count` 不符，或 `fail_on_manual_review` 为真且存在需要人工复核的碎片。`manual_review` 的内容、每片的 `needs_manual_review` 与 `manual_review_reasons` 都会写进 stdout 和 `pieces.json`。

复核（两条命令，都不需要 GPU、不需要相机）：

```bash
python -m unittest discover -s tests -v   # 34 项，应全绿
python docs/verify_stage2.py              # 59 项，预期 2 FAIL / 7 WARN、退出码 1（判读口径见报告 §5.4）
```

参数全部在 `config/extraction.example.json` 里，缺失的键用 `extract_pieces.py` 内 `DEFAULT_CONFIG_VALUES` 的默认值补齐。关键项：

| 键 | 含义 |
| --- | --- |
| `expected_piece_count` | 期望碎片数（本项目 16），数量不符直接判失败 |
| `ignore_border_mm` | 画布四周忽略的环形区域（mm）。阶段一画布 330 mm 宽、8 px/mm；本夹具碎片距画布边缘最近约 15 mm，故默认 5 mm，不能用模板里的 45 mm |
| `min_piece_area_px` | 连通域面积下限，过滤灰尘/反光噪点 |
| `min_quality` | 质量阈值，低于该值的碎片写入 `manual_review` |
| `threshold_mode` / `threshold_min` / `threshold_max` | 前景阈值策略：`otsu` 时把 Otsu 结果夹到 `[min, max]`。纸面印刷内容很暗，Otsu 会偏高，实测需夹到 25 才不会啃掉碎片边缘 |
| `open_ksize` / `close_ksize` | 形态学开闭核尺寸 |
| `piece_output_size_px` / `piece_margin_px` | 归一化画布尺寸与留白（本项目 512 / 20 px） |
| `approx_epsilon_ratio` | 粗多边形 `approxPolyDP` 的 epsilon（按周长比例） |
| `edge_sample_count` / `edge_profile_px` / `edge_profile_step_px` | 每条边沿法向采样的点数、半宽、步长（亚像素角点用） |
| `edge_localizer` | 边缘亚像素定位方式：`gradient_max`（默认，取剖面梯度绝对值最大处再抛物线细化，落在纸张真实边界上）或 `half_level`（1.0.0 的旧行为，取内/外中位数的半高交点，会向碎片内侧偏约 0.6 px） |
| `min_edge_contrast` / `min_edge_gradient` | `half_level` 的对比度门限 / `gradient_max` 的梯度峰值门限（响应单位/px） |
| `max_side_shift_px` | 单边保护：`gradient_max` 拟合出的某条边若与同一条边的 `half_level` 结果相差超过该值，就退回 `half_level` 的边（短边/弱对比边不会因为峰选错而整条边飞走） |
| `edge_fit_residual_px` / `max_corner_shift_px` | 直线拟合残差上限、角点相对粗多边形允许的位移上限 |
| `fail_on_manual_review` | 为真时 `manual_review` 非空即返回退出码 1（默认 `false`：质量标志是交接提示，不是流水线失败） |

## 算法

1. **背景差分**：`colour_response` 取两张图（同一相机、同一光照、同一桌面）的 Lab 欧氏距离；`edge_response` 在此基础上再加 0.5×（Laplacian 差分）。掩码用 `colour_response`（Laplacian 项会把 Otsu 阈值抬高）；`edge_response` 只给 `half_level` 定位用——`gradient_max` 改在 `colour_response` 上跑，因为 Laplacian 项会把梯度峰值往碎片内侧拉约 0.7 px（边缘偏移的证据见 `docs/REPRODUCTION_STAGE2.md` §5.1）。
2. **二值掩码**：Otsu（夹到 `[threshold_min, threshold_max]`）→ 开/闭运算 → 连通域 → 按面积/边界/凸性过滤，得到 16 个候选。
3. **亚像素角点**：
   - `approxPolyDP` 在凸包上生成长度 ≤ `max_vertices` 的粗多边形（3..6 边）；
   - 对每条边沿外法向在半宽 `edge_profile_px` 内双线性采样响应剖面；`edge_localizer="gradient_max"` 时在 `colour_response` 剖面取梯度的绝对值最大点（用相邻三点抛物线细化到亚像素），即纸张的真实边界；`"half_level"` 时取内/外两侧中位数之半高交点（印刷内容会让剖面非单调，所以不是"唯一交点"而是"最近交点"）；
   - 每条边用 `cv2.fitLine` + 残差剔除拟合出单位法向直线 `a·x+b·y+c=0`；
   - **单边保护**（`_merge_side_lines`）：`gradient_max` 的某条边若与同一条边的 `half_level` 结果相差超过 `max_side_shift_px`（默认 3 px），说明该短边/弱对比边峰选偏了，改用 `half_level` 的边；
   - 相邻直线求交得到亚像素角点；某条边拟合失败或交点相对粗角点位移超过 `max_corner_shift_px` 时退回粗角点；
   - 角点排序后做凸性检查与近共线顶点剔除（`prune_near_collinear`），保证 3..6 边。
   - 用 `gradient_max` 时同一组边还会用 `half_level` 再拟合一遍，两套角点的平均偏差记为 `agreement_shift_px`，作为"该片角点是否可信"的独立旁证（不参与加权质量分）。
4. **归一化**：`normalization_matrix` 用角点二阶矩主轴（特征向量取 x≥0 的符号）作为规范朝向，转正后以统一尺度缩放到 `piece_output_size_px` 画布、按旋转后包围盒中心居中、四周留 `piece_margin_px`。用 `warpPerspective` 正向变换（OpenCV 内部取逆）把原图与解析多边形一起搬到局部画布，掩码用解析多边形填充（直边，不掺桌面像素）。
5. **编号与质量**：按扫描顺序分配 `S01..S16`——先把中心 y 相差在 `SCAN_ROW_TOLERANCE_PX`（2 px）以内的碎片归为同一行，行自上而下、行内自左向右（`scan_order_indices`）；这样碎片在亚像素级上下抖动时编号不会乱跳。质量分 `0.35×fit + 0.25×shape + 0.20×support + 0.10×vertices + 0.10×convex`，其中 `fit = exp(-rms/3px)`（`gradient_max`；`half_level` 用 2 px，因为梯度峰值的抖动更大）、`support = min(1, 有效采样比例/0.6)`、`shape = 细化多边形与粗掩膜的 IoU`，各项都写进 `quality_breakdown`。`needs_manual_review` 在以下任一条件成立时为真并附 `manual_review_reasons`：`quality < min_quality`（`quality_below_min`）、最短边 < 100 px（`short_side`）、两套定位方法角点偏差 > 3 px（`localizer_disagreement`）。

## 交接契约（pieces.json）

必需字段（README §4.2）：`schema_version`、`coordinate_system="rectified_px"`、`piece_size_px=[512,512]`、`pieces[]`：
`piece_id`、`rgba_path`、`mask_path`、`corners_px`、`center_px`、`area_px`、`quality`。

实现额外提供（阶段三/四可选用）：

- 顶层：`algorithm_version`、`session_id`（输入字节 + 配置 + 算法版本的 sha256 前 16 位，**不含时间戳**，同一输入两次运行输出逐字节相同）、`canvas_px`、`px_per_mm`、`corner_order`、`expected_piece_count`、`detected_piece_count`、`inputs`、`segmentation`（阈值、前景占比、被丢弃候选）、`summary`（质量/面积统计）、`config` 回显、`manual_review`。
- 每片：`debug_path`、`corner_count`、`area_mm2`、`bbox_px`、`perimeter_px`、`quality_breakdown`（含 `fit_rms_px`、`edge_support`、`mask_iou`、`min_side_px`、`agreement_shift_px`）、`needs_manual_review`、`manual_review_reasons`、`normalized_corners_px`、`normalized_short_side_px`、`rotation_deg`、`rectified_to_piece_matrix`（3×3，整流画布 → 局部画布）。

约定：

- **路径解析**：`rgba_path` / `mask_path` / `debug_path` 是相对 `pieces.json` 所在目录（即 `output/`）的 POSIX 路径，换个地方放只需整体搬 `output/`；`inputs` 里的四个路径是**仓库根相对**的 POSIX 路径（仓库外的输入才退化为绝对路径），因此 JSON 不绑定某台机器。
- 所有坐标都是整流画布像素坐标，x 向右、y 向下（README §2.1）。
- `corners_px` 的绕向为"y 向下像素坐标系下鞋带面积为正"（即屏幕上顺时针），起点取最上、其次最左的顶点；`corner_order` 字段里有原文说明。
- `S01..S16` 只表示扫描顺序（行自上而下、行内自左向右，同行判定容差 `SCAN_ROW_TOLERANCE_PX = 2 px`），不代表源碎片 `P01..P16`；对应关系由阶段三的特征匹配给出。
- 归一化只做相似变换（等比缩放 + 旋转 + 平移），不做透视校正：碎片本身是平的，阶段一已经把它整流到桌面平面，再做透视只会引入插值误差。`rotation_deg` 是规范朝向相对画布坐标系的旋转角，阶段三可据此缩小搜索范围。

## 当前结果（本仓库数据）

`python stage2_piece_extraction/extract_pieces.py`：

```text
Pieces: 16/16 (canvas 2640x2640, normalized 512x512)
Segmentation: threshold=25.0 foreground=33.73% border_ignore=40px
Quality: min=0.8894 median=0.9248
Area px: min=114134 max=165124 total=2313710
Manual review: S06 (short_side)
```

- 16 片全部为 4 边形（与 `jigsaw_layout.json` 的 16 个 4 边形一致），每片 `fit_rms` 0.55–1.07 px，质量 0.889–0.935。`manual_review` 只有 `S06`，原因是最短边 66 px（`short_side`）；它的质量 0.890 本身高于 0.85，属于"这一片几何上确实最难"的交待，不是流水线失败。
- 检测总面积 2,313,710 px²，比按打印比例换算的期望值 2,296,369 px² 大 **+0.76%**；同时拟合尺度比 PDF 期望大 +0.34%。两者一致说明碎片是沿印刷线外侧约 0.2–0.24 mm 剪下来的——这是真实几何，不是提取误差（1.1.0 的 +0.03% 反而是 0.69 px 内缩抵消出来的假精度）。
- 独立验证（`docs/verify_stage2.py`，59 项检查，只读仓库、只写 `tmp/`，不参与交付）：以与所报多边形无关的 `grad-quad`（每边 13 条剖面的梯度最大值拟合直线 + 相邻求交）为参考，64 条边的垂直偏移有符号均值/中位数为 **+0.047 / +0.055 px**（无系统偏差），绝对值均值 0.587 / 中位 0.364 / p90 1.231 / max 3.622 px，55/64 ≤ 1 px；角点误差 mean 1.583 / median 0.859 / max 9.281 px，最大的几项都是被 `1/sin(θ/2)` 放大的锐角（S06 的 66 px 短边 + 53.8° 角）。与 1.0.0 相比 16 片里 13 片严格变好、没有一片变差；编号 `S14/S15` 交替的问题已随行聚类排序消失。跑法与判读口径（预期 `59 checks, 2 FAIL, 7 WARN`、退出码 1，以及为什么那两个 FAIL 与提取器无关）见 `docs/REPRODUCTION_STAGE2.md` §5。
- 确定性：同一输入连续两次运行，`output/` 下 50 个文件 SHA256 全部相同。
- 人工检查：`docs/stage2_preview.png`（16 片归一化缩略图）与每片 `S*_debug.png` 显示轮廓贴合纸张边界、无桌面残留。
- 测试：`python -m unittest discover -s tests -v` 34 项全绿（其中 `tests/test_stage2_extraction.py` 22 项，含合成 16 片端到端、真实数据集成与确定性）。
- 归一化后的短边（`normalized_short_side_px`）：中位 220 px，最薄的三片是 S06 = 45 px、S14 = 76 px、S13 = 163 px，**低于 README §2.1 建议的“短边 ≥ 300–400 px”**。原因是“装进 512 画布”的等比缩放按长边定标（缩放 0.69–0.95），薄片的短边被一起缩小；在 8 px/mm 的整流图里 S06 的原生短边本来就只有 66 px，再怎么归一化也到不了 300 px。需要高分辨率纹理时，Stage 3 应直接用 `pieces_rectified.png` + `corners_px`（或 `rectified_to_piece_matrix` 的逆矩阵）在原生分辨率上裁剪，而不是放大 512 画布。

## 已知限制

- 阈值依赖 `pieces_rectified.png` 与 `empty_rectified.png` 严格同机位、同光照（阶段一保证了这一点）；若重新拍摄而桌面反光变化较大，需要重新标定 `threshold_max`。
- 碎片若相互接触或叠放会合并成一个连通域，导致检测数 < 16，此时以退出码 1 结束并打印数量告警；`segmentation.discarded_candidates` 与面积分布可用于定位问题。
- 归一化朝向由形状主轴决定：近似正方形/圆形的碎片刻度上主轴不稳定（本夹具都是细长四边形，未受影响）；阶段三仍然需要对旋转做搜索，因此不影响正确性。
- 目前只做了相似变换归一化，未实现"局部透视归一化"。若后续用倾斜相机拍摄，需要把 `normalization_matrix` 换成基于四角单应的 `getPerspectiveTransform`（对角线选取需固定规则）。当前阶段一已输出正射整流图，相似变换等价且误差更小，故未采用。
- 亚像素精度的物理下限来自手剪：本次数据里单条边的直线度（同一侧采样点的 p90−p10）为 1.6–5.8 px，最短边只有 66 px，最锐角 53.8°，角点误差等于边缘偏移被 `1/sin(θ/2)` 放大（典型 ×2.7），因此 S06 残留 5–9 px 的角点误差是数据本身的属性，不是算法没调好。
- `max_side_shift_px` 的单边保护在本数据集上一次都没触发（两种定位方法在 64 条边上相差都 < 1.5 px，它们修好的那两条坏边靠的是把 `gradient_max` 换到 Lab 颜色响应上）；它保留为换数据时的廉价保险。
