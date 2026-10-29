# Stage 2 output

`pieces_base_edge.json` 是阶段三的**交接主格式**，`pieces.json` 是同一批碎片的完整记录。
运行 `python stage2_piece_extraction/extract_pieces.py` 会写出：

- `pieces_base_edge.json`：16 条最小记录
  `{"id": "S01", "baseLength": 561.79, "p2": [505.14, 252.99], "p3": [6.29, 258.41]}`，
  全精度、无四舍五入。`baseLength` 是最长边的真实长度（整流画布像素），基边起点
  `P0=(0,0)`、终点 `P1=(baseLength,0)`，`P2`/`P3` 在基边下方，四点在 x 向右、y 向下
  的坐标系里顺时针；所有碎片共用同一单位（`unit_scale: 1.0`）。
- `pieces.json`：交接契约（`schema_version`、`coordinate_system="rectified_px"`、
  `piece_frame`、`piece_scale`、`piece_margin_px`，`pieces[]` 的 `piece_id` / `rgba_path` /
  `mask_path` / `corners_px` / `center_px` / `area_px` / `quality`，外加 `base_edge`、
  `base_edge_corners_px`、`base_length_px`、`base_side_index`、`piece_size_px`、
  `image_corners_px`、`rectified_to_piece_matrix`、`rectified_to_base_edge_matrix`、
  `quality_breakdown`、`manual_review_reasons` 等扩展字段）。
- `pieces/S01_rgba.png` … `S16_rgba.png`：RGBA，透明背景，尺寸逐片不同（如 614×307），
  与整流画布 1:1；
- `pieces/S01_mask.png` … `S16_mask.png`：0/255 二值掩码，与 rgba 的 alpha 逐像素一致；
- `pieces/S01_debug.png` … `S16_debug.png`：棋盘格底 + 基边裁剪轮廓 + `P0..P3` 编号，
  基边（P0–P1）画成橙色，供人工复核。

阶段三消费方式：

- `rgba_path` / `mask_path` / `debug_path` 是**相对本目录**（`pieces.json` 所在目录）的 POSIX 路径；
- 每片已放在自己的基边坐标系里（长边沿 +x、起点在原点），`rotation_deg` 是基边相对整流画布
  x 轴的旋转角，只作搜索范围提示，**不代表**与源碎片 `P01..P16` 的对应关系——对应关系由
  阶段三的特征匹配给出；
- 裁剪是 1:1 的，没有缩放：`baseLength` 可直接跨片比较，`piece_size_px` 只是该片裁剪画布大小；
- 最薄的三片（S06/S14/S13）短边为 66/119/163 px，这是 8 px/mm 整流图里纸片本身的尺寸；
  需要更高分辨率纹理时，可回到 `pieces_rectified.png` 用 `corners_px` 在原生分辨率裁剪。

本目录产物随仓库提交（便于下游直接复现实验）；重新运行
`python stage2_piece_extraction/extract_pieces.py` 可随时重建，同一输入两次运行逐字节相同。
