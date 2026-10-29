# Stage 2 output

`pieces.json` 是阶段三唯一的碎片清单来源。运行
`python stage2_piece_extraction/extract_pieces.py` 会写出：

- `pieces.json`：16 片的交接契约（`schema_version`、`coordinate_system="rectified_px"`、
  `piece_size_px=[512,512]`、`pieces[]` 的 `piece_id` / `rgba_path` / `mask_path` / `corners_px` /
  `center_px` / `area_px` / `quality`，外加 `normalized_corners_px`、`rotation_deg`、
  `rectified_to_piece_matrix`、`quality_breakdown`、`manual_review_reasons` 等扩展字段）。
- `pieces/S01_rgba.png` … `S16_rgba.png`：512×512 RGBA，透明背景；
- `pieces/S01_mask.png` … `S16_mask.png`：512×512 0/255 二值掩码，与 rgba 的 alpha 逐像素一致；
- `pieces/S01_debug.png` … `S16_debug.png`：棋盘格底 + 归一化轮廓 + 角点编号，供人工复核。

阶段三消费方式：

- `rgba_path` / `mask_path` / `debug_path` 是**相对本目录**（`pieces.json` 所在目录）的 POSIX 路径；
- 每片已是规范朝向（形状主轴转正）的 512×512 图，`rotation_deg` 是相对整流画布坐标系的旋转角，
  只作搜索范围提示，**不代表**与源碎片 `P01..P16` 的对应关系——对应关系由阶段三的特征匹配给出；
- 薄片（S06 短边 45 px、S14 76 px）的归一化短边小于 README §2.1 建议的 300–400 px；
  需要原生分辨率纹理时改用 `pieces_rectified.png` + `corners_px` 裁剪（见 `../README.md`）。

本目录产物为本地生成、未提交（可用 `python stage2_piece_extraction/extract_pieces.py`
随时重建；同一输入两次运行逐字节相同）。
