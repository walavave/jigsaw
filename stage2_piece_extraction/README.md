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
- `output/pieces/S01_rgba.png`
- `output/pieces/S01_mask.png`
- `output/pieces/S01_debug.png`

当前状态：目录和交接契约已建立，处理脚本待实现。

