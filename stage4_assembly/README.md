# Stage 4 - Assembly

输入：

- 阶段一的 `jigsaw_layout.json` 和原始参考图
- 阶段二的碎片 RGBA 和 masks
- 阶段三的 `assignments.json`

任务：

1. 根据 `H_scan_to_source` 把碎片变换回源坐标系。
2. 按源多边形掩码粘贴到完整画布。
3. 在 Lab 空间做保守颜色校正。
4. 对多边形接缝做羽化或 multi-band blending。
5. 输出 SSIM、PSNR 和接缝报告。

输出：

- `output/assembled.png`
- `output/assembly_report.json`
- `output/seams.png`

当前状态：目录和交接契约已建立，拼接脚本待实现。

