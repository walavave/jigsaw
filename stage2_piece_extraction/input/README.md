# Stage 2 input

脚本默认**直接读阶段一的仓库内结果**，不需要先把文件拷到这里：

- `stage1_capture_calibration/output/rectified/pieces_rectified.png`
- `stage1_capture_calibration/output/rectified/empty_rectified.png`
- `stage1_capture_calibration/output/rectified/rectification.json`
- `stage1_capture_calibration/output/cut_pdf/jigsaw_layout.json`

这个目录保留给两种情况：把一份输入打包给别人（把上面四份拷进来，用
`--pieces/--empty/--rectification/--layout` 指过来），或者做离线回归夹具。

注意：`pieces_rectified.png` 与 `empty_rectified.png` 必须是**同一机位、同一光照**的整流图，
否则背景差分失效；`rectification.json` 提供 `canvas_px` 与 `px_per_mm`（面积、mm² 换算要用）。
