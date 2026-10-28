# Stage 1 - 切分、拍摄、标定与俯视矫正

本阶段把参考图变成可打印裁剪的 16 块碎片，并把桌面照片统一矫正到固定的毫米坐标画布。Stage 2 只读取本阶段落盘的图片和 JSON。

默认标定路线 **不需要 ArUco**：

1. 相机内参来自棋盘格标定照片，或者直接导出 Intel RealSense 出厂彩色相机内参。
2. 在桌面上平放一次同一张棋盘格，拍 `plane.jpg`，用于估计桌面到标准工作区的单应矩阵。
3. ArUco 仅保留为可选诊断/兼容路线，不属于主流程。

## 目录

```text
stage1_capture_calibration/
├── config/
│   ├── checkerboard_layout.json       # 主流程：棋盘格尺寸和工作区位置
│   ├── aruco_layout.json              # 可选旧路线
│   └── aruco_layout.example.json
├── input/
│   ├── calibration/                   # 15-25 张不同姿态的棋盘格照片
│   └── raw/
│       ├── plane.jpg                  # 一张棋盘格平放在工作区的照片
│       ├── empty.jpg                  # 空工作区，无标定板、无碎片
│       └── pieces.jpg                 # 16 块碎片散落
├── output/
│   ├── cut_pdf/
│   ├── reference/source_pieces/
│   ├── calibration_targets/
│   ├── calibration/
│   └── rectified/
├── checkerboard_plane.py
├── export_realsense_intrinsics.py
└── *.py
```

## 1. 生成可打印切割 PDF

在项目根目录执行：

```bash
python3 stage1_capture_calibration/make_print_pdf.py --image pic.jpg
```

输出：

- `output/cut_pdf/jigsaw_16_pieces.pdf`：A4 横向，2 页。
- 第 1 页：一比一裁剪页，沿洋红色边界裁剪。
- 第 2 页：编号与切分结构核对页，不用于裁剪。
- `output/cut_pdf/jigsaw_layout.json`：16 个源多边形及 `P01..P16` 编号。
- `output/reference/reference.png`：统一格式的原始参考图。
- `output/reference/source_pieces/`：每块源图的 RGBA 和 mask。

打印必须选择 **A4 横向、100%、实际大小**，关闭“适合页面”和“自动缩放”。打印后先量第 1 页底部的 `100 mm` 校准线。

## 2. 生成棋盘格

```bash
python3 stage1_capture_calibration/make_calibration_board.py
```

输出 `output/calibration_targets/checkerboard_9x6_18mm.pdf`。按 A4、100% 打印并粘在平整硬板上。默认参数是：

- 内角点：`9 x 6`
- 方格边长：`18 mm`
- 棋盘格外尺寸：`180 x 126 mm`

如果更换靶纸，必须同步修改 `config/checkerboard_layout.json` 和命令行参数。

## 3. 准备相机内参

以下两种方式选一种即可。正式采集期间不能再改变相机分辨率、焦距或彩色视频流配置。

### 3.1 RealSense 出厂内参（推荐先用这个）

安装 RealSense Python 绑定：

```bash
python3 -m pip install -r requirements-realsense.txt
```

按正式采集所用的彩色流导出内参：

```bash
python3 stage1_capture_calibration/export_realsense_intrinsics.py \
  --width 1280 --height 720 --fps 30
```

输出 `output/calibration/intrinsics.json`。之后拍摄的 `plane.jpg`、`empty.jpg`、`pieces.jpg` 必须使用同一分辨率。RealSense 的出厂彩色内参通常足够用于本项目这种短距离、二维平面任务；如果发现矫正后的直线仍明显弯曲，再改用棋盘格内参。

导出器直接支持 Brown-Conrady 畸变模型。若设备报告非零的 Inverse Brown-Conrady 等 OpenCV 不能直接消费的模型，脚本会明确报错，此时应使用棋盘格标定或由 RealSense SDK 先完成彩色图去畸变。

### 3.2 棋盘格内参

保持正式拍摄时的分辨率、焦距、曝光和白平衡，拍 15-25 张棋盘格照片。照片要覆盖画面中心、边缘和不同倾斜角度，不能都是同一个姿态。

把照片放入 `input/calibration/`，然后运行：

```bash
python3 stage1_capture_calibration/calibrate_intrinsics.py \
  --images stage1_capture_calibration/input/calibration \
  --cols 9 \
  --rows 6 \
  --square-mm 18
```

输出 `output/calibration/intrinsics.json` 和 `undistorted_example.png`。建议 RMS 重投影误差小于 `1.0 px`；超过时应重拍模糊、低倾斜或边缘覆盖不足的视图。

## 4. 拍摄平面照片、空桌面和碎片

相机固定后依次拍三张，期间不要移动相机、工作区或灯光：

1. `input/raw/plane.jpg`：棋盘格平放在工作区内，棋盘格长边与工作区 `x` 轴平行。默认配置假设棋盘格外框左上角位于工作区 `(110, 77) mm`，即棋盘格在工作区内居中。
2. `input/raw/empty.jpg`：拿走棋盘格和碎片，只拍空工作区。
3. `input/raw/pieces.jpg`：16 块碎片正面朝上散落，互不重叠、不遮挡。

默认工作区为 `400 x 280 mm`，画布为 `3200 x 2240 px`。如果你的桌面区域不同，修改 `config/checkerboard_layout.json` 中的 `workspace_mm`、`canvas_px` 和 `board_outer_origin_mm`，再重新拍摄。

普通棋盘格没有方向 ID，程序只能根据俯拍画面的左上方向消除 180 度歧义。因此 `plane.jpg` 中的棋盘格必须按上述方向放置，不能旋转 90 度，也不要斜放。

## 5. 执行俯视矫正

```bash
python3 stage1_capture_calibration/rectify_capture.py \
  --pieces stage1_capture_calibration/input/raw/pieces.jpg \
  --empty stage1_capture_calibration/input/raw/empty.jpg \
  --plane stage1_capture_calibration/input/raw/plane.jpg \
  --intrinsics stage1_capture_calibration/output/calibration/intrinsics.json \
  --checkerboard-layout stage1_capture_calibration/config/checkerboard_layout.json
```

输出：

- `output/rectified/pieces_rectified.png`
- `output/rectified/empty_rectified.png`
- `output/rectified/checkerboard_detection_debug.png`
- `output/rectified/rectification.json`

`pieces_rectified.png` 和 `empty_rectified.png` 使用同一组相机内参和单应矩阵，可直接做背景差分。检查 debug 图中棋盘格角点是否全部检出，`rectification.json` 中的 `rms_reprojection_px` 应尽量小于 `2 px`。

## 6. 可选：ArUco 旧路线

如果棋盘格方向无法固定，或需要一套带 ID 的四角标记，可以继续使用旧路线：

```bash
python3 stage1_capture_calibration/make_aruco_markers.py
python3 stage1_capture_calibration/rectify_capture.py \
  --pieces stage1_capture_calibration/input/raw/pieces.jpg \
  --empty stage1_capture_calibration/input/raw/empty.jpg \
  --intrinsics stage1_capture_calibration/output/calibration/intrinsics.json \
  --aruco-layout stage1_capture_calibration/config/aruco_layout.json
```

设置 `--aruco-layout` 后会覆盖默认的棋盘格平面路线。主流程不需要执行这一步。

## 7. 交给 Stage 2

把以下文件复制到 `stage2_piece_extraction/input/`：

```text
jigsaw_layout.json
reference.png
source_pieces/
rectification.json
pieces_rectified.png
empty_rectified.png
```

详细原理和排错说明见项目根目录的 `docs/camera_calibration.md`。
