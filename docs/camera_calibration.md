# 相机标定与桌面矫正

## 1. 两个问题，不一定需要两套标记

Stage 1 需要解决两个几何问题：

1. **相机内参**：估计焦距、主点和镜头畸变，回答“照片里的直线为什么会弯”。
2. **桌面到工作区的平面变换**：估计单应矩阵，回答“斜着拍的桌面如何变成固定毫米坐标的正视图”。

传统 RealSense 使用流程中，内参通常直接来自设备出厂标定；外部棋盘格标定主要用于更高精度或重新标定。对于本项目，推荐组合是：

- 内参：RealSense 出厂彩色内参，或者棋盘格多姿态标定。
- 平面位姿：在工作区内平放一次棋盘格，拍 `plane.jpg`，估计桌面到标准工作区的单应矩阵。

因此 **ArUco 不是必需的**。同一张棋盘格可以同时用于两件事，但两次拍摄用途不同：内参需要 15-25 张不同姿态照片；平面矫正只需要一张平放在桌面上的照片。

## 2. 固定相机和采集设置

相机安装到俯拍支架后，正式采集期间不要改变：

- 相机位置和角度；
- 分辨率、焦距、彩色视频流配置；
- ISO、快门、白平衡和对焦；
- 工作区位置和灯光。

如果使用 RealSense，`plane.jpg`、`empty.jpg`、`pieces.jpg` 必须和 `intrinsics.json` 中记录的图像尺寸完全一致。改变分辨率后应重新导出或标定内参。

## 3. 获取相机内参

### 3.1 RealSense 出厂内参

安装：

```bash
python3 -m pip install -r requirements-realsense.txt
```

导出与正式采集相同的彩色流参数：

```bash
python3 stage1_capture_calibration/export_realsense_intrinsics.py \
  --width 1280 --height 720 --fps 30
```

输出：

```text
stage1_capture_calibration/output/calibration/intrinsics.json
```

脚本读取 librealsense 提供的 `fx`、`fy`、`ppx`、`ppy` 和畸变系数。Brown-Conrady 畸变可直接交给 OpenCV；如果设备报告非零的 Inverse Brown-Conrady 等不同模型，脚本会拒绝生成会误导后续处理的 JSON。此时可以：

1. 用 RealSense SDK 先输出去畸变彩色图；
2. 改用棋盘格内参标定；
3. 在确认模型和系数顺序后扩展导出器。

RealSense 的出厂内参通常足以支持本项目的工作距离和二维平面任务。它不代表绝对完美，仍要通过 `checkerboard_detection_debug.png` 和矫正后直线检查实际效果。

### 3.2 棋盘格内参

1. 运行 `python3 stage1_capture_calibration/make_calibration_board.py`。
2. 按 A4、100%、实际大小打印，量一下纸上的 100 mm 校准线。
3. 拍 15-25 张。覆盖画面中心和四角，并改变标定板倾斜角度。
4. 照片放入 `stage1_capture_calibration/input/calibration/`。

默认靶纸为 9 x 6 个**内角点**、方格边长 18 mm。这里的“内角点”不是黑白方格总数；如果使用别的靶纸，必须同步修改参数。

运行：

```bash
python3 stage1_capture_calibration/calibrate_intrinsics.py \
  --images stage1_capture_calibration/input/calibration \
  --cols 9 \
  --rows 6 \
  --square-mm 18
```

输出：

```text
stage1_capture_calibration/output/calibration/
├── intrinsics.json
└── undistorted_example.png
```

验收建议：

- 接受视图至少 12 张，最好 15-25 张；
- `rms_reprojection_px` 尽量小于 1.0 px；
- `undistorted_example.png` 中直线不应继续明显弯曲。

## 4. 用棋盘格确定桌面平面

### 4.1 布置工作区

默认配置 `config/checkerboard_layout.json` 为：

```json
{
  "cols": 9,
  "rows": 6,
  "square_mm": 18,
  "workspace_mm": [400, 280],
  "canvas_px": [3200, 2240],
  "board_outer_origin_mm": [110, 77]
}
```

棋盘格的外框尺寸是 `(9+1) x (6+1)` 个方格，即 `180 x 126 mm`。`board_outer_origin_mm` 指的是棋盘格**最外层黑白方格的外框左上角**在工作区毫米坐标中的位置。默认值让棋盘格居中。

布置要求：

1. 工作区实际尺寸与 `workspace_mm` 一致。
2. 棋盘格平放在硬质、平整表面上，不能翘曲。
3. 棋盘格长边与工作区 `x` 轴平行，不能旋转 90 度或斜放。
4. 棋盘格外框放在配置指定的位置，四边都要在相机画面中可见。

普通棋盘格没有方向 ID，只靠黑白格无法区分 0 度和 180 度。当前代码假设相机俯拍方向正常，并选择更靠近图像左上角的棋盘格角点作为起点。因此必须按上述方向摆放；如果无法保证方向，使用可选 ArUco 路线，或换成带非对称方向标记的 ChArUco 板。

### 4.2 拍摄三张桌面图

相机完全不动：

```text
stage1_capture_calibration/input/raw/
├── plane.jpg    # 棋盘格平放在工作区
├── empty.jpg    # 拿走棋盘格和碎片后的空桌面
└── pieces.jpg   # 16 块碎片散落
```

`plane.jpg` 负责求单应矩阵；`empty.jpg` 和 `pieces.jpg` 会被同一个单应矩阵映射。三张图的分辨率必须一致。

### 4.3 执行矫正

```bash
python3 stage1_capture_calibration/rectify_capture.py \
  --pieces stage1_capture_calibration/input/raw/pieces.jpg \
  --empty stage1_capture_calibration/input/raw/empty.jpg \
  --plane stage1_capture_calibration/input/raw/plane.jpg \
  --intrinsics stage1_capture_calibration/output/calibration/intrinsics.json \
  --checkerboard-layout stage1_capture_calibration/config/checkerboard_layout.json
```

程序会：

1. 检查 `pieces.jpg`、`empty.jpg`、`plane.jpg` 的分辨率与内参一致。
2. 对所有图执行同一组去畸变。
3. 在 `plane.jpg` 中检测棋盘格内角点。
4. 根据棋盘格的已知毫米尺寸和 `board_outer_origin_mm` 建立图像点与工作区点的对应关系。
5. 用 `cv2.findHomography` 求单应矩阵。
6. 将碎片图和空桌面图映射到 `canvas_px` 画布。

输出：

```text
stage1_capture_calibration/output/rectified/
├── pieces_rectified.png
├── empty_rectified.png
├── checkerboard_detection_debug.png
└── rectification.json
```

检查 `checkerboard_detection_debug.png` 中角点和棋盘格是否对齐。`rectification.json` 中的 `rms_reprojection_px` 建议小于 2 px。这个 RMS 只反映棋盘格角点的单应拟合误差，不包含棋盘格本身翘曲或工作区尺寸测量误差。

## 5. 可选 ArUco 路线

ArUco 路线的优势是 marker 自带 ID，不存在棋盘格的 180 度方向歧义，并且可以直接用四角定义工作区。代价是需要额外打印和布置 marker。保留命令：

```bash
python3 stage1_capture_calibration/make_aruco_markers.py
python3 stage1_capture_calibration/rectify_capture.py \
  --pieces stage1_capture_calibration/input/raw/pieces.jpg \
  --empty stage1_capture_calibration/input/raw/empty.jpg \
  --intrinsics stage1_capture_calibration/output/calibration/intrinsics.json \
  --aruco-layout stage1_capture_calibration/config/aruco_layout.json
```

设置 `--aruco-layout` 会覆盖默认的棋盘格平面路线。它不是主流程的依赖。

## 6. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 棋盘格检测不到 | 参数写成方格总数，或靶纸反光、弯曲 | 确认 9 x 6 是内角点；换平整哑光纸 |
| 内参 RMS 很大 | 姿态单一、模糊、靶纸翘曲 | 重拍中心和四角的倾斜视图 |
| `plane.jpg` 检测不到 | 棋盘格没有完整入镜或尺寸参数错误 | 让整块板入镜；同步修改 cols、rows、square-mm |
| 矫正后方向颠倒 | 棋盘格相对相机旋转了 180 度 | 按正常俯拍方向摆正；不要旋转 90 度 |
| 矫正后整体偏移 | `board_outer_origin_mm` 与实物位置不一致 | 重新量棋盘格外框左上角，更新配置 |
| 碎片图和空桌面错位 | 拍摄中移动了相机或工作区 | 固定相机后重新拍三张图 |
| 白纸碎片边缘难分 | 背景和白纸太接近 | 用深色哑光背景，裁掉碎片白边 |
| RealSense 内参导出报畸变模型错误 | 设备使用 OpenCV 不能直接消费的模型 | 棋盘格标定，或先用 SDK 输出去畸变图 |

## 7. 代码边界

Stage 1 只负责输出几何一致的原图、碎片源资源、相机内参、俯视桌面图和空桌面图。碎片分割、源位置匹配和最终拼接分别由 Stage 2、3、4 完成。
