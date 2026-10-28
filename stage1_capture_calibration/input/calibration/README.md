# Chessboard photos

把 15-25 张 `jpg/png` 棋盘格照片放在此目录。保持与碎片照片相同的分辨率、焦距、曝光和白平衡。

如果使用 Intel RealSense 出厂彩色内参，可以不在此目录放照片，改为从项目根目录运行：

```bash
python3 stage1_capture_calibration/export_realsense_intrinsics.py \
  --width 1280 --height 720 --fps 30
```

无论内参来自哪里，桌面平面矫正仍需要一张平放棋盘格的 `input/raw/plane.jpg`。

然后从项目根目录运行：

```bash
python3 stage1_capture_calibration/calibrate_intrinsics.py \
  --images stage1_capture_calibration/input/calibration \
  --cols 9 --rows 6 --square-mm 18
```
