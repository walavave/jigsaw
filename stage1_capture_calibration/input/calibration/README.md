# Chessboard photos

把 15-25 张 `jpg/png` 棋盘格照片放在此目录。保持与碎片照片相同的分辨率、焦距、曝光和白平衡。

手机横拍和竖拍的照片可以混放。程序会读取 EXIF 方向，并把长宽互换的图片自动旋转到首张照片的坐标系。

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
  --cols 6 --rows 8 --square-mm 25
```
