# Stage 1 output

本目录由脚本生成：

```text
output/
├── cut_pdf/
│   ├── jigsaw_16_pieces.pdf
│   └── jigsaw_layout.json
├── reference/
│   ├── reference.png
│   └── source_pieces/
├── calibration_targets/
├── calibration/
└── rectified/
    ├── pieces_rectified.png
    ├── empty_rectified.png
    ├── checkerboard_detection_debug.png
    └── rectification.json
```

不要手工修改标定参数或单应矩阵；重新运行脚本并保留原始输入。

