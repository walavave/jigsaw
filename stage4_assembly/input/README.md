# Stage 4 input

默认直接读取仓库中 Stage 2/3 的输出，不要求复制到本目录。

独立交接时携带完整 `pieces.json` 及其 `rgba_path/mask_path` 指向的图像，
保持这些路径相对于 `pieces.json` 所在目录。
另携带 Stage 3 的几何 `placements` 结果，通过 `--pieces/--placements` 指定。

若使用早期参考图分配协议，携带 `assignments.json`、`jigsaw_layout.json`，
通过 `--assignments/--layout` 指定。参考图为可选 `--reference`，用于朝向、校色与评价。
只给 `pieces_base_edge.json` 无法渲染真实纹理，因为它没有图像与裁剪坐标信息。

