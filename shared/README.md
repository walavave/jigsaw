# Shared

`jigsaw_common/` 只放四个阶段都需要的无状态工具：

- `cut_layout.py`：递归两刀切分、碎片分离、打印几何。
- `io.py`：支持中文路径的图像和 JSON 读写。
- `schema.py`：交接 JSON 的版本常量。

阶段专属的标定、分割、匹配和拼接算法不要放进这里。
