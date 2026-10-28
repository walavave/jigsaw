# Stage 3 - Feature matching

输入：

- 阶段一的 `jigsaw_layout.json` 和 `reference/source_pieces/`
- 阶段二的 `pieces.json` 和 `pieces/*`

任务：

1. 对扫描碎片和源多边形模板提取 SIFT 特征。
2. Lowe ratio test + RANSAC 估计几何变换。
3. 构造 16 x 16 匹配分数矩阵。
4. 用匈牙利算法求一一分配。
5. 输出低置信度人工复核项。

输出：

- `output/assignments.json`
- `output/matches/`：每块碎片的关键点匹配可视化。

当前状态：目录和交接契约已建立，匹配脚本待实现。

