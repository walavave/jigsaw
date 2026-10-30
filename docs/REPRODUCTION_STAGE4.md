# Stage 4 实现与复现记录

日期：2026-10-29。基础分支 `step3`，固定提交
`12f0f7bf1fdb4a885af069a29052e27ec386cc3d`；它已包含 `step2` 的
`64d689d4b1006407f4e7db03f289b910b75474a2`。

## 输入与实现范围

- `stage2_piece_extraction/output/pieces.json`：`stage2-extract-1.4.0`，
  session `9ab5743b5eec79a1`，16 个最长边坐标系中的 1:1 裁剪图。
- `stage3_feature_matching/quad_puzzle/examples/output.json`：
  `polygon-assembly-result/v2`，状态 approximate，16 个旋转平移位姿。
- 参考图：`stage1_capture_calibration/output/reference/reference.png`，1280×720。
- 交付：真实纹理数字拼接、保守校色、接缝诊断、质量指标、人工摆放计划。
  不实现机械臂部分，不重新求解 Stage 3，不拿原图补缺口。

早期根 README 设想 SIFT + 源编号分配，但当前 step3 实际实现为无图像内容参与的几何求解。
本阶段同时支持两者，默认使用已存在的几何结果。关键转换为
`H_PNG_to_assembly = G @ T @ B @ inverse(C)`，具体含义见 Stage 4 README。
从原桌面转动的角度由 `H_PNG_to_assembly @ C` 计算，不能直接抄基边角度。

## 环境与命令

验证环境：Python 3.12，NumPy 2.3.5，OpenCV 5.0.0，Shapely 2.2.0，
SciPy 1.17.0，scikit-image 0.26.0。项目依赖中新增 scikit-image，用于 SSIM。
实现只用 CPU，无需 CUDA 或额外模型权重。

```bash
python -m pip install -r requirements.txt
python stage4_assembly/assemble.py \
  --reference stage1_capture_calibration/output/reference/reference.png
python -m unittest discover -s tests -v
```

版本控制仅保留源码、配置、测试和说明。生成的拼接图、报告与逐步人工摆放图
写入 `stage4_assembly/output/`，由 `.gitignore` 排除；运行上述命令即可重新生成。

`--reference` 只用于选择整图朝向、Lab 统计校色、评价和比较图。
本例候选 0° 的灰度相关系数为 −0.31697，180° 为 0.92756，最终选 180°。
画布保持测量尺度；参考图显式重采样到 2028×1143 后才评价，相关信息在报告中记录。

## 客观结果

| 项目 | 结果 |
| --- | ---: |
| 碎片数 | 16/16 |
| 连续矩形宽×高 | 2027.016623×1142.954974 px |
| 栅格图像宽×高 | 2028×1143 px |
| 连续几何覆盖率 | 99.446573% |
| 栅格有效覆盖率 | 99.551813% |
| 栅格未覆盖像素 | 10,389 |
| 原始 RGB PSNR / 灰度 SSIM | 18.6555 dB / 0.73938 |
| 校色 RGB PSNR / 灰度 SSIM | 20.4931 dB / 0.74444 |
| 有效接缝统计 | 24 条，均有两侧样本 |
| 人工顺序连通性 | 1 个锚点；余下 15 步均对齐已放邻片 |
| 目标外框物理尺寸 | 约 253.38×142.87 mm，8 px/mm |

连续几何覆盖率与上游报告在 1e−7 内一致；差异仅来自 Stage 2 完整 JSON
中顶点的小数取舍。栅格覆盖率因填充边界和 alpha 插值而稍高，不应混用。
校色改善亮度/颜色统计，但实拍件已有明显颜色损失，不声称恢复了原图颜色。
SSIM 未达到最初文档建议的 0.85，当前没有做任何隐藏的参考图填充或局部重新配准。

最终 `status=review`，保留上游近似解/严格验收未通过，以及 S06 的复核提示。
默认退出 0 只代表产物生成成功；严格流水线可设 `fail_on_manual_review=true`。

## 验证

全仓库 **84 项测试通过**，其中 Stage 4 新增 **17 项**。
新增测试包括：

- 具有已知逐像素真值、非零 PNG 留白、倾斜原桌面坐标的独立合成样本；
- 基边负 x 坐标与仅有角点偏移时的转换；
- 透明边的分数像素插值不混入黑色；
- 两种输入协议、从整张俯视图出发的单应合成；
- 缺口在提供参考图时仍保留；公共角点不误判为公共边；
- 旋转整图时同步旋转目标多边形和人工角度；
- 重复/遗漏 ID、错误单位、无解状态、NaN、空掩码、输出覆盖输入等拒绝路径；
- 复核项传播、禁用步骤图、重新运行移除过期产物；
- 仓库实拍 16 块端到端回归及校色前后指标。

静态检查：`ruff check --isolated --select E,F --ignore E501` 通过；`git diff --check` 通过。
详细 CLI 和各输出说明见 [`stage4_assembly/README.md`](../stage4_assembly/README.md)。
