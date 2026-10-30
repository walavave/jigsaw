# Stage 4 output

此目录用于保存仓库现有 16 块实拍碎片的重建结果。
除本说明外，生成图像、JSON 报告和人工操作表均由 `.gitignore` 排除，
首次克隆后请在仓库根目录执行以下命令：

```bash
python -m pip install -r requirements.txt
python stage4_assembly/assemble.py \
  --reference stage1_capture_calibration/output/reference/reference.png
```

所有输入已随 Stage 2/3 提交；运行后生成下列文件。

| 文件 | 内容 |
| --- | --- |
| `assembled.png` / `assembled_raw.png` | 保守校色版 / 未校色版 |
| `assembly_report.json` | 输入摘要、配置、几何/栅格覆盖、SSIM、PSNR、逐片与接缝指标 |
| `comparison.png` | 参考图、未校色、校色版并排，仅有参考图时生成 |
| `seams.png` / `coverage_mask.png` | 黄边为多边形边界、洋红为缺口、青色为重叠；覆盖白、空缺黑 |
| `placement_plan.json` | 供人工逐片执行的坐标/角度/邻片计划 |
| `placement_guide.md` | 中文操作表和步骤图片链接 |
| `placement_order.png` / `piece_index.png` | 目标顺序总览 / 碎片正面取片索引 |
| `scan_locations.png` | 原桌面 S 编号位置，原俯视图可读时生成 |
| `steps/01.png..16.png` | 每一步增加一块，绿边为当前碎片 |

样本报告 `status=review` 是预期：上游近似解、S06 人工复核项仍然存在。
参考图没有用于缺口填充。输出可追溯到 Stage 2 和 Stage 3 JSON 的 SHA256。

