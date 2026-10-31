# Stage 4 交付包

本包包含 stage4_assembly 的完整实现、17 项新增测试、仓库实拍样本所需输入和全部示例结果。
基础版本：walavave/jigsaw 的 step3 提交 12f0f7bf1fdb4a885af069a29052e27ec386cc3d。
本地实现提交：d36b417；尚未推送到远端。

## 独立运行

解压后进入本目录，执行：

```bash
python -m pip install -r requirements.txt
python stage4_assembly/assemble.py --reference stage1_capture_calibration/output/reference/reference.png
python -m unittest discover -s tests -p 'test_stage4_assembly.py' -v
```

打开 stage4_assembly/output/placement_guide.md 查看人工取片、转角、对齐邻片和 16 步顺序图。
详细说明见 stage4_assembly/README.md，复现报告见 docs/REPRODUCTION_STAGE4.md。

## 合入现有仓库

建议先从原仓库检出 step3，再将本包中的同路径文件复制覆盖。
stage4_source.patch 是代码、文档与 JSON 的文本补丁，可在干净的 step3 上执行 git apply。
该补丁不含 PNG；图片已随包提供，按目录复制即可。不要在复制覆盖后再次应用补丁。
本包省略了与 Stage 4 无关的上游程序和照片；完整流水线请使用原仓库。
全仓库 84 项测试已在完整 checkout 中通过；本包可单独运行 Stage 4 的 17 项测试。

## 已知结果

16/16 块，连续几何覆盖率 99.446573%，校色后 RGB PSNR 20.4931 dB、灰度 SSIM 0.74444。
Stage 3 近似解与 S06 复核项仍保留，报告状态 review 是预期结果。参考图没有用于补缺口。
