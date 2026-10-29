# Backgrounds - 程序化拼图底面场景

本目录生成“换底面”的合成场景（木地板 / 大理石地面 / 水泥地面，以及按真实采集协议铺黑色垫布的变体），
用来检验第二阶段碎片提取在不同桌面上的鲁棒性。纸片是阶段一实拍结果里反预乘取出的真实像素，
几何等于 `stage2_piece_extraction/output/pieces.json`，所以每个场景都自带真值，可逐片量化。

```bash
python backgrounds/make_backgrounds.py                     # 生成五个场景（产物约 130 MB，已 gitignore）
python backgrounds/evaluate_backgrounds.py --rerun \
  --config backgrounds/config/extraction_light_bg.example.json   # 逐场景跑阶段二并与真值比较
```

产物：`backgrounds/output/<scene>/{empty,pieces,alpha}.png`、`scene.json`（真值几何与渲染参数）、
`overlay.png`（品红真值 vs 绿色提取）、`stage2/`（阶段二结果），以及 `preview.png`、`evaluation.json`。

一句话结论：**铺黑色垫布的浅色地面可以把精度做到 bench 同级**（角点中位 0.39-0.40 px，
IoU≈0.993），裸木地板次之（中位 2.28 px），裸大理石/水泥因投影并入掩膜而失败（物理限制，
已量化）。完整方法、评估表、原因分析与已知限制见 [`docs/BACKGROUNDS.md`](../docs/BACKGROUNDS.md)。
