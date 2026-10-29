# Backgrounds - 程序化拼图底面场景

本目录生成“换底面”的合成场景，用来检验第二阶段碎片提取在**不同桌面/地面**上的鲁棒性。
真实采集（阶段一）只有黑绒布桌面，这里补上木地板、大理石地面、水泥地面，以及按真实采集协议
铺了黑色垫布的变体。

纸片本身是**真实拍摄的**：从阶段一整流结果 `pieces_rectified.png` 里按 alpha 反预乘取出纸片像素，
再贴着新的底面重新合成。因此合成场景同时具备两个好处：

- 纸面纹理、印刷内容、手剪边缘都是真实数据，不是贴图；
- 几何完全等于 `stage2_piece_extraction/output/pieces.json`（阶段一/二在 bench 上得到的几何），
  所以每个场景都自带“真值”，分割结果可以逐片量化。

```text
backgrounds/
├── make_backgrounds.py                 生成器
├── evaluate_backgrounds.py             逐场景跑阶段二并与真值比较
├── config/
│   ├── backgrounds.example.json        底面/合成/光照参数
│   └── extraction_light_bg.example.json 阶段二的“浅色纹理底面”profile
└── output/                             生成产物（已 gitignore，约 130 MB）
    ├── preview.png                     五场景预览图
    ├── evaluation.json                 评估结果
    └── <scene>/{empty,pieces,alpha}.png, scene.json, overlay.png, stage2/
```

## 运行

```bash
# 1) 生成五个场景（约十几秒，2640x2640 / 8 px/mm，与阶段一整流画布同尺度）
python backgrounds/make_backgrounds.py

# 2) 逐场景跑阶段二并评估（--rerun 强制重跑；默认复用已有 stage2 结果）
python backgrounds/evaluate_backgrounds.py --rerun --config backgrounds/config/extraction_light_bg.example.json

# 只跑一个场景 / 只重算表格
python backgrounds/evaluate_backgrounds.py --scenes wood_floor
python backgrounds/evaluate_backgrounds.py --summary backgrounds/output/evaluation.json
```

`evaluate_backgrounds.py` 的退出码：`0` = 所有场景检出数等于期望值；`1` = 有场景数量不符。
表格里 `corner mean / median / p90 / max` 是“提取角点 → 真值多边形边界”的距离（单位 px），
`haus max` 是双向最大距离，`IoU mean` 是多边形 IoU，`area` 是面积比（检测/真值）。

## 场景一览

| 场景 | 底面 | 说明 |
| --- | --- | --- |
| `wood_floor` | 暖棕色木地板（顺纹 + 板缝 + 树结） | 裸木地板，纸片与木色有明显色差 |
| `marble_floor` | 浅灰白大理石（脉络） | 裸浅色地面，投影在视觉上很“重” |
| `concrete_floor` | 中灰水泥（骨料颗粒 + 污渍） | 裸水泥地面 |
| `marble_floor_mat` | 大理石 + 黑色绒布垫布 | **推荐的采集协议**：按阶段一实拍做法铺垫布 |
| `concrete_floor_mat` | 水泥 + 黑色绒布垫布 | 同上 |

每个场景输出两帧同名几何的图：`empty.png`（只有底面）和 `pieces.png`（底面 + 16 片纸片）。
两帧的照明抖动、传感器噪声**各自独立**（模拟“拍两张时灯光微变”），避免背景差分退化成
“同图相减 = 0”的平凡情形。`alpha.png` 是纸片覆盖度，`scene.json` 记录真值几何与渲染参数
（`generator_version`、`seed`、`rendered_with`）。

生成的场景是确定性的：第 i 个场景用 `seed + i*1013` 初始化 RNG，同参数重跑得到逐字节相同的图。

## 合成细节

1. **取纸片**：`unmultiply_pieces(observed, empty, alpha)` 用
   `piece = (observed - (1-alpha)·empty) / max(alpha, 0.20)` 把纸片从黑绒布背景里“抠”出来，
   避免把黑桌面边缘带进新场景。alpha 来自真值多边形 4× 超采样 + 0.7 px 羽化。
2. **造底面**：`make_wood_floor` / `make_marble_floor` / `make_concrete_floor` 用多尺度值噪声
   （`fbm`）合成纹理，参数在 `config/backgrounds.example.json` 里（板宽、脉络宽度、骨料尺度……）。
3. **垫布**（`_mat` 变体）：`_mat_alpha` 取 16 片真值多边形包围盒外扩 `margin_mm` 铺一块矩形
   黑绒布（羽化边），`_add_mat` 再叠一层织物纹理后与底面混合。
4. **光照**：`lighting_field` + `apply_lighting` 给整幅图加方向性渐变、暗角与增益抖动；
   纸片像素按同一场的相对增益（`piece_modulation`）一起调制，保证“纸片跟着灯一起亮/暗”。
5. **投影**：`composite_scene` 在碎片下方合成软阴影（偏移 16-22 px、σ≈16 px）与贴边接触阴影
   （σ≈4 px），再叠加纸片。
6. **噪声**：两帧各自 `add_sensor_noise`（σ 由 config 的 `lighting.noise` 控制）。

## 评估结果（交付版本）

阶段二用 `backgrounds/config/extraction_light_bg.example.json`：在默认配置基础上打开
`shadow_rejection`、`mask_erode_px=2.0`、`close_ksize=9`、`half_level_response="colour"`，
并把 `min_quality` 放宽到 0.78（合成场景的纸边是 0.7 px 羽化的混合边，拟合残差天然略大）：

| 场景 | 检出 | corner mean | median | p90 | max | IoU mean | 面积比 | q min | 顶点数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `wood_floor` | 16/16 | **2.63 px** | 2.28 | 4.80 | 5.11 | 0.9779 | 0.9836 | 0.787 | 全 4 边 |
| `marble_floor` | 12/16 | 232.78 px | 318.23 | 395.12 | 524.82 | 0.3684 | 1.5307 | 0.7384 | 5/6 边混入 |
| `concrete_floor` | 16/16 | 121.52 px | 56.75 | 311.47 | 848.44 | 0.5607 | 0.6630 | 0.7494 | 5/6 边混入 |
| `marble_floor_mat` | 16/16 | **0.73 px** | 0.39 | 1.42 | 2.41 | 0.9925 | 0.9945 | 0.899 | 全 4 边 |
| `concrete_floor_mat` | 16/16 | **0.77 px** | 0.40 | 1.50 | 2.72 | 0.9927 | 0.9947 | 0.896 | 全 4 边 |

对照：同一套阶段二在真实黑绒布 bench 上，角点误差（对照片轮廓）median 0.86 / mean 1.58 px，
面积比 1.00755（见 `docs/REPRODUCTION_STAGE2.md` §5.1）。

逐片细节（`backgrounds/output/evaluation.json`）：

- 两个 `_mat` 场景几乎完美：每片 corner 0.22–2.72 px，`agreement_shift_px` 0.25–2.24，
  quality 0.896–0.963；只有 S06 因为最短边 66 px 被标 `short_side`（与 bench 一致）。
- 裸木地板：每片 corner 0.88–5.11 px（中位 2.28），但 12/16 被标 `localizer_disagreement`
  —— 木纹上的半对比度参考线本身不可靠（`agreement_shift_px` 3.09–5.23 px），而 `gradient_max`
  的角点是准的。这类标记是“值得人工看一眼”的三分类提示，不影响流水线运行。
- `overlay.png`（每场景一张，品红=真值、绿=提取）用于肉眼复核：`_mat` 场景里两套轮廓基本重合。

## 为什么裸浅色地面会失败（物理原因，已量化）

阶段二的掩膜来自两帧的 Lab 欧氏距离，阈值夹在 25：

- 大理石地面 L≈215，纸片投影把地面压到 L≈140，ΔL≈79 ≫ 25 → 掩膜把“纸片 + 投影”连成一片；
  相邻碎片的投影还会连起来，于是 16 片变成 12 个连通域。这不是参数问题：用
  `shadow_rejection` 也救不了——这类底面的高通纹理太弱（`shadow_min_texture` 之下），
  相关度证据被整体归零。
- 黑绒布 L≈32，投影后 ΔL≈17 < 25 → 投影根本不越阈值。**这就是真实采集要铺黑垫布的原因**，
  也是 `_mat` 变体存在的意义。
- 木地板是唯一的例外：它同时满足“纸片与底面有明显色差”和“有足够高通纹理让
  `shadow_rejection` 判定‘只是变暗’”，所以裸木地板能 16/16。

换句话说：纸面与地面同为中性亮色、且投影很深时，只看两帧差分在物理上无法把投影和纸片分开；
铺一块比纸片暗得多的垫布（或用彩色/深色地面）是最省事、也最贴近真实协议的办法。

## 给阶段二新增的开关

| 键 | 作用 |
| --- | --- |
| `shadow_rejection` + `shadow_blur_px` / `shadow_window_px` / `shadow_min_texture` / `shadow_correlation` | 用“两帧高通后的局部相关度”识别“同一纹理只是变暗”的投影像素并从掩膜剔除；默认关闭 |
| `mask_erode_px` | 掩膜额外腐蚀半径，只为让粗多边形/逐边定位有个更干净的初值；默认 0（bench 逐字节不变） |
| `half_level_response` | `half_level` 参考线改用颜色响应（`colour`）而不是含 Laplacian 的 `edge` 响应，纹理底面更稳 |
| `min_quality` | 合成场景下可放宽（本 profile 用 0.78），因为羽化边让 fit/shape 项略低 |

## 已知限制

- **裸浅色 + 强投影**：`marble_floor` 12/16、`concrete_floor` 角点误差上百 px，是物理限制而非调参问题，
  已量化写在上表；工程上应改用垫布或深色/彩色底面。
- **投影是合成的**：高斯软阴影 + 接触阴影是近似，真实遮挡（纸片互相投影、环境光反弹）会更复杂；
  本目录的结论是“相对比较”，不能替代真实拍摄验证。
- **真值是阶段二的产出几何**：这些场景衡量的是“换底面后能否复原同一几何”，不是绝对物理真值
  （绝对精度见 `docs/REPRODUCTION_STAGE2.md` §5）。
- **只评估到阶段二**：阶段三/四没有在这些场景上重跑。
- 垫布按碎片包围盒外扩 20 mm 铺放；若碎片铺得离垫布边缘太近，投影会落到垫布外，`_mat` 场景也会退化。
- **配置覆盖是整块替换**：`main()` 用 `dict(DEFAULT_CONFIG_VALUES).update(read_json(...))`，所以
  `lighting`、`piece_composite` 这类嵌套块一旦在配置文件里出现就必须写全（只写一个字段会
  `KeyError: 'vignette'`）。`_mat` 场景同理：要么不写 `mat`（自动用 `mat_default`），要么写完整的覆盖块。
- `_mat_alpha` 经模糊后峰值 1.0000001（未裁剪），当严格覆盖率用需自行 `clip(0,1)`；
  `unmultiply_pieces` 在 α≈0 处有 float32 抵消误差（量级 1e-4），只影响边缘极少数像素。
- 测试覆盖：`tests/test_backgrounds.py`（30 项）覆盖噪声/底面生成、垫布合成、`build_scene` 端到端与
  `shadow_suspicion` / `foreground_mask`；`python -m unittest discover -s tests -v` 现为 67 项全绿。

## 扩展一种新底面

1. 在 `make_backgrounds.py` 里加 `make_xxx_floor(shape, rng, params, px_per_mm)`，注册进 `GENERATORS`，
   并在 `BACKGROUND_LABELS` 加中文名；
2. 在 `config/backgrounds.example.json` 的 `backgrounds` 列表里加场景名（要垫布就加 `xxx_floor_mat`），
   并按需给 `xxx_floor` 一段参数块；
3. `python backgrounds/make_backgrounds.py --only xxx_floor` 生成，再用
   `python backgrounds/evaluate_backgrounds.py --scenes xxx_floor` 看精度。
