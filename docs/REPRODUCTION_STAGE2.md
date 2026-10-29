# 第二部分（Stage 2：碎片提取）实现与验证报告

本文件记录 `stage2_piece_extraction/` 的实现过程、关键决策与客观验证结果。
第一部分的复现报告见 `docs/REPRODUCTION.md`。

## 1. 目标

按仓库 README §4.2 / §4.4 的交接契约，从第一阶段整流后的俯视图里

1. 用空桌面图做背景差分，把 16 个手剪纸片从深色桌面上分割出来；
2. 对每片拟合亚像素精度的直边凸多边形角点；
3. 做局部归一化，输出 512×512 的 RGBA / mask / debug 图；
4. 分配扫描编号 `S01..S16`（与源编号 `P01..P16` 无关，由 Stage 3 匹配），
   并把结果写成 `pieces.json` 交给 Stage 3。

验收标准（README §4.4）：16/16 检出、mask IoU > 0.98、角点误差 < 2 px、
编号/朝向稳定、给出质量分与人工复核标记、不把阴影/桌沿/灰尘/印刷图案误检成碎片。

## 2. 环境

复用第一部分的 Python 环境，**没有新增依赖**：

- Python 3.13.11（`C:\Users\86768\miniconda3\python.exe`）
- `opencv-contrib-python` 5.0.0.93（`cv2` 5.0.0）、`numpy` 2.2.6
- 只用到 `shared/jigsaw_common/io.py` 的 unicode 安全读写（本仓库路径含中文，
  `cv2.imread` 会失败，必须走 `np.fromfile` + `cv2.imdecode`）

## 3. 运行

```bash
# 默认输入/输出（阶段一的仓库内结果）
python stage2_piece_extraction/extract_pieces.py
```

实测耗时约 2.1 s（RTX 5060 无关，纯 CPU）。退出码：`0` = 检出数量等于
`expected_piece_count`；`1` = 数量不符，或 `fail_on_manual_review` 为真且
`manual_review` 非空。`manual_review` 非空本身只是“请人工看一眼”的告警。

当前输出：

```text
Pieces: 16/16 (canvas 2640x2640, normalized 512x512)
Segmentation: threshold=25.0 foreground=33.73% border_ignore=40px
Quality: min=0.8894 median=0.9248
Area px: min=114134 max=165124 total=2313710
Manual review: S06 (short_side)
```

产物：`stage2_piece_extraction/output/pieces.json` + `output/pieces/`
下 16×3 张图（`S01_rgba.png`、`S01_mask.png`、`S01_debug.png`，共 48 张）。

交付版本是 `stage2-extract-1.2.1`：相对 1.2.0 只改**元数据**——`inputs` 从本机绝对路径改成仓库根相对
POSIX 路径、每片新增 `normalized_short_side_px`、修正告警文案（`Manual review: S06 (short_side)`，
而不是笼统的 "below min_quality"）。几何完全没动，证据是 48 张产物图与 1.2.0 的 SHA256 逐字节相同
（`pieces.json` 因元数据变化而不同）。

## 4. 算法与关键决策

完整算法见 `stage2_piece_extraction/README.md`；这里只记录“为什么这样做”和对应的证据。

| 决策 | 理由 / 证据 |
| --- | --- |
| 掩码用 **Lab 颜色差** 做 Otsu，Laplacian 差分只给 `half_level` 定位用 | 把梯度项加进 Otsu 会把阈值抬到 45+（纸面印刷内容太暗），在 25 处夹住才不啃边；而梯度项参与 `gradient_max` 定位会把峰值系统性内拉 **0.69 px**（见下一行） |
| `gradient_max` 定位跑在 **Lab 颜色响应** 上（不是边缘响应） | 同一条边用 4 种绝对边缘定义（灰度梯度峰、灰度 50%、遮挡差分梯度峰、遮挡差分 50%）互相验证，散布仅 0.14 px；1.1.0 的内缩就是 Laplacian 项造成的，换响应后 64 条边的有符号偏移从 −0.660 px 回到 **+0.047 px** |
| 每边加 **±3 px 单边保护**（`max_side_shift_px`），越界退回 `half_level` 的边 | 短边/弱对比边容易峰选偏（1.1.0 里 S06 e3 偏 5.7 px、S05 e0 偏 5.9 px）；保护是把“整条边飞走”变成“退回保守结果”。本数据集上从未触发（两套定位在 64 条边上差都 < 1.5 px），保留作换数据时的保险 |
| 归一化只做**相似变换**（等比缩放 + 旋转 + 平移），不做透视 | 碎片是平的，第一阶段已整流到桌面平面；再做四角单应只会增加插值误差。README §4.2 允许“局部透视归一化”，这里等价且更稳 |
| 归一化朝向取**角点二阶矩主轴**（特征向量取 x≥0 的符号） | 比 `minAreaRect` 的角度约定更不容易踩坑，且完全由形状决定（同一片两次运行朝向一致）。近似正方形的主轴不稳定，本夹具都是细长四边形，不触发 |
| 编号用**行聚类**（y 差 ≤ 2 px 视为同一行，行内按 x） | 1.1.0 里 S14/S15 的中心 y 只差 0.10 px 却因亚像素抖动交换了编号；行聚类把“同一视觉行”的判定变成相对量，编号恢复稳定 |
| 质量分保留 0.35/0.25/0.20/0.10/0.10 权重；`needs_manual_review` 是**三分支告警** | 曾试图把“两套定位方法的分歧”加权进质量分，会把正常 1.5 px 的分歧也罚下 0.85 线，已回退；改为独立告警分支：`quality_below_min` / `short_side`（<100 px）/ `localizer_disagreement`（>3 px），并写进 `manual_review_reasons` |
| 退出码语义改为 `fail_on_manual_review`（默认 false） | 质量告警是交接提示，不是流水线失败：Stage 3 仍应拿到几何结果 |

相对仓库模板 `config/extraction.example.json` 的默认值，本数据集必须偏离两处（已写进配置与 README）：

- `ignore_border_mm: 45 → 5`：45 mm = 360 px，会把距画布边缘仅约 15 mm 的 S01/S04/S16 裁掉；
- `threshold_max: 60 → 25`：见上，纸面印刷内容让 Otsu 偏高。

交接路径约定（1.2.1 起）：`pieces.json` 里每片的 `rgba_path` / `mask_path` / `debug_path`
是相对 `pieces.json` 所在目录的 POSIX 路径，`inputs` 四个路径是仓库根相对 POSIX 路径
（仓库外输入才退化为绝对路径）。这样整份 JSON 不绑定本机盘符，换机器/换目录都能直接读。

## 5. 独立验证

验证脚本 **`docs/verify_stage2.py`**（59 项检查，只读仓库、只写 `tmp/`）用一套与被检结果无关的
参考量做对照：

- **参考四边形（grad-quad）**：对每条边取 13 条剖面，用灰度梯度的最大值 + 抛物线细化定位，
  再 TLS 拟合直线、相邻求交。它不使用 `pieces.json` 的任何角点，因此可以给不同版本的提取器排名；
- **伪真值**：把每个碎片当作 `jigsaw_layout.json` 里某个源多边形的刚体相似副本
  （旋转 + 等比缩放 + 平移，不允许镜像），用 16×16 匈牙利匹配求 1:1 对应，再算角点残差；
- **打印比例**：从提交的 `jigsaw_16_pieces.pdf` 第 1 页的 `cm` 矩阵反解 mm/源像素
  （0.19731 mm/px），与 `px_per_mm = 8` 一起换算出画布上的期望面积。

```bash
python docs/verify_stage2.py     # 写 tmp/verify_stage2_result.json 与 overlay 图
```

> 脚本最终 verdict 会是 FAIL，因为其中两项判据是“相对 `jigsaw_layout.json` 的伪真值”
> （角点 < 2 px、IoU > 0.98）。而脚本自己也会给出 WARN：这份数据是**手剪**的，
> 同一组统计量在照片上对 layout 多边形就已经是 mean 3.9 / max 10.7 px，
> 所以这两项对任何提取器都不可达——它们衡量的是“剪得像不像印刷线”，不是提取精度。
> 真正说明精度的是下面的边缘偏移与对角参考的角点误差。

### 5.1 三个版本的客观对比（参考 = grad-quad，64 角点 / 64 条边）

| 版本 | 角点误差 mean / median / p90 / max (px) | ≤1 px 角点 | 边缘有符号偏移 mean / median (px) | 边缘 \|偏移\| mean / median / p90 / max (px) | 边缘 ≤1 px |
| --- | --- | --- | --- | --- | --- |
| 1.0.0（半高交点） | 1.743 / 1.185 / 4.272 / 9.197 | 27/64 | −0.079 / −0.070 | 0.712 / 0.443 / 1.447 / 3.635 | 52/64 |
| 1.1.0（梯度峰，边缘响应） | 1.656 / 1.131 / 2.446 / 12.351 | 24/64 | −0.660 / −0.647 | 0.955 / 0.678 / 1.464 / 5.853 | 50/64 |
| **1.2.0（梯度峰，颜色响应 + 单边保护）** | **1.583 / 0.859 / 3.472 / 9.281** | **37/64** | **+0.047 / +0.055** | **0.587 / 0.364 / 1.231 / 3.622** | **55/64** |

结论：1.2.0 的**系统偏差消失**（有符号偏移 +0.05 px，四种绝对边缘定义之间只差 0.14 px），
单边绝对偏移的中位数 0.36 px，55/64 条边 ≤ 1 px；角点误差中位数 0.86 px。
与 1.0.0 相比 16 片里 13 片严格变好，没有一片变差超过 0.11 px。1.1.0 的“面积真值 +0.03%”
其实是 0.69 px 内缩抵消出来的假精度（面积靠内缩“变准”，但每条边都偏了）。
交付版本 1.2.1 的几何与 1.2.0 逐字节相同（48 张产物图 SHA256 一致），因此这一行就是交付几何的实测值。

### 5.2 面积与打印尺度

| 量 | 1.0.0 | 1.1.0 | 1.2.0 |
| --- | --- | --- | --- |
| 检测总面积 / 期望面积 | 1.00662 | 1.00027 | **1.00755** |
| 每条边相对打印线的偏离 | — | — | 约 0.2–0.24 mm 外侧 |

拟合出的相似尺度比 PDF 反解的期望值大 **+0.344%**，与面积的 +0.76%（线性 2 倍）自洽：
这批碎片是沿印刷线**外侧**剪下来的，属于真实几何，不是提取误差，因此不应去“修正” `px_per_mm`。

### 5.3 其它检查

- 编号稳定性：本次的 `pieces.json` 文件顺序等于按 (中心 y, 中心 x) 排序，16 片匹配到的 layout 多边形
  与 1.0.0 完全一致（S14/S15 的交换已修复）；S14/S15 中心 y 相差 1.03 px，行聚类后按 x 排序正确。
- mask / rgba 一致：`mask == alpha`，无桌面像素渗入；`quality_breakdown.mask_iou` 为 0.970–0.986。
  注意这个 IoU 是**提取器自报**（细化多边形 vs 粗掩膜），仓库里没有人工标注掩码，因此 README §4.4
  的“mask IoU > 0.98（对人工标注）”这一条**没有被独立认证**；保存的 `mask.png` 本身按定义就是
  所报多边形的像素栅格（重建 IoU ≈ 1.0），拿它再比一次是循环论证。
- 确定性：同一输入连续两次运行，`output/` 下 50 个文件 SHA256 全部相同。
- 单元测试：`python -m unittest discover -s tests -v` → **64 项全绿**（本部分 22 + `tests/test_backgrounds.py` 30 + 阶段一 12）
  （`tests/test_stage2_extraction.py` 22 项：纯函数、合成 16 片端到端（含确定性）、真实数据集成；
  另有第一阶段 12 项）。合成用例里 16 片的角点误差 ≤ 1.03 px。
- 人工检查：`docs/stage2_preview.png`（16 片归一化缩略图）与每片 `S*_debug.png`
  显示轮廓贴纸边、无桌面残留；`tmp/crop_S06.png` 是难点碎片 S06 的放大对照。

### 5.4 验证脚本的预期输出与判据口径

先对 README §4.4 的五条验收逐条表态，免得两个红色 FAIL 被误读成整体不达标：

| §4.4 验收项 | 结论 |
| --- | --- |
| 16/16 检出 | ✅ 满足 |
| mask IoU > 0.98（对人工标注） | ⚠️ **无法认证**：仓库没有人工掩码标注；自报 `mask_iou` 0.970–0.986 |
| 角点误差 < 2 px | ✅ 对**照片轮廓**满足（median 0.86 / mean 1.58 px）；❌ 对 **layout 伪真值**不可能满足（mean 4.07 / max 10.66 px，手剪差距） |
| 编号/朝向稳定、质量分 + 人工复核标记 | ✅ 满足（行聚类编号，S06 进 `manual_review` 并带原因） |
| 不误检阴影/桌沿/灰尘/印刷图案 | ✅ 满足（16 片，无额外连通域，`discarded_candidates` 为空） |

`python docs/verify_stage2.py` 在这份冻结数据上的**预期**结果是
`SUMMARY: 59 checks, 2 FAIL, 7 WARN`、verdict FAIL、**退出码 1**（不要把它当成复现失败）：

- 2 个 FAIL 都是“相对 `jigsaw_layout.json` 的伪真值”判据（角点 mean 4.074 / median 2.776 / max 10.663 px；
  IoU mean 0.98349 / median 0.98893 / min 0.94785），而脚本自己会用 WARN 说明：同样的统计量在照片上
  对 layout 就已经是 mean 3.897 / median 2.694 / max 10.745 px，所以这两条对任何提取器都不可达，
  它们衡量的是“手剪得像不像印刷线”。角点精度应看**对照片轮廓**的那一项（mean 1.583 / median 0.859 px）。
- 7 个 WARN 也都是稳定且预期的：mask 循环性、mask IoU 无法独立评估、起始角点局部规则 5/16、
  规范朝向偏差最多 21.02°、对照片四边形的一致性 1.735/1.182/6.235、两种照片估计量互差
  median 0.955 / max 5.415（仅 S06 超 3 px）、伪真值前提本身不成立。
- 判读上的两个坑：
  1. **绕向必须先规范化**。`jigsaw_layout.json` 的多边形是 y-up 存储的（在 y-down 像素坐标下鞋带面积为负），
     而 `pieces.json` 按约定是正面积；不先按有符号面积统一绕向，无镜像相似拟合会给出 200+ px 的残差，
     看起来像灾难性失败。
  2. **只有 mask-free 的 `grad-quad` 参考能用于跨版本比较**。以掩码/所报多边形为起点的参考
     （TLS 侧边重拟合、IoU 最优四边形）会被起点带偏：同一份 v1.2.0 数据，mask-free 角点
     mean/median/max = 1.583/0.859/9.281，mask-TLS = 1.735/1.182/6.235。§5.1 的表固定用 mask-free 参考。
- 脚本的独立性：不 import 任何 Stage 2 代码（只用标准库 + cv2/numpy/scipy），输入只有阶段一的 JSON/图像
  与阶段二产物；自检只在合成相似变换上验证匹配器（0.25 px 边缘噪声下 0.507 px），是 sanity floor，
  不对提取器或照片参考做精度声明。
- 面积期望值的推导（如需改画布/分辨率请重算）：期望总面积 = 921,600 源像素² ×
  (0.197315 mm/源像素 × 8 px/mm)² = 2,296,367 px²，其中 0.197315 由脚本从提交 PDF 的 `cm` 矩阵
  （715.9261×402.7085 pt / 1280×720）独立反解，和文档值相对误差 0.000%。
- 归档文件重映射：1.1.0 的 `pieces.json`/诊断里 `S14`/`S15` 编号是交换的，重新推导 A/B 时若不做
  物理碎片重映射会得到不可能的 134 px 误差（证据：拟合尺度交叉匹配 1.5611↔1.5647、1.5852↔1.5844）。

## 6. 已知限制

- 亚像素精度的物理下限来自**手剪**：单条边的直线度（同侧采样点 p90−p10）为 1.6–5.8 px，
  最短边 66 px，最锐角 53.8°；角点误差等于边缘偏移被 `1/sin(θ/2)` 放大（典型 ×2.7）。
  所以 S06 残留的 5–9 px 角点误差是数据属性（verifier 的两种独立估计在该片上也互差 5.4 px），
  不是算法没调好；它走 `short_side` 人工复核。
- 前景阈值依赖两张图严格同机位同光照；换拍摄条件需要重标 `threshold_max`。
- 碎片相互接触/叠放会合并，检出数 < 16 → 退出码 1。
- 归一化朝向上，近似正方形/圆形的主轴不稳定；Stage 3 仍会搜索旋转，因此不影响正确性。
- **归一化后的短边小于 README §2.1 的建议值**：`normalized_short_side_px` 中位 220 px，S06 = 45 px、
  S14 = 76 px、S13 = 163 px（建议 300–400 px）。原因是"装进 512 画布"按长边定标（缩放 0.69–0.95），
  薄片的短边被同比缩小；而 S06 在 8 px/mm 整流图里的原生短边本来就只有 66 px，归一化到 300 px
  就意味着长边要 3100 px，与 512 契约冲突。若 Stage 3 需要更高分辨率纹理，应回到
  `pieces_rectified.png` 用 `corners_px`（或 `rectified_to_piece_matrix` 的逆）在原生分辨率上裁剪。

## 7. 文件清单（本部分新增/修改）

| 路径 | 说明 |
| --- | --- |
| `stage2_piece_extraction/extract_pieces.py` | 主体实现（约 1300 行，算法版本 `stage2-extract-1.3.0`） |
| `stage2_piece_extraction/config/extraction.example.json` | 完整参数（含 `edge_localizer`、`max_side_shift_px`、`fail_on_manual_review`） |
| `stage2_piece_extraction/README.md` | 中文实现文档：参数表、算法、契约（含路径解析约定）、实测结果、限制 |
| `stage2_piece_extraction/input/README.md`（修改） | 说明默认直接读阶段一结果、此目录只用于打包/回归夹具 |
| `stage2_piece_extraction/output/README.md`（修改） | 产物清单 + 阶段三消费方式（路径相对 `pieces.json`、薄片高分辨率用法） |
| `stage2_piece_extraction/output/pieces.json` + `output/pieces/*.png` | 交付产物（50 个文件） |
| `tests/test_stage2_extraction.py` | 22 项测试（合成端到端 + 真实数据集成） |
| `docs/verify_stage2.py` | 独立验证脚本（59 项检查，只写 `tmp/`） |
| `docs/stage2_preview.png` | 16 片归一化缩略图 |

## 8. 后续扩展：换底面鲁棒性

阶段一只在黑绒布 bench 上采集，为了检验“换一张桌子还行不行”，另加了 `backgrounds/`：
程序化生成木地板 / 大理石 / 水泥三种底面，把**同一批真实纸片**按 `pieces.json` 的几何重新合成上去，
于是每个合成场景都自带真值。结论：铺黑色垫布时精度与 bench 同级（角点中位 0.39-0.40 px、IoU≈0.993），
裸木地板次之（中位 2.28 px），裸大理石/水泥因投影并入掩膜而失败（物理限制，已量化）。
方法与数据见 [`docs/BACKGROUNDS.md`](BACKGROUNDS.md)；`backgrounds/output/` 约 130 MB，已 gitignore。

生成的产物与第一阶段一致地**保留在工作区但未提交**；若要回到原始状态执行
`git checkout -- stage2_piece_extraction/output`（产物本身是 untracked，直接删除即可）。

`tmp/` 被 `.gitignore` 忽略，里面是本次迭代的诊断脚本与中间证据（`verify_stage2_report.md`、
`diag_ab_v120_*.{json,md}`、`crop_S06.png` 等），属于“本地产物、可重新生成”，不进仓库；
`docs/verify_stage2.py` 是其中唯一被固化下来的入口。
