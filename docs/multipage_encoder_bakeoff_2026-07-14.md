# 视觉编码器路线核实 + bake-off：切块(gundam) vs 原生分辨率

日期：2026-07-14
状态：**调研 + 决策记录**。本文是 `DESIGN.md §0.5「北极星：多页统一 markdown」` 的补充档，只记 §0.5 没覆盖的三件事：①竞品(HunyuanOCR 1.0/1.5)核实；②「切块 vs 原生分辨率」两条轴的分析与「光学压缩」反转；③一个用来终结争论的 bake-off 方案。R-SWA 的机制事实见 `DESIGN.md §1`，北极星/跨页合并/数据策略见 `DESIGN.md §0.5`，不在此重复。

> **决策（2026-07-14 已定）**：multi_gundam **暂缓**（不现在建）；**先用 `multi_base` 跑通「多页 → 一份合并 markdown」**。bake-off（§6）与 multi_gundam（§7）留作后续，由「multi_base 撞分辨率天花板 / 确需多页高清」触发再启。理由链见全文。

---

## 0. TL;DR

1. **目标没变**：多页 PDF/图 → 一份跨页合并的统一 markdown（不是逐页 `<PAGE>` 拼）。这仍是市面空白——**连 HunyuanOCR-1.5 都没做**（它的多页是 QA，不是合并解析）。
2. **两条轴要分开看**：轴①视觉编码器（切块 gundam vs 原生分辨率）；轴②解码器长上下文（R-SWA vs 稠密 128K）。「DeepSeek 视觉不最优」和「R-SWA 可行」是两条轴上的两句话，别混。
3. **R-SWA 的成本瓶颈在 reference（图像），不在输出**：输出侧 KV 恒定（利于长合并 markdown ✓）；图像 reference 全程被每个输出 token 全注意力、不 window（`DESIGN.md §1`）。=> **每页视觉 token 数 × 页数 = 真正的预算瓶颈。**
4. **「光学压缩」反转**：既然 reference 是瓶颈，**每页 token 效率**才是最该优化的轴，而 DeepEncoder/gundam 恰是为此而生（SAM→16× 卷积压缩→CLIP）。真实文档页上 gundam(~1513 tok) 比原生分辨率(~2500+ tok) **更省**。所以「换原生分辨率」在最疼的地方（reference）反而更贵——**这给 gundam 一个真正站得住的理由，但前提是北极星是 token 效率、不是峰值精度。**
5. **唯一未决的是一条经验曲线**：gundam 的压缩是有损的，「同 token 预算下 gundam vs 原生谁精度高」对我们的文档**未测**。→ 用第 6 节的 bake-off 测出来，一张 Pareto 图终结三个问题（gundam 值不值 / 要不要原生 / 要不要 multi_gundam）。

---

## 1. 两条轴（理解全局的框架）

| 轴 | 选项 | 决定什么 | 现状判断 |
|---|---|---|---|
| ①视觉编码器 | 切块(gundam) vs 原生分辨率(native-res) | 单页精度、每页 token 数 | DeepSeek 切块**峰值精度**不占优，但**token 效率**可能占优（见 §4） |
| ②解码器长上下文 | R-SWA（滑窗+全 reference，KV 恒定） vs 稠密 128K | 能塞几页、能吐多长合并 markdown | **R-SWA 是 UOCR 值得留的资产**（长合并输出恒定内存） |

UOCR = 切块视觉 + R-SWA。HunyuanOCR = 原生分辨率 + 稠密 128K。两者各占一轴，没有谁全面赢。

---

## 2. 竞品核实：HunyuanOCR 1.0 / 1.5（2026-07-14 查）

**HunyuanOCR-1.0**（arxiv 2511.19575，权重已开源 `tencent/HunyuanOCR`）
- 1B：0.4B **原生分辨率 ViT**（基于 SigLIP-v2-400M，自适应 patch，任意分辨率）+ 0.5B 混元 LLM + **XD-RoPE**（text/H/W/time 四子空间）+ 自适应 MLP 压 token。
- **单图**模型；五大任务（检测识别 / 文档解析→md·html·latex / 字段抽取→json / 视频字幕 / 拍照翻译）。文档解析是**对一张图**出 markdown。
- License = **Tencent Hunyuan Community License**（总体可商用，有 MAU 上限等条款，需逐条看）。

**HunyuanOCR-1.5**（arxiv 2607.04884，约 2026-07；权重"will release"，可能尚未放）
- **同 backbone 不重设计**；ViT 最大分辨率 **2K→4K**；context **扩到 128K**；**DFlash** 推理加速（Transformer 6.37× / vLLM 2.14×，OmniDocBench 上 0.706 页/秒 ≈ DeepSeek-OCR 的 3.88×）。
- **OmniDocBench v1.6 Overall = 94.74，端到端 OCR 专家模型 SOTA**（**单页**解析）。古文字/图表/多语种长尾全面强过 DeepSeek-OCR。
- **多页 = 多图/多页 QA**（benchmark 是 DUDE：跨页检索、比较、证据聚合），**不是**「N 页→一份合并 markdown 的解析」。论文**未提** reading order 跨页保持、跨页表格/段落合并、多页拼一篇。

> **结论**：1.5 补上了单页精度、速度、多页*理解*；**唯独没补「多页→合并 markdown *解析*」**。我们的北极星（`DESIGN.md §0.5`）**挺过了 1.5，仍是空白**。
> 注意：arxiv 页标的 `CC BY-SA 4.0` 是**论文**协议，不是**权重**协议。

---

## 3. 「原生分辨率」≠「得用 NaViT」

- **NaViT**（2023 Google）只是原生分辨率的一个祖师爷配方，招牌是 **Patch-n-Pack**（多图打包进一条序列）+ factorized pos-emb + token dropping。**生产里赢下来的不是它的打包，而是 RoPE 化的动态分辨率**：Qwen2-VL/2.5-VL（2D/M-RoPE）、SigLIP2-NaFlex（Hunyuan 的底子）、Pixtral、Hunyuan XD-RoPE。
- **你不会去*实现* NaViT，只会*继承*一个原生分辨率基座**（Qwen2.5-VL 最稳/许可宽松；HunyuanOCR-1.x 同量级专用）。
- **两层解耦**：
  - *原生分辨率* = ViT 层、**单页内**的事 → 每页独立编码，继承现成的即可。
  - *多页* = LLM 层、**跨页**的事 → 各页视觉 token 顺次拼进 context，靠 M-RoPE/XD-RoPE 的 page/time 维 + 长 context 建关系。**差异化（合并 markdown / 跨页缝合）全在这一层**——即 `DESIGN.md §0.5` 说的、R-SWA「边看原图边输出统一 markdown」的角度。
- => NaViT 的「打包多图」不是多页能力的来源；多页来源是 LLM context。**若换基座，差异化工作几乎全在解码器侧训练，视觉侧白嫖原生分辨率。**

---

## 4. 「光学压缩」反转：为什么 gundam 在 R-SWA 里可能*更省* token

R-SWA 里图像 reference 全程被 attend、不 window（`DESIGN.md §1`），所以 **reference 的每页 token 数是硬瓶颈**，且要 ×页数。而 **DeepEncoder（SAM→16× 卷积压缩→CLIP）整套就是为「每页最少 token 装下可 OCR 信息」设计的**（DeepSeek "contexts optical compression" 的命题）。原生分辨率 ViT 优化保真/灵活，不是极限压 token。

**每页视觉 token（估算，10 页 = reference 规模）：**

| 模式 | 每页 token | ×10 页 | 备注 |
|---|---|---|---|
| multi_base | ~273 | ~2.7K | 最省，小字糊 |
| single/multi_gundam(12 块) | ~1,513（n×100+256） | ~15K | 中，较清（= 旧 plan B3 的 ~15K） |
| 原生分辨率(Qwen 类，真实 A4 页) | ~2,500–3,000（估） | ~25–30K | 保真，token ~2× 于 gundam |
| 原生分辨率 4K(Hunyuan 类，压缩前) | 数千–上万 | 40K+ | 最清，reference 直接爆 |

**读法**：真实文档页上 gundam 比原生分辨率**省 ~1.5–5×**，且这个倍数在多页下被直接放大。**在 reference 是瓶颈的 R-SWA 里，这正是最该省的地方 → gundam 的光学压缩有真优势。**

**但两个 caveat（别当免费午餐）：**
1. gundam 的省来自**有损压缩**（16× 卷积会丢小字）——省 token 与掉精度是同一枚硬币，是 **Pareto 权衡**不是全面赢。
2. 「同精度下 gundam 是否真比原生省」对我们的文档**未测**。DeepSeek 说它的压缩 Pareto 更优；Hunyuan 的 OmniDocBench 分更高（峰值）；且 Hunyuan 的 adaptive MLP 压缩比**未公开**。→ 必须测。

> 反转的意义：**若北极星是「token 高效地吃很多页」，UOCR（DeepEncoder gundam + R-SWA）是自洽且匹配的架构，gundam 拿到真正的理由；若北极星是「单页峰值精度」，则该考虑换原生分辨率基座。** 这条北极星必须先认领。

---

## 5. 关键未知 = 一条经验曲线

把前面所有争论压成一个可测量：**在我们的目标文档上，各方案的「OCR 精度 vs 每页视觉 token」Pareto 曲线长什么样。** 谁在我们的目标 token 带里 Pareto 占优，谁就对。

---

## 6. Bake-off 方案（可执行；per-page、零训练、不用先建 multi_gundam）

**为什么 per-page 就够**：多页 reference 成本 = 每页 token × 页数（推出来的），所以编码器之争是**单页问题**。gundam 那条臂**用现成 single_gundam**（扫 tile 数得多个点），**无需先实现 multi_gundam**；R-SWA/训练这一轮全不碰。

**臂（同量级才公平）：**
- **A. single_base**（UOCR）——~273 tok 的省 token 地板。
- **B. single_gundam**（UOCR）——扫 base_size / 最大 tile 数 → {base273, gundam-6, gundam-12…} 多点。
- **C. HunyuanOCR-1.0**（`tencent/HunyuanOCR`）——**公平的原生分辨率臂**（同 1B、同 OCR 专用、零训练）。
- **D.（可选）Qwen2.5-VL**——原生分辨率「天花板参照」，规模是混淆变量，只看上限不下结论。

**怎么扫 token 轴（原生分辨率臂）**：token ∝ 喂进去的像素 → **控输入分辨率**。
- 通法：同一页缩放到不同长边（如 {768,1024,1536,2048,原生}），每档一个 (token, 精度) 点。
- Qwen2.5-VL 更干净：设 processor `max_pixels/min_pixels`，token 数确定可控。
- ⚠️ 有的模型有最小分辨率地板 → **token 数一律实测，别按名义分辨率算**。

**统一 token 计量（一把尺量三家）**：`vision_tokens = (input_ids == image_pad_id).sum()`（= processor 展开的 image 占位符数；UOCR 侧即 `image_seq_mask.sum()`）。

**精度侧**：
- 测试集：~100–300 张带 GT markdown 的单页（olmOCR-mix 已有 / OmniDocBench），**务必含密排小字页**（切块 vs 原生的差异活在小字上）。
- 指标：文本用归一化编辑相似度（复用 `eval_infer`），表格 **TEDS**，公式可选 CDM。
- 公平化：所有臂**统一输出纯 markdown、同指标、同批页**；strip 掉 `<PAGE>` 等标记；每模型用**其推荐 prompt**（prompt 是混淆源）。

**产出**：一张图，x=每页视觉 token(log)、y=精度、每系统一条线。读目标 token 带内谁 Pareto 占优：
- single_gundam 在 HunyuanOCR **左上方** → 光学压缩假设成立，**留 UOCR，认领 compression 北极星**。
- 重合 / Hunyuan 更优 → 省的 token 不值精度，**考虑换原生分辨率基座**。
- base 已够 → 连 gundam 都省，**只做输出侧合并 markdown**。

---

## 7. multi_gundam 定位与决策状态

与 `DESIGN.md §0.5` 的踏脚石一致（§0.5 已把 multi_gundam 列为「正交能力向坑，另计」），本轮进一步收敛：

- **justification 更新**：multi_gundam 的理由**不是「切块提分辨率」本身，而是「在 R-SWA reference 里 token 高效地做多页高清」**（§4）。它只在「北极星=token 效率」且「目标文档需要多页高清」时才成立。
- **R-SWA ≠ 救 multi_gundam**：R-SWA 只 window 输出，不 window 图像 reference；multi_gundam 的成本在 reference 侧（页×tile），R-SWA 不减（旧 plan B3 的 ~15K 照旧）。所以「R-SWA 让 multi_gundam 可行」这一步**不成立**——正确说法是「reference 是瓶颈 → 要选每页最省的编码器」。
- **无悔动作（与 multi_gundam 解耦、可先做）**：`multi_base + 输出侧训练成合并 markdown`（干掉 `<PAGE>`，跨页缝合）——直接命中北极星、吃 R-SWA 长输出优势、避开 token 爆炸。对应 §0.5 踏脚石 2/3（跨页表格合并 → 统一 markdown）。
- **go/no-go（2026-07-14 已定）**：**multi_gundam 暂缓，不现在建。** 先用 `multi_base` 跑通「多页 → 一份合并 markdown」（上面的「无悔动作」转正为主线）。multi_gundam 与 §6 bake-off 留作后续，由「`multi_base` 撞到分辨率天花板 / 需要多页高清」触发再启。

---

## 8. 待确认输入 + 下一步

**待确认（决定 multi_gundam 命运的三个输入）：**
1. **北极星**：token 高效地吃很多页（compression） vs 单页峰值精度？
2. **目标文档小字密度**：密排小字多不多（决定 base 够不够、要不要 gundam）。
3. **规模/预算**：一次要塞几页、reference/延迟预算 → 每页可用 token 带。

**下一步**：先跑第 6 节 bake-off（零训练、per-page），用一张 Pareto 图把 §5 的未知测掉；同时 `multi_base + 合并 markdown` 作为无悔动作可并行推进。

---

## 9. 参考

- HunyuanOCR-1.5：https://arxiv.org/abs/2607.04884
- HunyuanOCR Technical Report：https://arxiv.org/abs/2511.19575
- tencent/HunyuanOCR（权重）：https://huggingface.co/tencent/HunyuanOCR
- NaViT (Patch n' Pack)：arXiv:2307.06304
- 本项目：`DESIGN.md §0.5`（北极星/差异化/数据）、`DESIGN.md §1`（R-SWA 机制）、`MODES.md`（mode 现状）、`../plan_multi_gundam_and_docs.md`（旧 multi_gundam 施工计划，含 codex review）
