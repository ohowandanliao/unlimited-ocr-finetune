# 输入 mode:base/gundam × single/multi（现状、目标、怎么训）

## mode 是什么
输入图像的分辨率/切块策略，决定 image token 数、能看多细、一次几页。
- **base**：整张 1024×1024，不切块，~256（本模型实测 273）token。省 token、小字易糊。
- **gundam**：动态切块高清 —— n×640×640 局部 crop + 1×1024×1024 全局，token = n×100 + 256。本模型 gundam 用 12 块（3×4 网格）→ ~1513 token，看得清小字。
- **single / multi**：一次喂 1 页 / 多页。

## Unlimited-OCR 只保留 2 种 mode（DeepSeek-OCR 原生 5 种 → 砍到 2）
- **single_gundam**：单页 → gundam 高清切块（crop_mode=True, base_size=1024）。
- **multi_base**：多页 → base 1024、**不切块**（多页若每页都切块，token 会爆）。
- 即 **单页用 gundam、多页用 base**；R-SWA（滑窗）负责长文/多页的 KV cache 不爆。
- 对比：DeepEncoder 原生 5 种（Tiny512 / Small640 / Base1024 / Large1280 + Gundam），Unlimited-OCR 只留 base + gundam。

## DeepSeek-OCR 是「多 mode 混合训练」
- 5 种分辨率 mode **一起训**（gundam 与 4 个原生 mode 同时训），让**一个模型支持所有分辨率**，推理时按需选 token 预算（这就是"contexts optical compression"的灵活性）。
- 训练分阶段：Stage1 单训 DeepEncoder（视觉）；Stage2 编码器-解码器联合训，数据是 OCR+视觉+文本混合；再 SFT。
- 结论：**不是只用 gundam 训 —— 是多 mode 混着训。**

## 我们的现状 vs 目标
- **现状（smoke）**：只训 `single_gundam` 一种（configs 里 `mode:` 写死，覆盖数据里的 mode）。这是简化，不是最终方案。
- **风险**：只训一种 mode 可能弱化其它 mode（尤其全参微调时）；LoRA 冻结 base 相对安全。
- **目标 = multi_gundam**（多页 + 高清切块）：**超出 Unlimited-OCR 原生能力**（它多页只用 base）。这是真正的差异化/北极星，但需新写 forward 的多页 crop 分支（见 DESIGN.md「另计」）。
- **建议**：正式训练应**混 mode**（至少 `single_gundam` + `multi_base`，对齐 Unlimited-OCR 的两种），而非 gundam 一种；`multi_gundam` 作为新增能力单独实现后再并入混合训练。

## 来源
- DeepSeek-OCR 论文 arXiv:2510.18234；GitHub `deepseek-ai/DeepSeek-OCR`
- Unlimited-OCR：`baidu/Unlimited-OCR`（HF）、MarkTechPost 报道、`deepwiki/baidu/Unlimited-OCR`
