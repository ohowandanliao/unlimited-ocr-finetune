# 多页 OCR 训练可行性验证（2026-07-16）

目标：验证 Unlimited-OCR 能否通过后训练学会"多页 PDF 页图 -> 一份跨页合并的 markdown"
（北极星形态：连续、无 `<PAGE>` 分隔、跨页表格/段落接续）。定位：小而快的**可行性探针**，
不是建大语料、不追求产品质量。

---

## 结论（TL;DR）

1. 训练**管线**可行：多页输入能读（3–11 页全读、按页序、OCR 内容准确）、R-SWA 在多页序列生效、能训不 OOM。
2. "多页 -> 合并 markdown"的**输出行为没训出来**：自由生成仍是基座**原生格式**
   （`<PAGE>` 分页 + `<|det|>类型 [bbox]<|/det|>` 版面 grounding），不是干净合并 markdown。
3. **teacher-forcing 训练 loss（2.21 -> 0.25）是假象**：只说明"给定 GT 前缀能预测下一 token"，
   不代表生成行为改变。差点凭它误报"可行=是"。
4. 要真做出来：需**大得多的训练**（几百上千篇数据 + 更多步/epoch + 可能 full-decoder 或更高 LoRA rank）。
   本质是"格式/行为迁移"，基座原生 grounding 先验极强，28 样本 / 40 步 / r16 的小 LoRA 完全掰不动。

---

## 做了什么

### 1. arxiv2md 自建清洗管线（已弃用，**保留作记录**，`scripts/arxiv2md/`）

做法：arxiv e-print(LaTeX) -> pandoc 转 markdown -> 清洗掉源码态残留(citation、交叉引用、
`{#attr}`、`:::` fenced、HTML、图) -> 得"渲染态近似"markdown 当 target。

**它不是被某个技术死结卡死的，是"用错了力气"。两层原因：**

**(a) 技术上脆——让人在它上面耗太久的直接原因。** 早期是拿 pandoc 输出的 markdown **字符串**
再用正则清洗，等于把 pandoc 已解析过的语法重新识别一遍，天生脆：
- **公式吞正文**：`$`/`$$` 配对被相邻或畸形的 `$$` 翻转，把两个公式块之间的大段正文连同 markup
  一起当公式吞掉、逃过清洗。修了好几版(合并正则 -> 左到右扫描器 -> "闭合 `$$` 超 2000 字符判错配"
  启发式)，典型论文治好了，但**重结构论文(定理/算法环境、多行 align)仍治不干净**。
- 还有：转义 `\$5`(钱)被当公式、pandoc 表格被"多空格收拢"毁掉、`f()` 被空括号清理误删、
  HTML 标签正则无词边界(`<emphasis>` 被当 `<em>`)等。
- codex(gpt-5.6-sol) review 列出这些，根治建议是**别在字符串上清、改用 pandoc AST(Lua filter)**。
  照做后在**合成测试样例**上把上述坑都修好了(表格保住、`\$5` 正确、`f()` 不被杀)——**所以技术上它没"死"，
  Lua 版看着能用；只是还没用它跑完整批 build 就转向了**(未经完整语料验证)。

**(b) 用错力气——真正被弃的策略原因：**
- **重复造轮子**：调研发现 `marcodsn/arxiv-markdown` 已用 docling 把 3269 篇 arxiv 转成文档级 markdown、
  现成可下。自建 arxiv->pandoc/Lua 本质在重做别人做好的事。
- **超出定位**：这是"造数据/打标"的工程，而本次任务定位是小而快的可行性验证(已被叫停"别过度打磨管线")。
- **同样的质量天花板**：pandoc 与 docling 的 target 都是"转换器近似"、非 PDF 渲染真身(引用编号等还原不了)。
  既然都有此限制，用现成的比自维护一套脆管线划算。

**状态与去留**：整条弃用、转 Path A(marcodsn)。**代码保留作记录不删**——若哪天要追求
比 docling **更高保真**的 target(公式逐字、跨页更准)，这套 **Lua-AST 思路**(或 Nougat 的 LaTeXML 路线)
仍有捡回价值。文件：`scripts/download/fetch_arxiv.py`(下载,已入 download/) + `scripts/arxiv2md/{build_dataset,build_batch,clean_md,clean.lua}` +
相关 config，均已于 `6aae56d` 提交并 push（2026-07-20 核实）。

### 2. 选 Path A：用现成的 marcodsn/arxiv-markdown 当 target
调研（deep-research）结论：没有大量现成的"多页->合并 markdown"数据；文档级 markdown 主要来自
arxiv-source 派生集。选 `marcodsn/arxiv-markdown`（HF，CC-BY-4.0，单个 96MB parquet，3269 篇，
`{arxiv_id, markdown}`，docling 生成的整篇连续 markdown = 天然合并形态）作为 target，最省事。
（备选先例：OCRFlux 的"整表切开-再合并"配方，是真跨页合并数据的可借思路，留待后续。）

### 3. 数据管线（新，`scripts/marcodsn/`）
- `build_marcodsn.py --phase select`：读 parquet 挑 N 篇（按 target 字符长度过滤），产 selected.jsonl + ids。
- `scripts/download/fetch_pdfs.py`（Mac 跑，arxiv 直连快）：按 id 下 PDF（只 PDF、不要 e-print，不碰 LaTeX），magic 校验 + 退避重试。
- `build_marcodsn.py --phase build`：fitz 渲每页图，组 multi_base 样本
  （输入=页图，target=marcodsn markdown 剥掉 `![Image](url)` 图链，prompt=`<image>Multi page merge.`）。
- 产出：**28 个多页合并样本**（2–12 页/篇，target 4.5k–17.4k 字符，132 张页图）。控 target<=25k 字符以适配上下文。

### 4. Smoke 训练
`configs/marcodsn_smoke.yaml`：lora_decoder + `rswa_train: true` + GC，40 步。
结果：loss first=2.2105 last=0.2495 finite；R-SWA mask 在多页 2054-token 序列 APPLIED；
84 LoRA 模块 / 3.98M 可训（routed experts=0）；无 OOM；adapter 存 `outputs/marcodsn_smoke`。

### 5. 生成 eval（关键，`scripts/marcodsn/eval_merge.py`）
合并 adapter 后用 `infer_multi`（多页 + merge prompt + 全页图，save_results=False 取原始生成）在 3 个样本上生成，与 GT 比。
- 相似度 0.18 / 0.67 / 0.58（均值 0.47），格式差异主导。
- 生成实际是**原生格式**：`<PAGE><|det|>title [336,97,670,113]<|/det|>...`，带版面 grounding + 分页。
- 而全部训练 target 是 docling 干净 markdown（`## 标题` + 正文，**无 `<|det|>`、无 bbox、无 `<PAGE>`**）。

---

## 为什么"现在就能坐实"（不用再跑 base 对比）

生成输出里的 `<|det|>[bbox]<|/det|>` grounding + `<PAGE>` 这套 markup，**在 28 篇训练 target 里一个都没有**。
若 LoRA 对生成有任何实质影响，输出至少会出现向"干净 markdown"偏移的痕迹；但输出**零偏移、纯原生**。
故"输出纯原生"本身即证明：smoke LoRA 没改动生成分布。base 对比只会重复确认，无需再跑。

---

## 教训

1. **别信 teacher-forcing 训练 loss** 判"学会没"——必须跑自由生成 eval 看实际输出。这次差点被 loss 骗成"成功"。
2. **autodl 从开机就计费**（不看 GPU 用没用）。数据准备（下 PDF、渲页图）该在服务器**关着**时于 Mac 侧做，服务器只为训练/eval 那几分钟开。本轮此处安排失当、有浪费。
3. 基座**原生 grounding 先验极强**：小规模后训练掰不动，要改输出格式/行为需大得多的训练压力。

---

## 下一步（若继续推进）

> 注（2026-07-19）：用户已拍板**维持、不走量**。下列均为"若将来重启放量"的路线图，非当前待办；当前决定以 `../status.md` 为准。

- 放量数据：几百上千篇（marcodsn 有 3269 篇可选；Path A 管线现成、按需扩 N 即可）。
- 加训练压力：更多步/epoch、更高 LoRA rank、或 full_decoder（48G 单卡 full_decoder ~53G 差一点，需 ZeRO/FSDP 或换卡）。
- 数据/prompt 上加强区分度，帮模型条件化到"合并"而非原生。
- 每轮小规模先跑**生成 eval** 验行为是否真变，再决定是否放量。

---

## 产物位置（双份，关机不丢）

- 数据集 + adapter：服务器 `autodl-fs:/autodl-fs/data/unlimited-ocr-finetune/{data/marcodsn_v1, outputs/marcodsn_smoke}`（持久）
  且 Mac `/Volumes/SharedData/marcodsn_v1/`（train.jsonl 28 样本、images 132、pdfs、adapter_marcodsn_smoke）。
- 脚本：`scripts/download/fetch_pdfs.py`（下载）+ `scripts/marcodsn/{build_marcodsn.py, eval_merge.py}` + `configs/marcodsn_smoke.yaml`（已于 `6aae56d` 提交并 push）。
- marcodsn parquet：`autodl-fs:/autodl-fs/data/marcodsn-arxiv-markdown/train.parquet`（92M，可从 hf-mirror 重下）。
