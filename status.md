# Unlimited-OCR 后训练 status（活日志，持续更新）

最后更新：2026-07-16
设计权威文档：`docs/DESIGN.md`（结构稳定）。本文件是"做到哪、改了啥、踩过啥"的活日志。
北极星：多页 PDF -> 一份统一、跨页合并的 markdown（目前无此能力、公开无现成数据；只记录不实现，见 design 0.5 节）。

---

## 当前状态

干净训练工程 `train_uocr` 端到端跑通（单页）。链路：JSONL -> dataset -> processor -> collator -> model(eager) -> forward -> loss -> backward(GC non-reentrant) -> optim -> save adapter -> reload，全绿。

已实测：
- `eval_forward`：single_gundam×2、single_base、multi_base(两页) 的 forward->loss 均有限
- `lora_attn` 20 步：loss 1.20->0.05，存 adapter，reload 后真实图 forward 通过
- `lora_decoder` 3 步：84 模块(attn48 + dense3 + shared33，routed=0) / 1.99M

工程文件（`/mnt1/yixuan/unlimited-ocr-posttrain/train_uocr`）：
- src/uocr_train：constants dataset processor collator model_loader train_modes
- scripts：pull_sample_olmocr eval_forward train reload_check
- configs：smoke_lora_attn.yaml lora_decoder.yaml
- data/samples_olmocr：30 条真实 olmOCR 单页（PDF->PNG）
- outputs/smoke_lora_attn：LoRA adapter

## 训练设计要点（当前）

- **视觉全程冻结、只训 LLM decoder**。不只是选择：模型 forward 用 `no_grad` 包住视觉(modeling_unlimitedocr.py:493)，梯度进不去视觉；要训视觉得改 forward。
- **必须 eager 加载、禁 FA2**（use_mla=false 时 ATTENTION_CLASSES 只有 mha_eager）。
- **训练必须用 R-SWA mask**（`rswa_train: true`）：reference(prompt+image) 全通 + output 只 attend 最近 128，匹配推理。全 causal 训练有害（长输出崩溃，已实测，见变更日志）。实现见 `src/uocr_train/rswa.py`。
- **LoRA**：默认 lora_attn(q/k/v/o)；lora_decoder **有意排除 64 routed experts**（每 token 只激活 6/64，稀疏+爆参），只加 attn + dense_mlp(layer0) + shared_experts。
- 单卡长序列必须开 **gradient checkpointing**。

## mode 实现状态

| mode | 状态 | 说明 |
|---|---|---|
| single_gundam | 已实现+实测 | base1024/crop640，动态 crop |
| single_base | 已实现+实测 | 1024 无 crop |
| multi_base | 已实现+实测 | 多页各 1024 base view、无 crop；对齐 infer_multi，每页 273 token |
| **multi_gundam** | **未实现** | 需改 UnlimitedOCRModel.forward 的 crop 分支支持"每页各自 local crops"。分辨率向的坑（paper 自承多页只 base、40+页小字丢），与北极星正交，暂缓 |
| multi_gundam_global | 未单列 | 多页各 1024 global、无 crop，约等于 multi_base |

## 变更日志（倒序）

### 2026-07-16｜多页可行性验证:Path A + 生成 eval 坐实(详见 docs/multipage_feasibility_2026-07-16.md)
- **转向 Path A**:放弃自建 arxiv->pandoc/Lua 清洗(codex gpt-5.6-sol review 指出根子脆),改用现成 `marcodsn/arxiv-markdown`(docling 文档级 markdown)当 target。新 `scripts/marcodsn/`(fetch_pdfs / build_marcodsn / eval_merge)。
- **28 样本多页合并 smoke**:lora_decoder + rswa 40 步,loss 2.21->0.25 finite,R-SWA 在 2054-token 多页序列 APPLIED,无 OOM,adapter 存 outputs/marcodsn_smoke。
- **生成 eval 坐实关键结论**:自由生成是 **100% 原生 `<PAGE>` + `<|det|>grounding` 格式**(训练 target 里根本没有这套 markup)-> smoke LoRA 没掰动基座先验;**teacher-forcing loss 2.21->0.25 是假象**。可行性=管线可行、但"多页->合并 markdown"行为要大得多训练才出得来(几百上千篇/更多步/full-decoder 或更高 rank)。
- **教训**:别信训练 loss 必看自由生成 eval;autodl 从开机计费、数据准备(下PDF/渲图)应在服务器关着时于 Mac 做。
- **scripts 重组**:6 个下载脚本统一入 `scripts/download/`(fetch_arxiv / fetch_pdfs / download_olmocr / download_olmocr_full / download_mer17m / pull_sample_olmocr);pipeline 文件夹(arxiv2md / marcodsn)只留 build/处理;DATA/GUIDE/DESIGN/usage 引用已同步。
- **待提交(等用户说"提交")**:`scripts/download/*`(下载脚本重组)、`scripts/marcodsn/{build_marcodsn,eval_merge}.py`、`configs/marcodsn_smoke.yaml`、`docs/multipage_feasibility_2026-07-16.md`。数据+adapter 双份(autodl-fs 持久 + Mac /Volumes/SharedData/marcodsn_v1)。

### 2026-07-15(续:P2 批量化 + 数据契约)
- **fetch_arxiv.py 加固(根因修)**:原完整性判据 `psz>1000 and esz>200` 把 arxiv 对未渲染论文返回的 HTML 占位页(~7666B)当成功收下。改为按内容 magic 校验(PDF=`%PDF`、e-print=gzip/tar),HTML/0字节/无LaTeX源的 PDF-only 投稿一律拒;加 `--start` 偏移避开最前沿未 ready 的;加失败退避重试扛 arxiv 限流(实测:sleep=1 首篇后连续限流返 0 字节,sleep=3+重试后 5/6 good)。
- **build_batch.py(新)**:读 manifest.jsonl 逐篇调 build_dataset,单篇失败只跳过计数不中断——补上 fetch->build 之间缺的批量 driver。
- **数据契约:合并任务另起专用 prompt**。核实 `"Multi page parsing."` 是 UOCR 专属(写在 UOCR 自己的 `infer_multi`;DeepSeek-OCR 用的是 `Free OCR.` / `<|grounding|>Convert the document to markdown.`,措辞与结构都不同)。故按"仅 UOCR 用则另起"原则,新增 `DEFAULT_MERGE_PROMPT="<image>Multi page merge."` 给 multi_page_merge 样本,**不覆盖**原生逐页 parsing;推理可按 prompt 选合并/逐页。
- **新机(第 4 台,bjb1)**:connect.bjb1.seetacloud.com:57531;持久盘 `/autodl-fs/data` 有仓库+模型+olmOCR-mix;base conda 齐(torch2.5.1/tf4.57.1/peft0.19.1/fitz/yaml/pandoc)。SSH 别名 `uocr`。注:`nvidia-smi` 权限报错待查(build 用不到 GPU,训练前再查)。
- **进行中**:15 篇 arxiv 小批端到端验通(Mac fetch 20 -> scp raw -> 服务器 build_batch -> 抽查 train.jsonl 质量)。

### 2026-07-15
- **P2 起步｜arxiv->合并 markdown pipeline 单篇跑通**：新增 `scripts/arxiv2md/`(clean_md.py + build_dataset.py)。链路:Mac 下 arxiv PDF+e-print(222KB/s;服务器直连仅 18KB/s 会截断,故下载放 Mac)-> 服务器 build:解压 e-print、找主 tex、pandoc(`-t markdown --wrap=none --standalone`)、clean_md 清洗 -> fitz 渲每页 PNG -> 组 multi_base 样本(target=整篇连续 markdown、**无 `<PAGE>`**、task=multi_page_merge)。首篇 1706.03762:15 页 + title/abstract/正文/公式/表格/图注俱全、44525 字符。踩坑:pandoc 默认把 title/abstract 当 metadata 丢 ->`--standalone`+解析 YAML 注入;嵌套 subfigure 逐标签清;公式先抽出占位保护再清洗;图只留 caption 不进图内容(OCR≠chart 理解);author 块 pandoc YAML 太脏暂跳过;References 无 .bbl 缺失(参考文献页、价值低)。0 页坏 PDF 直接报错(不静默产空样本)。
- **新机复现（第 3 台 4090，这台 48G）**：换 autodl 新机（RTX4090 48G / 驱动580）。autodl-fs 持久：模型 Unlimited-OCR(6.4G) + 仓库 clone 都在；依赖镜像自带(torch2.5.1 / tf4.57.1 / peft0.19.1)；免密 SSH。旧 4090(/mnt1/yixuan)弃、权重不搬（可重训）。
- **P1｜multi_base 训练冒烟通过**：`make_multipage` 拼 150 个 2 页 multi_base 样本（`<PAGE>` 分隔、仅打通路径），跑 `configs/lora_multibase_smoke.yaml`（lora_decoder + rswa_train）。40 步 loss 0.82->0.19 finite，R-SWA mask 在 2 页 1199-token 序列 build 时 APPLIED，84 LoRA 模块，adapter 存 outputs/lora_multibase_smoke，train_ok。**此前 multi_base 只 eval_forward 验过 forward，这次坐实端到端能训。** 拼页只验路径，真"多页->合并 markdown"数据见下方 P2。
- **full 参数模式（补记，上个会话）**：train_modes 加 `full_backbone`（attn + dense_mlp + shared_experts、正则排除 routed experts）+ full 模式 fp32 master 上溢；`full_decoder` 解冻全部 model.layers。full_backbone 存档 6.6G（本地副本已删）。

### 2026-07-09
- **开源**：从 train_uocr 抽出干净仓库 `unlimited-ocr-finetune`（27 文件，仅训练代码+configs+DESIGN，无权重/数据/env；模型路径读 `UOCR_MODEL_DIR` 环境变量；README 以 R-SWA 训练为核心）。已 push 到 `github.com/ohowandanliao/unlimited-ocr-finetune`。服务器副本在 `/mnt1/yixuan/unlimited-ocr-finetune`。
- **markdown 清理**：删 server_status(被 status.md 取代)、旧 plan+audit(archive 留底)；两份数据调研合并为 `ocr_vlm_dataset_deep_dive`（放 /mnt1/yixuan/data）。当前 doc：design / status.md / README（服务器）+ deep_dive（数据）。
- **multi_base + R-SWA 打通**：多页训练路径端到端验证（mask 在 2030-token 多页序列生效，无为多页加特判）。多页数据用 `<PAGE>` 拼页仅供验证；真实目标数据 target 应为一份连续合并 markdown、不带 `<PAGE>`。

### 2026-07-08
- **发现并修复"训练没用 R-SWA"（本课题最关键的一条）**：之前训练走全 causal（R-SWA 滑窗只在推理 decode 的 ring-buffer 生效），与推理不一致。生成式 eval 坐实其害：全 causal 训练后**长输出崩成数字垃圾**（held-out 样本0/1 相似度 0.003/0.000；崩溃只发生在长样本=mismatch 指纹）。实现 `rswa.py`（R-SWA 训练 mask：reference=prompt+image 全通、output 只 attend 最近 128；patch `modeling_deepseekv2._prepare_4d_causal_attention_mask`，运行时打印确认 mask 生效），`train.py` 加 `rswa_train` 开关。三方 eval(n=6 held-out)：**base 0.787 / 全causal训练 0.465 / R-SWA训练 0.709**；崩溃长样本 0/1 从 0.003/0.000 恢复到 0.822/0.493。结论：**Unlimited-OCR 后训练必须用 R-SWA mask，全 causal 训练有害。** R-SWA训练后仍略低 base=olmOCR 无能力增益（base 本就强 + 纯文本 vs 原生 layout 格式漂移），gap 已不是注意力问题而是数据/目标格式问题。
- **首次真训练完成**：lora_decoder r16 on 800 页 olmOCR（150 步×accum2，warmup+cosine，每50步 checkpoint）。loss first 0.097/last 0.068，checkpoints step_50/100/最终都存，reload 后 forward 通过(loss 0.019)。**管线在真实规模/长跑/LR调度/checkpoint 全 OK**；loss 噪声大无大降 = base 对 olmOCR 本就强（预期内，非能力增益）。train.py 加了 LR 调度 + 周期 checkpoint。产物 outputs/lora_decoder_olmocr_v1。
- olmOCR-mix-1025 全量原始态已下到 `/mnt1/yixuan/data/olmOCR-mix-1025`（84 块/72GB/~26.8 万 train 页，parquet + pdf_tarballs）。建 `convert_olmocr.py`（吃本地、逐 tarball 流式抽取渲染、可配子集/条数/过滤）；转首批 800 页真实单页 -> `data/olmocr_train_v1`（char_len 中位 2092，57 含表格）。注：base 对 olmOCR 本就强，单页训练=管线/规模验证，非北极星能力。
- 清 band-aid：删 model_loader 投机 try/except(直接 eager) + 误导性 GC 死参数(GC 归 train.py 在 peft 后开)；删 train_modes full 分支三层冗余防御；`torch_dtype`->`dtype`。回归通过。
- lora_decoder 改为**排除 routed experts**（84 模块/1.99M，routed=0，实测）。
- codex review 采纳 4 条（<image>数校验 / 空数据集死循环 / mode 白名单 / full 模式 fp32-master 警告）。
- **训练闭环打通**：lora_attn 20 步 + reload 通过；lora_decoder 3 步。
- 解决 peft+GC in-place 冲突：GC 必须在 apply_train_mode(peft) 之后开 + use_reentrant=False。
- 解决单卡 OOM：开 gradient checkpointing + expandable_segments。
- 实现 train_modes / train / reload_check + configs。
- 实现 constants/dataset/processor/collator/model_loader + eval_forward，真实数据 forward->loss 通过（含 multi_base 两页）。
- 抽样 30 条真实 olmOCR 单页（pull_sample_olmocr，PDF->PNG，走 hf-mirror）。
- 服务器纯组织整理（README + docs/，缓存保留）。
- 架构核实：UOCR 结构没动，只 decoder 注意力 MLA->128 滑窗 MHA ring-buffer(R-SWA)；多页输出逐页 <PAGE> 不合并。
- 多页方向对齐：北极星=统一跨页 markdown（记录不实现）。

## 踩过的坑（累积）

1. **peft + GC 的 in-place masked_scatter_ 冲突**（requires-grad leaf）-> GC 在 peft 之后开 + `use_reentrant=False`。
2. 单卡长序列 backward OOM -> gradient checkpointing（+ expandable_segments）。
3. 必须 eager，禁 FA2（mha 分支没注册 FA2/sdpa）。
4. 训练全 causal vs 推理滑窗 mismatch（v1 跟官方，滑窗训练做 ablation 开关）。
5. olmOCR 图在 ~1GB tarball(PDF)，抽样需整块下 + fitz 渲染。
6. **随机拼单页 != 教跨页合并**（北极星数据要 arXiv 等天然连续文档 + PubTabNet-cross）。
7. HF 直连被墙 -> hf-mirror + HF_HUB_DISABLE_XET=1（服务器同本机）。

## 下一步

- **P2 续（当前主线）：批量数据。** fetch+build 两端都通:`fetch_arxiv.py`(Mac,已加固:magic 校验 / `--start` 偏移 / 退避重试)+ `build_batch.py`(服务器,读 manifest 逐篇 build)。当前:15 篇小批端到端验通 + 抽查合并 markdown 质量,通过后放量到可训练规模。可选提质:References(用源码 .bbl)、author 块、图密集/文本稀疏页过滤;必要时 LaTeXML 换保真。
- **之后：文档整理 + 正确性审计**（README 加导航表；DESIGN/status/README 对齐代码、删过时"待实现/全 causal v1"结论；dataset doc 改名 DATASET_SURVEY）。见本地 `plan_multi_gundam_and_docs.md`（含 codex review）。
- **待提交（等用户说"提交"）**：`configs/lora_multibase_smoke.yaml`（P1 config）、`scripts/convert_olmocr.py`（筛选开关）、`scripts/download/fetch_arxiv.py` + `scripts/arxiv2md/{clean_md.py,build_dataset.py,build_batch.py,clean.lua}`（P2 pipeline,fetch 已入 download/）、`src/uocr_train/constants.py`（新增 DEFAULT_MERGE_PROMPT),均本地未提交。
- **暂缓**：multi_gundam（改 forward 的 crop 分支、分辨率向、与北极星正交）；full_decoder 多卡（accelerate 已装、deepspeed 未装；需 FSDP/ZeRO 分片，48G 单卡 full_decoder ~53G 差一点）。
