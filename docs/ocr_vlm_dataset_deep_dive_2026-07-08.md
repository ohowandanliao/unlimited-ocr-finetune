# OCR/VLM 图片转 Markdown 开源训练集深度调研

日期：2026-07-08  
本轮重点：补齐每个数据集的输入、输出/GT、数据格式、是否可直接用于 `image/PDF -> Markdown` 后训练，并扩大到 DeepSeek-OCR、PaddleOCR-VL、GOT-OCR2、Dolphin、MonkeyOCR、MinerU、dots.mocr、FireRed-OCR 等 OCR-VLM 路线。

## 结论先行

如果目标是收集“图片/PDF 转 Markdown”的开源训练集，优先级建议如下：

| 优先级 | 数据源 | 结论 |
|---|---|---|
| S | `allenai/olmOCR-mix-1025` | 当前最接近“整页 PDF 图像 -> 自然文本/HTML 表格/类 Markdown”的公开训练源之一。字段和 GT 明确，DeepOCR 也实际拿它转成 LLaVA/VILA 格式。 |
| A | `Vary-tiny-600k` / Vary-600k | 60 万中英文 PDF 页面图文对，适合 OCR 预训练和 warm-up。GT 是段落文本，使用 `<lb>` 分段，不是严格 Markdown。 |
| A | `SynthDoG` EN/ZH | 合成文档图像 + `ground_truth` JSON，适合视觉-文字对齐和基础 OCR，但版面/表格/公式能力有限。 |
| A- | `MonkeyDoc` | MonkeyOCR 官方 2026-01-30 宣称释放并提供数据生成 pipeline。公开入口存在，但 ModelScope 页面抓取到的是前端壳，本轮未能直接验证文件字段；建议下载后做字段复核。 |
| B+ | `OmniDocBench` | 标注极强：layout、reading order、text、LaTeX、HTML table。但规模小，本质 benchmark。更适合 holdout/eval 和少量格式校准，不能大规模混入训练。 |
| B+ | `PubTabNet`、`UniMER-1M`、`MathWriting` | 表格/公式 element-level 强监督。不能直接训练整页 Markdown，但非常适合补结构短板。 |
| B | `DocLayNet`、`PubLayNet`、`DocBank` | layout 检测/阅读顺序辅助数据。没有整页 Markdown GT，需要转成检测/布局任务或生成中间 JSON。 |
| C | `DocVQA`、`ChartQA`、`TextOCR`、`SROIE/CORD/FUNSD/XFUND` | 有图像和 GT，但任务是 QA、检测、KIE、receipt/form extraction，不应直接当整页 Markdown 训练集。适合增强问答、局部 OCR、结构理解。 |

几个容易误判的点：

- PaddleOCR-VL、Dolphin、FireRed-OCR、dots.mocr、MinerU 这些项目很有参考价值，但大多没有直接释放完整训练集。它们公开的是模型、推理代码、benchmark 或数据构造方法。
- DeepSeek-OCR/DeepSeek-OCR-2 官方仓库主要是推理/benchmark，远端 `research_repos` 里没有发现官方可下载训练数据。
- QA 数据集不能直接混成 Markdown SFT。`DocVQA` 的输出是 answer，不是页面全文；`ChartQA` 的输出是答案或数值，不是 chart-to-table Markdown。
- plain text 不等于 Markdown。Vary/SynthDoG 可作为 OCR/warm-up，但要用 `target_format=plain_text` 标清楚，避免把模型教成“所有内容都只输出纯文本”。

## 调研边界和证据来源

本轮做了三类核验：

1. 远端服务器核验：`<GPU-HOST-已迁移>:/mnt1/yixuan/unlimited-ocr-posttrain/research_repos`
2. 公开 GitHub/Hugging Face/arXiv 页面核验
3. 训练脚本/转换脚本字段级核验

网络限制：GitHub raw 多次返回 `429 Too Many Requests`，Hugging Face 直连下载/API 偶发 reset。因此对能打开的 GitHub 页面、HF dataset viewer、arXiv HTML 作为主证据；对远端仓库用 `ssh` 直接读脚本和配置。

## 可直接或近似直接用于 OCR/Markdown SFT 的数据卡

### 1. allenai/olmOCR-mix-1025

| 字段 | 内容 |
|---|---|
| 输入 | PDF 单页，字段 `pdf_relpath` 指向 tarball 内 PDF 页面；DeepOCR 脚本会把第一页渲染成 PNG。 |
| 输出/GT | `natural_text`。HF viewer 显示其中包含纯文本，也包含 `<table>...</table>`、`<html><table>...` 等表格片段。 |
| 原始格式 | Hugging Face parquet；字段包括 `url`、`page_number`、`pdf_relpath`、`primary_language`、`is_rotation_valid`、`rotation_correction`、`is_table`、`is_diagram`、`natural_text`、`id` 等。 |
| 远端转换格式 | DeepOCR 先合并 parquet，再转成 LLaVA conversation：human `Free OCR.\n<image>`，assistant `natural_text`，image 为渲染 PNG。 |
| 是否有 GT | 有。 |
| 是否开源可下载 | 是，HF dataset：`https://huggingface.co/datasets/allenai/olmOCR-mix-1025`，license 标为 `odc-by`。 |
| 能否直接用于图片转 Markdown | 高。建议作为整页 OCR/Markdown-like SFT 主力，但要保留 `target_format=natural_text_or_html`，不要强行标成纯 Markdown。 |
| 风险 | `natural_text` 是自然文本/HTML table 混合，格式不完全等价于你最终想要的 Markdown。需要后处理：HTML table -> Markdown table 或保留 HTML table 二选一。 |

远端证据：

- `Princeton-AI2-Lab_DeepOCR/data_prepare/README.md`：Stage 2 使用 `allenai/olmOCR-mix-1025`，约 260k。
- `data_prepare/olmOCR-mix-1025/convert_to_llava.py`：读取 `pdf_relpath` 和 `natural_text`，把 PDF 页面转 PNG，输出 conversation。
- `llava/data/registry/datasets/default.yaml`：`olmOCR-mix-pretrain` 指向 `transformed_data_png.json` 和 `png_tarballs`。

公开证据：

- HF dataset viewer 显示 `00_documents` 233k rows，`01_books` 17.5k，`02_loc_transcripts` 9.99k，`03_national_archives` 10k；字段中明确有 `pdf_relpath` 和 `natural_text`：<https://huggingface.co/datasets/allenai/olmOCR-mix-1025>

### 2. Vary-600k / Vary-tiny-600k

| 字段 | 内容 |
|---|---|
| 输入 | PDF 页面图像，中英文各约 30 万页。 |
| 输出/GT | 页面文本。段落用 `<lb>` 分隔。 |
| 原始格式 | 官方通过百度网盘发布，README 描述为 PDF image-text pairs。DeepOCR 远端注册里有 `vary_pdf_cn_30w`、`vary_pdf_en_30w` 的 LLaVA JSON 路径。 |
| 是否有 GT | 有，来自 PDF 文本抽取和处理。 |
| 是否开源可下载 | 可下载，但入口是百度网盘；数据许可为 research use / CC BY-NC 4.0 风格提示。 |
| 能否直接用于图片转 Markdown | 中高。适合 OCR pretrain/warm-up；不适合作为表格/公式/严格 Markdown 主数据。 |
| 风险 | 输出是文本段落，不是完整结构化 Markdown；换行符被替换为 `<lb>`，训练前要统一到你的 target 格式。 |

公开证据：

- 官方 README 写明 Vary-600k 是“about 30W English and 30W Chinese pages”的 PDF image-text pair dataset，并说明 Fitz 抽取、BERT 合并句子、段落用 `<lb>` 分隔：<https://github.com/Ucas-HaoranWei/Vary-tiny-600k>

### 3. SynthDoG EN/ZH

| 字段 | 内容 |
|---|---|
| 输入 | 合成文档图像。 |
| 输出/GT | `ground_truth` 字符串，JSON 内含 `gt_parse.text_sequence`。 |
| 原始格式 | HF parquet；`synthdog-en` viewer 显示 `image` 和 `ground_truth` 两列，默认 66k rows。 |
| 是否有 GT | 有。 |
| 是否开源可下载 | 是。EN: `naver-clova-ix/synthdog-en`；ZH 也常用于 OCR 预训练。 |
| 能否直接用于图片转 Markdown | 中。适合 OCR/文档视觉-文本对齐，不适合训练复杂 Markdown、表格、公式。 |
| 风险 | 合成数据域偏移明显；`text_sequence` 常含人为断字/空格噪声，需要清洗或作为低权重数据。 |

公开证据：

- HF viewer 显示 `image` 列和 `ground_truth` 列，样例为 `{"gt_parse": {"text_sequence": "..."} }`：<https://huggingface.co/datasets/naver-clova-ix/synthdog-en>

### 4. MonkeyDoc / MonkeyOCR 数据路线

| 字段 | 内容 |
|---|---|
| 输入 | 文档 PDF/图片；MonkeyOCR 支持 end-to-end parsing，也支持 text/formula/table 单任务。 |
| 输出/GT | README 描述输出最终 Markdown、layout PDF、中间 JSON；MonkeyDoc 宣称包含数据生成 pipeline 细节。 |
| 原始格式 | 本轮无法从 ModelScope 页面直接抓到字段，页面只返回前端壳。需要实际下载核验。 |
| 是否有 GT | 官方宣称数据集释放，理论上有；本轮未验证具体字段。 |
| 是否开源可下载 | 官方 README 链接 ModelScope `Yuliang/MonkeyDoc`。 |
| 能否直接用于图片转 Markdown | 潜力高，但必须下载后做字段抽样。 |
| 风险 | 公开入口存在不等于字段可直接训练。需要确认是否包含原图、Markdown、layout JSON、元素类型、阅读顺序、license。 |

公开证据：

- MonkeyOCR README 2026-01-30 news：发布 MonkeyDoc 并提供数据生成 pipeline 细节；推理输出包含 `your.md`、`your_layout.pdf`、`your_middle.json`：<https://github.com/Yuliang-Liu/MonkeyOCR>
- ModelScope 入口：<https://www.modelscope.cn/datasets/Yuliang/MonkeyDoc>

### 5. OmniDocBench

| 字段 | 内容 |
|---|---|
| 输入 | PDF 页面图像。 |
| 输出/GT | 高质量人工标注：block-level 和 span-level layout，文本 OCR，公式 LaTeX，表格 LaTeX/HTML，reading order。 |
| 原始格式 | JSON。核心字段包括 `layout_dets`、`category_type`、`poly`、`order`、`text`、`latex`、`html`、`line_with_spans`、`page_info.image_path`。 |
| 是否有 GT | 有，且质量高。 |
| 是否开源可下载 | 是，HF/OpenDataLab/GitHub。 |
| 能否直接用于图片转 Markdown | 不建议大规模训练；适合 eval、少量格式校准、reward/verifier 开发。 |
| 风险 | 是 benchmark，规模约 1651 PDF pages。混入训练会污染评测；建议保留为 holdout。 |

公开证据：

- README 说明覆盖 10 类文档、5 类 layout、5 类语言，包含 28 个 block-level 和 4 个 span-level 元素，文本/LaTeX/HTML/reading order 标注齐全，并给出 JSON 格式：<https://github.com/opendatalab/OmniDocBench>

## 元素级强监督数据集

这些数据集不能单独完成整页 Markdown，但对提升 OCR-VLM 的表格、公式、chart、layout 能力很关键。

| 数据集 | 输入 | 输出/GT | 格式 | 是否可用于 Markdown 能力 | 建议用途 |
|---|---|---|---|---|---|
| PubTabNet | 表格图像 | HTML table structure、cell tokens、非空 cell bbox | JSONL，字段 `filename/split/imgid/html.structure.tokens/html.cell.tokens/bbox` | 是，表格子任务 | table image -> HTML/Markdown table SFT；建议保留 HTML，最后统一渲染/转换。 |
| UniMER-1M | 公式图像 | LaTeX | 公开项目/数据 | 是，公式子任务 | formula crop -> LaTeX，补公式识别。 |
| MathWriting | 手写数学表达式 | 表达式 GT，可用于 LaTeX/结构识别 | online/offline handwriting data | 是，手写公式子任务 | 手写公式和复杂符号鲁棒性。 |
| ChartQA | chart image + question | answer | QA 格式 | 间接 | 不做整页 Markdown；可构造成 chart reasoning 或 chart-to-table 辅助任务。 |
| PlotQA / DVQA / Chart2Text / UniChart | chart 图像 | QA、底层表格、caption/summary 等 | 各自格式 | 间接 | PaddleOCR-VL 也将这些作为 chart coverage 来源；适合 chart-to-table/summary。 |
| DocLayNet | 文档页面 | 11 类 bbox layout 标注 | COCO format | 间接 | layout detection、reading-order 前置模块。 |
| PubLayNet | 科研 PDF 页面 | layout bbox/polygon | COCO-like | 间接 | 科研文章 layout 预训练。 |
| DocBank | 文档 token/line/layout 标注 | token-level layout labels | 来自 arXiv LaTeX/PDF 对齐 | 间接 | token/块级 layout 辅助。 |
| FUNSD/XFUND | 表单图像 | entity、linking、文本框 | JSON + image | 间接 | form understanding/KIE，不要当整页 Markdown。 |
| CORD/SROIE | 收据图像 | OCR + key fields | JSON/TXT bbox | 间接 | receipt OCR/KIE，可做局部结构化输出。 |
| TextOCR | 自然图像 | 文字实例和 bbox，多边形 | JSON | 间接 | 场景文字识别，不是文档 Markdown。 |
| DocVQA | 文档图像 + question | answer(s) | QA JSON | 间接 | 文档问答，不要混成全文 OCR。 |

公开证据摘录：

- PubTabNet：568k+ table images，GT 为对应 HTML representation，annotation 为 JSONL：<https://github.com/ibm-aur-nlp/PubTabNet>
- DocLayNet：80,863 pages，COCO format，11 classes：<https://arxiv.org/abs/2206.01062>
- DocVQA：50,000 questions，12,000+ document images：<https://arxiv.org/abs/2007.00398>
- TextOCR：900k annotated words on real images：<https://arxiv.org/abs/2105.05486>
- ChartQA：9.6k human-written + 23.1k generated chart QA：<https://arxiv.org/abs/2203.10244>
- SROIE：1000 scanned receipt images，text localization/OCR/KIE 三任务：<https://arxiv.org/abs/2103.10213>
- UniMER-1M：100 万公式识别训练实例，输出公式结构/LaTeX：<https://arxiv.org/abs/2404.15254>
- MathWriting：230k 人写 + 400k synthetic handwritten math samples：<https://arxiv.org/abs/2404.10690>

## 远端 research_repos 核验结果

### Princeton-AI2-Lab_DeepOCR

| 项 | 结果 |
|---|---|
| 主数据 | Stage 1: `liuhaotian/LLaVA-CC3M-Pretrain-595K`; Stage 2: `allenai/olmOCR-mix-1025`。 |
| OCR 预训练混合 | `olmOCR-mix-pretrain + vary_pdf_cn_30w + vary_pdf_en_30w + synthdog-en + synthdog-zh`。 |
| 输入输出 | `convert_to_llava.py` 从 `pdf_relpath` 找 PDF，渲染 PNG；assistant 输出 `natural_text`。 |
| 格式 | LLaVA/VILA conversation JSON：`id`、`image`、`conversations`。 |
| 对你是否有价值 | 高。这个仓库提供了可复用的 OCR-VLM 数据转换 schema。 |

### baggie11_Finetuning_Deepseek_OCR

| 项 | 结果 |
|---|---|
| 数据集 | `5CD-AI/Viet-Handwriting-OCR-v2` |
| 输入 | `sample["image"]` |
| 输出/GT | `sample["text"]` |
| 格式 | HF dataset -> conversation：user prompt + image，assistant text。 |
| 用途 | 越南语手写 OCR 微调；不适合 Markdown，但适合多语/手写 OCR 增强。 |

### WT-ever_deepseek-ocr-lora

| 项 | 结果 |
|---|---|
| 数据集 | 本地 JSONL，无公开数据源。 |
| 输入 | `IMAGE_FOLDER/sample["image"]` |
| 输出/GT | `sample["suffix"]` |
| 格式 | JSONL 字段：`image`、`prefix`、`suffix`；user 为 `<image>\n` + `prefix`，assistant 为 `suffix`。 |
| 用途 | 这个 schema 很适合你自己的 image-to-markdown 数据：`prefix=docparse`，`suffix=markdown`。 |
| 限制 | 没有公开训练集，只有格式模板价值。 |

### HaCTang_MolSeek-OCR / ChemSeek-OCR

| 项 | 结果 |
|---|---|
| 数据集 | MolScribe：`pubchem.zip`、`uspto_mol.zip`；训练 CSV 如 `pubchem/train_1m.csv`、`uspto_mol/train_680k.csv`。 |
| 输入 | 分子结构图像，可 dynamic render 或 realistic image。 |
| 输出/GT | SMILES。 |
| 格式 | CSV 中查找 `SMILES/smiles/canonical_smiles` 列；训练时转 conversation。 |
| 用途 | 化学结构 OCR/图像转 SMILES，不适合 Markdown 主任务。 |

额外发现：该仓库 misc 示例还引用 `hezarai/parsynth-ocr-200k`，字段为 `image_path` 和 `text`，用于 Persian OCR 示例。它是语言 OCR 辅助数据，不是项目主训练集。

### DeepSeek-OCR / DeepSeek-OCR-2 / baidu_Unlimited-OCR

| 项 | 结果 |
|---|---|
| 数据 | 未发现官方训练数据下载或训练数据字段。 |
| 内容 | 主要是推理、模型、prompt、benchmark、部署。 |
| 用途 | 可参考 prompt 和输出格式；不要期待里面藏有可直接用的大训练集。 |

### archive 低价值仓库中的可用线索

| 仓库 | 数据线索 | 价值 |
|---|---|---|
| `DinhXuanKhuong_Finetune-DeepseekOCR-with-Vietnamese-Dataset` | UIT-HWDB Vietnamese handwriting dataset，约 80k/100k 手写 OCR 图像，GT text。 | 多语手写 OCR 辅助。 |
| `ichthyosaur_Post-training-of-DeepSeek-OCR` | `lmms-lab/DocVQA`，图像 + question -> answer。 | 文档 QA 辅助，不是全文 Markdown。 |
| `elihoole_deepseek-ocr-finetutning` | 泛化 `annotations.json/csv`，字段 `image_path,text`。 | 格式示例，非公开数据源。 |
| `ricyoung_Justitia...` | Synthea 生成医疗 PDF + PHI annotation pipeline。 | 可做合成领域数据，不是现成 Markdown GT。 |

## OCR-VLM 项目的训练数据配方启发

### PaddleOCR-VL

| 维度 | 调研结果 |
|---|---|
| 是否公开训练集 | 没有看到完整训练集释放。 |
| 训练规模 | 论文写 Stage 1 使用 29M high-quality image-text pairs；Stage 2 使用 2.7M curated samples。 |
| 数据来源 | open-source datasets、synthetic dataset、network accessible data、in-house dataset。 |
| 任务 | OCR、Table Recognition、Formula Recognition、Chart Recognition。 |
| 输出目标 | text、OTSL table、LaTeX formula、chart -> Markdown table，最终 post-process 成 Markdown/JSON。 |
| 对你的启发 | 不要只训“整页输出”。需要拆成 layout、text、table、formula、chart，再汇总成 Markdown。 |

公开证据：

- PaddleOCR-VL 论文：两阶段架构，PP-DocLayoutV2 做 layout/reading order，PaddleOCR-VL 做 text/table/formula/chart element recognition，最终输出 Markdown/JSON；训练数据 29M + 2.7M：<https://arxiv.org/html/2510.14528>

### Dolphin

| 维度 | 调研结果 |
|---|---|
| 是否公开训练集 | 没有看到训练数据释放；公开模型、推理代码、Fox-Page benchmark。 |
| 训练规模 | 论文摘要写 over 30M samples，multi-granularity parsing tasks。 |
| 输出 | page-level structured JSON and Markdown；element-level text/table/formula/code。 |
| 架构 | analyze-then-parse：先 layout + reading order，再并行 element-wise parsing。 |
| 对你的启发 | 训练数据应包含 page-level 和 element-level 两种样本，并保留元素类型。 |

公开证据：

- README 写 page-level parsing 输出 structured JSON/Markdown，element-level parsing 支持 text/table/formula/code：<https://github.com/ByteDance/Dolphin>
- arXiv 摘要写构造 over 30 million samples：<https://arxiv.org/abs/2505.14059>

### GOT-OCR2.0

| 维度 | 调研结果 |
|---|---|
| 是否公开训练集 | 公开 benchmarks、train sample、fine-tune schema；未公开完整 stage-1/stage-2 训练集。 |
| 任务 | plain text OCR、format text OCR、fine-grained OCR、multi-crop OCR。 |
| 格式 | ms-swift fine-tune JSONL：`query`、`response`、`images`；query 必须含 `<image>`。 |
| 对你的启发 | JSONL schema 可直接复用；format OCR 任务可以对齐 Markdown/HTML/LaTeX 输出。 |

公开证据：

- README Train/Fine-tune 段落给出 `{"query": "<image>...", "response": "...", "images": [...]}` 格式，并说明 quick finetune 可用 `latex-ocr-print#5000`：<https://github.com/Ucas-HaoranWei/GOT-OCR2.0>

### FireRed-OCR

| 维度 | 调研结果 |
|---|---|
| 是否公开训练集 | 未看到训练数据释放；公开模型、推理、数据工程路线。 |
| 数据路线 | “Geometry + Semantics Data Factory”；multi-task pre-alignment、specialized SFT、format-constrained GRPO。 |
| SFT 输出 | 标准化 full-image Markdown。 |
| RL reward | Formula Syntax、Table Integrity、Hierarchical Closure、Text Accuracy。 |
| 对你的启发 | 后训练不要只做 SFT，最后应加格式验证或 RL/GRPO；至少做 rejection sampling。 |

公开证据：

- README Key Features 和 Model Architecture 明确三阶段：detection/region/layout-to-markdown pre-alignment，Markdown SFT，format-constrained GRPO：<https://github.com/FireRedTeam/FireRed-OCR>

### dots.mocr

| 维度 | 调研结果 |
|---|---|
| 是否公开训练集 | 未看到训练集释放。 |
| 输出 | document parsing、web parsing、scene spotting、image-to-SVG。HF demo prompt 要求 layout JSON：bbox、category、text；formula 用 LaTeX，table 用 HTML，其他 Markdown。 |
| 对你的启发 | 很适合作为统一 schema 参考：`bbox + category + text/html/latex + reading order`，而不是只输出一坨 Markdown。 |

公开证据：

- README prompt 示例要求 JSON 输出，layout categories 包括 Caption/Formula/Table/Text/Title 等，并规定 formula LaTeX、table HTML、others Markdown：<https://github.com/rednote-hilab/dots.mocr>

### MinerU / MinerU2.5

| 维度 | 调研结果 |
|---|---|
| 是否公开训练集 | MinerU 是工具/引擎，不是训练集仓库。 |
| 输出 | Markdown、JSON、intermediate formats；PDF/image/DOCX/PPTX/XLSX 输入；tables -> HTML，formulas -> LaTeX。 |
| 对你的启发 | 可以作为伪标注/数据生产工具，但必须抽样人工校验，避免把工具错误蒸馏进模型。 |

公开证据：

- README 写 MinerU 将 PDF/image/Office/web page 转为 Markdown/JSON，支持 formulas -> LaTeX、tables -> HTML、reading order、header/footer removal：<https://github.com/opendatalab/MinerU>

## 推荐训练数据混合方案

### Stage 0：投影层/视觉语言对齐

可选，不一定针对 OCR：

- `LLaVA-CC3M-Pretrain-595K`：DeepOCR Stage 1 使用。它不是 OCR 数据，但可初始化 projector。
- 如果你的 base model 已经是成熟 VLM，这一步可以弱化或跳过。

### Stage 1：OCR dense pretrain / warm-up

目标：让模型稳定读整页文字，不追求复杂 Markdown。

建议混合：

| 数据 | 权重建议 | 任务 |
|---|---:|---|
| `olmOCR-mix-1025` | 40-50% | full page `image/PDF -> natural_text/html-table` |
| `Vary-600k` | 25-35% | full page `image -> text` |
| `SynthDoG EN/ZH` | 5-15% | synthetic doc OCR |
| 多语/手写 OCR：`5CD-AI/Viet-Handwriting-OCR-v2`、UIT-HWDB、`hezarai/parsynth-ocr-200k`、CASIA-HWDB | 5-10% | handwriting/multilingual OCR |
| TextOCR/scene text | 0-5% | scene text robustness |

### Stage 2：结构化 Markdown SFT

目标：表格、公式、阅读顺序、标题/段落/list/code/table/formula 的格式一致性。

建议数据：

- `olmOCR-mix-1025` 中 `is_table=true`、`is_diagram=true` 或含 `<table>`/`<html>` 的样本上采样。
- `PubTabNet`：table image -> HTML table 或 Markdown table。
- `UniMER-1M` / `MathWriting`：formula image -> LaTeX。
- `OmniDocBench`：只用 train/dev 小比例或自己划一小部分做格式校准；主要保留为 eval。
- `DocLayNet/PubLayNet/DocBank`：做 layout task，不要伪装成 Markdown。
- `ChartQA/PlotQA/DVQA/Chart2Text/UniChart`：转成 chart-to-table 或 chart summary，不要直接当页面 Markdown。

### Stage 3：格式约束和 verifier

参考 FireRed-OCR：

- Markdown AST 可解析。
- HTML table 可解析，行列数稳定。
- LaTeX/KaTeX 可编译。
- 表格用 TEDS 或类似结构相似度。
- 文本用 edit distance / CER / WER。
- reading order 用 block order edit distance。

如果暂时不做 RL/GRPO，也建议做 rejection sampling：模型生成 Markdown 后跑 verifier，失败样本进入 hard negative / retry 数据。

## 统一 JSONL 格式建议

建议你不要把所有数据强行转成一个 `text` 字段，而是加 `target_format` 和 `task`：

```json
{
  "id": "olmocr_000001",
  "source": "allenai/olmOCR-mix-1025",
  "image": "png_tarballs/xxx.png",
  "prompt": "<image>\nConvert the page to Markdown, preserving reading order, tables, and formulas.",
  "target": "....",
  "target_format": "markdown_or_html_table",
  "task": "full_page_parse",
  "meta": {
    "language": "en",
    "is_table": true,
    "is_diagram": false,
    "page_number": 7,
    "license": "odc-by"
  }
}
```

元素级数据也保留同一 schema：

```json
{
  "id": "pubtabnet_000001",
  "source": "PubTabNet",
  "image": "table.png",
  "prompt": "<image>\nRecognize the table and output valid HTML.",
  "target": "<table>...</table>",
  "target_format": "html_table",
  "task": "table_recognition",
  "meta": {
    "has_cell_bbox": true
  }
}
```

`WT-ever_deepseek-ocr-lora` 的 `image/prefix/suffix` 格式可以作为最简训练格式，但建议扩展 metadata，否则后面难以按任务、语言、license、格式做采样。

## 采集清单

第一批建议真正下载和抽样：

1. `allenai/olmOCR-mix-1025`
2. `Vary-tiny-600k`
3. `naver-clova-ix/synthdog-en` / `synthdog-zh`
4. `PubTabNet`
5. `UniMER-1M`
6. `OmniDocBench`，只做 eval/format sanity
7. `MonkeyDoc`，必须先字段级验证

第二批辅助数据：

1. `DocLayNet`
2. `PubLayNet`
3. `DocBank`
4. `DocVQA`
5. `ChartQA` / `PlotQA` / `DVQA` / `Chart2Text`
6. `SROIE` / `CORD` / `FUNSD` / `XFUND`
7. 多语/手写：`5CD-AI/Viet-Handwriting-OCR-v2`、UIT-HWDB、`hezarai/parsynth-ocr-200k`、CASIA-HWDB

## 下载后必须做的字段级验证

每个数据集至少抽样 20 条，记录：

| 检查项 | 通过标准 |
|---|---|
| 图像能否打开 | PIL/OpenCV 能正常读取，尺寸合理。 |
| GT 是否为空 | target 非空，长度分布正常。 |
| GT 是否对应图像 | 随机可视化图像 + target，人工确认。 |
| target 格式 | Markdown/HTML/LaTeX/plain_text/JSON 明确。 |
| 表格是否闭合 | HTML/Markdown table 可解析。 |
| 公式是否可编译 | LaTeX/KaTeX parser 通过。 |
| license | 可用于你的训练场景。 |
| 是否可能评测泄漏 | OmniDocBench、Fox、olmOCR-bench 等 benchmark 默认不混入训练。 |

## 最后建议

当前最务实的数据策略是：

1. 用 `olmOCR-mix-1025 + Vary-600k + SynthDoG` 建立 full-page OCR 底座。
2. 用 `PubTabNet + UniMER/MathWriting + DocLayNet/DocBank` 补元素级结构能力。
3. 把 `OmniDocBench` 留作主评测和格式 verifier 开发，不要混入训练主集。
4. `MonkeyDoc` 值得优先下载核验，一旦字段包含 `image + markdown/json layout GT`，它会成为第一梯队数据。
5. 用 MinerU/MonkeyOCR/dots.mocr/FireRed 的输出 schema 和 reward 思路设计自己的数据生产 pipeline，但不要把这些项目等同于“已释放训练集”。

---

## 附录：research_repos 数据集来源溯源（并入自 dataset_survey，2026-07-08）

这份优先级表按"哪个远端 research_repo 引用了哪个数据集"整理，作为数据来源溯源。远端目录 `<GPU-HOST-已迁移>:/mnt1/yixuan/unlimited-ocr-posttrain/research_repos`。

| 优先级 | 数据集/来源 | 远端项目线索 | 任务形态 | 建议 |
| --- | --- | --- | --- | --- |
| P0 | `allenai/olmOCR-mix-1025` | `Princeton-AI2-Lab_DeepOCR` | PDF/页面图像→`natural_text`，DeepOCR 标 260k | OCR 后训练主线 |
| P0 | Vary-600k `pdf_cn_30w`/`pdf_en_30w` | `Princeton-AI2-Lab_DeepOCR` | 中英文 PDF-dense OCR | 补中文页面，确认 CC BY-NC 4.0 |
| P1 | `SynthDoG` EN/ZH | `Princeton-AI2-Lab_DeepOCR` | 合成文档图像+结构文本 | 小比例(5-15%)混入 |
| P1 | `LLaVA-CC3M-Pretrain-595K` | `Princeton-AI2-Lab_DeepOCR` | 图像-caption，projector 对齐 | 只用于 alignment，非 OCR 主数据 |
| P1 | `5CD-AI/Viet-Handwriting-OCR-v2` | `baggie11_Finetuning_Deepseek_OCR` | 越南语手写→文本 | 低比例多语种补充 |
| P2 | `lmms-lab/DocVQA` | 归档 `ichthyosaur_*` | 文档 VQA | instruction tuning 补充，非 OCR target |
| P2 | MolScribe `pubchem`/`uspto_mol` | `HaCTang_MolSeek-OCR` | 分子图→SMILES | 化学专项 adapter，别混主训练 |
| P2 | SROIE/ESTVQA/POIE/ReCTS/LSVT/MTWI/TextOCR/UniChart/ChartQA | `DeepOCR` mixture | 票据/场景文字/图表/VQA | 按任务小比例辅助 |
| P3 | Synthea 医疗 PDF/PHI | 归档 `ricyoung_Justitia-*` | 合成医疗 PDF+PHI | 自造管线，注意 PHI |

prompt 溯源（与部署对齐）：
- DeepSeek/DeepSeek-OCR-2 markdown：`<image>\n<|grounding|>Convert the document to markdown.`；plain OCR：`<image>\nFree OCR.`
- Unlimited-OCR 单图：`<image>document parsing.`；多页：`<image>Multi page parsing.`
- 规则：plain text target 用 `Free OCR.`，只有 Markdown/HTML/表格 target 才用 markdown prompt，否则会教模型把无结构文本当 Markdown（本项目已实测：olmOCR 是 plain natural_text，与模型原生 layout 格式漂移会伤生成）。

