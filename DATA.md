# 数据集准备（olmOCR-mix-1025）

训练/评测数据来自 HuggingFace `allenai/olmOCR-mix-1025`（约 270K PDF 页，GPT-4.1 转写的
`natural_text` 作为 GT，ODC-BY 许可）。本仓库**不含数据**（见 `.gitignore`），需在训练机上
按下述步骤重建。**国内 GPU 机下载源有坑，先看下方"下载源"一节。**

## 下载源（国内 GPU 机实测，重要）
- **基座模型 → ModelScope**：`PaddlePaddle/Unlimited-OCR`（= HF `baidu/Unlimited-OCR`，同一模型，6.8G）。HF 那个在国内 GPU 机**下不动**：hf-mirror 限速到 ~0.1MB/s、huggingface.co 直连 unreachable、AutoDL `network_turbo` 也频繁 timeout；ModelScope 直连 ~10MB/s，十几分钟到手。
- **olmOCR 数据 → HF**（脚本走 hf-mirror，慢但一次性能成）。ModelScope 上只有 `allenai/olmOCR-mix-0225`（**旧版 0225**，非本仓库用的 1025），要用需改脚本 subset/版本，暂不推荐。
- **pip 依赖 → 清华源** `-i https://pypi.tuna.tsinghua.edu.cn/simple`，且**别开 network_turbo**（turbo 会拖慢非 HF/GitHub 的源）。**git clone → 开 turbo** `source /etc/network_turbo`（学术加速，直连 GitHub 慢）。
- AutoDL：大件（模型/数据/venv）放持久盘 `/root/autodl-fs`（200G），别塞系统盘 `/`（30G）；conda 在 `/root/miniconda3`（非交互 shell 需 `source /etc/profile` 或用绝对路径）。

## 环境搭建（conda base + 依赖）
以 AutoDL PyTorch 镜像为例（自带 torch 2.5.1 / CUDA 12.4；conda 在 `/root/miniconda3`）：
```bash
# 用镜像自带 conda base（已有 torch），只补其余依赖；清华源、别开 network_turbo
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
- **不需要 flash-attn**（全程 eager）。remote code 的额外依赖 `addict/easydict/timm/matplotlib` 已在 requirements.txt 里，缺了会在 `from_pretrained` 报 ModuleNotFoundError。
- 非交互 SSH 里 conda 不在 PATH：用绝对路径 `/root/miniconda3/bin/python`，或先 `source /etc/profile`。
- 跑任何脚本前先 `export UOCR_MODEL_DIR=<模型目录>`（model_loader 从这读）。

## 数据结构（HF 仓库内）
- `{subset}_{split}.parquet` —— 每行一页：`natural_text`(GT) + `pdf_relpath`(指向 tarball 内某 PDF) + `page_number`/`primary_language`/`is_table`/`is_diagram`/`is_rotation_valid`/`id`
- `pdf_tarballs/{subset}_{split}_{块号}.tar.gz` —— 对应 PDF
- subset：`00_documents` / `01_books` / `02_loc_transcripts` / `03_national_archives`

## 环境变量
```bash
export HF_ENDPOINT=https://hf-mirror.com     # HF 镜像（必须）
export HF_HUB_DISABLE_XET=1                   # 关 xet
export OLMOCR_DIR=./olmOCR-mix-1025           # 数据根，convert/download 共用
export UOCR_MODEL_DIR=./models/Unlimited-OCR   # 基座模型目录（ModelScope: PaddlePaddle/Unlimited-OCR）
```

## 冒烟复现（重建 results/ 里那批实验用的数据）
```bash
# 0) 基座模型（~6.8G）—— 用 ModelScope（国内 GPU 机 HF 下不动，见上方"下载源"）
pip install -q modelscope
modelscope download --model PaddlePaddle/Unlimited-OCR --local-dir "$UOCR_MODEL_DIR"

# 1) 下训练块：00_documents/train 第 0 块（parquet + 1 个 tarball），约 1-3GB
python scripts/download/download_olmocr.py --subset 00_documents --split train --chunks 00000

# 2) 转训练 JSONL（最多 800 页，渲染 PNG，长边 1600）
python scripts/convert_olmocr.py --subset 00_documents --split train \
    --chunks 00_documents_train_00000 --max-samples 800 --skip-bad-rotation \
    --out data/olmocr_train_v1
#   -> data/olmocr_train_v1/train.jsonl（train_jsonl，训练 config 里指向它）

# 3) 生成 30 条 eval 探针（脚本自动下 eval 块 ~191MB）
python scripts/download/pull_sample_olmocr.py
#   -> data/samples_olmocr/train.jsonl（eval_infer.py 的 --jsonl，取前 6 即那张对照表）
```

## 体积（估）
- eval 块 `00000` ~191MB；train 块 + parquet ~1-3GB（冒烟）
- 渲染产物：`olmocr_train_v1` ~300MB、`samples_olmocr` ~200MB
- 全量很大（~270K 页），**按需增量下**，别一次拉全

## 扩量
- 多块：`download_olmocr.py --chunks 00000,00001,00002`，convert 用 `--chunks 00_documents_train_00000,00_documents_train_00001,...`（download 跑完会打印现成的 convert 参数）
- 换 subset：`--subset 01_books` 等，同理

## 注意
- `convert_olmocr.py` 只读本地、不下载；下载统一用 `download_olmocr.py`（train）或 `pull_sample_olmocr.py`（eval 探针，自带下载）。
- 三个脚本的路径都已 env 化（`OLMOCR_DIR` / `OLMOCR_SAMPLE_OUT` / `UOCR_MODEL_DIR`），无写死的服务器路径。
- `debug/` 下的调试脚本仍有历史硬编码路径，仅作参考、不参与数据/训练流程。
