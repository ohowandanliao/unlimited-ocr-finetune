# 数据集准备（olmOCR-mix-1025）

训练/评测数据来自 HuggingFace `allenai/olmOCR-mix-1025`（约 270K PDF 页，GPT-4.1 转写的
`natural_text` 作为 GT，ODC-BY 许可）。本仓库**不含数据**（见 `.gitignore`），需在训练机上
按下述步骤重建。国内直连 huggingface.co 被墙，全程走 hf-mirror。

## 数据结构（HF 仓库内）
- `{subset}_{split}.parquet` —— 每行一页：`natural_text`(GT) + `pdf_relpath`(指向 tarball 内某 PDF) + `page_number`/`primary_language`/`is_table`/`is_diagram`/`is_rotation_valid`/`id`
- `pdf_tarballs/{subset}_{split}_{块号}.tar.gz` —— 对应 PDF
- subset：`00_documents` / `01_books` / `02_loc_transcripts` / `03_national_archives`

## 环境变量
```bash
export HF_ENDPOINT=https://hf-mirror.com     # HF 镜像（必须）
export HF_HUB_DISABLE_XET=1                   # 关 xet
export OLMOCR_DIR=./olmOCR-mix-1025           # 数据根，convert/download 共用
export UOCR_MODEL_DIR=./models/baidu_Unlimited-OCR   # 基座模型目录
```

## 冒烟复现（重建 results/ 里那批实验用的数据）
```bash
# 0) 基座模型（~10-16G，以实际为准）
huggingface-cli download baidu/Unlimited-OCR --local-dir "$UOCR_MODEL_DIR"

# 1) 下训练块：00_documents/train 第 0 块（parquet + 1 个 tarball），约 1-3GB
python scripts/download_olmocr.py --subset 00_documents --split train --chunks 00000

# 2) 转训练 JSONL（最多 800 页，渲染 PNG，长边 1600）
python scripts/convert_olmocr.py --subset 00_documents --split train \
    --chunks 00_documents_train_00000 --max-samples 800 --skip-bad-rotation \
    --out data/olmocr_train_v1
#   -> data/olmocr_train_v1/train.jsonl（train_jsonl，训练 config 里指向它）

# 3) 生成 30 条 eval 探针（脚本自动下 eval 块 ~191MB）
python scripts/pull_sample_olmocr.py
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
