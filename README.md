# unlimited-ocr-finetune

给 [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 做后训练（LoRA / 全参 SFT）的一套干净训练代码。

Unlimited-OCR 相对 DeepSeek-OCR 只改了解码器注意力：把 MLA 换成标准 MHA + **128 滑窗 ring-buffer（官方称 R-SWA, Reference Sliding Window Attention）**——每个生成 token 只 attend 全部视觉/prompt reference + 最近 128 个输出 token，KV cache 恒定，因此能一次前向读完几十页长文档。

## 核心：训练要用 R-SWA，别用全 causal

官方 remote code 的训练 forward 默认走**全 causal**（滑窗只在推理 decode 的 ring-buffer 生效）。直接这样后训练，会造成 train/infer 注意力不一致，**长输出会崩**。本仓库实现了 R-SWA 训练 mask（`src/uocr_train/rswa.py`）让训练注意力与推理一致：reference（prompt+image）全通、output 只 attend 最近 128（含自身，与官方 ring-buffer 逐 token 对齐）。

一个 held-out olmOCR 上的对照（生成相似度，越高越好）：

| | base | 全 causal 训练 | R-SWA 训练 |
|---|---|---|---|
| 均值(6) | 0.787 | 0.465（长样本崩成垃圾） | **0.709**（长样本恢复） |

所以：**后训练一律开 `rswa_train: true`。**

## 安装

```bash
pip install -r requirements.txt   # torch / transformers / peft / pyarrow / pymupdf / pyyaml / pillow
export UOCR_MODEL_DIR=/path/to/baidu_Unlimited-OCR   # 本地 Unlimited-OCR 权重目录
```

加载约定：`attn_implementation="eager"`（`use_mla=false` 时 mha 分支只注册了 eager，FA2/sdpa 会报错）；`use_cache=False`。这些 `model_loader.py` 已处理。

## 快速开始

```bash
# 1) 合成 smoke 数据 -> forward 自检
python scripts/make_synth_smoke_data.py
python scripts/eval_forward.py --jsonl data/synthetic_markdown_smoke/train.jsonl

# 2) LoRA smoke 训练（20 步，验证闭环：forward->loss->存 adapter）
python scripts/train.py --config configs/smoke_lora_attn.yaml

# 3) reload adapter 做推理自检
python scripts/reload_check.py --adapter outputs/smoke_lora_attn --jsonl <你的 jsonl>

# 4) 生成式 eval（会打印 R-SWA 确认 + 与 GT 的相似度）
python scripts/eval_infer.py --adapter <adapter> --jsonl <held-out jsonl> --n 6
```

## 数据

自带 JSONL，每行一个样本（`dataset.py` 解析）：

```json
{"id":"x","image":"images/x.png","mode":"single_gundam",
 "prompt":"<image>document parsing.","target":"...页面文本/markdown...",
 "target_format":"markdown","task":"full_page_parse","meta":{}}
```

多页：`mode:"multi_base"`，`image` 换成 `images:[...]`。图像路径相对 JSONL 所在目录。

一个 olmOCR-mix-1025 -> JSONL 的转换器（本地 parquet + pdf_tarballs 流式渲染）：

```bash
python scripts/convert_olmocr.py --data-root /path/to/olmOCR-mix-1025 \
    --subset 00_documents --chunks 00_documents_train_00000 --max-samples 800 --out data/olmocr_v1
```

## 训练模式（`train_modes.py`）

- `lora_attn`：只在 `self_attn.{q,k,v,o}_proj` 加 LoRA（最小 smoke）
- `lora_decoder`：attn + dense-mlp + shared_experts（**有意排除 64 个 routed experts**：MoE 每 token 只激活 6/64，expert 上 LoRA 梯度稀疏且参数爆炸）
- `full_decoder` / `full_lm`：解冻 `model.layers.*`（全参需多卡 DeepSpeed ZeRO / FSDP，单卡装不下）

视觉侧（SAM+CLIP+projector）全程冻结——模型 forward 本就用 `no_grad` 包住视觉，梯度进不去。

关键 config 项：`train_mode / mode / rswa_train / gradient_checkpointing / lora_r / lr / max_steps / grad_accum`。单卡长序列必须 `gradient_checkpointing: true`。

## 目录

```
src/uocr_train/   constants dataset processor collator model_loader train_modes rswa
scripts/          make_synth_smoke_data convert_olmocr make_multipage eval_forward train reload_check eval_infer
configs/          *.yaml
docs/DESIGN.md    架构与设计（R-SWA、image-token 构造、训练两坑、北极星目标等）
```

## 现状与边界

- 单页 LoRA/R-SWA 训练链路已端到端验证；`multi_base` + R-SWA 训练路径已打通。
- 多页最终目标是"多页图 -> 一份统一、跨页合并的 markdown"（不用 `<PAGE>` 分页）。管线已就绪，**缺的是 target 数据**：一份跨页合并好的连续 markdown（拼单页做不出，需天然连续文档如 arXiv 渲染）。`make_multipage.py` 的拼页只用于打通验证，非能力数据。
- 未做：`multi_gundam`（每页 crop 提分辨率，需改模型 forward）、full 参多卡脚手架。

详见 `docs/DESIGN.md`。

## License

MIT（与 base 模型一致）。本仓库只含训练代码，不含任何模型权重或数据集。
