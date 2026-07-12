# 复现 Runbook（新 GPU 机从零到训练+eval）

在一台干净的国内 GPU 机（示例：AutoDL 单卡 RTX 4090 24G，PyTorch 镜像 = Ubuntu22.04 + torch2.5.1/cu124 + conda 在 `/root/miniconda3`）上，照抄下面命令即可跑通。
细节：数据见 [DATA.md](DATA.md)，两个训练坑见 [docs/DESIGN.md](docs/DESIGN.md) §2。

> 本 runbook 已于 2026-07-12 在 AutoDL 4090D 实测跑通：rswa_v1 训练 150 步、eval mean≈0.905。

## 0. 变量与约定
```bash
ROOT=/root/autodl-fs                          # 持久盘：放仓库/模型/数据（别塞 30G 系统盘）
REPO=$ROOT/unlimited-ocr-finetune
PY=/root/miniconda3/bin/python                # 非交互 SSH 里 conda 不在 PATH，一律用绝对路径
export UOCR_MODEL_DIR=$ROOT/models/Unlimited-OCR   # model_loader 从这读模型（必须）
export OLMOCR_DIR=$ROOT/olmOCR-mix-1025            # olmOCR 原始数据根
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
```
坑：`network_turbo`（AutoDL 学术加速）**只**给 GitHub/HF 用，且**只在子 shell 里 source**（会拖慢 pip 等其它源）；pip 一律清华源、不开 turbo。**不需要 flash-attn**（全程 eager）。

## 1. 克隆仓库（GitHub 走学术加速）
```bash
( source /etc/network_turbo; git clone https://github.com/ohowandanliao/unlimited-ocr-finetune.git "$REPO" )
cd "$REPO"
```

## 2. 装依赖（清华源，不开 turbo）
```bash
/root/miniconda3/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
torch 用镜像自带（2.5.1 即可）；requirements.txt 已含 remote-code 依赖（addict/easydict/timm/matplotlib），缺了会在加载模型时 ModuleNotFoundError。

## 3. 下基座模型（ModelScope）
```bash
/root/miniconda3/bin/modelscope download --model PaddlePaddle/Unlimited-OCR --local-dir "$UOCR_MODEL_DIR"
```
国内 GPU 机 HF 下不动（hf-mirror 限速、hf.co 不通、turbo 也 timeout）→ 走 ModelScope（~10MB/s，6.8G 十几分钟）。

## 4. 备数据（olmOCR，少量冒烟）
```bash
$PY scripts/download_olmocr.py --subset 00_documents --split train --chunks 00000
$PY scripts/convert_olmocr.py  --subset 00_documents --split train \
      --chunks 00_documents_train_00000 --max-samples 300 --skip-bad-rotation \
      --out data/olmocr_train_v1
$PY scripts/pull_sample_olmocr.py            # 30 条 eval 探针 -> data/samples_olmocr/train.jsonl
```

## 5. 前向自检（可选但强烈建议，1~2 分钟）
```bash
$PY scripts/eval_forward.py --gpu 0 --jsonl data/olmocr_train_v1/train.jsonl
```
期望：single_gundam / single_base / multi_base 三模式 loss 均 `finite=True`，末行 `eval_forward_ok`。过了说明 环境+模型+数据 全通。

## 6. 训练（R-SWA，150 步）
```bash
nohup $PY -u scripts/train.py --config configs/lora_decoder_rswa_v1.yaml > train_rswa.log 2>&1 &
tail -f train_rswa.log
```
看点：`R-SWA training mask ON (window=128)`（坑1 规避生效）、`routed_experts=0`（正确排除 64 路由专家）、step 递增；结束标志 `train_ok`，adapter → `outputs/lora_decoder_rswa_v1/adapter_model.safetensors`（~16M）。

## 7. Eval
```bash
$PY scripts/eval_infer.py --adapter outputs/lora_decoder_rswa_v1 \
      --jsonl data/samples_olmocr/train.jsonl --n 6 --gpu 0
```
期望：末行 `mean similarity over 6 held-out: ~0.7–0.9`。

## 全参微调（可选，LoRA 的替代；都只训 LLM decoder，与 LoRA 互不影响）
两种范围，代码里是独立 `train_mode`（`src/uocr_train/train_modes.py`）；全参会把 trainable 参数 upcast fp32 作 master weights。
```bash
# 【B】全参 backbone —— attn+dense+shared，排除 64 routed experts（~180M，fp32 master ≈17G，4090 24G 够）
#     = lora_decoder 同款模块的「全参版」，做 LoRA-vs-全参 对照最干净
nohup $PY -u scripts/train.py --config configs/full_backbone_rswa_v1.yaml > train_full_backbone.log 2>&1 &
tail -f train_full_backbone.log

# 【A】真·全参 decoder —— 含 64 routed experts（~2604M，fp32 master ≈53G，需 80G 单卡 A100/H100；多卡 DeepSpeed 另接）
nohup $PY -u scripts/train.py --config configs/full_decoder_rswa_v1.yaml > train_full_decoder.log 2>&1 &
```
- 看点：`full_backbone: 解冻 84 个参数张量`、`84 个 trainable 张量 upcast fp32 作 master`、step 递增、`train_ok`。（full_backbone 已在 24G 实测跑通。）
- 显存分水岭：routed experts 占 decoder 93%，训它们才需 80G，不训则 24G 够（experts 每 token 只激活 6/64，小数据上全参它们既费显存又低效——故 backbone 版默认冻住）。
- **全参 eval 与 LoRA 不同**（TODO：尚未一键化）：full 模式 `save_pretrained` 存的是**完整模型**（非 adapter）。eval 时把 `UOCR_MODEL_DIR` 指到该 checkpoint、且**去掉 `--adapter`**；若报缺 modeling 文件，把基座里的 `*.py`/`config*.json` 拷进 checkpoint 目录。

## A/B（可选，重现「坑1」）
`configs/lora_decoder_olmocr_v1.yaml` = 与 rswa_v1 同数据同超参、唯一区别 `rswa_train=false`（全 causal 训练）。跑它 + eval，长文档会复读崩（4090 上 mean 0.465 vs rswa 0.709），实证 train/infer 注意力必须一致。

## 常见坑速查
- `python: command not found`：非交互 SSH 没加载 conda → 用 `/root/miniconda3/bin/python` 或先 `source /etc/profile`。
- 加载模型 `ModuleNotFoundError`：remote code 缺依赖 → 已在 requirements.txt（addict/easydict/timm/matplotlib）。
- 模型/数据下载卡死：HF 在国内 GPU 机不可靠 → 模型走 ModelScope；数据走 hf-mirror（慢但一次性）。
- 磁盘不够：大件全放 `/root/autodl-fs`（持久 200G），别用系统盘 `/`。
