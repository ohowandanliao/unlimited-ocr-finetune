# Unlimited-OCR 训练代码改造说明

日期：2026-07-07

目标：在 `/mnt1/yixuan/unlimited-ocr-posttrain` 下写一套能训练 Unlimited-OCR 的代码，先跑通小规模 post-training，再扩展到论文式 LLM 全参继续训练和部署验证。

这份文档只记录实现相关的结论，删掉了前一版里比较泛的方案讨论。

## 1. 当前结论

### 1.1 不先改 remote model code

当前 Unlimited-OCR remote code 的训练 forward 可以直接跑。

已在服务器实测：

- train mode batch=1 forward/backward：通过。
- `gradient_checkpointing_enable()` + backward：通过。
- 普通 SFT forward 使用 `use_cache=False`：通过。

所以第一阶段不要先改：

```text
hf_cache/modules/transformers_modules/baidu_Unlimited_hyphen_OCR/modeling_unlimitedocr.py
models/baidu_Unlimited-OCR/modeling_unlimitedocr.py
```

训练代码应先作为外部 pipeline 写在：

```text
/mnt1/yixuan/unlimited-ocr-posttrain/train_uocr/
```

### 1.2 “UOCR-native” 是什么意思

`UOCR-native` 不是现成框架名。这里指：训练 pipeline 直接按照 Unlimited-OCR 自己的 forward 输入协议来写，而不是先套 LLaMA-Factory/ms-swift 的通用 VLM processor。

Unlimited-OCR 训练 forward 的关键接口是：

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    images=images,
    images_seq_mask=images_seq_mask,
    images_spatial_crop=images_spatial_crop,
    use_cache=False,
    return_dict=True,
)
```

其中：

- `images` 是 list，长度等于 batch size。
- 每个元素是 `(images_crop, images_ori)`。
- `images_seq_mask` 标出哪些 token 位置要被视觉 embedding 替换。
- `images_spatial_crop` 控制 crop/no-crop 特征拼接。

这个输入协议和通用 VLM 框架不完全一样，尤其是 multi-page。

### 1.3 为什么不先用 LLaMA-Factory

LLaMA-Factory 更适合标准 VLM：通常是一张图对应一个 image placeholder。

Unlimited-OCR 的 multi-page path 是：

```text
一个 <image> token + 多张 page image
```

源码位置：

```text
modeling_unlimitedocr.py:1139 infer_multi()
```

它把多页图全部塞进一个 `<image>` 位置，视觉 token 连在一起。这和很多通用 VLM dataset/template 的假设不同。

所以：

- LLaMA-Factory 不作为第一训练入口。
- ms-swift 后续可以考虑，因为它支持 custom model/custom dataset/plugin，但也需要先有我们自己的 UOCR processor/collator。

### 1.4 第一版推荐训练入口

第一版用 Transformers/PEFT 或薄 PyTorch loop：

```text
JSONL
  -> UOCRDataset
  -> UOCRProcessor/UOCRCollator
  -> UnlimitedOCRForCausalLM.forward(use_cache=False)
  -> loss
  -> checkpoint / LoRA adapter / merged model
```

等 UOCR-native pipeline 跑通，再考虑把同一套数据和 processor 注册到 ms-swift。

### 1.5 `multi_gundam` 要写进设计，但不能当稳定模式

文档里应该出现 `multi_gundam`，但它和 `multi_base` 的地位不同。

当前官方源码里：

```text
modeling_unlimitedocr.py:1139 infer_multi()
```

注释明确写了：

```text
Multi-image inference. Does NOT support crop mode.
```

并且实现方式是：

```text
images_crop = zeros
images_ori = stack(all_page_images)
images_spatial_crop = [[1, 1], [1, 1], ...]
```

这会进入 `UnlimitedOCRModel.forward()` 的 no-crop 分支，逐页循环 `image_ori`。所以 `multi_base` 是当前源码支持的稳定多页路径。

真正的 `multi_gundam` 意味着：

```text
一个样本里有多页
每页既有 global view 又可能有 local crops
每页可能有不同 crop_ratio
```

但当前 crop 分支只按单个 `crop_shape` reshape：

```text
local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
```

它没有逐页处理不同 crop shapes。因此，真正 `multi_gundam` 需要改模型/processor 协议，不能只在 collator 里加一个 mode。

后续可以定义三种状态：

```text
multi_base              # 当前稳定支持
multi_gundam_global     # 多页 + 1024 global view，但不做 local crops；可作为实验近似
multi_gundam            # 多页 + 每页 local crops；需要改 UnlimitedOCRModel.forward
```

第一版训练先实现 `multi_base`。文档和配置里保留 `multi_gundam`，但标注为实验/待改模型侧。

## 2. 训练模式必须同时支持 LoRA 和全参

LoRA 只是先跑通和做快速实验，不应作为唯一训练方式。代码应从一开始支持多种 `train_mode`。

建议配置：

```yaml
train_mode: lora_attn
freeze_vision: true
freeze_projector: true
train_lm_head: false
train_embed_tokens: false
```

### 2.1 `lora_attn`

用途：最小 smoke test，验证数据、collator、loss、保存、reload。

训练范围：

```text
model.layers.*.self_attn.q_proj
model.layers.*.self_attn.k_proj
model.layers.*.self_attn.v_proj
model.layers.*.self_attn.o_proj
```

特点：

- 参数很少。
- 不追效果。
- 最适合先 debug。

### 2.2 `lora_decoder`

用途：正式 LoRA baseline。

训练范围：

```text
model.layers.*.self_attn.{q,k,v,o}_proj
model.layers.*.mlp.*.{gate,up,down}_proj
```

注意 Unlimited-OCR decoder 是 MoE，`model.layers.1+` 里每层有很多 expert，所以 LoRA 参数量不小。

已估算：

- rank 8：约 38.75M LoRA 参数。
- rank 16：约 77.51M LoRA 参数。

### 2.3 `full_decoder`

用途：贴近论文 setting 的主实验。

训练范围：

```text
model.layers.*
```

冻结：

```text
model.sam_model
model.vision_model
model.projector
model.image_newline
model.view_seperator
```

默认不训练：

```text
model.embed_tokens
lm_head
```

原因：

- 论文说冻结 DeepEncoder，只训练 LLM 参数。
- 是否训练 `lm_head/embed_tokens` 可以单独做 ablation。

### 2.4 `full_lm`

用途：更激进的 LLM 全参实验。

训练范围：

```text
model.layers.*
lm_head
model.embed_tokens   # 可选
```

建议：

- 先不要默认开启 `embed_tokens`。
- 如果目标格式加入新 token，才考虑 tokenizer 扩展和 embedding 训练。

### 2.5 不建议第一阶段训练视觉侧

源码里视觉编码分支在：

```text
modeling_unlimitedocr.py:493 with torch.no_grad()
```

第一阶段不要改这里。除非后续明确要训练 projector/vision encoder，否则保持冻结。

## 3. 数据格式

训练数据可以设计得兼容 ms-swift，但 encode 时必须走 Unlimited-OCR 的 `format_messages(..., sft_format="plain", system_prompt="")`，不要用普通 chat template。

### 3.1 单页样本

```json
{
  "id": "single_000001",
  "mode": "single_gundam",
  "images": ["data/synthetic/images/single_000001.png"],
  "prompt": "<image>document parsing.",
  "target": "Invoice No: A-001\nDate: 2026-07-07\nTotal: 123.45",
  "source": "synthetic"
}
```

### 3.2 多页样本

```json
{
  "id": "multi_000001",
  "mode": "multi_base",
  "images": [
    "data/synthetic/images/page_000001.png",
    "data/synthetic/images/page_000002.png"
  ],
  "prompt": "<image>Multi page parsing.",
  "target": "<PAGE>\n第一页解析结果\n<PAGE>\n第二页解析结果",
  "source": "synthetic_concat"
}
```

多页用 `<PAGE>`，因为当前 `infer_multi()` 保存结果时按 `<PAGE>` 切页。

### 3.3 target 格式

第一阶段支持两种 target。

纯文本/markdown：

```text
Invoice No: A-001
Date: 2026-07-07
Total: 123.45
```

layout target：

```text
<|ref|>Invoice No: A-001<|/ref|><|det|>[[72,88,310,125]]<|/det|>
<|ref|>Total: 123.45<|/ref|><|det|>[[700,820,930,870]]<|/det|>
```

坐标建议统一到 `0-999`，因为源码 `process_image_with_refs()` 用的是 `coord / 999 * image_width`。

## 4. Processor / Collator 要怎么改

不要直接照搬 DeepSeek-OCR 社区 collator。应该从已经验证过的：

```text
/mnt1/yixuan/unlimited-ocr-posttrain/debug/debug_batch2_forward.py
```

抽出正式版本。

### 4.1 单页 `single_gundam`

对齐 `model.infer()`：

```text
base_size=1024
image_size=640
crop_mode=True
```

构造：

```text
global view: 1024 x 1024
local crops: dynamic_preprocess(image_size=640)
images = [(images_crop, images_ori)]
images_spatial_crop = [[width_crop_num, height_crop_num]]
```

### 4.2 单页 `single_base`

对齐 single image base mode：

```text
base_size=1024
image_size=1024
crop_mode=False
```

### 4.3 多页 `multi_base`

对齐 `infer_multi()`：

```text
prompt 里只有一个 <image>
images 里有多张 page image
crop_mode=False
image_size=1024
```

构造：

```text
images_ori = torch.stack(page_images, dim=0)
images_crop = torch.zeros((1, 3, image_size, image_size))
images_spatial_crop = [[1, 1], [1, 1], ...]  # 每页一个
```

注意：batch=2 表示两个独立样本，不是一个样本两张图。

### 4.4 多页 `multi_gundam`

需要区分两个概念。

`multi_gundam_global`：

- prompt 仍然只有一个 `<image>`。
- 一个样本内有多页。
- 每页用 1024 global view。
- 不生成 local crop，即 `images_crop` 仍然为 0。
- token 数按每页 1024 global view 生成。
- 这可能能走当前 no-crop forward 分支，但它不是真正 gundam crop。

真正 `multi_gundam`：

- 每页都可能动态裁剪。
- 每页都有自己的 `[width_crop_num, height_crop_num]`。
- 需要 processor 输出 per-page crop metadata。
- 需要模型 forward 在 crop 分支里逐页组装：

```text
for each page:
  encode local crops for this page
  encode global view for this page
  concat local + global + view_separator
concat pages
masked_scatter into the single <image> span
```

因此第一版不要直接承诺 `multi_gundam` 可训练。合理顺序：

1. 先跑 `multi_base`。
2. 再试 `multi_gundam_global` 作为 debug mode。
3. 如果确实需要多页 crop，再改 `UnlimitedOCRModel.forward()` 支持真正 `multi_gundam`。

### 4.5 labels mask

规则：

```text
pad token -> -100
image token -> -100
user prompt token -> -100
assistant target token -> 正常训练
```

实现时要生成 prompt-only version 来得到 prompt length。不要手写估计长度。

### 4.6 必须加的 sanity check

每个 batch forward 前至少检查：

```python
assert input_ids.shape == labels.shape == images_seq_mask.shape
assert len(images) == input_ids.shape[0]
assert torch.isfinite(input_ids.float()).all()
assert (labels != -100).any()
```

对每个样本检查：

```python
image_token_count = images_seq_mask[i].sum().item()
```

这个数量必须和模型视觉分支实际拼出的 embedding 数一致，否则 `masked_scatter_` 会报错或产生难查的 shape bug。

## 5. 具体工程结构

建议创建：

```text
/mnt1/yixuan/unlimited-ocr-posttrain/train_uocr/
  configs/
    smoke_lora_attn.yaml
    smoke_lora_decoder.yaml
    full_decoder_debug.yaml
  scripts/
    make_synth_smoke_data.py
    train_smoke_lora.sh
    train_full_decoder_debug.sh
  src/uocr_train/
    __init__.py
    constants.py
    dataset.py
    processor.py
    collator.py
    model_loader.py
    train_modes.py
    train.py
    eval_forward.py
    infer_transformers.py
    export_adapter.py
  data/
    synthetic/
      train.jsonl
      eval.jsonl
      images/
  outputs/
```

### 5.1 `constants.py`

放固定路径和特殊 token：

```python
ROOT = Path("/mnt1/yixuan/unlimited-ocr-posttrain")
MODEL_DIR = ROOT / "models" / "baidu_Unlimited-OCR"
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
DEFAULT_SINGLE_PROMPT = "<image>document parsing."
DEFAULT_MULTI_PROMPT = "<image>Multi page parsing."
PAGE_TOKEN = "<PAGE>"
```

### 5.2 `dataset.py`

职责：

- 读取 JSONL。
- 解析相对图片路径为绝对路径。
- 校验字段。
- 输出统一 sample dict。

不要在 dataset 里做图像 tensor 变换。

### 5.3 `processor.py`

职责：

- 封装 `format_messages`、`text_encode`、`BasicImageTransform`、`dynamic_preprocess`。
- 实现：

```python
process_single_gundam(sample)
process_single_base(sample)
process_multi_base(sample)
```

输出单样本 tensor：

```python
{
    "input_ids": ...,
    "labels": ...,
    "images_seq_mask": ...,
    "images": (images_crop, images_ori),
    "images_spatial_crop": ...,
    "meta": ...
}
```

### 5.4 `collator.py`

职责：

- pad `input_ids/labels/images_seq_mask`。
- 生成 `attention_mask`。
- 保持 `images` 为 list，长度等于 batch size。
- `images_spatial_crop` 不要盲目 cat；要和 `UnlimitedOCRModel.forward` 的 zip 行为一致。

返回：

```python
{
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
    "images": images,
    "images_seq_mask": images_seq_mask,
    "images_spatial_crop": images_spatial_crop,
}
```

### 5.5 `model_loader.py`

职责：

- 设置 HF cache 到项目目录。
- 加载本地 Unlimited-OCR 权重。
- 默认：

```python
torch_dtype=torch.bfloat16
trust_remote_code=True
use_safetensors=True
model.config.use_cache = False
```

如果配置开启：

```python
model.gradient_checkpointing_enable()
```

这个已经实测可用。

### 5.6 `train_modes.py`

职责：

- 根据 `train_mode` 冻结/解冻参数。
- 根据 `train_mode` 添加 LoRA。

建议提供：

```python
apply_train_mode(model, cfg)
print_trainable_parameters(model)
```

冻结视觉侧：

```python
freeze_prefixes = [
    "model.sam_model",
    "model.vision_model",
    "model.projector",
]
freeze_names = [
    "model.image_newline",
    "model.view_seperator",
]
```

全参 decoder：

```python
requires_grad = name.startswith("model.layers.")
```

LoRA decoder：

只给 `model.layers.*` 下的 module 加 LoRA，避免误加到视觉侧。不要只靠后缀字符串不加过滤。

### 5.7 `train.py`

第一版可以用 `Trainer`，但训练代码不要依赖 `datasets` 必须存在；也可以用 PyTorch `Dataset` + `DataLoader`。

关键配置：

```python
remove_unused_columns=False
bf16=True
use_cache=False
gradient_checkpointing=True
```

如果用 `Trainer`，环境需要安装：

```text
peft
datasets
pyyaml
```

服务器当前环境缺这些包。

### 5.8 `eval_forward.py`

这个文件非常重要，用来在训练前验证数据和 collator。

它应该跑：

```text
batch=1 single_gundam
batch=2 single_gundam
batch=1 multi_base
```

输出：

```text
input_ids shape
labels shape
images_seq_mask sum
target token count
loss
```

### 5.9 `export_adapter.py`

职责：

- 保存 LoRA adapter。
- 可选 merge 到 base model。
- 输出一个可部署模型目录。

部署侧 vLLM/SGLang 第一版建议用 merged model，不要先依赖 runtime LoRA。

## 6. 部署反推训练约束

训练数据和推理部署必须对齐。

Transformers inference 当前官方用法：

```python
model.infer(
    tokenizer,
    prompt="<image>document parsing.",
    image_file="your_image.jpg",
    base_size=1024,
    image_size=640,
    crop_mode=True,
    no_repeat_ngram_size=35,
    ngram_window=128,
)
```

multi-page：

```python
model.infer_multi(
    tokenizer,
    prompt="<image>Multi page parsing.",
    image_files=[...],
    image_size=1024,
    no_repeat_ngram_size=35,
    ngram_window=1024,
)
```

SGLang/vLLM request 侧要保持：

```text
skip_special_tokens=False
temperature=0
custom no-repeat-ngram processor
single image: window_size=128
multi-page: window_size=1024
```

因此训练时不要随便加 system prompt，也不要换成普通 chat template。

## 7. 数据路线

### 7.1 第一阶段：合成 smoke 数据

用 PIL 生成 20-200 条简单单页图：

- 发票
- 收据
- 表格
- 段落

再合成少量 2 页 multi-page。

目的：

- debug collator。
- 验证 loss。
- 验证 LoRA/full_decoder 的训练代码。

### 7.2 第二阶段：OmniDocBench 小样本转换

用 50-100 条先转换：

- page image
- text/layout annotation
- reading order

生成 UOCR target。

注意：OmniDocBench 更适合 eval，不要把评测集污染到训练。

### 7.3 第三阶段：DocLayNet / PubLayNet / PubTabNet

这些更适合补 layout/table 能力。

需要写 converter：

```text
dataset annotation -> normalized bbox -> UOCR target
```

### 7.4 第四阶段：自建 PDF + OCR 标注

更贴论文：

```text
PDF
  -> render pages
  -> PaddleOCR / PP-Structure / MinerU 标注
  -> block text + bbox
  -> single-page JSONL
  -> random concat multi-page JSONL
```

## 8. 实施顺序

建议按这个顺序写代码：

1. 创建 `train_uocr/` 目录结构。
2. 从 `debug_batch2_forward.py` 抽出 `processor.py` 和 `collator.py`。
3. 先实现 `single_gundam/single_base/multi_base`。
4. 文档和 config 里预留 `multi_gundam_global/multi_gundam`，但不要先实现真正 multi crop。
5. 写 `scripts/make_synth_smoke_data.py`。
6. 写 `eval_forward.py`，先只验证 forward loss。
7. 参考 MolSeek `full_sft.py` 写 `train_modes.py`，支持 `lora_attn/lora_decoder/full_decoder/full_lm`。
8. 参考 MolSeek 的 full SFT 配置，支持 `freeze_modules/freeze_layers/split_lrs/resume/accelerate`。
9. 写 `train.py`，先跑 `lora_attn` 20 steps。
10. 跑 `lora_decoder` 20 steps。
11. 跑 `full_decoder_debug` 5 steps。
12. 写 `infer_transformers.py`，验证保存后的模型/adapter。
13. 写 `export_adapter.py`，输出 merged model。
14. 最后再接 SGLang/vLLM 验证。

## 9. 第一版验收标准

第一版完成时至少要满足：

- 能生成 synthetic JSONL 和图片。
- 能读取单页和多页样本。
- `eval_forward.py` 能跑 batch=1 和 batch=2。
- 能输出有限 loss，非 NaN。
- `lora_attn` 能跑 20 steps 并保存 adapter。
- `lora_decoder` 能跑 20 steps。
- `full_decoder_debug` 能跑至少 5 steps。
- 能 reload adapter 或 merged model 做 Transformers 推理。

## 10. 当前服务器环境

已有：

```text
torch 2.10.0+cu128
transformers 4.57.1
accelerate
```

缺少：

```text
peft
datasets
bitsandbytes
vllm
sglang
ms-swift
llamafactory
```

训练第一版至少要装：

```text
peft
datasets
pyyaml
```

vLLM/SGLang 可以等训练和 export 跑通后再单独配。

## 11. 有用参考路径

服务器代码：

```text
/mnt1/yixuan/unlimited-ocr-posttrain/debug/debug_batch2_forward.py
/mnt1/yixuan/unlimited-ocr-posttrain/hf_cache/modules/transformers_modules/baidu_Unlimited_hyphen_OCR/modeling_unlimitedocr.py
/mnt1/yixuan/unlimited-ocr-posttrain/research_repos/WT-ever_deepseek-ocr-lora/datacollator.py
/mnt1/yixuan/unlimited-ocr-posttrain/research_repos/WT-ever_deepseek-ocr-lora/finetune.py
/mnt1/yixuan/unlimited-ocr-posttrain/research_repos/baidu_Unlimited-OCR/infer.py
```

## 12. DeepSeek-OCR 训练代码参考结论

已经看过的训练实现：

```text
WT-ever_deepseek-ocr-lora/finetune.py
WT-ever_deepseek-ocr-lora/datacollator.py
baggie11_Finetuning_Deepseek_OCR/train.py
baggie11_Finetuning_Deepseek_OCR/utils/collator.py
HaCTang_MolSeek-OCR/lora_sft.py
HaCTang_MolSeek-OCR/misc/full_sft.py
HaCTang_MolSeek-OCR/progressive_sft.py
HaCTang_MolSeek-OCR/DeepSeek_OCR_2.py
```

### 12.1 WT-ever / baggie11 的价值

适合参考基础链路：

- conversation dataset。
- `train_on_responses_only` labels mask。
- `remove_unused_columns=False`。
- `Trainer + data_collator`。
- LoRA target modules。

但它们偏简单，主要是单图路径，不适合直接照搬到 Unlimited-OCR 多页。

### 12.2 HaCTang/MolSeek 的价值更高

MolSeek 的代码更接近我们需要的训练工程：

- `lora_sft.py` 支持 LoRA SFT。
- `misc/full_sft.py` 支持 full SFT。
- `progressive_sft.py` 支持从 merged LoRA 继续 full SFT。
- 支持多个 train_sets/val_sets。
- 支持 accelerate 多卡启动。
- 支持 resume checkpoint。
- 支持 gradient checkpointing。
- 支持 `freeze_layers` 和 `freeze_modules`。
- full SFT 里有 split learning rates：

```text
vision_learning_rate
language_learning_rate
```

这对我们很有用。Unlimited-OCR 里可以改成：

```text
vision/projector lr = 0 或冻结
decoder lr = language_learning_rate
lm_head/embed lr = 可选
```

### 12.3 需要从 MolSeek 借鉴的具体实现

应该借鉴：

- YAML config dataclass 化。
- `_maybe_launch_with_accelerate()`。
- `_resolve_resume_checkpoint()`。
- `_torch_load_resume_compat()`。
- `_print_trainable_parameters()`。
- `_apply_freeze_config()`。
- `_build_optimizer_with_split_lrs()`。
- 多 train_sets concat。
- periodic validation callback。

不应直接照搬：

- 它的 DeepSeek-OCR2 image token 计数逻辑，因为 Unlimited-OCR 的 token pattern 不完全一样。
- 它的 `multi-image == 多个 <image>` 假设。
- 它的 SMILES accuracy metric。

### 12.4 对本项目训练代码的影响

`train_uocr` 第一版不要只写一个玩具 Trainer。结构应该兼容后续 full SFT：

```text
train_modes.py:
  apply_train_mode()
  apply_freeze_config()
  build_optimizer_with_split_lrs()

train.py:
  load yaml config
  build multiple datasets
  build collator
  apply train mode
  Trainer/optimizer/resume
```

这样 LoRA 和全参路线可以共用同一套数据/processor/collator。

外部参考：

- vLLM Unlimited-OCR recipe: https://recipes.vllm.ai/baidu/Unlimited-OCR
- ms-swift: https://github.com/modelscope/ms-swift
- LLaMA-Factory data format: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
- OmniDocBench: https://github.com/opendatalab/OmniDocBench
- DocLayNet: https://github.com/DS4SD/DocLayNet
