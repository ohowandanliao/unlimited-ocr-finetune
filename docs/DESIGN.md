# Unlimited-OCR 后训练：干净工程设计（权威版）

日期：2026-07-08
状态：设计定稿，待实现。本文件是 `train_uocr` 工程的唯一事实源，取代/收敛 `unlimited_ocr_training_plan_2026-07-07.md` 中散落的方案讨论（旧 plan 仅留作历史）。

远端根目录：`<GPU-HOST-已迁移> /mnt1/yixuan/unlimited-ocr-posttrain`
本地镜像：`unlimited-ocr-posttrain-local/`

---

## 0. 一句话目标

基于 DeepSeek-OCR 的训练思路，写一套能加载 `baidu/Unlimited-OCR` 权重继续 post-training 的干净工程；先用真实抽样数据（olmOCR / LightOnOCR）打通 forward→loss→保存，再扩到全参和多页。多页的最终目标见下节北极星。

---

## 0.5 北极星目标：多页统一 markdown（最终目标，暂不实现，先记录认识）

最终想做的能力：**多页 PDF -> 一份统一、跨页合并的 markdown**——一次前向输出，跨页的表格/段落重接成连续文档，不用 `<PAGE>` 把内容切断。**目前 Unlimited-OCR 没有这个能力，公开也没有现成的这种数据。** 本节只记录认识和方向，不是现在动手。

### 现状（证据）
- UOCR 多页输出是**逐页 + `<PAGE>` 分隔**：`modeling_unlimitedocr.py:1269` 的 `infer_multi` 做 `outputs.split('<PAGE>')`，模型被训练成页间吐 `<PAGE>`、逐页解析。跨页表格/段落在页边界被切断，不合并。
- paper 5.4 "Long-horizon Parsing"：卖点是"单次前向 prefill 几十~上百页、KV cache 恒定、逐页连续转录"，指标是 edit distance / distinct-n（去重类），**没有跨页结构合并的主张或指标**。"one-shot" 指**推理一次**，不是输出合并成一篇。
- paper 自承坑：多页只用 base(1024×1024 无 crop)，40+ 页误差"主要来自小字看不清"（分辨率），非 R-SWA 迷失方向。

### 同行怎么做跨页合并（都不是端到端）
- **OCRFlux**(3B VLM)：两阶段——逐页 VLM 解析 markdown + **单独一个 LLM** 做"给定相邻两页 markdown，识别该合并元素的 index"（段落拼接；表格处理表头重复/跨页多行单元格/纵向拆分）。监督/评测：OCRFlux-bench-cross(1000 人标 merge index)、pubtabnet-cross(9064 拆分表格对)。
- **MinerU2.5-Pro / PaddleOCR-VL-1.5**：把"截断段落合并 + 跨页表格合并"当长文档特性内置。
- 共同点：合并那步只看两页的 **markdown 文本，已经丢了图像**。

### 我们的差异化角度（大而深的主张）
R-SWA = 每个输出 token 对**所有页的图像 token 全注意力 reference（永不滑出）**，只有输出文本走 128 滑窗。=> UOCR 一次前向里所有页图像始终在场，**天然能"边看原图边输出统一跨页 markdown"**，合并发生在"还能重读原图"时，而非事后文本对齐。这是两阶段管线结构上做不了的。
- 难点/真问题：128 的**输出窗** => 局部跨页合并（边界续接）靠重读图像可行；但**全局一致性**（跨几十页的标题编号、目录）受限——这是核心实验点。且需要"合并好的多页 GT"这种目前没有的数据。

### 数据启示
- **随机拼单页 = 教模型逐页，不教合并，错。**
- 要天然连续文档：**arXiv 论文**(LaTeX/HTML 源本就连续 -> 渲染多页图 + 连续 markdown GT，跨页表格/段落天然合并)、书、长报告；加 **PubTabNet-cross 式拆分表格对**补表格合并。

### 通往北极星的踏脚石（当前工程已在路上）
1. `multi_base` forward->loss 已通（eval_forward 实测，见 status doc）——踏脚石，非目标。
2. 近期宜先做窄而硬的子问题：**跨页表格合并**（PubTabNet-cross 造数据，TEDS 评测）。
3. 再扩到整页**统一 markdown**（arXiv 连续文档数据）。
4. `multi_gundam`（每页 crop 提分辨率）是正交的能力向坑，需改 `UnlimitedOCRModel.forward` 的 crop 分支，另计。**其 justification（token 效率而非提分辨率本身）、与 R-SWA 的关系、go/no-go 闸门，见 `multipage_encoder_bakeoff_2026-07-14.md`。**

> 竞品核实（HunyuanOCR 1.0/1.5：仍无「多页→合并 markdown 解析」）、切块 vs 原生分辨率两轴、光学压缩反转、以及编码器 bake-off 方案 → 见 `multipage_encoder_bakeoff_2026-07-14.md`（2026-07-14 补充档）。

参考：OCRFlux https://github.com/chatdoc-com/OCRFlux ; PaddleOCR-VL-1.5 https://ernie.baidu.com/blog/posts/paddleocr-vl-1.5/ ; MinerU https://github.com/opendatalab/mineru

---

## 1. 关键架构事实：没改结构，只改了 attention

结论：Unlimited-OCR 的 backbone 就是 DeepSeek-OCR 那套 DeepSeek-V2 解码器（12 层 MoE，hidden 1280，64 expert，6 激活，v_head_dim 128）+ DeepEncoder(SAM-b + CLIP-l) + linear projector，**层结构、权重接口、视觉侧全没动**。唯一实质改动在解码器注意力：

| 维度 | 原版 DeepSeek-OCR | Unlimited-OCR |
|---|---|---|
| 注意力 | MLA（Multi-head Latent Attention，KV 低秩压缩 + rope/nope 拆分） | 标准 Llama MHA + 128 滑窗 ring-buffer |
| config | `use_mla=true`，`kv_lora_rank/q_lora_rank/qk_nope_head_dim/qk_rope_head_dim` 有值 | `use_mla=false`，上述全清零/`null`，新增 `sliding_window=128` / `sliding_window_size=128` |
| 实现类 | `DeepseekV2Attention` | `SlidingWindowLlamaAttention` |

选择逻辑：`DeepseekV2DecoderLayer.__init__` 按 `config.use_mla` 拼 `"mla_"` 或 `"mha_"` 前缀，再查 `ATTENTION_CLASSES`（`modeling_deepseekv2.py:1380,1398`）。`use_mla=false` → `mha_eager` → `SlidingWindowLlamaAttention`。

"Unlimited" 的机制：`infer/infer_multi` 里把 `config.sliding_window` 临时设 `None`、改读 `config._ring_window`，让 decode 时用环形缓冲维护 KV cache（`modeling_deepseekv2.py:1310-1377`）——prompt/image token 全保留，自己输出只保留最近 128 个（环形覆盖）。KV cache 有界 → 任意长文档/多页解码不 OOM。

术语精确化（避免混淆）：**「滑窗」指注意力范围**——每个输出 token 只 attend 最近 128 个输出 + 全部 reference（prompt/image，永不滑出，即 R-SWA 里的「Reference」）；**「环形 / ring-buffer」指这 128 的 KV cache 在 decode 时的实现**——`SlidingWindowLlamaAttention` 类（类注释原文 "sliding window KV cache using a ring buffer during decode"），`modeling_deepseekv2.py:1366` 的 `ring_pos = (ring_pos + 1) % 128` 循环覆盖最旧输出。二者常合称 R-SWA / 「滑窗环形缓冲」，但一个是**注意力机制**、一个是 **KV cache 实现**，别当成一件事。

证据文件：
- `models/baidu_Unlimited-OCR/config.json`（`use_mla`, `sliding_window`）
- `models/baidu_Unlimited-OCR/modeling_deepseekv2.py`：`SlidingWindowLlamaAttention`(1232)、`ATTENTION_CLASSES`(1380)、DecoderLayer 选择(1398)、Model.forward mask(1719)
- `models/baidu_Unlimited-OCR/modeling_unlimitedocr.py`：`infer`(787)、`infer_multi`(1139)、ring-window 开关(998-1045,1233-1257)

---

## 2. 训练必须显式处理的两个坑

### 2.1 train/infer 注意力不一致（全 causal vs 滑窗）

`DeepseekV2Model.forward` 在 `modeling_deepseekv2.py:1719` 调 `_prepare_4d_causal_attention_mask` 时**没传 `sliding_window`** → 训练是**全 causal**：每个 output token attend 全部历史。
但推理 decode 走 ring-buffer → 只 attend（全部 prompt/image + 自己最近 128 个 output）。

后果：训练与推理的 output-to-output 注意力范围不同。对 OCR（输出基本随图从左到右、局部性强）通常不致命。

决策：
- v1：**跟官方，训练用全 causal**（最简单，也最可能贴近 baidu 自己的训练）。
- 预留一个 config 开关 `train_sliding_window: false`，置 true 时在训练 collator/mask 里构造滑窗带状 causal mask，做 ablation。默认 false。

### 2.2 必须 eager 加载，禁用 FlashAttention2

`ATTENTION_CLASSES` 里 `mha_flash_attention_2` 被注释掉（`modeling_deepseekv2.py:1388`），`use_mla=false` 时只有 `mha_eager`。若 `config._attn_implementation` 落到 `flash_attention_2` 或 `sdpa`，查表会 KeyError。

debug 脚本没显式指定也能跑，是因为自定义模型未声明 sdpa 支持、transformers 回退到 eager。为保险，**训练/推理 loader 都显式 `attn_implementation="eager"`**，且严禁 FA2。模型小（3.3B）+ 序列可控，eager 训练可接受。

补充（推理也 eager，故 FA2 不构成 train/infer 不一致）：FA2 在这个 MHA 分支**根本没接**（`mha_flash_attention_2` 注释于 1388），所以**推理也只能 eager**——`infer/infer_multi` 不设 `attn_implementation`，transformers 回退到 eager。即训练与推理在**注意力实现**上一致，不存在「train eager / infer FA2」的错位；「坑」仅是「显式指定 FA2 → 查表 KeyError」。真正需要 train/infer 一致的是 **mask**（见 §2.1），已由 `rswa.py` 处理。对照：DeepSeek-OCR 原版 MLA 分支保留了 `mla_flash_attention_2`，Baidu 只把 MHA 分支的 FA2 注释掉了。

### 2.3 loss：标准 SFT（官方 forward 算），外加 MoE 均衡 aux loss（全参训 experts 重点）

**主 loss 不是我们写的，是官方 forward 算的。** `model(..., labels=...)` 走 `UnlimitedOCRForCausalLM.forward`（`modeling_unlimitedocr.py:668-679`）：shift 一位 + `CrossEntropyLoss()`（默认 `ignore_index=-100`）。我们只负责喂对 labels，不碰 loss 数学 → 与模型自身算法天然一致。

**labels 掩码 = 标准 SFT**（`processor.py:155-157`）：`labels = input_ids.clone(); labels[:prompt_len]=-100; labels[images_seq_mask]=-100` → **只有 target（natural_text）算 loss**，prompt / image token 全 mask。与 DeepSeek-OCR 的 SFT 同范式（它另有一阶段单训 DeepEncoder，那是视觉预训练、不是这个文本 SFT loss）。

**MoE 负载均衡 aux loss（已查实生效）**：`modeling_deepseekv2.py:508-537` 有专家均衡 aux loss，`if self.training and alpha>0` 才算。config.json **未显式写** `aux_loss_alpha`，用类默认——transformers 实际加载确认为 **`aux_loss_alpha=0.001`（>0）、`seq_aux=True`**。它经 `AddAuxiliaryLoss`（540-558）**只往反向注入梯度、不改 `out.loss` 的数值** → 监控到的 loss 是纯 SFT loss。

各 train_mode 下的行为（**全参重点**）：

| 模式 | router(gate) | MoE aux loss 效果 |
|---|---|---|
| lora_* / full_backbone | 冻结 | aux 梯度落到冻结 router → **无效、不影响**；loss 是纯 SFT |
| **full_decoder** | 训练 | **aux 生效，自动均衡 64 experts（alpha=0.001）→ 无需手动加**，同 DeepSeek-V2 标准 |

全参训 experts 时注意：
- reported `out.loss` 是纯 SFT，专家均衡是**隐式梯度**，别以为"没均衡"。
- grad-ckpt(non-reentrant) + `AddAuxiliaryLoss` 自定义 autograd 的组合，建议**实测确认**（跑几十步看专家使用分布 / router 是否塌到少数专家）。
- 若哪天把 `aux_loss_alpha` 显式设为 0，就等于关掉均衡，full_decoder 会有专家不均衡/坍塌风险。

---

## 3. 硬件与训练模式可行性

服务器：4× RTX 4090 24G。模型 6.67GB bf16（约 3.3B，绝大多数是 MoE experts）。`/mnt1` 剩 ~21T。

| train_mode | 训练范围 | 单卡 4090 | 备注 |
|---|---|---|---|
| `lora_attn` | `layers.*.self_attn.{q,k,v,o}_proj` | 轻松 | 最小 smoke，验证数据/collator/loss/保存 |
| `lora_decoder` | attn + `mlp.*.{gate,up,down}_proj`（含 expert） | 可 | 正式 LoRA baseline，rank8≈39M / rank16≈78M |
| `full_decoder` | `model.layers.*` 全参（含 3B experts） | **装不下**（AdamW master+m+v≈46GB） | 需 4 卡 accelerate/DeepSpeed ZeRO-2/3，或 8bit-adam+offload |
| `full_lm` | `layers.* + lm_head (+embed 可选)` | 同上 | 更激进，embed 默认不训 |

冻结（所有模式）：`model.sam_model / vision_model / projector / image_newline / view_seperator`。视觉编码分支在 `modeling_unlimitedocr.py:493 with torch.no_grad()`，第一阶段不碰。

结论：LoRA 单卡即可；full 系列必须走多卡脚手架 —— 这正是要借 MolSeek `_maybe_launch_with_accelerate` 的原因。

---

## 4. 干净工程目录结构

```text
train_uocr/
  configs/
    smoke_lora_attn.yaml
    lora_decoder.yaml
    full_decoder.yaml
  scripts/
    download/                     # 下载类脚本(Mac 侧联网抓数据)
      pull_sample_olmocr.py       # 抽样真实数据 -> images/ + train.jsonl
    make_synth_smoke_data.py      # 已有
    train.sh / train_accelerate.sh
  src/uocr_train/
    __init__.py
    constants.py                  # 路径、IMAGE_TOKEN_ID=128815、prompt、PAGE_TOKEN
    dataset.py                    # JSONL -> 统一 sample dict（不做 tensor 变换）
    processor.py                  # 从 debug_batch2_forward.py 抽：single_gundam/single_base/multi_base
    collator.py                   # pad + images list + spatial_crop list
    model_loader.py               # eager 加载 + gradient_checkpointing + use_cache=False
    train_modes.py                # apply_train_mode / apply_freeze / split_lr / print_trainable
    train.py                      # yaml -> dataset -> collator -> train_mode -> Trainer/loop
    eval_forward.py               # 训练前 sanity：batch1/2 single + multi_base -> loss
    export_adapter.py             # 存 LoRA / merge / 导出可部署目录
    infer_transformers.py         # reload 验证
  data/
    synthetic_markdown_smoke/     # 已有
    samples_olmocr/               # 本次抽样落地
  outputs/
```

原则：不改 `models/` 下的 remote model code（第一阶段官方 forward 直接可训）；训练代码全部作为外部 pipeline。

---

## 5. 各模块接口（以已验证的 debug_batch2_forward.py 为准）

### 5.1 processor —— image token 构造（这是最易错的地方）

常量：`patch_size=16`，`downsample_ratio=4`，`IMAGE_TOKEN_ID=128815`。

`single_gundam`（对齐 `model.infer`，base_size=1024, image_size=640, crop_mode=True）：
- `num_queries = (image_size//16 + 3)//4 = 10`（每个 local crop tile 边）
- `num_queries_base = (1024//16 + 3)//4 = 16`（global view 边）
- global view token 串：`([IMG]*16 + [IMG]) * 16 + [IMG]` = 273（16x16 网格，每行后跟一个 newline token，末尾一个 view 分隔）
- 若 `crop_ratio=[W,H]` 有裁剪（W>1 或 H>1）：追加 `([IMG]*(10*W) + [IMG]) * (10*H)`
- 图 <=640x640 时 `crop_ratio=[1,1]`、无 local crop
- 输出图张量：`images_ori`=stack(global_views)，`images_crop`=stack(crops) 或 zeros，`images_spatial_crop=[W,H]`

`single_base`（base mode，image_size=1024, crop_mode=False）：
- `num_queries=16`，token 串 `([IMG]*16+[IMG])*16 + [IMG]`=273，`spatial_crop=[1,1]`

`multi_base`（对齐 `infer_multi`，多页塞进一个 `<image>`，crop_mode=False, image_size=1024）：
- `images_ori = stack(page_images)`，`images_crop = zeros`，`images_spatial_crop = [[1,1]]*页数`
- 注意 batch=2 是两个独立样本，不是一个样本两页

`multi_gundam`：暂不实现真正多页 crop（需改 `UnlimitedOCRModel.forward`）。config 里保留 `multi_gundam_global`（多页各 1024 global view、无 local crop）作为近似 debug mode。

单样本输出 dict：`{input_ids, labels, images_seq_mask, images:(images_crop,images_ori), images_spatial_crop, prompt_len, image_tokens}`。

### 5.2 labels mask

`labels = input_ids.clone(); labels[:prompt_len] = -100; labels[images_seq_mask] = -100`。
`prompt_len` 必须靠 **prompt-only 版本重新 encode** 得到，不能手算长度。encode 走 `remote_mod.format_messages(conversation, sft_format="plain", system_prompt="")` + `text_encode`，**不要用普通 chat template、不要加 system prompt**（否则和部署侧不一致）。

### 5.3 collator

pad `input_ids`(pad_id)、`labels`(-100)、`images_seq_mask`(False)；`attention_mask = input_ids.ne(pad_id)`；`images` 保持 list（len=batch，每元素 `(crop,ori)`）；`images_spatial_crop` 保持 list。返回 dict 直接喂 `model(**batch, use_cache=False, return_dict=True)`。

### 5.4 model_loader

```python
AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, use_safetensors=True,
                          torch_dtype=torch.bfloat16, attn_implementation="eager")
model.config.use_cache = False
if cfg.gradient_checkpointing: model.gradient_checkpointing_enable()  # 已实测可用
```
HF 环境：`HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`、`HF_HOME=ROOT/hf_cache`（debug 脚本已内置）。

### 5.5 每 batch sanity（eval_forward 必查）

```python
assert input_ids.shape == labels.shape == images_seq_mask.shape
assert len(images) == input_ids.shape[0]
assert (labels != -100).any()
# 每样本：images_seq_mask[i].sum() 必须等于视觉分支实际拼出的 embedding 数，否则 masked_scatter_ 报错
```

---

## 6. 数据 JSONL schema（抽样与正式统一）

不要只塞一个 `text` 字段。用带 `task/target_format/meta` 的 schema，便于后续按任务/语言/license/格式采样：

```json
{
  "id": "olmocr_000001",
  "source": "allenai/olmOCR-mix-1025",
  "image": "data/samples_olmocr/images/olmocr_000001.png",
  "mode": "single_gundam",
  "prompt": "<image>document parsing.",
  "target": "...natural_text 或 markdown/html-table...",
  "target_format": "natural_text_or_html",
  "task": "full_page_parse",
  "meta": {"language": "en", "is_table": true, "page_number": 7, "license": "odc-by"}
}
```

多页：`mode="multi_base"`，`image` 换成 `images:[...]`，多页 target 用 `<PAGE>` 分隔（`infer_multi` 按 `<PAGE>` 切页）。

坐标类 layout target 统一到 0-999（源码 `process_image_with_refs` 用 `coord/999*width`）。

---

## 7. 从 MolSeek(HaCTang) 借什么 / 弃什么

MolSeek = 化学分子 OCR（图→SMILES），数据是 RDKit 现渲的分子图。**数据、`dataset.py`、SMILES-accuracy metric、"multi-image==多个 `<image>`" 假设，全弃。**

只借通用训练脚手架：
- `_maybe_launch_with_accelerate()`（多卡启动）
- `_apply_freeze_config()`（freeze_modules / freeze_layers）
- `_build_optimizer_with_split_lrs()`（vision_lr / language_lr → 我们改成 vision=0/冻结，decoder=language_lr）
- `_resolve_resume_checkpoint()` / `_torch_load_resume_compat()`
- `_print_trainable_parameters()`
- 多 train_sets concat、periodic validation callback、YAML dataclass 化

---

## 8. 实施顺序与验收

顺序：
1. `constants.py` + `dataset.py`
2. 从 `debug_batch2_forward.py` 抽 `processor.py`（single_gundam/single_base/multi_base）+ `collator.py`
3. `model_loader.py`（eager + gc + use_cache=False）
4. `eval_forward.py`：先只验 forward loss（batch1/2 single、batch1 multi_base）
5. `train_modes.py`（lora_attn/lora_decoder/full_decoder/full_lm + freeze + split_lr）
6. `train.py`：lora_attn 跑 20 步 → lora_decoder 20 步 → full_decoder 5 步
7. `export_adapter.py` + `infer_transformers.py`：reload 验证
8. 最后接 SGLang/vLLM

第一版验收：
- 能读单页/多页样本；`eval_forward` batch1/2 出有限 loss（非 NaN）
- `lora_attn` 20 步存 adapter；`lora_decoder` 20 步；`full_decoder` ≥5 步（多卡）
- reload adapter / merged model 做 Transformers 推理正常

环境缺包（要装）：`peft datasets pyyaml`（+ 全参多卡时 `deepspeed` 或 `bitsandbytes`）。

---

## 9. 与真实抽样数据的适配点（本轮重点）

用 olmOCR / LightOnOCR 抽样驱动，而不是只靠 synthetic：
- olmOCR-mix-1025 的 GT 是 `natural_text`（含 `<table>`/HTML 片段），不是纯 Markdown → `target_format=natural_text_or_html`，先不强转 Markdown。
- 图像来源：olmOCR 的图是 PDF 页（tarball 内），抽样脚本要 PDF→PNG 渲染；LightOnOCR 若为 inline image 更省事，抽样时二选一验证。
- 抽样后立即跑 `eval_forward` 于真实图，验证：真实长文本 target 的 token 长度分布、image_seq_mask 计数与视觉 embedding 对齐、loss 有限。
- 数据主线（全量混合、清洗、verifier）由你另开 session 处理；本工程只需能吃统一 JSONL。
