"""加载 Unlimited-OCR：eager attention、bf16、use_cache=False。

必须显式 eager：use_mla=false 时 ATTENTION_CLASSES 只有 mha_eager，FA2/sdpa 会 KeyError（见 design doc 2.2）。
gradient checkpointing 不在这里开——必须在 apply_train_mode(peft) 之后开，否则 peft 给 embeddings 挂
require-grad hook，撞上模型 forward 对 inputs_embeds 的 in-place masked_scatter_（见 train.py）。
"""
import os
import importlib

import torch
from transformers import AutoModel, AutoTokenizer

from .constants import MODEL_DIR, HF_ENV


def load_model(device: str = "cuda"):
    for k, v in HF_ENV.items():
        os.environ.setdefault(k, v)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, use_safetensors=True,
        dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.config.use_cache = False
    if device:
        model = model.to(device)

    remote_mod = importlib.import_module(model.__class__.__module__)
    print(f"[model_loader] loaded; _attn_implementation="
          f"{getattr(model.config, '_attn_implementation', '?')}; use_cache={model.config.use_cache}")
    return tokenizer, model, remote_mod
