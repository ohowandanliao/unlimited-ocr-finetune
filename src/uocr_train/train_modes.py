"""按 train_mode 冻结/解冻参数、加 LoRA。视觉侧全程冻结（且模型 forward 已用 no_grad 包住视觉），只训 decoder。

train_mode:
  lora_attn     只在 layers.*.self_attn.{q,k,v,o}_proj 上加 LoRA（最小 smoke）
  lora_decoder  attn + dense-mlp(layer0) + shared_experts 上加 LoRA（有意排除 64 routed experts）
  full_decoder  解冻 model.layers.* 全参（含 experts），视觉/embed/lm_head 冻结
  full_lm       full_decoder + lm_head（可选 embed_tokens）
"""
import re

import torch

_RE_LORA_ATTN = re.compile(r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
# lora_decoder 有意【排除 64 个 routed experts】：MoE 每 token 只激活 6/64，routed expert 上的
# LoRA 梯度极稀疏、参数量还爆炸。只放 attention + dense 层(layer0) mlp + 常激活的 shared_experts。
# 若确要给 routed experts 加 LoRA：把下面 shared_experts 那行换成 (experts\.\d+\.|shared_experts\.)。
_RE_LORA_DECODER = re.compile(
    r"^model\.layers\.\d+\.("
    r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj)"                     # 仅 dense 层(layer0)
    r"|mlp\.shared_experts\.(gate_proj|up_proj|down_proj)"     # 常激活 shared expert
    r")$"
)

_ALLOWED_MODES = {"lora_attn", "lora_decoder", "full_decoder", "full_lm"}


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train_modes] trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M "
          f"({100 * trainable / max(total, 1):.3f}%)")
    return trainable, total


def _lora_targets(model, pattern: re.Pattern):
    targets = sorted({n for n, _ in model.named_modules() if pattern.search(n)})
    if not targets:
        raise RuntimeError("LoRA target_modules 匹配到 0 个模块 —— 模块命名与预期不符，"
                           "检查 model.named_modules() 前缀是否为 'model.layers.'")
    return targets


def apply_train_mode(model, cfg: dict):
    """就地设置 requires_grad / 加 LoRA，返回（可能被 peft 包装后的）模型。"""
    mode = cfg["train_mode"]
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"unknown train_mode {mode!r}, expected one of {sorted(_ALLOWED_MODES)}")

    if mode.startswith("lora"):
        from peft import LoraConfig, get_peft_model

        targets = _lora_targets(model, _RE_LORA_ATTN if mode == "lora_attn" else _RE_LORA_DECODER)
        n_attn = sum(1 for t in targets if "self_attn" in t)
        n_shared = sum(1 for t in targets if "shared_experts" in t)
        n_routed = sum(1 for t in targets if ".experts." in t)  # 期望 0（有意排除）
        print(f"[train_modes] {mode}: {len(targets)} LoRA modules (attn={n_attn}, "
              f"dense_mlp={len(targets) - n_attn - n_shared - n_routed}, "
              f"shared_experts={n_shared}, routed_experts={n_routed})")
        lora = LoraConfig(
            r=int(cfg.get("lora_r", 8)), lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)), bias="none",
            target_modules=targets, task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora)  # peft 冻结全部 base、只训 LoRA；视觉不在 targets，天然冻结

    # ---- 全参模式 ----
    # 视觉侧(model.sam_model / vision_model / projector / image_newline / view_seperator)不在
    # model.layers. 下，freeze-all 后只解冻 model.layers.*，视觉自然全程冻结（且 forward 已 no_grad）。
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if n.startswith("model.layers."):
            p.requires_grad_(True)
    if mode == "full_lm":
        for n, p in model.named_parameters():
            if n.startswith("lm_head") or (cfg.get("train_embed_tokens", False) and "embed_tokens" in n):
                p.requires_grad_(True)
    return model


def build_optimizer(model, cfg: dict):
    """默认单一 lr；可选给 lm_head 拆分 lr（视觉已冻结，无需 vision lr）。"""
    lr = float(cfg.get("lr", 1e-4))
    # 全参 + 裸 AdamW + bf16 参数：无 fp32 master weights，精度受损。正式全参走多卡 DeepSpeed ZeRO
    # (自带 fp32 master) 或对 trainable 参数 upcast fp32。LoRA 不受影响(peft adapter 默认 fp32)。
    if cfg.get("train_mode", "").startswith("full"):
        print("[train_modes][WARN] full 模式裸 AdamW+bf16 无 fp32 master；正式全参请用 DeepSpeed ZeRO 或 upcast fp32")
    lm_head_lr = cfg.get("lm_head_lr")
    if lm_head_lr is None:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    decoder_params, head_params = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (head_params if n.startswith("lm_head") else decoder_params).append(p)
    return torch.optim.AdamW(
        [{"params": decoder_params, "lr": lr}, {"params": head_params, "lr": float(lm_head_lr)}]
    )
