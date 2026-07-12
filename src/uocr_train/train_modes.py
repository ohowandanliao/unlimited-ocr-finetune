"""按 train_mode 冻结/解冻参数、加 LoRA。视觉侧全程冻结（且模型 forward 已用 no_grad 包住视觉），只训 decoder。

train_mode:
  lora_attn     只在 layers.*.self_attn.{q,k,v,o}_proj 上加 LoRA（最小 smoke）
  lora_decoder  attn + dense-mlp(layer0) + shared_experts 上加 LoRA（有意排除 64 routed experts）
  full_backbone 解冻 attn + dense-mlp + shared_experts 全参（排除 64 routed experts；= lora_decoder 同款模块的全参版，~182M，24G 可跑）
  full_decoder  解冻 model.layers.* 全参（含 experts，~2604M；fp32 master ~53G，需 80G/单卡或 DeepSpeed），视觉/embed/lm_head 冻结
  full_lm       full_decoder + lm_head（可选 embed_tokens）

全参模式：trainable 参数 upcast 到 fp32 作 master weights（见 build_optimizer），forward 在 autocast(bf16) 下算。
LoRA 与全参是独立分支，互不影响。
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

# full_backbone: 与 lora_decoder 同款模块（attn + dense mlp + shared_experts），但全参而非 LoRA。
# 匹配【参数名】：`mlp\.(gate|up|down)_proj` 只命中 dense 层（MoE 层的 mlp 是 mlp.experts.N.* / mlp.gate 路由器，均不匹配）→ 天然排除 64 routed experts。
_RE_FULL_BACKBONE = re.compile(
    r"^model\.layers\.\d+\.("
    r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj)"
    r"|mlp\.shared_experts\.(gate_proj|up_proj|down_proj)"
    r")\."
)

_ALLOWED_MODES = {"lora_attn", "lora_decoder", "full_backbone", "full_decoder", "full_lm"}


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
    # model.layers. 下，freeze-all 后只解冻目标，视觉自然全程冻结（且 forward 已 no_grad）。
    for p in model.parameters():
        p.requires_grad_(False)

    if mode == "full_backbone":
        # 只解冻 attn + dense-mlp + shared_experts 全参，排除 64 routed experts（= lora_decoder 同款模块）
        hit = [n for n, _ in model.named_parameters() if _RE_FULL_BACKBONE.search(n)]
        if not hit:
            raise RuntimeError("full_backbone 匹配到 0 个参数 —— 检查参数命名前缀是否为 'model.layers.'")
        for n, p in model.named_parameters():
            if _RE_FULL_BACKBONE.search(n):
                p.requires_grad_(True)
        print(f"[train_modes] full_backbone: 解冻 {len(hit)} 个参数张量"
              f"（attn+dense_mlp+shared_experts，排除 routed experts）")
        return model

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
    # 全参：把 trainable 参数 upcast 到 fp32 作 master weights（AdamW 状态随之 fp32），forward 在
    # autocast(bf16) 下自动降精度计算 —— 单卡即得 fp32 master，避免裸 bf16 优化的精度损失。
    # 冻结参数保持 bf16。LoRA 不走这里（peft adapter 默认 fp32）。多卡全参可另接 DeepSpeed ZeRO（自带
    # master + 分片），本仓库训练循环暂为单卡。
    if cfg.get("train_mode", "").startswith("full"):
        up = 0
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
                up += 1
        print(f"[train_modes] full 模式：{up} 个 trainable 张量 upcast fp32 作 master weights")
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
