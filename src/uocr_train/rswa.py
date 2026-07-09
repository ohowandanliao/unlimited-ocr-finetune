"""R-SWA 训练 mask：让训练的注意力匹配推理时的 Reference Sliding Window Attention。

推理时 R-SWA 的语义（见 modeling_unlimitedocr infer + SlidingWindowLlamaAttention ring-buffer）：
  - prefill = prompt + image tokens（"reference"），全程保留、被所有位置全注意力 attend、永不滑出；
  - output = assistant 生成的 token，每个只能 attend（全部 prefill）+（自己最近 window 个 output token）。
训练默认走全 causal（modeling_deepseekv2.py:1719 的 _prepare_4d_causal_attention_mask 不带 sliding_window），
与推理不一致 -> 长输出失稳。本模块构造 R-SWA 4D 加性 mask 并 patch 进模型 mask 构造。

注意：不能用普通带状滑窗（那会把 image/prefill 也划进窗口）；reference 必须全通。
prefill 边界 = labels 第一个 != -100 的位置（assistant 响应起点 = prompt_len）。
"""
import torch

# 训练循环每个 batch 前 set_mask(...)，forward 后 clear()。单进程单线程，用模块级变量即可。
_CURRENT = {"mask": None}


def build_rswa_mask(labels: torch.Tensor, attention_mask: torch.Tensor, window: int, dtype: torch.dtype):
    """labels/attention_mask: (B, L)。返回 (B,1,L,L) 加性 mask（0=可 attend，min=屏蔽）。"""
    B, L = labels.shape
    device = labels.device
    min_val = torch.finfo(dtype).min
    q = torch.arange(L, device=device)[:, None]        # (L,1)
    k = torch.arange(L, device=device)[None, :]        # (1,L)
    causal = k <= q                                     # (L,L)
    within = (q - k) < window                           # (L,L) 最近 window 个（含自身）
    masks = []
    for b in range(B):
        nz = (labels[b] != -100).nonzero(as_tuple=False)
        boundary = int(nz[0].item()) if nz.numel() else L   # prefill 边界
        ref = k < boundary                              # (1,L) -> broadcast：k 属于 prefill/reference
        # output q(>=boundary)：attend 全部 prefill(ref) 或最近窗(within)；prefill q(<boundary)：纯 causal
        allow = causal & (ref | within)
        allow = allow & attention_mask[b].bool()[None, :]   # 屏蔽 pad 列
        masks.append(torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                                 torch.full((), min_val, dtype=dtype, device=device)))
    return torch.stack(masks, dim=0).unsqueeze(1)       # (B,1,L,L)


def install(remote_deepseekv2_module):
    """patch modeling_deepseekv2 的 _prepare_4d_causal_attention_mask：设了 _CURRENT 就用 R-SWA mask，否则原逻辑。"""
    M = remote_deepseekv2_module
    if getattr(M, "_rswa_installed", False):
        return
    orig = M._prepare_4d_causal_attention_mask

    def patched(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window=None):
        cur = _CURRENT["mask"]
        if cur is not None:
            if not getattr(M, "_rswa_logged", False):
                print(f"[rswa] R-SWA mask APPLIED at mask-build: shape={tuple(cur.shape)}")
                M._rswa_logged = True
            return cur.to(inputs_embeds.dtype)
        return orig(attention_mask, input_shape, inputs_embeds, past_key_values_length)

    M._prepare_4d_causal_attention_mask = patched
    M._rswa_installed = True
    print("[rswa] patched modeling_deepseekv2._prepare_4d_causal_attention_mask")


def deepseek_module_of(model):
    """从加载好的模型拿到 modeling_deepseekv2 模块对象（SlidingWindowLlamaAttention 定义在那）。"""
    import importlib
    return importlib.import_module(type(model.model.layers[1].self_attn).__module__)


def set_mask(m):
    _CURRENT["mask"] = m


def clear():
    _CURRENT["mask"] = None


if __name__ == "__main__":
    # 自测：boundary=3, window=2, L=7 -> 检查 output token 只 attend prefill + 最近2
    labels = torch.tensor([[-100, -100, -100, 5, 6, 7, 8]])
    attn = torch.ones((1, 7), dtype=torch.long)
    m = build_rswa_mask(labels, attn, window=2, dtype=torch.float32)[0, 0]
    ok = (m == 0)
    # q=5(output) 应 attend k in {0,1,2 (prefill)} + {4,5 (window=2)}，不 attend 3
    row5 = ok[5].tolist()
    assert row5 == [True, True, True, False, True, True, False], row5
    # q=2(prefill) 纯 causal：attend 0,1,2
    assert ok[2].tolist() == [True, True, True, False, False, False, False], ok[2].tolist()
    print("rswa self-test ok")
