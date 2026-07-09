#!/usr/bin/env python3
"""薄训练循环（第一版，单页为主）。YAML -> dataset -> processor/collator -> train_mode -> loop -> save。

用法：python scripts/train.py --config configs/smoke_lora_attn.yaml
"""
import argparse
import os
import sys
import pathlib

import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from uocr_train.model_loader import load_model
from uocr_train.dataset import UOCRJsonlDataset
from uocr_train.processor import UOCRProcessor
from uocr_train.collator import UOCRCollator
from uocr_train.train_modes import apply_train_mode, build_optimizer, print_trainable_parameters


def to_cuda(batch):
    out = dict(batch)
    for k in ("input_ids", "attention_mask", "labels", "images_seq_mask"):
        out[k] = batch[k].cuda()
    out["images"] = [(c.cuda(), o.cuda()) for c, o in batch["images"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    print("[train] config:", cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu", "0"))
    # 抗显存碎片（eager 全注意力长序列易碎片）；须在首次 CUDA 分配前设置
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(int(cfg.get("seed", 0)))

    want_gc = bool(cfg.get("gradient_checkpointing", False))
    want_rswa = bool(cfg.get("rswa_train", False))
    tok, model, remote = load_model()
    from uocr_train import rswa
    if want_rswa:
        rswa.install(rswa.deepseek_module_of(model))   # patch mask 构造为 R-SWA（匹配推理）
        print(f"[train] R-SWA training mask ON (window={cfg.get('rswa_window', 128)})")
    model = apply_train_mode(model, cfg)
    print_trainable_parameters(model)
    if want_gc:
        # peft 包装之后再开 GC：避免 peft 自动给 input embeddings 挂 require-grad hook，
        # 与模型 forward 的 in-place masked_scatter_ 冲突；non-reentrant 不需要输入 require-grad。
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        print("[train] gradient_checkpointing enabled (non-reentrant, post-peft)")
    model.train()

    proc = UOCRProcessor(tok, remote)
    coll = UOCRCollator(tok)
    ds = UOCRJsonlDataset(cfg["train_jsonl"])
    assert len(ds) > 0, f"empty dataset: {cfg['train_jsonl']}"
    forced_mode = cfg.get("mode")  # 训练时统一 mode（覆盖 jsonl 里的 mode）

    def collate(rows):
        enc = [proc.encode(dict(r, mode=forced_mode) if forced_mode else r) for r in rows]
        return coll(enc)

    dl = torch.utils.data.DataLoader(
        ds, batch_size=int(cfg.get("batch_size", 1)), shuffle=True,
        collate_fn=collate, num_workers=0, drop_last=False,
    )

    optim = build_optimizer(model, cfg)
    accum = int(cfg.get("grad_accum", 1))
    max_steps = int(cfg.get("max_steps", 20))
    clip = float(cfg.get("max_grad_norm", 1.0))
    save_every = int(cfg.get("save_every", 0))
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=int(cfg.get("warmup_steps", 0)), num_training_steps=max_steps)

    step, micro = 0, 0
    optim.zero_grad()
    losses = []
    while step < max_steps:
        for batch in dl:
            b = to_cuda(batch)
            if want_rswa:
                rswa.set_mask(rswa.build_rswa_mask(
                    b["labels"], b["attention_mask"], int(cfg.get("rswa_window", 128)), torch.bfloat16))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"],
                    images=b["images"], images_seq_mask=b["images_seq_mask"],
                    images_spatial_crop=b["images_spatial_crop"], use_cache=False, return_dict=True,
                )
                loss = out.loss / accum
            loss.backward()
            if want_rswa:
                rswa.clear()
            micro += 1
            if micro % accum == 0:
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, clip)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                lv = float(out.loss.detach().cpu())
                losses.append(lv)
                print(f"[train] step {step}/{max_steps} loss={lv:.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
                if save_every and step % save_every == 0 and step < max_steps:
                    ckpt = os.path.join(cfg["output_dir"], f"step_{step}")
                    model.save_pretrained(ckpt)
                    print(f"[train] checkpoint -> {ckpt}", flush=True)
                if step >= max_steps:
                    break

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)          # LoRA -> adapter；full -> 完整权重
    tok.save_pretrained(out_dir)
    print(f"[train] saved to {out_dir}")
    if losses:
        print(f"[train] loss first={losses[0]:.4f} last={losses[-1]:.4f} "
              f"finite={all(l == l for l in losses)}")
    print("train_ok")


if __name__ == "__main__":
    main()
