#!/usr/bin/env python3
"""生成式 eval：(1) 确认加载的是 Unlimited-OCR 的 R-SWA；(2) 合并 LoRA adapter 后用 model.infer
在 held-out 图上生成，和 GT 算相似度。

R-SWA 确认点：use_mla=False、decoder self_attn 类=SlidingWindowLlamaAttention、sliding_window=128；
infer 内部会 config._ring_window=128 + sliding_window=None 再 generate（推理时 ring-buffer 激活）。

用法：python scripts/eval_infer.py --adapter outputs/lora_decoder_olmocr_v1 \
        --jsonl data/samples_olmocr/train.jsonl --n 6 --gpu 0
"""
import argparse
import difflib
import os
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from uocr_train.model_loader import load_model
from uocr_train.dataset import UOCRJsonlDataset


def confirm_rswa(model, tag):
    cfg = model.config
    attn_cls = type(model.model.layers[1].self_attn).__name__
    sw = getattr(cfg, "sliding_window", None)
    sws = getattr(cfg, "sliding_window_size", None)
    print(f"[R-SWA:{tag}] use_mla={getattr(cfg, 'use_mla', None)} "
          f"self_attn(layer1)={attn_cls} sliding_window={sw} sliding_window_size={sws}")
    assert attn_cls == "SlidingWindowLlamaAttention", f"不是 R-SWA！实际 {attn_cls}"
    assert getattr(cfg, "use_mla", True) is False, "use_mla 不是 False"
    return attn_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录；空=只评 base")
    ap.add_argument("--jsonl", default="data/samples_olmocr/train.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--mode", choices=["gundam", "base"], default="gundam")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tok, model, remote = load_model(device="cuda")
    confirm_rswa(model, "base-load")

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()          # 把 LoRA 焊进 base，得回纯 UnlimitedOCRForCausalLM
        confirm_rswa(model, "after-merge")         # 合并不应改变注意力类
        print(f"[eval] merged adapter: {args.adapter}")
    model = model.eval().cuda()

    base_size, image_size, crop_mode = (1024, 640, True) if args.mode == "gundam" else (1024, 1024, False)
    out_tmp = str(pathlib.Path(args.adapter or "outputs").parent / "_infer_tmp")

    ds = UOCRJsonlDataset(args.jsonl)
    ratios = []
    for i in range(min(args.n, len(ds))):
        s = ds[i]
        gt = s["target"]
        gen = model.infer(
            tok, prompt="<image>document parsing.", image_file=s["images"][0], output_path=out_tmp,
            base_size=base_size, image_size=image_size, crop_mode=crop_mode, eval_mode=True,
            no_repeat_ngram_size=35, ngram_window=128, max_length=8192, temperature=0.0,
        )
        gen = gen or ""
        r = difflib.SequenceMatcher(None, gt, gen).ratio()
        ratios.append(r)
        print(f"\n[{i}] similarity={r:.3f}  gt_len={len(gt)} gen_len={len(gen)}")
        print("  GT :", gt[:120].replace("\n", " "))
        print("  GEN:", gen[:120].replace("\n", " "))

    if ratios:
        print(f"\n=== mean similarity over {len(ratios)} held-out: {sum(ratios) / len(ratios):.3f} ===")
    print("eval_infer_ok")


if __name__ == "__main__":
    main()
