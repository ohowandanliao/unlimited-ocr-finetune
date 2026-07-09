#!/usr/bin/env python3
"""验收：reload 保存的 LoRA adapter + 真实图 forward，确认 adapter 能回读并前向。

用法：python scripts/reload_check.py --adapter outputs/smoke_lora_attn --jsonl data/samples_olmocr/train.jsonl --gpu 0
"""
import argparse
import os
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from uocr_train.model_loader import load_model
from uocr_train.dataset import UOCRJsonlDataset
from uocr_train.processor import UOCRProcessor
from uocr_train.collator import UOCRCollator
from uocr_train.constants import DEFAULT_SINGLE_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--jsonl", default="data/samples_olmocr/train.jsonl")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from peft import PeftModel

    tok, base, remote = load_model(device="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print(f"[reload] adapter loaded from {args.adapter}")

    proc = UOCRProcessor(tok, remote)
    coll = UOCRCollator(tok)
    ds = UOCRJsonlDataset(args.jsonl)
    sample = dict(ds[0], mode="single_gundam", prompt=DEFAULT_SINGLE_PROMPT)
    batch = coll([proc.encode(sample)])

    b = dict(batch)
    for k in ("input_ids", "attention_mask", "labels", "images_seq_mask"):
        b[k] = batch[k].cuda()
    b["images"] = [(c.cuda(), o.cuda()) for c, o in batch["images"]]

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"],
                images=b["images"], images_seq_mask=b["images_seq_mask"],
                images_spatial_crop=b["images_spatial_crop"], use_cache=False, return_dict=True,
            )
    loss = float(out.loss.detach().cpu())
    print(f"[reload] forward ok: logits={tuple(out.logits.shape)} loss={loss:.4f} finite={loss == loss}")
    print("reload_ok")


if __name__ == "__main__":
    main()
