#!/usr/bin/env python3
"""训练前 sanity：用真实 olmOCR 抽样，验证 single_gundam / single_base / multi_base 的
forward -> loss。多页是最终目标，重点看 multi_base 两页拼接能否跑通（masked_scatter 不报错、loss 有限）。

用法：python scripts/eval_forward.py --gpu 0 --jsonl data/samples_olmocr/train.jsonl
"""
import argparse
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from uocr_train.model_loader import load_model
from uocr_train.dataset import UOCRJsonlDataset
from uocr_train.processor import UOCRProcessor
from uocr_train.collator import UOCRCollator
from uocr_train.constants import DEFAULT_SINGLE_PROMPT, DEFAULT_MULTI_PROMPT, PAGE_TOKEN


def sanity(batch):
    ii, lb, sm = batch["input_ids"], batch["labels"], batch["images_seq_mask"]
    assert ii.shape == lb.shape == sm.shape, (ii.shape, lb.shape, sm.shape)
    assert len(batch["images"]) == ii.shape[0]
    assert len(batch["images_spatial_crop"]) == ii.shape[0]
    assert (lb != -100).any(), "no supervised token"


def to_cuda(batch):
    out = dict(batch)
    out["input_ids"] = batch["input_ids"].cuda()
    out["attention_mask"] = batch["attention_mask"].cuda()
    out["labels"] = batch["labels"].cuda()
    out["images_seq_mask"] = batch["images_seq_mask"].cuda()
    out["images"] = [(c.cuda(), o.cuda()) for c, o in batch["images"]]
    return out


def run_case(name, samples, proc, coll, model):
    print(f"\n===== {name} =====")
    enc = [proc.encode(s) for s in samples]
    for e in enc:
        print(f"  [{e['id']}] mode={e['mode']} ids={tuple(e['input_ids'].shape)} "
              f"img_tokens={e['image_tokens']} prompt_len={e['prompt_len']} "
              f"target_tokens={int((e['labels'] != -100).sum())}")
    batch = coll(enc)
    sanity(batch)
    print(f"  batch: input_ids={tuple(batch['input_ids'].shape)} "
          f"attn_sum={batch['attention_mask'].sum(1).tolist()} "
          f"seq_mask_sum={batch['images_seq_mask'].sum(1).tolist()} "
          f"spatial={batch['images_spatial_crop']}")
    print(f"  image tensor shapes: {[(tuple(c.shape), tuple(o.shape)) for c, o in batch['images']]}")
    b = to_cuda(batch)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"],
                images=b["images"], images_seq_mask=b["images_seq_mask"],
                images_spatial_crop=b["images_spatial_crop"], use_cache=False, return_dict=True,
            )
    loss = float(out.loss.detach().cpu()) if out.loss is not None else None
    print(f"  logits={tuple(out.logits.shape)} loss={loss} finite={loss is not None and loss == loss}")
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--jsonl", default="data/samples_olmocr/train.jsonl")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tok, model, remote = load_model(device="cuda")
    model.eval()
    proc = UOCRProcessor(tok, remote)
    coll = UOCRCollator(tok)

    ds = UOCRJsonlDataset(args.jsonl)
    print(f"dataset: {len(ds)} rows from {args.jsonl}")
    rows = [ds[i] for i in range(min(len(ds), max(args.n, 2)))]

    # single_gundam x2（真实两页各自单独）
    sg = [dict(r, mode="single_gundam", prompt=DEFAULT_SINGLE_PROMPT) for r in rows[:2]]
    run_case("single_gundam x2", sg, proc, coll, model)

    # single_base x1
    sb = [dict(rows[0], mode="single_base", prompt=DEFAULT_SINGLE_PROMPT)]
    run_case("single_base x1", sb, proc, coll, model)

    # multi_base：两页拼一个样本，target 用 <PAGE> 分隔（对齐 infer_multi 切页）
    multi = dict(
        id="multi_demo_2p", mode="multi_base",
        images=[rows[0]["images"][0], rows[1]["images"][0]],
        prompt=DEFAULT_MULTI_PROMPT,
        target=f"{rows[0]['target']}\n{PAGE_TOKEN}\n{rows[1]['target']}",
        meta={},
    )
    run_case("multi_base 2pages (ULTIMATE GOAL path)", [multi], proc, coll, model)

    print("\neval_forward_ok")


if __name__ == "__main__":
    main()
