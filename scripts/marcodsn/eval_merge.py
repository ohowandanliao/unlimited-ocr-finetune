#!/usr/bin/env python3
"""多页合并生成 eval:合并训好的 adapter 后,用 infer_multi 在多页图上生成合并 markdown,和 GT 比。

区别于 eval_infer.py(单页 infer + "document parsing." prompt、只喂第 1 页):本脚本走
infer_multi(多页)+ 样本自带的 "Multi page merge." prompt + 全部页图;save_results=False 时
infer_multi 返回原始生成文本(不做 <PAGE> 切分),正是合并输出。R-SWA ring buffer 生成时激活。

用法:python scripts/marcodsn/eval_merge.py --adapter outputs/marcodsn_smoke \
        --jsonl data/marcodsn_v1/train.jsonl --n 3
"""
import argparse
import difflib
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from uocr_train.model_loader import load_model  # noqa: E402
from uocr_train.dataset import UOCRJsonlDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录;空=只评 base")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--image-size", type=int, default=1024)   # multi_base=1024
    ap.add_argument("--max-length", type=int, default=10000)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tok, model, remote = load_model(device="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()          # LoRA 焊进 base
        print(f"[eval] merged adapter: {args.adapter}")
    attn = type(model.model.layers[1].self_attn).__name__
    print(f"[eval] self_attn(layer1)={attn} use_mla={getattr(model.config,'use_mla',None)}")
    model = model.eval().cuda()

    ds = UOCRJsonlDataset(args.jsonl)
    out_tmp = "outputs/_infer_merge_tmp"
    pathlib.Path(out_tmp).mkdir(parents=True, exist_ok=True)
    ratios = []
    for i in range(min(args.n, len(ds))):
        s = ds[i]
        gt = s["target"]
        prompt = s.get("prompt", "<image>Multi page merge.")
        gen, ntok = model.infer_multi(
            tok, prompt=prompt, image_files=s["images"], output_path=out_tmp,
            image_size=args.image_size, max_length=args.max_length,
            no_repeat_ngram_size=35, ngram_window=128, temperature=0.0,
        )
        gen = gen or ""
        r = difflib.SequenceMatcher(None, gt, gen).ratio()
        ratios.append(r)
        print(f"\n[{i}] {s['id']} pages={len(s['images'])} prompt={prompt!r} "
              f"sim={r:.3f} gt_len={len(gt)} gen_len={len(gen)} gen_tokens={ntok}")
        print("  GT  head:", gt[:200].replace("\n", " "))
        print("  GEN head:", gen[:200].replace("\n", " "))
        print("  GEN tail:", gen[-160:].replace("\n", " "))

    if ratios:
        print(f"\n=== mean similarity over {len(ratios)}: {sum(ratios) / len(ratios):.3f} ===")
    print("eval_merge_ok")


if __name__ == "__main__":
    main()
