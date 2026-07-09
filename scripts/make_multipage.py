#!/usr/bin/env python3
"""把单页 JSONL 配成多页 JSONL：相邻 N 页拼一个 multi_base 样本，target 用 <PAGE> 分隔。

仅用于打通/验证 multi_base + R-SWA 训练路径。注意：随机拼页不教跨页合并（见 design 0.5），
不是北极星能力数据；真正的多页能力数据来自数据 session 的天然连续文档。

输出与输入同目录（保持 image 相对路径可解析）。
用法：python scripts/make_multipage.py --in data/olmocr_train_v1/train.jsonl --pages 2
"""
import argparse
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from uocr_train.constants import PAGE_TOKEN, DEFAULT_MULTI_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="", help="默认与输入同目录、名后缀 _multi")
    ap.add_argument("--pages", type=int, default=2, help="每个样本拼几页")
    args = ap.parse_args()

    inp = pathlib.Path(args.inp)
    out = pathlib.Path(args.out) if args.out else inp.with_name(inp.stem + "_multi.jsonl")

    rows = [json.loads(line) for line in inp.read_text().splitlines() if line.strip()]

    samples = []
    for g in range(0, len(rows) - args.pages + 1, args.pages):
        group = rows[g:g + args.pages]
        samples.append(dict(
            id=f"multi_{len(samples):06d}",
            source=group[0].get("source", ""),
            images=[r["image"] for r in group],       # 相对路径，与输入同目录
            mode="multi_base",
            prompt=DEFAULT_MULTI_PROMPT,
            target=f"\n{PAGE_TOKEN}\n".join(r["target"] for r in group),
            target_format="natural_text_or_html",
            task="multi_page_parse",
            meta=dict(n_pages=args.pages, page_ids=[r["id"] for r in group]),
        ))

    out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    print(f"wrote {len(samples)} multi-page samples ({args.pages} pages each) -> {out}")


if __name__ == "__main__":
    main()
