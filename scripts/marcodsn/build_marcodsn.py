#!/usr/bin/env python3
"""
build_marcodsn.py — Path A:用现成的 marcodsn/arxiv-markdown(文档级、跨页连续 markdown)当 target,
自己渲 arxiv PDF 每页图当输入,组多页合并样本(mode=multi_base, task=multi_page_merge)。
不碰 LaTeX/pandoc——target 是别人 docling 好的整篇 markdown,只剥掉图链。

两阶段(适配 Mac 下 PDF / 服务器渲染的分工):
  --phase select:读 parquet,挑 N 篇(markdown 长度在区间内)-> 产出
      {out}/selected.jsonl   每行 {arxiv_id, target}(target=剥图链的 markdown)
      {out}/selected_ids.txt  供 Mac 端 fetch_pdfs.py
  --phase build:读 selected.jsonl + PDF 目录,逐篇渲页图组样本 -> {out}/train.jsonl

用法(服务器):
  python build_marcodsn.py --phase select --parquet .../train.parquet --out data/marcodsn_v1 --n 50
  # Mac: python scripts/download/fetch_pdfs.py --ids selected_ids.txt --out pdfs ; scp 到 {out}/pdfs
  python build_marcodsn.py --phase build --out data/marcodsn_v1 --pdf-dir data/marcodsn_v1/pdfs --dpi 144
"""
import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from uocr_train.constants import DEFAULT_MERGE_PROMPT  # noqa: E402

IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")   # docling 的 ![Image](R2 url) 图链


def clean_target(md):
    md = IMG_RE.sub("", md)                    # 删图链:OCR target 不含图 URL(图内容不识别)
    md = re.sub(r"[ \t]+\n", "\n", md)         # 行尾空白
    md = re.sub(r"\n{3,}", "\n\n", md)         # 多空行
    return md.strip() + "\n"


def phase_select(a):
    import pyarrow.parquet as pq
    t = pq.read_table(a.parquet, columns=["arxiv_id", "markdown"])
    ids = t.column("arxiv_id").to_pylist()
    mds = t.column("markdown").to_pylist()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    picked = 0
    with open(out / "selected.jsonl", "w", encoding="utf-8") as sf, \
         open(out / "selected_ids.txt", "w") as idf:
        for aid, md in zip(ids, mds):
            if not aid or not md:
                continue
            tgt = clean_target(md)
            if not (a.min_chars <= len(tgt) <= a.max_chars):
                continue
            sf.write(json.dumps({"arxiv_id": aid, "target": tgt}, ensure_ascii=False) + "\n")
            idf.write(aid + "\n")
            picked += 1
            if picked >= a.n:
                break
    print(f"[select] 选 {picked} 篇(char∈[{a.min_chars},{a.max_chars}])-> "
          f"{out}/selected.jsonl (+ selected_ids.txt)")


def render_pages(pdf_path, out_img_dir, arxiv_id, dpi):
    import fitz
    out_img_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError("PDF 0 页")
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    safe = arxiv_id.replace("/", "_")
    rels = []
    for i in range(doc.page_count):
        pix = doc.load_page(i).get_pixmap(matrix=mat)
        name = f"{safe}_p{i:03d}.png"
        pix.save(str(out_img_dir / name))
        rels.append(f"images/{name}")
    return rels


def phase_build(a):
    out = pathlib.Path(a.out)
    pdf_dir = pathlib.Path(a.pdf_dir)
    rows = [json.loads(l) for l in open(out / "selected.jsonl", encoding="utf-8") if l.strip()]
    train = out / "train.jsonl"
    train.unlink(missing_ok=True)
    ok = 0
    with open(train, "w", encoding="utf-8") as tf:
        for i, r in enumerate(rows, 1):
            aid = r["arxiv_id"]
            if a.max_target_chars and len(r["target"]) > a.max_target_chars:
                print(f"[build] {i}/{len(rows)} {aid}: target {len(r['target'])} chars > "
                      f"{a.max_target_chars},跳过(控长以适配上下文/加速 smoke)")
                continue
            pdf = pdf_dir / f"{aid.replace('/', '_')}.pdf"
            if not pdf.exists():
                print(f"[build] {i}/{len(rows)} {aid}: 无 PDF,跳过")
                continue
            try:
                images = render_pages(pdf, out / "images", aid, a.dpi)
            except Exception as e:
                print(f"[build] {i}/{len(rows)} {aid}: 渲染失败({e}),跳过")
                continue
            sample = dict(
                id=aid, source="marcodsn/arxiv-markdown", images=images,
                mode="multi_base", prompt=DEFAULT_MERGE_PROMPT, target=r["target"],
                target_format="markdown", task="multi_page_merge",
                meta=dict(n_pages=len(images), arxiv_id=aid),
            )
            tf.write(json.dumps(sample, ensure_ascii=False) + "\n")
            ok += 1
            print(f"[build] {i}/{len(rows)} {aid}: {len(images)} 页, target {len(r['target'])} chars")
    print(f"[build] 完成:{ok}/{len(rows)} 篇成样 -> {train}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["select", "build"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet", help="select 阶段:marcodsn parquet 路径")
    ap.add_argument("--n", type=int, default=50, help="select 挑几篇")
    ap.add_argument("--min-chars", type=int, default=3000)
    ap.add_argument("--max-chars", type=int, default=120000)
    ap.add_argument("--pdf-dir", help="build 阶段:PDF 目录")
    ap.add_argument("--dpi", type=int, default=144)
    ap.add_argument("--max-target-chars", type=int, default=25000,
                    help="build 阶段:target 超此长度跳过(控上下文/加速 smoke;0=不限)")
    a = ap.parse_args()
    if a.phase == "select":
        if not a.parquet:
            sys.exit("select 需要 --parquet")
        phase_select(a)
    else:
        if not a.pdf_dir:
            sys.exit("build 需要 --pdf-dir")
        phase_build(a)


if __name__ == "__main__":
    main()
