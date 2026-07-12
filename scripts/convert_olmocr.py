#!/usr/bin/env python3
"""把本地 olmOCR-mix-1025（parquet + pdf_tarballs）转成训练用 JSONL + 渲染 PNG。

数据需先下到本地（见 scripts/download_olmocr.py），路径用环境变量 OLMOCR_DIR 指定
（或 --data-root 覆盖；默认 ./olmOCR-mix-1025）。逐 tarball 流式抽取，只渲染需要的页，
避免解压整块到磁盘。

用法示例：
  # 从 00_documents train 的第 0 块转最多 800 页
  python scripts/convert_olmocr.py --subset 00_documents --split train \
      --chunks 00_documents_train_00000 --max-samples 800 --out data/olmocr_train_v1
  # 不指定 chunks 则按 parquet 顺序跨块直到取够 max-samples
"""
import argparse
import json
import os
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

import fitz  # pymupdf
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("OLMOCR_DIR", "./olmOCR-mix-1025"))  # 或用 --data-root 覆盖
REPO = "allenai/olmOCR-mix-1025"
_CHUNK_RE = re.compile(r"pdf_tarballs/([^:]+\.tar\.gz):(.+)$")


def parse_relpath(relpath: str):
    """'pdf_tarballs/CHUNK.tar.gz:arcname' -> (chunk, arcname)。"""
    m = _CHUNK_RE.search(relpath or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def load_rows(parquet_path: Path):
    cols = ["pdf_relpath", "natural_text", "page_number", "primary_language",
            "is_table", "is_diagram", "is_rotation_valid", "id"]
    t = pq.read_table(parquet_path, columns=cols)
    d = {c: t.column(c).to_pylist() for c in cols}
    rows = []
    for i in range(t.num_rows):
        rows.append({c: d[c][i] for c in cols})
    return rows


def render_page(pdf_bytes: bytes, page_number, long_side: int):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pno = 0 if doc.page_count <= 1 else min(max((page_number or 1) - 1, 0), doc.page_count - 1)
    page = doc.load_page(pno)
    rect = page.rect
    zoom = long_side / max(rect.width, rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    doc.close()
    return pix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--subset", default="00_documents",
                    choices=["00_documents", "01_books", "02_loc_transcripts", "03_national_archives"])
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--chunks", default="", help="逗号分隔的 tarball chunk 名(不含.tar.gz)；空=按 parquet 顺序跨块")
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--render-long-side", type=int, default=1600)
    ap.add_argument("--lang", default="", help="只保留该 primary_language（如 en）；空=不过滤")
    ap.add_argument("--skip-bad-rotation", action="store_true", help="跳过 is_rotation_valid=False")
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = root / f"{args.subset}_{args.split}.parquet"
    print(f"[1/3] read {parquet_path.name} ...")
    rows = load_rows(parquet_path)

    # 按 chunk 分组需要的 arcname -> meta
    want_chunks = [c.strip() for c in args.chunks.split(",") if c.strip()] or None
    by_chunk = defaultdict(dict)      # chunk -> {arcname: meta}
    chunk_order = []                   # 保持 parquet 出现顺序
    for r in rows:
        txt = r["natural_text"]
        if not txt or not str(txt).strip():
            continue
        if args.lang and r["primary_language"] != args.lang:
            continue
        if args.skip_bad_rotation and r["is_rotation_valid"] is False:
            continue
        chunk, arc = parse_relpath(r["pdf_relpath"])
        if not chunk:
            continue
        chunk_base = chunk[:-len(".tar.gz")]
        if want_chunks is not None and chunk_base not in want_chunks:
            continue
        if chunk not in by_chunk:
            chunk_order.append(chunk)
        by_chunk[chunk][arc] = r
    print(f"    candidate rows: {sum(len(v) for v in by_chunk.values())} across {len(chunk_order)} chunks")

    print(f"[2/3] extract + render (max {args.max_samples}) ...")
    samples = []
    for chunk in chunk_order:
        if len(samples) >= args.max_samples:
            break
        tar_path = root / "pdf_tarballs" / chunk
        if not tar_path.exists():
            print(f"    [skip] tarball 不存在: {tar_path}")
            continue
        want = by_chunk[chunk]
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                if len(samples) >= args.max_samples:
                    break
                if not m.isfile():
                    continue
                name = m.name[2:] if m.name.startswith("./") else m.name
                meta = want.get(name) or want.get(m.name)
                if meta is None:
                    continue
                try:
                    pix = render_page(tf.extractfile(m).read(), meta["page_number"], args.render_long_side)
                    sid = f"olmocr_{len(samples):06d}"
                    pix.save(str(img_dir / f"{sid}.png"))
                except Exception as e:
                    print(f"    [skip] {name}: {repr(e)[:70]}")
                    continue
                samples.append(dict(
                    id=sid, source=REPO, image=f"images/{sid}.png", mode="single_gundam",
                    prompt="<image>document parsing.", target=str(meta["natural_text"]),
                    target_format="natural_text_or_html", task="full_page_parse",
                    meta=dict(chunk=chunk, arcname=name, olmocr_id=meta["id"],
                              img_w=pix.width, img_h=pix.height, char_len=len(str(meta["natural_text"])),
                              language=meta["primary_language"], is_table=meta["is_table"],
                              is_diagram=meta["is_diagram"], license="odc-by"),
                ))
        print(f"    {chunk}: total samples so far = {len(samples)}")

    print(f"[3/3] write JSONL ({len(samples)}) ...")
    with open(out / "train.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if samples:
        clens = [s["meta"]["char_len"] for s in samples]
        print("=== STATS ===")
        print(f"n={len(samples)}  out={out/'train.jsonl'}")
        print(f"char_len: min={min(clens)} med={int(statistics.median(clens))} max={max(clens)}")
        print(f"has_table={sum(1 for s in samples if s['meta']['is_table'])}")
    print("convert_olmocr_ok")


if __name__ == "__main__":
    main()
