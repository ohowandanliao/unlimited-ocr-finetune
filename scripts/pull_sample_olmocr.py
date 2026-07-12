#!/usr/bin/env python3
"""抽样 olmOCR-mix-1025 的一个小 eval 块 -> (page PNG, natural_text) 对，落成统一 JSONL。
只为看真实数据形状 + 驱动训练工程适配，不用于训练。"""
import os, io, json, tarfile, statistics
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import fitz  # pymupdf

REPO = "allenai/olmOCR-mix-1025"
CHUNK = "00_documents_eval_00000.tar.gz"
PARQUET = "00_documents_eval.parquet"
N = 30                      # 抽样条数
RENDER_LONG_SIDE = 1600     # 渲染 PDF 首页长边像素

# 输出目录：默认 ./data/samples_olmocr，可用环境变量 OLMOCR_SAMPLE_OUT 覆盖
OUT = Path(os.environ.get("OLMOCR_SAMPLE_OUT", "data/samples_olmocr"))
RAW = OUT / "_raw"
IMG = OUT / "images"
for d in (RAW, IMG):
    d.mkdir(parents=True, exist_ok=True)

print(f"[1/4] download parquet + tarball ({CHUNK}, ~191MB) via hf-mirror ...")
pq_path = hf_hub_download(REPO, PARQUET, repo_type="dataset", local_dir=str(RAW))
tar_path = hf_hub_download(REPO, f"pdf_tarballs/{CHUNK}", repo_type="dataset", local_dir=str(RAW))

print("[2/4] read parquet, index rows belonging to this chunk ...")
tbl = pq.read_table(pq_path)
cols = tbl.column_names
print("    parquet columns:", cols)
rel = tbl.column("pdf_relpath").to_pylist() if "pdf_relpath" in cols else [None] * tbl.num_rows
nat = tbl.column("natural_text").to_pylist() if "natural_text" in cols else [None] * tbl.num_rows
is_table = tbl.column("is_table").to_pylist() if "is_table" in cols else [None] * tbl.num_rows
is_diagram = tbl.column("is_diagram").to_pylist() if "is_diagram" in cols else [None] * tbl.num_rows
lang = tbl.column("primary_language").to_pylist() if "primary_language" in cols else [None] * tbl.num_rows

# arcname(in tar) -> row meta
want = {}
for i, r in enumerate(rel):
    if not r or CHUNK not in r or not nat[i] or not str(nat[i]).strip():
        continue
    arc = r.split(":", 1)[1] if ":" in r else r
    want[arc] = dict(text=str(nat[i]), is_table=is_table[i], is_diagram=is_diagram[i], lang=lang[i])
print(f"    rows in this chunk with non-empty natural_text: {len(want)}")

print("[3/4] extract PDFs from tar, render page-0 -> PNG ...")
samples = []
with tarfile.open(tar_path, "r:gz") as tf:
    # map possible member names (handle leading ./) to canonical arcname
    for m in tf.getmembers():
        if len(samples) >= N:
            break
        if not m.isfile() or not m.name.lower().endswith(".pdf"):
            continue
        name = m.name[2:] if m.name.startswith("./") else m.name
        meta = want.get(name) or want.get(m.name)
        if meta is None:
            continue
        try:
            data = tf.extractfile(m).read()
            doc = fitz.open(stream=data, filetype="pdf")
            page = doc.load_page(0)
            rect = page.rect
            zoom = RENDER_LONG_SIDE / max(rect.width, rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            sid = f"olmocr_{len(samples):04d}"
            png_rel = f"images/{sid}.png"
            pix.save(str(OUT / png_rel))
            doc.close()
        except Exception as e:
            print("    skip", name, "->", repr(e)[:80])
            continue
        samples.append(dict(
            id=sid, source=REPO, image=png_rel, mode="single_gundam",
            prompt="<image>document parsing.", target=meta["text"],
            target_format="natural_text_or_html", task="full_page_parse",
            meta=dict(chunk=CHUNK, arcname=name, img_w=pix.width, img_h=pix.height,
                      char_len=len(meta["text"]), is_table=meta["is_table"],
                      is_diagram=meta["is_diagram"], language=meta["lang"], license="odc-by"),
        ))

print(f"[4/4] write JSONL ({len(samples)} samples) + stats ...")
jsonl = OUT / "train.jsonl"
with open(jsonl, "w") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

if samples:
    clens = [s["meta"]["char_len"] for s in samples]
    ws = [s["meta"]["img_w"] for s in samples]
    hs = [s["meta"]["img_h"] for s in samples]
    print("=== SAMPLE STATS ===")
    print(f"n={len(samples)}  jsonl={jsonl}")
    print(f"target char_len: min={min(clens)} med={int(statistics.median(clens))} max={max(clens)}")
    print(f"img size: w[{min(ws)}..{max(ws)}] h[{min(hs)}..{max(hs)}]")
    print(f"has_table={sum(1 for s in samples if s['meta']['is_table'])}  has_diagram={sum(1 for s in samples if s['meta']['is_diagram'])}")
    print("--- first sample target (first 400 chars) ---")
    print(samples[0]["target"][:400])
print("olmocr_sample_ok")
