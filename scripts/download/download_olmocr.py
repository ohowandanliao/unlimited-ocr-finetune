#!/usr/bin/env python3
"""下载 olmOCR-mix-1025 指定 subset/split 的 parquet + 若干 pdf_tarballs 到本地，
供 convert_olmocr.py 使用（convert 本身不下载）。默认走 hf-mirror（国内直连 HF 被墙）。

用法：
  # 冒烟：下 00_documents/train 第 0 块（parquet + 1 个 tarball）
  python scripts/download/download_olmocr.py --subset 00_documents --split train --chunks 00000
  # 多块：
  python scripts/download/download_olmocr.py --subset 00_documents --split train --chunks 00000,00001
输出到 OLMOCR_DIR（默认 ./olmOCR-mix-1025），与 convert_olmocr.py 的 --data-root 一致。
"""
import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "allenai/olmOCR-mix-1025"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="00_documents",
                    choices=["00_documents", "01_books", "02_loc_transcripts", "03_national_archives"])
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--chunks", default="00000",
                    help="逗号分隔的块号，如 00000,00001 -> pdf_tarballs/{subset}_{split}_{块号}.tar.gz")
    ap.add_argument("--out", default=os.environ.get("OLMOCR_DIR", "olmOCR-mix-1025"),
                    help="本地数据根目录（= convert_olmocr.py 的 OLMOCR_DIR / --data-root）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[endpoint] {os.environ['HF_ENDPOINT']}  ->  {out}")

    parquet = f"{args.subset}_{args.split}.parquet"
    print(f"[1] parquet {parquet} ...")
    hf_hub_download(REPO, parquet, repo_type="dataset", local_dir=str(out))

    convert_chunk_names = []
    for c in [c.strip() for c in args.chunks.split(",") if c.strip()]:
        base = f"{args.subset}_{args.split}_{c}"
        tar = f"pdf_tarballs/{base}.tar.gz"
        print(f"[2] tarball {tar} ...")
        hf_hub_download(REPO, tar, repo_type="dataset", local_dir=str(out))
        convert_chunk_names.append(base)

    print("download_olmocr_ok  ->", out)
    print("下一步给 convert_olmocr.py 用：--chunks " + ",".join(convert_chunk_names))


if __name__ == "__main__":
    main()
