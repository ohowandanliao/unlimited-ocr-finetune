#!/usr/bin/env python3
"""
fetch_pdfs.py — 按 arxiv_id 列表下 PDF(只 PDF,不要 e-print;Path A 用现成 marcodsn markdown 当
target、不碰 LaTeX)。在 Mac 跑(arxiv 直连快),下完 scp 到服务器渲染页图。

读 --ids:纯文本(每行一个 id)或 jsonl(每行含 "arxiv_id" 字段)都行。按内容校验(%PDF)+
失败退避重试扛限流;已下过的跳过。

用法:
  python scripts/download/fetch_pdfs.py --ids selected_ids.txt --out data/marcodsn_v1/pdfs
"""
import argparse
import json
import pathlib
import subprocess
import time

UA = "unlimited-ocr-arxiv-fetch/1.0 (research; contact via repo)"


def curl(url, out_path, timeout=180):
    return subprocess.run(
        ["curl", "-sL", "-A", UA, "--max-time", str(timeout), "-o", str(out_path), url]
    ).returncode == 0


def is_pdf(path):
    try:
        with open(path, "rb") as f:
            return f.read(5).startswith(b"%PDF")
    except OSError:
        return False


def read_ids(path):
    ids = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ids.append(json.loads(line)["arxiv_id"] if line.startswith("{") else line)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="id 列表(纯文本每行一个 或 jsonl 含 arxiv_id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=3.0, help="每篇间隔秒(防 arxiv 限流)")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = read_ids(a.ids)
    print(f"[pdf] {len(ids)} ids -> {out}")

    ok = 0
    for i, aid in enumerate(ids, 1):
        pdf = out / f"{aid.replace('/', '_')}.pdf"
        if is_pdf(pdf):
            ok += 1
            print(f"[pdf] {i}/{len(ids)} {aid} cached")
            continue
        got = False
        for t in range(3):                      # 失败退避重试
            curl(f"https://arxiv.org/pdf/{aid}", pdf)
            if is_pdf(pdf):
                got = True
                break
            time.sleep(3.0 * (t + 1))
        if got:
            ok += 1
            print(f"[pdf] {i}/{len(ids)} {aid} OK {pdf.stat().st_size}")
        else:
            pdf.unlink(missing_ok=True)         # 不留脏文件
            print(f"[pdf] {i}/{len(ids)} {aid} FAIL")
        time.sleep(a.sleep)
    print(f"[pdf] done: {ok}/{len(ids)} good -> {out}")


if __name__ == "__main__":
    main()
