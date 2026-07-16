#!/usr/bin/env python3
"""
build_batch.py — 读 fetch_arxiv.py 产出的 raw/manifest.jsonl,逐篇调 build_dataset.py,
把整批 arxiv 论文组成一个 multi_base 数据集(一个 train.jsonl + images/)。

在服务器上跑(需 pandoc + pymupdf,见 build_dataset.py)。fetch 在 Mac、build 在服务器,
raw 目录 scp 过去后用本脚本批处理。

单篇失败(坏 PDF / 找不到 tex / pandoc 空)只跳过并计数,不中断整批——build_dataset.py
遇到这些情况 sys.exit 非零,这里捕获返回码继续下一篇。manifest 里的 title/authors
直接透传给 build_dataset(比 pandoc YAML 干净)。

用法(服务器):
  python build_batch.py --raw data/arxiv_v1/raw --out-dir data/arxiv_v1 [--dpi 144]
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="fetch_arxiv 输出目录(manifest.jsonl + *.pdf/*.eprint)")
    ap.add_argument("--out-dir", required=True, help="数据集输出目录(train.jsonl + images/)")
    ap.add_argument("--dpi", type=int, default=144)
    a = ap.parse_args()

    raw = pathlib.Path(a.raw)
    manifest = raw / "manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"[batch] 找不到 {manifest}(先在 Mac 跑 fetch_arxiv.py 再 scp 过来)")

    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        sys.exit(f"[batch] {manifest} 为空,没有可处理的论文")

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = out_dir / "train.jsonl"
    train.unlink(missing_ok=True)          # 全新一批:build_dataset 是 append,先清掉避免和上次叠加

    ok = 0
    for i, m in enumerate(rows, 1):
        aid = m["id"]
        cmd = [sys.executable, str(HERE / "build_dataset.py"),
               "--pdf", str(raw / m["pdf"]), "--eprint", str(raw / m["eprint"]),
               "--id", aid, "--out-dir", str(out_dir), "--dpi", str(a.dpi)]
        if m.get("title"):
            cmd += ["--title", m["title"]]
        if m.get("authors"):
            cmd += ["--authors", m["authors"]]
        if subprocess.run(cmd).returncode == 0:
            ok += 1
        else:
            print(f"[batch] {i}/{len(rows)} {aid}: build 失败,跳过")

    print(f"[batch] 完成:{ok}/{len(rows)} 篇成样 -> {train}")


if __name__ == "__main__":
    main()
