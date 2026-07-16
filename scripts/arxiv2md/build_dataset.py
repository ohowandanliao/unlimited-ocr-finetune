#!/usr/bin/env python3
"""
build_dataset.py — 把一篇 arxiv 论文(已下载的 PDF + e-print 源码包)做成一条 multi_base 训练样本。

链路:e-print 解压 -> 找主 tex -> pandoc(-t markdown --wrap=none) -> clean_md 清洗
      -> PDF 用 fitz 渲染每页 PNG -> 组 multi_base 样本。

北极星:多页 PDF -> 一份统一合并 markdown。故 target 是整篇连续 clean.md、**不加 <PAGE> 分隔**
(区别于 make_multipage 的随机拼页 multi_page_parse)。图内容不进 target,只保留图注(见 clean_md.py)。

用法:
  python build_dataset.py --pdf att.pdf --eprint att.eprint --id 1706.03762 \
      --title "Attention Is All You Need" --out-dir data/arxiv_v1 [--dpi 144]
输出:
  {out-dir}/images/{id}_p{NNN}.png   每页渲染图
  {out-dir}/train.jsonl              追加一行样本(target = 整篇连续 markdown)
"""
import argparse
import gzip
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from uocr_train.constants import DEFAULT_MERGE_PROMPT  # noqa: E402

LUA_FILTER = pathlib.Path(__file__).resolve().parent / "clean.lua"


def extract_eprint(eprint_path, dest):
    """arxiv e-print 可能是 gzip tar、纯 tar、或单个 gzip 文件(单 .tex)。解压到 dest。"""
    try:
        with tarfile.open(eprint_path, "r:*") as tf:
            tf.extractall(dest)
        return
    except tarfile.ReadError:
        pass
    with gzip.open(eprint_path, "rb") as f:                 # 单 gzip 文件
        (pathlib.Path(dest) / "main.tex").write_bytes(f.read())


def find_main_tex(srcdir):
    r"""找含 \documentclass 的 .tex;多个取含 \begin{document} 的,再退化到最大文件。"""
    texs = list(pathlib.Path(srcdir).rglob("*.tex"))
    if not texs:
        return None
    with_doc = [t for t in texs if re.search(r"\\documentclass", t.read_text(errors="ignore"))]
    for t in with_doc:
        if re.search(r"\\begin\{document\}", t.read_text(errors="ignore")):
            return t
    return with_doc[0] if with_doc else max(texs, key=lambda t: t.stat().st_size)


def pandoc_to_md(main_tex, pandoc_bin, lua_filter):
    r"""在源码目录里跑 pandoc(--input relative 可解析),经 clean.lua 在 AST 上清洗。
    禁掉会输出属性语法的 writer 扩展;Math/Table/Code 节点原样保留。返回 (markdown, returncode)。"""
    r = subprocess.run(
        [pandoc_bin, main_tex.name, "-f", "latex",
         "-t", "markdown-header_attributes-link_attributes-fenced_divs-bracketed_spans",
         "--wrap=none", "--standalone", "--lua-filter", str(lua_filter)],
        cwd=str(main_tex.parent), capture_output=True, text=True,
    )
    return r.stdout, r.returncode


def render_pages(pdf_path, out_img_dir, arxiv_id, dpi):
    import fitz
    out_img_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        sys.exit(f"[build] {arxiv_id}: PDF 打开得 0 页(下载不完整/损坏?): {pdf_path}")
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    safe_id = arxiv_id.replace("/", "_")
    rels = []
    for i in range(doc.page_count):
        pix = doc.load_page(i).get_pixmap(matrix=mat)
        name = f"{safe_id}_p{i:03d}.png"
        pix.save(str(out_img_dir / name))
        rels.append(f"images/{name}")
    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--eprint", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--authors", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dpi", type=int, default=144)
    ap.add_argument("--pandoc", default=shutil.which("pandoc") or "/root/miniconda3/bin/pandoc")
    ap.add_argument("--max-residue", type=float, default=0.5,
                    help="清洗后每 1k 字符源码态残留标记上限,超过则丢弃该样本(0=关闭过滤)")
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) e-print -> LaTeX -> pandoc -> clean markdown
    with tempfile.TemporaryDirectory() as td:
        extract_eprint(a.eprint, td)
        main_tex = find_main_tex(td)
        if main_tex is None:
            sys.exit(f"[build] {a.id}: 找不到 .tex,跳过")
        body, rc = pandoc_to_md(main_tex, a.pandoc, LUA_FILTER)
        if rc != 0:
            sys.exit(f"[build] {a.id}: pandoc 返回码 {rc}(主 tex={main_tex.name}),跳过")
        if not body.strip():
            sys.exit(f"[build] {a.id}: pandoc 输出为空(主 tex={main_tex.name}),跳过")

    # clean.lua 已在 AST 上处理 abstract+body;标题用 manifest 的干净标题(比 pandoc \title 干净)
    head = f"# {a.title}\n\n" if a.title else ""
    if a.authors:
        head += f"{a.authors}\n\n"
    md = head + body

    # 质量闸(backstop):AST 清洗后残留应≈0;度量只认真残留、不数 math 下标(旧尺子把 $x_{n=1}$
    # 误当残留)。超阈值仍丢弃并 log,不静默留脏样本。
    resid = len(re.findall(r'\[@|\{#|\{\.[a-zA-Z]|\{[a-zA-Z][\w-]*="|:::|<(?:span|div|figure)|\[\\\[|\]\(#', md))
    density = resid / max(len(md) / 1000, 1)
    if a.max_residue and density > a.max_residue:
        sys.exit(f"[build] {a.id}: 残留密度过高({resid} marks / {len(md)} chars = "
                 f"{density:.2f}/1k > {a.max_residue}),跳过")

    # 2) PDF -> 每页 PNG
    images = render_pages(a.pdf, out_dir / "images", a.id, a.dpi)

    # 3) 组 multi_base 样本(target = 整篇连续 markdown,无 <PAGE>)
    sample = dict(
        id=a.id,
        source="arxiv",
        images=images,
        mode="multi_base",
        prompt=DEFAULT_MERGE_PROMPT,              # 跨页合并专用 prompt(区别于原生逐页 parsing)
        target=md,
        target_format="markdown",
        task="multi_page_merge",                 # 北极星:跨页合并(区别拼页 multi_page_parse)
        meta=dict(n_pages=len(images), arxiv_id=a.id),
    )
    with open(out_dir / "train.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[build] {a.id}: {len(images)} 页, target {len(md)} chars -> {out_dir}/train.jsonl")


if __name__ == "__main__":
    main()
