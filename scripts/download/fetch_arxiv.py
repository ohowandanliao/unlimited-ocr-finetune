#!/usr/bin/env python3
"""
fetch_arxiv.py — 批量从 arxiv 下论文的 PDF + e-print 源码包(在 Mac 上跑,走代理快)。

arxiv API 查类别最近 N 篇 -> 拿 id/title/authors -> curl 下 PDF + e-print -> 写 manifest.jsonl。
所有网络请求走 curl(复用系统代理:Mac 直连 arxiv ~222KB/s,服务器仅 ~18KB/s 会截断,故下载放 Mac)。
只用标准库(xml/json/subprocess),不需在 Mac 装任何东西。

arxiv API 给的 authors 是干净作者名,直接进 manifest,供 build_dataset --authors 注入 target
(比 pandoc --standalone 的 YAML author 块干净得多)。

产出(供 build_dataset.py 逐篇处理):
  {out}/manifest.jsonl   每行 {id, title, authors, cats, pdf, eprint}
  {out}/{id}.pdf
  {out}/{id}.eprint

用法:
  python scripts/download/fetch_arxiv.py --cats cs.CL cs.LG --n 40 --out data/arxiv_v1/raw
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
API = "http://export.arxiv.org/api/query"
UA = "unlimited-ocr-arxiv-fetch/1.0 (research; contact via repo)"


def curl(url, out_path=None, timeout=180):
    """走系统代理的 curl。out_path 给了就下到文件(返回 True/False),否则返回 stdout bytes 或 None。"""
    cmd = ["curl", "-sL", "-A", UA, "--max-time", str(timeout)]
    if out_path:
        cmd += ["-o", str(out_path)]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=(out_path is None))
    if out_path:
        return r.returncode == 0
    return r.stdout if r.returncode == 0 else None


def _head(path, n):
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def is_pdf(path):
    """真 PDF 以 %PDF 开头。arxiv 对没渲好的论文会返回 HTML 页(<!DOCTYPE...),据此拒。"""
    return _head(path, 5).startswith(b"%PDF")


def is_eprint(path):
    """arxiv e-print 是 gzip(常见)或 POSIX tar;HTML 占位页两者都不是。"""
    b = _head(path, 264)
    return b[:2] == b"\x1f\x8b" or b[257:262] == b"ustar"


def fetch_file(url, path, validate, tries=3, backoff=3.0):
    """下到 path 并按内容 validate();失败(空/HTML/被限流)线性退避重试,最终仍坏返回 False。
    arxiv 对连续快速请求会限流(返回空体),重试给它喘息;真不存在的则一直坏,几次后放弃。"""
    for t in range(tries):
        curl(url, path)
        if validate(path):
            return True
        time.sleep(backoff * (t + 1))
    return False


def query_meta(cats, n, start=0):
    """arxiv API 查类别最近 n 篇(从第 start 篇起),返回 [{id,title,authors,cats}]。
    start>0 用来避开最前沿刚提交、PDF/源码还没渲染好的那批。"""
    q = "+OR+".join(f"cat:{c}" for c in cats)
    url = (f"{API}?search_query={q}"
           f"&sortBy=submittedDate&sortOrder=descending&start={start}&max_results={n}")
    data = curl(url)
    if not data:
        sys.exit("[fetch] arxiv API 查询失败(检查网络/代理)")
    root = ET.fromstring(data)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        abs_url = e.findtext(f"{ATOM}id") or ""
        aid = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])   # 去版本号 v1/v2
        title = " ".join((e.findtext(f"{ATOM}title") or "").split())
        authors = [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")]
        cats_e = [c.get("term") for c in e.findall(f"{ATOM}category")]
        out.append(dict(id=aid, title=title,
                        authors=", ".join(filter(None, authors)), cats=cats_e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="+", default=["cs.CL", "cs.LG"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=3.0, help="每篇间隔秒(对 arxiv 礼貌,防限流)")
    ap.add_argument("--start", type=int, default=0,
                    help="API 结果偏移;>0 避开最新还没渲好的论文")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    metas = query_meta(a.cats, a.n, a.start)
    print(f"[fetch] API 命中 {len(metas)} 篇({'+'.join(a.cats)}, start={a.start})")

    ok = 0
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for i, m in enumerate(metas, 1):
            aid = m["id"]
            safe = aid.replace("/", "_")
            pdf, ep = out / f"{safe}.pdf", out / f"{safe}.eprint"
            pdf_ok = fetch_file(f"https://arxiv.org/pdf/{aid}", pdf, is_pdf)
            ep_ok = fetch_file(f"https://arxiv.org/e-print/{aid}", ep, is_eprint)
            good = pdf_ok and ep_ok                  # 按内容校验:HTML 占位页/半成品一律拒
            psz = pdf.stat().st_size if pdf.exists() else 0
            esz = ep.stat().st_size if ep.exists() else 0
            tag = "OK" if good else f"SKIP(pdf={'ok' if pdf_ok else 'bad'},eprint={'ok' if ep_ok else 'bad'})"
            print(f"[fetch] {i}/{len(metas)} {aid:16s} pdf={psz} eprint={esz} {tag}")
            if good:
                mf.write(json.dumps(dict(m, pdf=pdf.name, eprint=ep.name), ensure_ascii=False) + "\n")
                ok += 1
            else:                                     # 不完整就删掉半成品,别留脏文件
                pdf.unlink(missing_ok=True)
                ep.unlink(missing_ok=True)
            time.sleep(a.sleep)
    print(f"[fetch] 完成:{ok}/{len(metas)} 成功 -> {out}/manifest.jsonl")


if __name__ == "__main__":
    main()
