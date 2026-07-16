#!/usr/bin/env python3
"""
clean_md.py — 把 pandoc(-t markdown --wrap=none --standalone)洗出的 arxiv markdown
清成适合当 OCR 训练 target 的“渲染态近似”markdown。

--standalone 让 pandoc 把 title/author/abstract 输出到开头的 YAML front matter
（否则 fragment 模式会把它们当 metadata 丢掉,正文缺 title 和 abstract）。
本脚本:分离 YAML -> 取 title/abstract -> 把 abstract 并入正文一起清洗 -> 前置 # title。
author 默认跳过(pandoc YAML 里作者块含大量 footnote/email/机构标记,太脏),需要时用 --authors 手动给。

清掉的源码态残留:
  - 交叉引用  Figure [2](#fig:x){reference-type=...}          -> Figure 2
  - citation  [@key] / [@k1; @k2]                             -> 删(无 bbl 无法渲成 [1])
  - pandoc 属性 {#id} {.class} {key=val} {reference-type=..}  -> 删
  - fenced div ::: center ... :::                            -> 删标记留内容
  - 图片  ![cap](path) / <img> / <embed> / <figure>          -> 删图与源路径,保留 caption 文字
  - 残留 HTML 行内标签 <span> <div> <p> ...                   -> 删标签留内容

公式安全:先把 $$...$$ / $...$ 抽成占位符,清洗后再放回,绝不破坏公式内的 {} < > 。
（图内容本身不进 target:OCR 只识别图注文字,不做 chart-to-text。）

用法:
  python clean_md.py --in raw.md --out clean.md [--title "..."] [--authors "..."]
"""
import argparse
import re

import yaml


def split_yaml(text):
    """分离 pandoc --standalone 输出开头的 YAML front matter,返回 (yaml_str, body)。"""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[4:end], text[end + 5:]
    return "", text


def protect_math(text):
    """把 $$...$$(display)和 $...$(inline)抽成占位符,后续清洗不误伤公式内 {} < > []。
    用左到右逐字符扫描而非全局正则配对:遇 $$ 找下一个 $$、遇 $ 找同行下一个 $、落单 $ 原样留。
    这样相邻 inline($a$$b$)会被逐个当 inline 消费,绝不形成假 $$ 去翻转 $$ 配对、把正文当公式吞掉。"""
    store, out, i, n = [], [], 0, len(text)
    while i < n:
        if text[i] == "$":
            if i + 1 < n and text[i + 1] == "$":            # display $$...$$(可跨行)
                j = text.find("$$", i + 2)
                # 真 display 公式有界;闭合 $$ 在 2000 字符外 = pandoc 吐的错配 $$,
                # 不当公式(否则会把大段正文吞进假公式、逃过清洗),当落单 $ 处理。
                if j != -1 and j - i <= 2000:
                    store.append(text[i:j + 2])
                    out.append(f"\x00M{len(store) - 1}\x00")
                    i = j + 2
                    continue
            else:                                            # inline $...$(同行、内部无 $)
                j = i + 1
                while j < n and text[j] not in "$\n":
                    j += 1
                if j < n and text[j] == "$" and j > i + 1:
                    store.append(text[i:j + 1])
                    out.append(f"\x00M{len(store) - 1}\x00")
                    i = j + 1
                    continue
        out.append(text[i])                                  # 落单的 $ 或普通字符,原样留下
        i += 1
    return "".join(out), store


def restore_math(text, store):
    return re.sub(r"\x00M(\d+)\x00", lambda m: store[int(m.group(1))], text)


def _caption_para(cap):
    cap = (cap or "").strip()
    return f"\n\n{cap}\n\n" if cap else "\n\n"


def _scrub(text):
    """核心清洗:公式保护下,清掉图/交叉引用/citation/属性/div/HTML 标签。"""
    text = text.replace("\xa0", " ")               # pandoc 把 LaTeX ~ 输出成 nbsp,归一成普通空格
    text, math = protect_math(text)

    # figure:逐标签处理,兼容嵌套 subfigure——先提 figcaption,再删 figure 壳
    text = re.sub(r"<figcaption>(.*?)</figcaption>",
                  lambda m: _caption_para(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r"</?figure[^>]*>", "", text)
    # markdown 图 ![cap](path){attr} -> 保留 cap,删图与路径
    text = re.sub(r"!\[(.*?)\]\([^)]*\)(\{[^}]*\})?",
                  lambda m: _caption_para(m.group(1)), text)
    text = re.sub(r"<img[^>]*/?>", "", text)
    text = re.sub(r"<embed[^>]*/?>", "", text)
    # pandoc \eqref: [\[label\]](#anchor)——方括号被转义,下面通用 xref 抓不到;PDF 上显示
    # 的是编号(pandoc 无法还原)-> v1「剪干净」直接删整个引用链接(含尾随 {attr})。
    text = re.sub(r"\[\\\[[^\]]*?\\\]\]\(#[^)]*\)(\{[^}]*\})?", "", text)
    # 交叉引用去链接 [TEXT](#anchor){attr} -> TEXT
    text = re.sub(r"\[([^\]]+)\]\(#[^)]*\)(\{[^}]*\})?", r"\1", text)
    # citation [@key] / [@k1; @k2] -> 删(PDF 上是编号,pandoc 不编译还原不了)
    text = re.sub(r"\[@[^\]]*\]", "", text)
    # fenced div ::: -> 删整行
    text = re.sub(r"^:::+.*$", "", text, flags=re.MULTILINE)
    # pandoc 属性 -> 删(公式已保护,不误伤公式内 {})
    text = re.sub(r"\{#[^}]*\}", "", text)
    text = re.sub(r"\{\.[^}]*\}", "", text)
    text = re.sub(r"\{[^{}]*=[^{}]*\}", "", text)
    # 残留 HTML 行内标签 -> 删标签留内容
    text = re.sub(r"</?(?:span|div|p|sup|sub|strong|em|br)[^>]*>", "", text)
    # 收拢:先删引用剪掉后残留的空括号/方括号,再删标点前孤立空格、多空格、多空行
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)     # 空 ()(如 "Eq. ()" -> "Eq.")
    text = re.sub(r"\[\s*\]", "", text)             # 空 []
    text = re.sub(r"[ \t]+([.,;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return restore_math(text.strip(), math)


def clean(text, title=None, authors=None):
    yaml_str, body = split_yaml(text)
    meta = {}
    if yaml_str:
        try:
            meta = yaml.safe_load(yaml_str) or {}
        except Exception:
            meta = {}
    if not title and isinstance(meta.get("title"), str):
        title = meta["title"].strip()
    abstract = meta.get("abstract")
    abstract = abstract.strip() if isinstance(abstract, str) else None

    # abstract 并入正文开头,和 body 一起清洗(去 cite/公式标记)
    doc = _scrub("\n\n".join(p for p in (abstract, body) if p))

    head = ""
    if title:
        head += f"# {title}\n\n"
    if authors:
        head += f"{authors}\n\n"
    return head + doc + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--authors", default=None)
    a = ap.parse_args()
    raw = open(a.inp, encoding="utf-8").read()
    out = clean(raw, title=a.title, authors=a.authors)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[clean_md] {a.inp} -> {a.out}  ({len(raw)} -> {len(out)} chars)")


if __name__ == "__main__":
    main()
