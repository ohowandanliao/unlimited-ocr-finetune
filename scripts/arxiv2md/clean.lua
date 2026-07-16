-- clean.lua — pandoc Lua filter:在 AST 上把 arXiv LaTeX 清成"渲染态近似" markdown 的 OCR target。
--
-- 为什么用 AST 而不是在 pandoc 输出的 markdown 字符串上正则清洗:字符串清洗要重新识别语法,
-- 对 dollar 配对($a$$b$/$$畸形/\$5 钱)、转义、表格、代码块都极脆(见 codex review)。AST 上
-- Math 是独立节点,原样通过即可,不存在 dollar 配对问题;Cite/Link/Image/Div/Span 按类型处理。
--
-- 不定义 Math/Code/CodeBlock/Table 的处理函数 -> 它们原样保留(公式、代码、表格是 PDF 可见内容)。

-- 文献引用(\cite/\citep/\citet/\eqref 里的 cite 等都是 Cite):PDF 上是编号,AST 还原不了 -> 删。
function Cite(el)
  return {}
end

-- 链接:内部交叉引用 \ref/\eqref(target 以 # 开头)PDF 上是编号、AST 还原不了 -> 整删;
-- 外链 \href -> 留可见文字、去 URL。
function Link(el)
  if (el.target or ""):sub(1, 1) == "#" then
    return {}
  end
  return el.content
end

-- 行内图:OCR 不识别图内容 -> 删图,保留 alt/caption 文字。
function Image(el)
  return el.caption
end

-- 图块(pandoc Figure):保留 caption 段落,删图本体(caption 空则整块删)。
function Figure(el)
  return (el.caption and el.caption.long) or {}
end

-- Span:去属性壳,留内容。
function Span(el)
  return el.content
end

-- Div(含 ::: fenced div、theorem/algorithm 环境的属性壳):去壳留内容。
function Div(el)
  return el.content
end

-- 裸 HTML / 残留 LaTeX(math 是 Math 节点、不在此):删。
function RawInline(el)
  return {}
end
function RawBlock(el)
  return {}
end

-- 脚注:marker+内容整体删(v1 剪干净)。
function Note(el)
  return {}
end

-- 标题:清掉 id/class/attr,免得 writer 出 "# Intro {#sec:intro}"。
function Header(el)
  el.attr = pandoc.Attr()
  return el
end

-- 把 abstract(--standalone 会放进 meta)提到正文开头,并清空 meta 免得 writer 出 YAML。
-- 元素函数(Cite/Link/...)会先遍历包括 meta 在内的全树,故这里拿到的 abstract 已清洗过。
function Pandoc(doc)
  local blocks = {}
  local abs = doc.meta.abstract
  if abs then
    -- Lua 里 meta 值已解包:MetaBlocks->Blocks、MetaInlines->Inlines(都无 .t,用 utils.type 辨)
    local t = pandoc.utils.type(abs)
    if t == "Blocks" then
      for _, b in ipairs(abs) do blocks[#blocks + 1] = b end
    elseif t == "Inlines" then
      blocks[#blocks + 1] = pandoc.Para(abs)
    end
  end
  for _, b in ipairs(doc.blocks) do blocks[#blocks + 1] = b end
  return pandoc.Pandoc(blocks)   -- 空 meta
end
