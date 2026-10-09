/* ============ Markdown 渲染（AI 辅助阅读专用）：不引第三方库的自带实现 ============
   用在两处：① AI 对话气泡（含流式增量重绘）；② 整理结果里的自由文本条目（总览 /
   要点 / 关联 / 疑问 / 下一步）。只求「AI 回复常见写法」好看：

     标题(#~######) / 无序·有序列表(可嵌套) / 粗体·斜体·删除线 / 行内 `代码` /
     围栏代码块(```) / 引用(>) / 分隔线(---) / 链接 / 表格(| a | b |) / 任务框(- [x])

   行内数学 `$…$`、`\(…\)`、`$$…$$` **不在这里转 KaTeX**：原样保留成文本节点，
   由 r2-ai.js 再调 r2-math.js 的 renderInlineMath() 就地替换（KaTeX 懒加载，
   没加载/渲染失败就显示原文，不会更糟）。

   安全：先把**所有**文本做 HTML 转义，再按固定语法组装标签；链接只放行
   http / https / mailto / 站内相对地址，其余按纯文本显示（挡 `javascript:` 等注入）。
   渲染结果只由「转义后的文本 + 白名单标签/属性」组成，innerHTML 注入是安全的。

   接口：R2Md.render(host, text)   整段渲染（覆盖 host 内容）
         R2Md.inline(host, text)   单行渲染（标题这类本来就是单行的文字用）
         R2Md.toHtml(text)         只取 HTML 字符串
         R2Md.hasMath(text)        文本里有没有数学定界符（决定要不要懒加载 KaTeX）
   ============================================================================ */
window.R2Md = (function () {
  "use strict";

  /* ---------- 基础 ---------- */
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // 占位符（Unicode 私用区，正常文本不会出现）：先抽走怕被后续语法改写的片段，
  // 全部语法处理完再还原，这样代码/公式里带 * _ ` 也不会被当成强调。
  const PH_OPEN = "\uE000", PH_CLOSE = "\uE001";
  const ph = i => PH_OPEN + i + PH_CLOSE;

  // 块级语法
  const RE_FENCE = /^\s{0,3}(`{3,}|~{3,})\s*([\w#+.-]*)\s*$/;
  const RE_HR = /^\s{0,3}([-*_])[ \t]*(?:\1[ \t]*){2,}$/;
  const RE_HEAD = /^\s{0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$/;
  const RE_QUOTE = /^\s{0,3}>[ \t]?/;
  const RE_ITEM = /^([ \t]*)([-+*]|\d{1,9}[.)])[ \t]+(.*)$/;

  /* ---------- 行内 ---------- */
  // 强调：先 ~~ 再 ** __，最后单 * 单 _。
  //   * 允许词中/中文里直接用（与 Markdown 一致），但开闭两侧都紧贴非空白字符，
  //     落单的 `**未闭合`（流式中间态）不会被吃成斜体；
  //   _ 只在词边界生效（否则 snake_case 会变斜体），边界含中文标点/引号/括号。 */
  function emph(s) {
    s = s.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
    s = s.replace(/\*([^*\s][^*\n]*?[^*\s]|[^*\s])\*/g, "<em>$1</em>");
    s = s.replace(/(^|[\s(（\[【，。、；;：:！!？?"“”'‘’])_([^_\s][^_\n]*?)_(?=$|[\s)）\]】}>,，。、；;：:！!？?.…"“”'‘’])/g, "$1<em>$2</em>");
    return s;
  }

  // 链接白名单：只认常见协议与站内相对地址；返回转义后的 href（空串 = 不放行）
  function safeUrl(u) {
    const raw = String(u).replace(/&amp;/g, "&");
    if (/^https?:\/\//i.test(raw) || /^mailto:/i.test(raw)) return esc(raw);
    if (/^[#/]/.test(raw) || /^\.{1,2}\//.test(raw)) return esc(raw);
    return "";
  }

  /* 单行文本 → HTML（转义 → 抽走代码/公式 → 反斜杠转义 → 链接 → 强调 → 还原占位符） */
  function inline(raw) {
    const st = [];                                 // 占位符仓库
    let s = esc(String(raw == null ? "" : raw));

    // 1) 行内代码：内部不再做任何 Markdown 处理
    s = s.replace(/`([^`]+)`/g, (m, code) => {
      st.push("<code>" + code + "</code>");
      return ph(st.length - 1);
    });

    // 2) 数学片段：原样保留定界符（等 renderInlineMath 来换 KaTeX），先抽走避免被强调语法改写。
    //    单 `$` 的防护口径与 r2-math.js 的 _splitInlineMath 一致：首尾不接空格且含 \ _ ^ { }，
    //    否则当成普通文字（货币 $100 之类）。
    s = s.replace(/\\\(([\s\S]*?)\\\)|\$\$([\s\S]*?)\$\$|\$([^$\n]{1,150}?)\$/g,
      (m, a, b, c) => {
        if (c != null && (/^\s|\s$/.test(c) || !/[\\_^{}]/.test(c))) return m;
        st.push(m);
        return ph(st.length - 1);
      });

    // 3) 反斜杠转义：\* \_ \# 之类按字面显示（转成占位符，别让后面的强调语法再盯上它）
    s = s.replace(/\\([\\`*_{}\[\]()#+\-.!~|>])/g, (m, ch) => {
      st.push(ch);
      return ph(st.length - 1);
    });

    // 4) 链接（锚文本允许再带强调；href 走白名单）
    s = s.replace(/\[([^\]\n]*)\]\(\s*([^)\s]+)\s*\)/g, (m, text, url) => {
      const href = safeUrl(url);
      if (!href) return m;                         // 不放行的协议：保持原文（已转义）
      st.push('<a href="' + href + '" target="_blank" rel="noopener noreferrer">'
        + emph(text) + "</a>");
      return ph(st.length - 1);
    });

    // 5) 强调
    s = emph(s);

    // 6) 还原占位符（可能嵌套：链接里套代码……多跑几轮直到没有占位符）
    for (let i = 0; i < 25 && s.indexOf(PH_OPEN) >= 0; i++) {
      s = s.replace(/\uE000(\d+)\uE001/g, (m, n) => (st[+n] != null ? st[+n] : ""));
    }
    return s;
  }

  /* ---------- 表格 ---------- */
  function splitCells(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    return s.split("|").map(c => c.trim());
  }
  // 对齐行（|---|:--:|--:|）→ ["", "center", "right"]；不是对齐行返回 null
  function tableAligns(line) {
    const cells = splitCells(line);
    if (!cells.length) return null;
    const out = [];
    for (const c of cells) {
      if (!/^:?-+:?$/.test(c)) return null;
      out.push(/^:.*:$/.test(c) ? "center" : /:$/.test(c) ? "right" : /^:/.test(c) ? "left" : "");
    }
    return out;
  }
  function cell(tag, text, align) {
    return "<" + tag + (align ? ' style="text-align:' + align + '"' : "") + ">"
      + inline(text) + "</" + tag + ">";
  }

  /* ---------- 列表 ---------- */
  // items: [{indent, ordered, text}]（按出现顺序）；idx/indent 是递归游标
  function listHtml(items, idx, indent) {
    const tag = items[idx].ordered ? "ol" : "ul";
    let html = '<' + tag + ' class="md-' + tag + '">';
    while (idx < items.length && items[idx].indent >= indent) {
      const it = items[idx];
      const base = it.indent;
      idx++;
      let content = inline(it.text.replace(/^\[([ xX])\][ \t]+/, (m, c) => (c === " " ? "☐ " : "☑ ")));
      while (idx < items.length && items[idx].indent > base) {   // 更深缩进 = 子列表（可多层）
        const [sub, nidx] = listHtml(items, idx, items[idx].indent);
        content += sub;
        idx = nidx;
      }
      html += "<li>" + content + "</li>";
    }
    return [html + "</" + tag + ">", idx];
  }

  // 收集一个列表（列表项行 + 它们的缩进续行）；空行结束（AI 回复里通常一条一块）
  function parseList(lines, start) {
    const items = [];
    let i = start;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; break; }
      const m = line.match(RE_ITEM);
      if (m) {
        // 缩进宽度把 tab 当两格（够用了；真代码放围栏里）
        items.push({
          indent: m[1].replace(/\t/g, "  ").length,
          ordered: /^\d/.test(m[2]),
          text: m[3].replace(/\s+$/, ""),
        });
        i++;
        continue;
      }
      if (items.length && /^\s+\S/.test(line)) {   // 缩进续行：接到上一条
        items[items.length - 1].text += " " + line.trim();
        i++;
        continue;
      }
      break;
    }
    return [listHtml(items, 0, 0)[0], i];
  }

  /* ---------- 块级 ---------- */
  function isBlockStart(lines, i) {
    const l = lines[i];
    return RE_FENCE.test(l) || RE_HR.test(l) || RE_HEAD.test(l) || RE_QUOTE.test(l)
      || RE_ITEM.test(l)
      || (l.indexOf("|") >= 0 && i + 1 < lines.length && tableAligns(lines[i + 1])
          && splitCells(l).length >= 2);
  }

  function blocks(lines) {
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }

      // 围栏代码块
      const fm = line.match(RE_FENCE);
      if (fm) {
        const close = new RegExp("^\\s{0,3}" + (fm[1][0] === "`" ? "`{3,}" : "~{3,}") + "\\s*$");
        const buf = [];
        i++;
        while (i < lines.length && !close.test(lines[i])) { buf.push(lines[i]); i++; }
        if (i < lines.length) i++;                 // 吃掉收尾围栏（没闭合就画到文末，流式中间态不炸）
        out.push('<pre class="md-pre"' + (fm[2] ? ' data-lang="' + esc(fm[2]) + '"' : "")
          + "><code>" + esc(buf.join("\n")) + "</code></pre>");
        continue;
      }

      // 分隔线
      if (RE_HR.test(line)) { out.push('<hr class="md-hr">'); i++; continue; }

      // 标题
      const hm = line.match(RE_HEAD);
      if (hm) {
        const lv = hm[1].length, txt = hm[2].replace(/\s+#+\s*$/, "");
        out.push("<h" + lv + ' class="md-h">' + inline(txt) + "</h" + lv + ">");
        i++;
        continue;
      }

      // 引用（连续 > 行一组，内部递归按块解析）
      if (RE_QUOTE.test(line)) {
        const buf = [];
        while (i < lines.length && RE_QUOTE.test(lines[i])) {
          buf.push(lines[i].replace(RE_QUOTE, ""));
          i++;
        }
        out.push('<blockquote class="md-quote">' + blocks(buf) + "</blockquote>");
        continue;
      }

      // 表格：本行含 |、下一行是 |---| 对齐行、至少两列（少一列容易误伤普通文字）
      if (line.indexOf("|") >= 0 && i + 1 < lines.length) {
        const aligns = tableAligns(lines[i + 1]);
        const head = splitCells(line);
        if (aligns && aligns.length === head.length && head.length >= 2) {
          let j = i + 2;
          const rows = [];
          while (j < lines.length && lines[j].trim() && lines[j].indexOf("|") >= 0) {
            rows.push(splitCells(lines[j]));
            j++;
          }
          let html = '<table class="md-table"><thead><tr>';
          head.forEach((c, k) => { html += cell("th", c, aligns[k]); });
          html += "</tr></thead><tbody>";
          for (const r of rows) {
            html += "<tr>";
            for (let k = 0; k < head.length; k++) html += cell("td", r[k] == null ? "" : r[k], aligns[k]);
            html += "</tr>";
          }
          html += "</tbody></table>";
          out.push(html);
          i = j;
          continue;
        }
      }

      // 列表
      if (RE_ITEM.test(line)) {
        const [html, ni] = parseList(lines, i);
        out.push(html);
        i = ni;
        continue;
      }

      // 段落：连续普通行；单个换行按 <br> 处理（聊天气泡里更接近原文的分行意识）
      const buf = [];
      while (i < lines.length && lines[i].trim() && !isBlockStart(lines, i)) {
        buf.push(lines[i]);
        i++;
      }
      out.push("<p>" + buf.map(inline).join("<br>") + "</p>");
    }
    return out.join("");
  }

  /* ---------- 对外接口 ---------- */
  function toHtml(text) {
    const src = String(text == null ? "" : text).replace(/\r\n?/g, "\n");
    if (!src.trim()) return "";
    return blocks(src.split("\n"));
  }
  function render(host, text) {
    if (host) host.innerHTML = toHtml(text);
  }
  function inlineTo(host, text) {
    if (host) host.innerHTML = inline(String(text == null ? "" : text));
  }
  // 是否有数学定界符（\(、$$、单个 $ 都可能；具体认不认由 renderInlineMath 定夺）
  function hasMath(text) {
    return /\\\(|\$/.test(String(text == null ? "" : text));
  }

  return { render: render, inline: inlineTo, toHtml: toHtml, hasMath: hasMath };
})();
