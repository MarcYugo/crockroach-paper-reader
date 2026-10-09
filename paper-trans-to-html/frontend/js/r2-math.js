/* ============ 公式渲染（KaTeX）：从 reader2.js 拆出来的独立模块 ============
   这块代码原来是 reader2.js 里的连续一段，**按原样搬过来**（缩进都没动，方便对照），
   只有一处改动：原来直接读外层闭包的 `DOC`，现在改成 `env.DOC`
   —— 换论文时 DOC 会整体替换，必须每次用的时候取最新值，不能在装配时抓一次快照。

   装配（reader2.js 底部）：
     const { renderInlineMath, … } = R2Math({ docId, findItem, itemById,
                                              get DOC() { return DOC; } });
   入参只放**跨模块共享**的东西，其余（KATEX_* 常量、_katexP/_katexReady、
   KATEX_EM、各种缓存）都跟着本文件走，不再污染 reader2.js 的作用域。

   对外接口在文件末尾的 return（原 17 项 + 页面公式框 2 项）——
   其中 _katexReady 是 let（加载完会置 true），不能按值导出，故给的是 katexReady()；
   `buildFormulaBoxes` / `finalizeFormulaBoxes` 是 **detection-service-group 公式框覆盖层**
   （解析侧下发 `pages[].formula_boxes`，按 `bbox_norm × 页面显示宽高` 定位，
   KaTeX 渲染失败时用框内裁图（base64）兜底，
   见 `backend/pdf_parser/backend_detection_service_group.py`）。
   ============================================================================ */
window.R2Math = function (env) {
  "use strict";
  const { docId, findItem, itemById, fitItemLines } = env;   // 这三个都是稳定引用，装配时取一次即可

  const KATEX_JS = "vendor/katex/katex.min.js";
  const KATEX_CSS = "vendor/katex/katex.min.css";
  const MATH_DELIM_RE = /\\\(|\\\[|\$/;     // 文本里是否含数学定界符(\(、\[、$、$$)
  let _katexP = null;
  let _katexReady = false;      // 组件是否已加载完(未就绪时公式回退原文字)

  function _loadScript(src) {
    return new Promise(res => {
      const s = document.createElement("script");
      s.src = src; s.async = true;
      s.onload = () => res(true); s.onerror = () => res(false);
      document.head.appendChild(s);
    });
  }
  /* 只加载项目内置的 KaTeX；加载失败就把公式当普通文字显示(不至于变成空白)。
     **必须等字体就绪再开渲**：字体没到位时量出来的公式宽度是后备字体的，
     宽度适配会算错，等字体一换公式就攞出框外被裁掉。 */
  function ensureKatex() {
    if (_katexP) return _katexP;
    _katexP = Promise.all([_loadCss(KATEX_CSS), _loadScript(KATEX_JS)])
      .then(([, ok]) => {
        if (!ok || !window.katex || typeof window.katex.render !== "function") return false;
        return _katexFontsReady().then(() => { _katexReady = true; return true; });
      })
      .catch(() => false);
    return _katexP;
  }
  function _loadCss(href) {
    return new Promise(res => {
      if (document.querySelector("link[data-katex-css]")) return res(true);
      const link = document.createElement("link");
      link.rel = "stylesheet"; link.href = href;
      link.setAttribute("data-katex-css", "1");
      link.onload = () => res(true); link.onerror = () => res(false);
      document.head.appendChild(link);
    });
  }
  function _katexFontsReady() {
    if (!document.fonts || !document.fonts.load) return Promise.resolve();
    const fams = ["KaTeX_Main", "KaTeX_Math", "KaTeX_AMS", "KaTeX_Caligraphic",
                  "KaTeX_Size1", "KaTeX_Size2", "KaTeX_Size3", "KaTeX_Size4"];
    const jobs = fams.map(f => {
      try { return document.fonts.load('1em "' + f + '"'); } catch (e) { return null; }
    }).filter(Boolean);
    return Promise.all(jobs).catch(() => {});
  }
  /* 文档里有没有需要数学排版的东西：detection-service-group 公式框、独立公式(latex)、
     Surya 块的行内数学区间/表格 <math>，或文本里的数学定界符。 */
  function docNeedsMath(doc) {
    for (const pg of ((doc && doc.pages) || [])) {
      if ((pg.formula_boxes || []).length) return true;   // 页面公式框覆盖层
      for (const t of (pg.texts || [])) {
        if (t.latex) return true;                         // 公式块（含 Surya kind="formula"）
        if (t.math && t.math.length) return true;         // Surya 块：解析侧标好的行内区间
        if (t.html && t.html.indexOf("<math") >= 0) return true;   // Surya 表格里的 <math>
        if (MATH_DELIM_RE.test(t.text || "")) return true;
      }
    }
    return false;
  }
  /* 行内数学：把文本里的 `\(…\)` / `$$…$$` 原地换成一个 KaTeX 元素。
     KaTeX 是同步的，所以直接扫文本节点、就地替换即可，不用排队也不用重排。 */
  function renderInlineMath(el) {
    if (!_katexReady || !el) return;
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    const jobs = [];
    let node;
    while ((node = walker.nextNode())) {
      const s = node.nodeValue || "";
      if (s.indexOf("\\(") < 0 && s.indexOf("$") < 0) continue;
      jobs.push(node);
    }
    for (const tn of jobs) {
      // `.md code / .md pre`：Markdown 容器（AI 面板）里的行内代码 / 代码块不换 KaTeX，
      // 保持原样显示（普通正文里没有这两个类，行为不变）。
      if (tn.parentNode && tn.parentNode.closest
          && tn.parentNode.closest(".mathbox, .katex, .md code, .md pre")) continue;
      const parts = _splitInlineMath(tn.nodeValue || "");
      // ⚠️ 不能只按长度判断：**整段就是一个公式**时 parts 只有一个元素(而且是个对象)，
      // 那也是要渲染的 —— 注入出来的 `\(…\)` 常常单独占一个 run(样式与前后文不同，
      // 合并不了)，按长度跳过就会把 `\(T^t\)` 原样显示成一串反斜杠。
      // 真正的「没有公式」是返回 `[原文]` 这一个**字符串**。
      if (parts.length < 2 && typeof parts[0] === "string") continue;
      const frag = document.createDocumentFragment();
      for (const p of parts) {
        if (typeof p === "string") { frag.appendChild(document.createTextNode(p)); continue; }
        const span = document.createElement("span");
        try {
          if (!katexRender(span, p.latex, false)) {
            frag.appendChild(document.createTextNode(p.raw));
            continue;
          }
        }
        catch (e) { frag.appendChild(document.createTextNode(p.raw)); continue; }
        frag.appendChild(span);
      }
      tn.parentNode.replaceChild(frag, tn);
    }
  }
  /* 按定界符切成 ['文字', {latex,raw}, '文字', …]。
     支持 `\(…\)`、`$$…$$`，以及 **单 `$…$`**（OCR 出来的文本里常见这种写法）。
     单 `$` 容易和货币符号撞车，所以额外要求内容「像数学」：首尾不接空格、
     且含 \ _ ^ { } 中的一个，否则不当公式处理（原样留作文字）。 */
  function _splitInlineMath(s) {
    const re = /\\\(([^]*?)\\\)|\$\$([^]*?)\$\$|\$([^$\n]{1,120}?)\$/g;
    const out = []; let last = 0, m;
    while ((m = re.exec(s))) {
      let latex = (m[1] != null ? m[1] : (m[2] != null ? m[2] : m[3]));
      if (m[3] != null && (/^\s|\s$/.test(latex) || !/[\\_^{}]/.test(latex))) continue;
      if (m.index > last) out.push(s.slice(last, m.index));
      out.push({ latex: latex.trim(), raw: m[0] });
      last = m.index + m[0].length;
    }
    if (!out.length) return [s];
    if (last < s.length) out.push(s.slice(last));
    return out;
  }

  /* ============ 行内公式：用 Surya 的 <math> LaTeX 渲染 ============
     正文里的行内数学在数据里是一堆碎片化的字形 run，而且常被切成**同一 y 的多段**
     （PyMuPDF 把上下标当成独立行），渲染出来会互相压在一起。
     Surya 的块 HTML 里本来就有正确的 LaTeX（`<math>close_{t,s}</math>`），
     这里把它对齐回块内的**字符区间**（canonical 偏移），在那个位置覆盖一个 KaTeX，
     原字形改成透明色（保留盒子与底色，选中/高亮仍可见）。
     对齐不可靠就**整块放弃**，保持原字形 —— 宁可不变，不可切错。 */
  const _LIG = { "\ufb02": "fl", "\ufb01": "fi", "\ufb00": "ff", "\ufb03": "ffi",
                 "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st" };
  const _DASH = { "\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-" };
  const _QUOT = { "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " " };

  /* 文本规整：连字/破折号/引号归一、空白并成一个空格、断行连字符（连字符+换行）丢掉。
     返回 {s, idx}，idx[i] 是规整串第 i 个字符在原串里的下标。 */
  function _normText(str) {
    const out = [], idx = [];
    let i = 0;
    while (i < str.length) {
      const c = str[i];
      if (c === "-" && i + 1 < str.length && /[ \t\r\n]/.test(str[i + 1])) {
        i++;                                                  // 断行连字符：连字符+换行一起丢
        while (i < str.length && /[ \t\r\n]/.test(str[i])) i++;
        continue;
      }
      const exp = _LIG[c];
      if (exp) { for (const e of exp) { out.push(e); idx.push(i); } i++; continue; }
      const ch = _DASH[c] || _QUOT[c] || c;
      if (/\s/.test(ch)) {
        let j = i;
        while (j < str.length && /\s/.test(str[j])) j++;
        out.push(" "); idx.push(i); i = j; continue;
      }
      out.push(ch); idx.push(i); i++;
    }
    return { s: out.join(""), idx };
  }
  /* LaTeX 的“纯文本形态”，用于和切出来的原字形比对 */
  function _plainMath(latex) {
    return String(latex || "")
      .replace(/\\[a-zA-Z]+/g, " ").replace(/\\[ ,;:!]/g, " ")
      .replace(/[{}_^]/g, "").replace(/[^0-9a-zA-Z]/g, "").toLowerCase();
  }
  function _countChars(s) {
    const m = new Map();
    for (const ch of s) m.set(ch, (m.get(ch) || 0) + 1);
    return m;
  }
  /* 乱序字符多重集相似度：LaTeX 里上下标的先后与 PDF 字形顺序不一致，顺序比对不可用 */
  function _bagRatio(latex, seg) {
    const a = _countChars(_plainMath(latex));
    const b = _countChars(String(seg).replace(/[^0-9a-zA-Z]/g, "").toLowerCase());
    let na = 0, nb = 0, inter = 0;
    a.forEach(v => { na += v; });
    b.forEach(v => { nb += v; });
    a.forEach((v, k) => { if (b.has(k)) inter += Math.min(v, b.get(k)); });
    return (na && nb) ? 2 * inter / (na + nb) : 0;
  }
  /* 把块 HTML 拆成 [文字, {latex}, …]；结构复杂（表格/多块）时返回 null */
  function _htmlMathParts(html) {
    if (!html || html.indexOf("<math") < 0) return null;
    if (/<(table|div|ul|ol|figure|tr|td|h[1-6])\b/i.test(html)) return null;
    const body = html.replace(/^\s*<p[^>]*>/i, "").replace(/<\/p>\s*$/i, "");
    if (!body) return null;
    const parts = [];
    const re = /<math[^>]*>([\s\S]*?)<\/math>/gi;
    let pos = 0, m;
    const txt = s => _unesc(s.replace(/<[^>]+>/g, ""));
    while ((m = re.exec(body))) {
      if (m.index > pos) parts.push(txt(body.slice(pos, m.index)));
      parts.push({ latex: txt(m[1]).trim() });
      pos = m.index + m[0].length;
    }
    if (pos < body.length) parts.push(txt(body.slice(pos)));
    return parts.some(p => typeof p !== "string") ? parts : null;
  }
  function _unesc(s) {
    return String(s).replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
      .replace(/&#0?39;/g, "'").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&");
  }

  /* 在规整串里找一个文字段：整段命中优先；找不到就退而用它的**最长前缀**命中
     （PDF 的行切分会在词中间插进碎片，例如 `formulaic` 被切成 `formu-`+`t`+`laic`，
     整段就永远匹配不上）。返回 {at, end, partial} 或 null。 */
  function _findLit(cn, lit, from) {
    if (!lit) return null;
    const j = cn.indexOf(lit, from);
    if (j >= 0) return { at: j, end: j + lit.length };
    const minLen = Math.min(8, lit.length);
    for (let n = lit.length - 1; n >= minLen; n--) {
      const jh = cn.indexOf(lit.slice(0, n), from);
      if (jh >= 0) return { at: jh, end: jh + n, partial: true };
    }
    return null;
  }
  /* 消费一个文字段，返回新游标。部分命中时再把**剩余部分的最长后缀**找回来，
     否则游标会停在词中间、后面的公式区间就会多圈进一段正文。 */
  function _consumeLit(cn, lit, from) {
    const f = _findLit(cn, lit, from);
    if (!f) return -1;
    if (!f.partial) return f.end;
    const tail = lit.slice(f.end - f.at);
    let cur = f.end;
    for (let m = tail.length; m >= 3; m--) {
      const k = cn.indexOf(tail.slice(-m), cur);
      if (k >= 0) { cur = k + m; break; }
    }
    return cur;
  }

  /* 把 [文字,{latex},…] 对齐到 canonical 的字符区间。
     做法：逐个文字段在 canonical（规整后）里顺序定位，公式区间 = 前一段结束 → 后一段开始；
     后一段是 `.` 这类极短锚点时，会命中公式内部的点（如 `...`），
     所以取多个候选终点，挑第一个“字形与 LaTeX 足够像”的组合。 */
  function _alignInlineMath(canon, parts) {
    const cn = _normText(canon);
    const out = [];
    let cur = 0;
    for (let k = 0; k < parts.length; k++) {
      const p = parts[k];
      if (typeof p === "string") {
        const lit = _normText(p).s.trim();
        if (lit) {
          cur = _consumeLit(cn.s, lit, cur);
          if (cur < 0) return null;               // 锚点断链 → 整块放弃
        }
        continue;
      }
      const nxt = (typeof parts[k + 1] === "string") ? _normText(parts[k + 1]).s.trim() : "";
      const ends = [];
      if (nxt) {
        let from = cur;
        for (let t = 0; t < 8; t++) {
          const f = _findLit(cn.s, nxt, from);
          if (!f) break;
          ends.push(f.at);
          from = f.at + 1;
        }
      } else if (k + 1 >= parts.length) {
        ends.push(cn.s.length);
      }
      if (!ends.length) return null;
      const ml = _plainMath(p.latex).length;
      let best = null;
      for (const bN of ends) {
        if (bN <= cur) continue;
        const a = cn.idx[cur] !== undefined ? cn.idx[cur] : canon.length;
        const b = cn.idx[bN] !== undefined ? cn.idx[bN] : canon.length;
        const seg = canon.slice(a, b).trim();
        const ls = (seg.match(/[0-9a-zA-Z]/g) || []).length;
        if (!ls || ls > Math.max(6, ml * 2.5)) continue;
        const r = _bagRatio(p.latex, seg);
        if (r >= 0.85) {
          // 去掉首尾空白：区间含空格的话，覆盖层会按“含空格的矩形”定位，
          // 整条公式会左移一个空格宽、右边多出一截 → 与前后文字的间隔就不对了
          const raw = canon.slice(a, b);
          const lead = raw.length - raw.replace(/^\s+/, "").length;
          const trail = raw.length - raw.replace(/\s+$/, "").length;
          best = { a: a + lead, b: Math.max(a + lead + 1, b - trail), latex: p.latex, endNorm: bN, ratio: r };
          break;
        }
      }
      if (!best) return null;
      out.push(best);
      cur = best.endNorm;
    }
    return out.length ? out : null;
  }
  /* 该块的行内公式区间（带缓存，同一块只算一次） */
  const _inlineCache = new Map();
  function inlineMathOf(t, canon) {
    const key = docId + "#" + t.id;
    if (_inlineCache.has(key)) return _inlineCache.get(key);
    let res = null;
    try {
      const parts = _htmlMathParts(t.html);
      if (parts) res = _alignInlineMath(canon, parts);
    } catch (e) { res = null; }
    _inlineCache.set(key, res);
    return res;
  }
  /* x..y 这个子区间落在哪条行内公式里？返回序号，不在任何公式里返回 -1 */
  function _mathIdxAt(list, x, y) {
    if (!list) return -1;
    for (let i = 0; i < list.length; i++) {
      const m = list[i];
      if (m.a <= x && y <= m.b) return i;
    }
    return -1;
  }
  /* 这个片段是不是“紧跟在某条行内公式之后”？两种形态：
     ① 跨公式 run 的尾部（公式在 run 中间结束）；
     ② 紧邻公式之后的另一个 run（中间最多隔 1 个空格）。
     用途：PyMuPDF 会把上标/小字号的标记**泄漏**到公式后面的正文 run 上
     （例：`…^{(i)}, r_t^{(h)})` 后面的 “ is the value vector of…” 被标成 up=true + 7pt，
     渲染成 0.7 倍，看着就是“公式偏大”）。 */
  function _rightAfterMath(list, gs, ge, x) {
    if (!list) return false;
    for (const m of list) {
      if (gs < m.b && m.b <= x && ge > m.b) return true;      // 跨公式 run 的尾部
      if (m.b <= x && x - m.b <= 2) return true;              // 紧跟在公式之后
    }
    return false;
  }
  /* 公式区间的基准字号：取区间内 run 的最大字号，再用**块内主字号**夹一道。
     不夹不行：区间恰好只盖住下标 run 时（全是 7pt），最大字号就是 7pt，
     公式会明显偏小；反过来 OCR/合成行也会给出离谱的大值。 */
  function _mathBaseSize(item, m) {
    let own = 0;
    for (const ld of item.linesData)
      for (const r of ld.runs)
        if (r.g < m.b && r.g + r.t.length > m.a && r.s > own) own = r.s;
    const body = (item.bodyRun && item.bodyRun.s) || 0;
    if (!own) own = body || 10;
    if (body) own = Math.min(Math.max(own, body * 0.85), body * 1.15);
    return own;
  }

  /* 量“这一行的正文基线”：往行首插一个 0 尺寸的 inline-block 探针，
     它的 margin-box 底边正好落在行基线上（inline-block 无内容时基线=底边）——
     比用字体度量反推可靠得多，也不受宿主行高影响。 */
  function _lineBaseline(lnEl) {
    if (!lnEl) return null;
    const p = document.createElement("span");
    p.style.cssText = "display:inline-block;width:0;height:0;overflow:hidden;padding:0;margin:0;border:0";
    lnEl.insertBefore(p, lnEl.firstChild);
    const y = p.getBoundingClientRect().bottom;
    p.remove();
    return y;
  }
  /* KaTeX 的基线 = 它的 strut 底边：strut 就是 KaTeX 用来给盒子定基线的，本身无降部 */
  function _katexBaseline(box) {
    const s = box.querySelector(".katex-html .strut");
    return s ? s.getBoundingClientRect().bottom : null;
  }

  /* 行内公式收尾：块已挂到文档上 → 量出原字形所占矩形，顶上一个 KaTeX */
  function finalizeInlineMath() {
    for (const id in itemById) {
      const item = itemById[id];
      const list = item.inlineMath;
      if (!list || !list.length || !item.el || !item.el.isConnected) continue;
      const br = item.el.getBoundingClientRect();
      for (let i = 0; i < list.length; i++) {
        if (item.el.querySelector('.mathbox.inline[data-mi="' + i + '"]')) continue;
        const spans = item.el.querySelectorAll('.mspan[data-mi="' + i + '"]');
        if (!spans.length) continue;
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        spans.forEach(s => {
          const r = s.getBoundingClientRect();
          if (!r.width && !r.height) return;
          x0 = Math.min(x0, r.left); y0 = Math.min(y0, r.top);
          x1 = Math.max(x1, r.right); y1 = Math.max(y1, r.bottom);
        });
        if (!isFinite(x0)) continue;
        const wPx = x1 - x0;
        const S = (item.linesData[0] && item.linesData[0].runs[0] && item.linesData[0].runs[0].S) || 1;
        const box = el("div", "mathbox inline");
        box.dataset.mi = i;
        box.dataset.latex = list[i].latex;
        box._blk = item.el;
        box._origW = wPx;
        box._ln = spans[0].closest(".ln");        // 让位时要挪的就是这一行的邻行
        box._x0 = Math.round(x0 - br.left);       // 基准左边(复原/让位都按它算)
        box.style.left = box._x0 + "px";
        // 字号：**统一**用基准字号（不逐条按宽度缩 —— 那会让同页公式大小不一），
        // 再除 KaTeX 内部的 1.21em（见 KATEX_EM），保证字高与正文一致。
        box.style.fontSize = mathSizePx(_mathBaseSize(item, list[i]) * S) + "px";
        // 垂直：先按原字形矩形放一个初始位置（下面会按实测基线纠正）
        box.style.top = Math.round(y0 - br.top) + "px";
        if (!katexRender(box, list[i].latex, false)) continue;   // 渲染失败 → 保持原字形
        item.el.appendChild(box);
        // 垂直对齐：**渲染后直接量、按差值平移**。
        // 不要用字体度量去推算“盒顶→基线”的距离：那个值受宿主元素的行高影响，
        // 在 body 里量出来的换到 .blk 里就不准（实测差 7~11px，公式整体下沉）。
        const tb = _lineBaseline(box._ln);
        const kb = _katexBaseline(box);
        if (tb != null && kb != null)
          box.style.top = Math.round(parseFloat(box.style.top) + (tb - kb)) + "px";
        // 隐原字形**靠 class**，不能只设内联色：选句/高亮/标注都会重建 span，
        // 重建后内联色就没了，原字形会从 KaTeX 底下冒出来变成**重影**。
        // 但 class 也只能加在“这一条公式”所在的 span 上（渲染失败的那条要保持可见）。
        if (!item.mathDone) item.mathDone = new Set();
        item.mathDone.add(i);
        const marked = item.el.querySelectorAll('.mspan[data-mi="' + i + '"]');
        marked.forEach(s => s.classList.add("mspan-hidden"));
      }
      _inlineMathRoom(item);          // 公式摆好之后，把周围的间距让出来
      // 让位会插 `.mspacer`（真的会加宽行）→ 行宽适配必须在这之后再跑一次，
      // 否则整行会溢出到栏外。rebuildItem 里那次跑得早，看不到这些 spacer。
      if (fitItemLines) fitItemLines(item);
    }
  }

  /* ============ 行内公式“让位”：不压扁、不缩字号，放不下就把周围让开 ============
     KaTeX 同字号下比原文宽 20~40%（字距/字体不同），塞进原字形框只有两条路：压扁（字形
     走形）、缩字号（同页公式大小不一），都不好。改成把空间从**周围**借出来：
       ① 横向：公式比原字形框宽出来的那点，补在**公式后面**（把同行后面的正文推开）；
       ② 纵向：公式带上标/下标/分式时比一行高，把**上一行往上、下一行往下**挪一点
          （只动紧邻的两行，行距变化最小；块的译文卡跟着往下让）。
     幂等要求：重排 + 选中/高亮/标注都会重建 .ln，所以每次先按基准值复原再重算：
       · 行位按 ld.baseTop，覆盖层按 box._x0（都不累计、不叠加）；
       · 横向用 spacer 元素（随 .ln 一起被清掉，不会残留）；
       · 纵向同一行被多条公式顶到时取**最大**让位量。 */
  function _inlineMathNaturalW(box) {
    const w0 = box.style.width;
    box.style.width = "max-content";
    const w = box.scrollWidth;          // max-width:100% 挡不住 scrollWidth，量到的是内容宽
    box.style.width = w0;
    return w;
  }
  function _lnScaleX(ln) {
    const m = /scaleX\(([\d.]+)\)/.exec(ln.style.transform || "");
    return m ? parseFloat(m[1]) : 1;    // fitLineEl 压过整行 → 行内长度要换算
  }
  function _shiftRoomLine(ld, delta) {
    if (!ld || !ld.el) return;
    const cur = ld._roomShift || 0;
    ld._roomShift = delta > 0 ? Math.max(cur, delta) : Math.min(cur, delta);
    ld.el.style.top = (ld.baseTop + ld._roomShift) + "px";
  }
  function _shiftRoomBox(box, dx) {
    box._roomShift = (box._roomShift || 0) + dx;
    box.style.left = (box._x0 + box._roomShift) + "px";
  }
  function _resetInlineRoom(item) {
    item.el.querySelectorAll(".mspacer").forEach(s => s.remove());
    for (const ld of item.linesData) {
      if (!ld.el) continue;
      if (ld.baseTop == null) ld.baseTop = parseFloat(ld.el.style.top) || 0;
      ld._roomShift = 0;
      ld.el.style.top = ld.baseTop + "px";
    }
    item.el.querySelectorAll(".mathbox.inline").forEach(b => {
      b._roomShift = 0;
      if (b._x0 != null) b.style.left = b._x0 + "px";
    });
  }
  function _inlineMathRoom(item) {
    const list = item.inlineMath;
    if (!list || !list.length) return;
    _resetInlineRoom(item);
    const byLine = new Map();          // 一行可能有好几条公式
    for (let i = 0; i < list.length; i++) {
      const b = item.el.querySelector('.mathbox.inline[data-mi="' + i + '"]');
      if (!b || !b._ln) continue;
      if (!byLine.has(b._ln)) byLine.set(b._ln, []);
      byLine.get(b._ln).push(b);
    }
    if (!byLine.size) return;
    const b0 = item.el.querySelector(".mathbox.inline");
    const em = (b0 && parseFloat(b0.style.fontSize)) || 15;   // 1em ≈ 公式字号
    // PUSH_MAX/VMOVE_MAX 都按 em 算：换缩放档位时不用调常量。
    // 上限是为了别把整行文字推得太离谱（表格/窄栏里尤其），超出的那点宁可留着少量重叠。
    const PUSH_MAX = 2.2 * em, VMOVE_MAX = 0.45 * em;
    const VMIN = 2;      // 小于 2px 的“重叠”是字体盒子的空隙，不值得动行距

    // ① 横向：从右往左处理（往右推只影响它右边的东西），把公式右边缘之后的正文推开
    for (const [ln, arr] of byLine) {
      const k = _lnScaleX(ln);
      for (let j = arr.length - 1; j >= 0; j--) {
        const box = arr[j];
        const spans = item.el.querySelectorAll('.mspan[data-mi="' + box.dataset.mi + '"]');
        const last = spans[spans.length - 1];
        if (!last) continue;
        // 公式可能**跨行**（Surya 的 <math> 区域会横跨换行）：覆盖层是单行的，
        // 续行后面的正文跟它没有可比性 → 只在“公式收尾还在本行”时才推。
        if (last.closest(".ln") !== ln) continue;
        const trail = last.nextElementSibling;
        // 后面没正文、或紧跟的是另一条公式的字形 → 没什么可推的
        if (!trail || trail.classList.contains("mspan")) continue;
        const d = Math.round(box.getBoundingClientRect().left + _inlineMathNaturalW(box) + 2
                             - trail.getBoundingClientRect().left);
        if (d <= 0) continue;
        const push = Math.min(d, PUSH_MAX);
        const sp = el("span", "mspacer");
        sp.setAttribute("aria-hidden", "true");
        sp.style.cssText = "display:inline-block;height:0;overflow:hidden;width:0";
        sp.style.width = (push / k).toFixed(2) + "px";   // 行内有 scaleX → 换算回行内长度
        ln.insertBefore(sp, trail);
        for (let m = j + 1; m < arr.length; m++) _shiftRoomBox(arr[m], push);
      }
    }

    // ② 纵向：公式比一行高 → 上一行往上、下一行往下，别压在邻行的字上
    for (const [, arr] of byLine)
      for (const box of arr) {
        const mi = box.dataset.mi;
        const spans = item.el.querySelectorAll('.mspan[data-mi="' + mi + '"]');
        const last = spans[spans.length - 1];
        const li = item.linesData.findIndex(ld => ld.el === box._ln);
        const li2 = last ? item.linesData.findIndex(ld => ld.el === last.closest(".ln")) : li;
        const r = box.getBoundingClientRect();
        const prev = li > 0 ? item.linesData[li - 1] : null;
        const n = Math.max(li, li2);
        const next = n >= 0 && n + 1 < item.linesData.length ? item.linesData[n + 1] : null;
        if (prev && prev.el) {
          const pr = prev.el.getBoundingClientRect();
          const d = Math.min(VMOVE_MAX, Math.round(pr.bottom - r.top));
          if (d >= VMIN) _shiftRoomLine(prev, -d);
        }
        if (next && next.el) {
          const nr = next.el.getBoundingClientRect();
          const d = Math.min(VMOVE_MAX, Math.round(r.bottom - nr.top));
          if (d >= VMIN) _shiftRoomLine(next, d);
        }
      }

    // 行被推下去 → 块高与译文卡跟着往下让，否则最后一行会顶到译文卡上
    let bottom = 0;
    for (const ld of item.linesData)
      if (ld.el) bottom = Math.max(bottom, ld.baseTop + (ld._roomShift || 0) + ld.el.offsetHeight);
    const h = Math.ceil(Math.max(item.contentH || 0, bottom));
    if (item._roomH !== h) {
      item._roomH = h;
      item.el.style.height = h + "px";
      if (item.exp) item.exp.style.top = h + "px";
    }
  }

  /* 独立公式的收尾：等块挂到文档上量出真实高度，再回填块高(KaTeX 内容本身是同步渲好的)。 */
  const _katexBoxes = new Set();      // 本轮待收尾的公式覆盖层
  const _mathFit = new Set();         // 已渲染的公式：字体/缩放变化后需要重新适配
  function finalizeDisplayMath() {
    if (!_katexBoxes.size) return;
    for (const box of [..._katexBoxes]) {
      _katexBoxes.delete(box);
      const blk = box._blk;
      if (!blk || !box.isConnected) { box.remove(); continue; }
      // KaTeX 渲染成功 → 有 .katex；失败时盒子里放的是 LaTeX 源码文本，也要留着
      // (公式块已经不渲染原字形了，丢掉这个盒子 = 这一块彻底空白)。
      if (!box.querySelector(".katex") && !box.textContent.trim()) { box.remove(); continue; }
      blk.classList.add("has-math");
      _mathFit.add(box);
      sizeMathBox(box);
    }
    // 兜底：万一首次渲染赶在字体之前，字体一到位就重新量一次(幂等，重复跑无副作用)
    if (document.fonts && document.fonts.ready)
      document.fonts.ready.then(refitMathBoxes).catch(() => {});
  }
  function refitMathBoxes() {
    for (const box of _mathFit) {
      if (!box.isConnected) { _mathFit.delete(box); continue; }
      sizeMathBox(box);
    }
  }
  function sizeMathBox(box) {
    // 兜底展示的 LaTeX 源码不做宽度适配(那会把它缩成一团小字)
    if (!box.classList.contains("failed")) fitMathBox(box);
    const blk = box._blk;
    if (!blk) return;
    const h = Math.max(box._contentH || 0, box.offsetHeight);
    blk.style.height = h + "px";
    const exp = blk.querySelector(".exp");
    if (exp) exp.style.top = h + "px";
  }

  /* 公式的“主体宽度”(pt)：从块左边到**除编号行以外**所有行的最右端。
     公式编号（如 "(1)"）在 PDF 里是贴栏右排的，和公式主体之间隔着一大段空白，
     而公式块的外框是“行并集”会把编号算进去 → 框被撑宽。
     宽度适配必须用主体宽度，否则带编号的公式永远轮不到缩小，
     看上去就比同页其它同类型的公式大一号（evaluation metrics 里的 (1) 就是这种）。 */
  const _TAG_LINE_RE = /^[（(]\s*\d+\s*[a-z]?\s*[)）]$/;
  function formulaBodyWidth(t) {
    let right = 0, found = false;
    for (const ln of (t.lines || [])) {
      // 公式块只带几何时由后端标 `tag`(见 `pdf_parser._TAG_LINE_RE`)；旧数据回退看行文本。
      // ⚠️ 两处规则必须一致。
      const isTag = ln.tag === true ||
        _TAG_LINE_RE.test((ln.runs || []).map(r => r.t || "").join("").trim());
      if (isTag) continue;                          // 编号行不算入主体
      right = Math.max(right, ln.x + ln.w);
      found = true;
    }
    return found ? Math.max(1, right - t.x) : t.w;
  }

  /* 很多解析结果把公式编号**写进了 LaTeX**（`…\qquad(1)` / `…\quad(1)`），
     PDF 里这个编号却是贴栏右排的。直接渲染会让内容多出一大截，宽度适配就把整条公式算宽了。
     → 把尾部编号摘出来改交给 KaTeX 的 `\tag`：它贴右排放，且**不参与内容宽度**，
     这样“内容宽 vs 主体宽”才是同一个口径。
     守卫：编号前必须有明确的空白命令(\qquad/\quad/\hspace/三连空格…)，
     免得把 `f(x)=g(1)` 结尾的真括号当成编号摘掉。 */
  const _TAIL_TAG_RE = /(?:\\(?:qquad|quad|hspace\*?(?:\{[^}]*\})?|,|;|:|!)|\s{3,})+\s*[（(]\s*(\d+[a-z]?)\s*[)）]\s*$/;
  function splitTailTag(latex) {
    const m = _TAIL_TAG_RE.exec(latex);
    if (!m) return { latex, tag: "" };
    const head = latex.slice(0, m.index).trim();
    return head ? { latex: head, tag: m[1] } : { latex, tag: "" };
  }

  /* KaTeX 内部把字号放大 1.21em（`.katex{font-size:1.21em}`）。那个系数是给“典型网页正文字体”
     配的：本项目里的“正文字号”是 PDF 的 pt 换算值，正文又用 Georgia 顶替原字体，
     两头折算下来必须再压 1/1.15 —— 不然公式的字高比正文大一截（实测 x-height +13%、
     cap +18%），看着就是“公式的字比正文大一号”。压回去之后实测 x-height −2%、cap +3%。
     顺带的好处：宽度也缩回 15%，KaTeX 同字号比原文宽 20~40% 的问题大半出在这个系数上。 */
  const KATEX_EM = 1.15;
  function mathSizePx(basePx) { return Math.max(1, basePx / KATEX_EM); }

  function inlineTextStyle(latex) {
    const source = String(latex || "");
    return /^\s*\\(?:textstyle|displaystyle)\b/.test(source)
      ? source
      : "\\textstyle " + source;
  }

  const KATEX_OPTS = { displayMode: true, throwOnError: true, strict: "ignore", trust: false };
  /* 渲染公式；成功返回 true。LaTeX 有问题就返回 false，调用方回退显示 PDF 原文字。
     display=false 时按行内样式渲染（上下限排在右上/右下）。 */
  function katexRender(box, latex, display) {
    const opts = (display === false) ? Object.assign({}, KATEX_OPTS, { displayMode: false }) : KATEX_OPTS;
    const source = display === false ? inlineTextStyle(latex) : latex;
    const tt = display === false ? { latex: source, tag: "" } : splitTailTag(source);
    if (tt.tag) {
      try {
        window.katex.render(tt.latex + "\\tag{" + tt.tag + "}", box, opts);
        box._hasTag = true;              // 调用方把覆盖层拉满块宽 → 编号贴在原框右边缘
        return true;
      } catch (e) {
        box.textContent = "";            // 清掉半成品，退回原样渲染
      }
    }
    try { window.katex.render(source, box, opts); return true; }
    catch (e2) { box.textContent = ""; return false; }
  }

  /* 宽度适配：KaTeX 的字距/字体与原文排版不同，**同一字号**下常比原框宽 20~30%，
     而公式块的外框就是 PDF 里这条公式的原始外框 → 不处理就会被块边界裁掉一截。
     按主体宽度等比缩放(下限 0.6)，让公式的占位比例贴近 PDF。
     _basePx 记着基准字号，每次都从基准重算 → 反复调用不会越缩越小。 */
  function fitMathBox(box, tolPx, floor) {
    const base = box._basePx || (box._basePx = parseFloat(box.style.fontSize) || 0);
    if (!base) return;
    box.style.fontSize = base + "px";
    // 可用宽度 = 原文里这条公式的**主体**宽度(不含贴右的编号)；拿不到就退回块宽。
    // 注意别用 box.clientWidth：.mathbox 是 shrink-to-fit，内容窄时它也窄，不是块宽。
    const avail = box._origW || (box._blk ? box._blk.clientWidth : 0);
    if (!avail) return;
    // 内容宽度：带编号的盒子被拉成 100% 宽(为了让 \tag 贴右)，这时 scrollWidth 给的是
    // 盒子宽度而非内容宽度 → 临时切成 max-content 量一次(max-width:100% 仍会兜住超宽)。
    const w0 = box.style.width;
    box.style.width = "max-content";
    const need = box.scrollWidth;
    box.style.width = w0;
    if (need <= avail + (tolPx || 1)) return;
    // 两段式：优先只缩到 floor（行内 0.85，字号基本统一），
    // 缩到 floor 后残留溢出还超过容差，才继续缩到完全装下（下限 0.6）——
    // 否则那几条特别宽的公式会明显压住后面的文字。
    const kFit = avail / need;
    const kSoft = Math.max(kFit, floor || 0.6);
    const over = need * kSoft - avail;
    const k = (over > (tolPx || 1)) ? Math.max(0.6, kFit) : kSoft;
    if (k < 0.995) box.style.fontSize = (base * k) + "px";
  }

  /* ============ 复制公式：给 LaTeX 源码，而不是排版后的显示内容 ============
     一个公式块在 DOM 里有三份“内容”：隐藏的 PDF 原文字行(.ln)、KaTeX 的 MathML 注解、
     KaTeX 的 HTML 渲染结果。直接复制会把三份都带上(字形还会重复)，且拿到的不是源码。
     这里接管 copy 事件：克隆一份选区，把里面的公式元素一律换回 LaTeX，再自己序列化成文本。
     KaTeX 会把源码存进 <annotation encoding="application/x-tex">，所以正文里内联的公式
     (\(…\) / $$…$$ / $…$) 也能一并还原成 `$…$`。
     显示公式块给的是**裸 LaTeX**（即解析出来的那个 `latex` 字段），方便直接贴到别处用。 */
  document.addEventListener("copy", onCopyMath);

  function onCopyMath(ev) {
    if (!ev.clipboardData) return;                     // 老浏览器：不插手
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return;
    const range = sel.getRangeAt(0);

    // ① 整段选区都在同一个公式块里 → 直接给它的 LaTeX
    const sb = closestMathBlk(range.startContainer);
    const eb = closestMathBlk(range.endContainer);
    if (sb && sb === eb) {
      const it = findItem(sb.dataset.id);
      if (it && it.latex) { writeClipboard(ev, it.latex); return; }
    }
    const startFormula = closestFormula(range.startContainer);
    const endFormula = closestFormula(range.endContainer);
    if (startFormula && startFormula === endFormula && startFormula.dataset.latex) {
      writeClipboard(ev, formulaSource(startFormula));
      return;
    }

    // ② 跨块选区：克隆后把公式逐个换回源码(没碰到公式就不插手，走浏览器默认行为)
    const holder = range.cloneContents();
    const formulaHosts = holder.querySelectorAll(".fbox, .mathbox.inline, .blk.has-math .mathbox");
    formulaHosts.forEach(box => {
      if (!box.dataset.latex) return;
      box.replaceWith(document.createTextNode(formulaSource(box)));
    });
    const katexEls = holder.querySelectorAll(".katex");
    const mathBlks = holder.querySelectorAll(".blk.has-math");
    if (!formulaHosts.length && !katexEls.length && !mathBlks.length) return;
    mathBlks.forEach(b => b.querySelectorAll(".ln").forEach(n => n.remove()));  // 去掉隐藏的原文字行
    katexEls.forEach(k => {
      const blk = k.closest(".blk.has-math");            // 整块公式：优先用解析出来的 latex，
      const it = blk ? findItem(blk.dataset.id) : null;  // 口径和“单选一个公式块”保持一致
      k.replaceWith(document.createTextNode(
        (it && it.latex) ? it.latex : katexSource(k)));
    });
    const text = fragToText(holder)
      .replace(/[ \t]+\n/g, "\n").replace(/\n{2,}/g, "\n").trim();
    if (text) writeClipboard(ev, text);
  }

  function writeClipboard(ev, text) {
    ev.clipboardData.setData("text/plain", text);
    ev.preventDefault();
  }
  function closestMathBlk(node) {
    const el = node && (node.nodeType === 1 ? node : node.parentElement);
    return (el && el.closest) ? el.closest(".blk.has-math") : null;
  }
  function closestFormula(node) {
    const el = node && (node.nodeType === 1 ? node : node.parentElement);
    return el && el.closest
      ? el.closest(".fbox, .mathbox.inline, .blk.has-math .mathbox")
      : null;
  }
  function formulaSource(host) {
    const latex = host.dataset.latex || "";
    return host.matches('.fbox[data-cls="InlineFormula"], .mathbox.inline')
      ? "$" + latex + "$" : latex;
  }
  /* 从 KaTeX 的标注节点里取回源码；显示公式给裸 LaTeX，行内公式用 $…$ 包起来
     (行内混在正文里，不包就分不清哪段是公式)。 */
  function katexSource(katexEl) {
    const a = katexEl.querySelector('annotation[encoding="application/x-tex"]');
    const tex = ((a && a.textContent) || "").trim();
    if (!tex) return katexEl.textContent || "";
    return katexEl.closest(".katex-display") ? tex : "$" + tex + "$";
  }
  /* 公式块的“句子文本”是 PDF/OCR 抽出来的字形串(可能乱码)，复制时一律换成 LaTeX */
  function mathTexOf(item) { return (item && item.latex) ? item.latex : ""; }

  const _BLOCK_TAG = /^(div|p|li|h[1-6]|tr|table|section|article|blockquote)$/;
  /* 把克隆出来的片段序列化成文本：块级元素前后加换行，手感和浏览器默认复制一致 */
  function fragToText(root) {
    let out = "";
    (function walk(n) {
      for (const c of n.childNodes) {
        if (c.nodeType === 3) { out += c.nodeValue; continue; }
        if (c.nodeType !== 1) continue;
        const tag = c.tagName.toLowerCase();
        if (tag === "br") { out += "\n"; continue; }
        const block = _BLOCK_TAG.test(tag);
        if (block) out += "\n";
        walk(c);
        if (block) out += "\n";
      }
    })(root);
    return out;
  }

  /* 该文字块用什么渲染？返回 {latex} 或 null(按普通文字渲染)。
     这是**旧数据**的兼容路径（PaddleOCR/Surya 时代把 LaTeX 挂在块上：Paddle 从 OCR
     文本里抽、Surya 从 `<math>` 里抽）；新数据里公式是**页面级公式框**
     （`pg.formula_boxes`，见 buildFormulaBoxes），不走这里。
     只认「整块就是公式」的块；正文里内联的 `\(…\)` 数学交给 renderInlineMath。 */
  function mathBlockOf(t) {
    if (!t || t.kind !== "formula") return null;
    const latex = typeof t.latex === "string" ? t.latex.trim() : "";
    return latex ? { latex } : null;
  }

  /* 页面正文的基准字号(PDF 点)：取该页**非公式**文字块里字数最多的那个 run 字号。
     公式块自己的字号常是 OCR/合成行估出来的(同一页能给出 8~17pt 的离谱值)，
     正文的 run 字号来自 PyMuPDF 真实字体信息，可靠得多。 */
  const _bodySizeCache = new Map();
  function pageBodySize(pi) {
    const key = docId + "#" + pi;          // 换论文后不能复用上一份的缓存
    if (_bodySizeCache.has(key)) return _bodySizeCache.get(key);
    const pg = (env.DOC && env.DOC.pages && env.DOC.pages[pi]) || null;
    const texts = (pg && pg.texts) || [];
    const cnt = new Map();
    for (const t of texts) {
      if (t.kind === "formula") continue;
      for (const ln of (t.lines || []))
        for (const r of (ln.runs || [])) {
          const k = Math.round((r.s || 0) * 2) / 2;      // 0.5pt 一档，避开毛刺
          if (k > 0) cnt.set(k, (cnt.get(k) || 0) + Math.max(1, (r.t || "").length));
        }
    }
    let best = 0, bestN = 0;
    for (const [k, n] of cnt) if (n > bestN) { bestN = n; best = k; }
    if (!best) {                          // Surya 块：没有逐 run 字号，用解析侧估的块字号
      const cnt2 = new Map();
      for (const t of texts) {
        if (t.kind === "formula" || !(t.size > 0)) continue;
        const k = Math.round(t.size * 2) / 2;          // 0.5pt 一档，按文本长度加权
        cnt2.set(k, (cnt2.get(k) || 0) + Math.max(1, (t.text || "").length));
      }
      for (const [k, n] of cnt2) if (n > bestN) { bestN = n; best = k; }
    }
    if (!best) {                                   // 整页都是 OCR 合成行 → 退回中位数
      const all = [];
      for (const t of texts)
        for (const ln of (t.lines || []))
          for (const r of (ln.runs || [])) if (r.s > 0) all.push(r.s);
      all.sort((a, b) => a - b);
      best = all.length ? all[all.length >> 1] : 0;
    }
    _bodySizeCache.set(key, best);
    return best;
  }

  /* 公式的主体字号：取块内**最大**的字号。
     公式里夹着下标/上标(7pt 之类的小号 run)，取第一个 run 会随 run 顺序忽大忽小
     ——`IC_i=\frac{1}{T}\sum…` 那种公式第一行就是求和上限的小号 `T`，
     按它渲染整条公式会明显偏小。再拿页面正文字号夹一道，挡住离谱的估算值。
     字号来源按新旧数据两条路取：公式块不渲染行元素(`linesData` 为空)，所以
     主用后端下发的行级 `ln.s`；旧数据 / 行内场景仍看 runs。 */
  function formulaBaseSize(linesData, pi, lines) {
    let own = 0;
    for (const ld of (linesData || []))
      for (const r of (ld.runs || [])) if (r.s > own) own = r.s;
    for (const ln of (lines || [])) {
      for (const r of (ln.runs || [])) if (r.s > own) own = r.s;
      if ((ln.s || 0) > own) own = ln.s;
    }
    if (!own) own = 10;
    const ref = pageBodySize(pi);
    if (!ref) return own;
    return Math.min(Math.max(own, ref * 0.8), ref * 1.25);
  }

  /* ============ 页面公式框覆盖层（detection-service-group） ============
     新版数据里公式的落地方式是「覆盖层」：解析侧把公式字形擦成等长空格
     （位置照旧、宽度由行框反推），服务组（formula_table_service_group）返回每条
     公式的 **bbox_norm**（归一化中心 xywh，与 YOLO 标签同口径，见 yolov13
     `inference.py::build_detections`）与 **LaTeX**；这里按「bbox_norm × 当前页面
     显示宽高」把 KaTeX 盖到页面上 —— 位置来自公式框、内容来自服务组。

     与旧的“公式块”渲染（`mathBlockOf`，t.latex）的区别：那个是**文本块内**的公式
     （块自带几何、覆盖层随块走）；这里是**页面级**的独立框（`pg.formula_boxes`），
     定位与正文块无关。字号定尺与 `fitMathBox` 同一套路：先按页面正文字号渲染，
     再等比缩放到框内（KaTeX 同字号常比原排版宽 20~40%）。 */
  const _fboxFit = new Set();          // 待收尾/重新定尺的公式框内层
  let _fboxFontsHooked = false;        // fonts.ready 兜底只挂一次

  /* 在页面元素上建公式框覆盖层；返回新建的框数（由 render 逐页调用）。
     `pg` 是数据里的页对象（pg.w/pg.h 为 PDF pt），`pageEl` 是该页的 .page 元素，
     `S` 是 pt→布局 px 的换算比，`pi` 是页下标（取正文字号用）。

     渲染优先级：KaTeX（latex）→ **框内裁图**（`fb.crop`，服务组回传的 base64
     图像，KaTeX 未就绪/这条 LaTeX 渲染不出来时铺满整个框）→ LaTeX 源码文本。 */
  function buildFormulaBoxes(pg, pageEl, S, pi) {
    const boxes = (pg && pg.formula_boxes) || [];
    let n = 0;
    for (const fb of boxes) {
      const b = fb && fb.bbox_norm;
      const latex = String((fb && fb.latex) || "").trim();
      const crop = String((fb && fb.crop) || "");   // 框内裁图（data URI，可空）
      if (!b || b.length !== 4 || (!latex && !crop)) continue;
      const w = Number(b[2]) || 0, h = Number(b[3]) || 0;
      if (!(w > 0) || !(h > 0)) continue;          // 退化框（没尺寸）：不画
      const isInline = fb.class_name === "InlineFormula";
      const detectedH = h * pg.h * S;
      const host = el("div", "fbox");
      if (latex) host.dataset.latex = latex;
      // bbox_norm 是**中心点** + 宽高 → 左上角 = (中心 − 半宽/半高)
      host.style.left = ((b[0] - w / 2) * pg.w * S) + "px";
      host.style.top = ((b[1] - h / 2) * pg.h * S) + "px";
      host.style.width = (w * pg.w * S) + "px";
      host.style.height = detectedH + "px";
      if (isInline) {
        host._formulaCenterY = b[1] * pg.h * S;
        host._formulaDetectedH = detectedH;
        host._formulaLineFallback = (pageBodySize(pi) || 0) * S * 1.2;
        _inlineFormulaBoxes.add(host);
      }
      if (fb.class_name) host.dataset.cls = fb.class_name;
      // ⚠️ 模块内部用 `_katexReady`；`katexReady()` 是导出给 reader2.js 的函数，这里没有
      if (latex && _katexReady) {
        const inner = el("div", "katex-fit");
        if (katexRender(inner, latex, isInline ? false : undefined)) {
          // 基准字号：页面正文字号（与正文观感一致）；实际大小由 fitFormulaBox 定尺
          const ref = pageBodySize(pi) || 10;
          inner.style.fontSize = mathSizePx(ref * S) + "px";
          host.appendChild(inner);
          _fboxFit.add(inner);
          pageEl.appendChild(host);
          n++;
          continue;
        }
        // katexRender 失败时会把盒子清空 —— 不挂上去，继续走兜底分支
      }
      if (crop) {
        // 兜底一：框内裁图（服务组按 keep_crops 回传的 base64 图像）铺满整个框
        const img = el("img");
        img.src = crop;
        img.alt = latex || "formula";
        if (latex) img.title = latex;            // 悬停看 LaTeX 源码
        host.classList.add("cropped");
        host.appendChild(img);
      } else {
        // 兜底二：没有裁图（keep_crops=false / 老数据）→ 显示 LaTeX 源码（同公式块策略）
        const inner = el("div", "katex-fit");
        inner.textContent = latex;
        host.classList.add("failed");
        host.appendChild(inner);
      }
      pageEl.appendChild(host);
      n++;
    }
    return n;
  }

  /* 公式框收尾：页已挂到文档上 → 把每个框里的 KaTeX 等比缩到框内。
     幂等（从基准字号重算），字体加载完再跑一遍 —— 早量会用后备字体的宽度。 */
  function finalizeFormulaBoxes() {
    if (!_fboxFit.size && !_inlineFormulaBoxes.size) return;
    refitFormulaBoxes();
    if (!_fboxFontsHooked && document.fonts && document.fonts.ready) {
      _fboxFontsHooked = true;
      document.fonts.ready.then(refitFormulaBoxes).catch(() => {});
    }
  }
  const _inlineFormulaBoxes = new Set();
  function refitFormulaBoxes() {
    for (const host of [..._inlineFormulaBoxes]) {
      if (!host.isConnected) { _inlineFormulaBoxes.delete(host); continue; }
      fitInlineFormulaHeight(host);
    }
    for (const inner of [..._fboxFit]) {
      if (!inner.isConnected) { _fboxFit.delete(inner); continue; }
      fitFormulaBox(inner);
    }
  }
  /* 以同一水平位置最近的正文行高定尺，公式框保持原中心点，避免移离原排版基线。 */
  function fitInlineFormulaHeight(host) {
    const page = host.closest(".page");
    if (!page) return;
    const box = host.getBoundingClientRect();
    const cx = (box.left + box.right) / 2;
    const cy = host._formulaCenterY + page.getBoundingClientRect().top;
    let lineH = 0, best = Infinity;
    page.querySelectorAll(".blk .ln").forEach(line => {
      const r = line.getBoundingClientRect();
      if (!r.height) return;
      const dx = cx < r.left ? r.left - cx : (cx > r.right ? cx - r.right : 0);
      const dy = Math.abs((r.top + r.bottom) / 2 - cy);
      const score = dy * 10 + dx;
      if (score < best) { best = score; lineH = r.height; }
    });
    if (!lineH) lineH = host._formulaLineFallback;
    if (!lineH) return;
    // Clamp to 1.0–1.3 times the neighboring text line height.
    const h = Math.min(Math.max(host._formulaDetectedH, lineH), lineH * 1.3);
    host.style.height = h + "px";
    host.style.top = (host._formulaCenterY - h / 2) + "px";
  }
  /* 把一个框里的 KaTeX 等比缩放到框内（含轻微放大：检测框通常比字形紧）。
     内容尺寸在**变换前**量（offsetWidth/Height 不受 transform 影响）；
     `.katex-fit` 是 `width: max-content`，量的就是自然尺寸。 */
  function fitFormulaBox(inner) {
    const host = inner.parentElement;
    if (!host) return;
    const availW = host.clientWidth, availH = host.clientHeight;
    if (!availW || !availH) return;
    inner.style.transform = "translate(-50%, -50%)";
    const needW = inner.offsetWidth, needH = inner.offsetHeight;
    if (!needW || !needH) return;
    let k = Math.min(availW / needW, availH / needH);
    k = Math.max(0.2, Math.min(3, k));           // 防退化框把公式拉爆/压没
    inner.style.transform = "translate(-50%, -50%) scale(" + k.toFixed(4) + ")";
  }

  /* ============ 对外接口 ============ */
  return {
    ensureKatex, docNeedsMath, renderInlineMath, inlineMathOf, _mathIdxAt, _rightAfterMath,
    finalizeInlineMath, _inlineMathRoom, finalizeDisplayMath, formulaBodyWidth, mathSizePx,
    katexRender, mathTexOf, mathBlockOf, formulaBaseSize, _katexBoxes,
    buildFormulaBoxes, finalizeFormulaBoxes,
    katexReady: () => _katexReady,      // let，加载完会置 true → 必须用函数按需取
  };
};
