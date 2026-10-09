/* 阅读器 v2：句子级交互
 *  - 版式还原(绝对坐标) + 每行按 PDF 线框适配宽度，避免双栏文字侵入中间空隙
 *  - 句子级高亮标注(黄/绿/粉，可清除)
 *  - 句子级笔记：直接内嵌在句子下方(与正文同层，默认展开)，右侧面板只做“全部笔记”总览
 *  - 段落级批量翻译(非实时、缓存)、阅读进度保存、缩放
 */
(function () {
  "use strict";

  const CSS_W = 900;                       // zoom=1 时页面目标宽度
  const ZOOMS = [0.8, 1, 1.2, 1.5];
  const FONT = {
    sans: '"Segoe UI","Helvetica Neue",Arial,"Microsoft YaHei",sans-serif',
    serif: 'Georgia,"Times New Roman","SimSun",serif',
    mono: '"Consolas","Courier New",monospace',
  };

  const docId = queryId("id");
  const pagesEl = document.getElementById("pages");

  let DOC = null, META = null;
  let zoomIdx = 1;
  const ZOOM = () => ZOOMS[zoomIdx];
  /* 缩放是“整体缩放”：把缩放比写到 <html> 的 zoom 上，顶栏 / 目录 / 笔记栏 / AI 面板
     连同正文一起放大，而不只是把页面撑宽。zoom 生效后测量 API 分成两套坐标，别混用：
       · getBoundingClientRect() / offsetWidth|Height             → 布局坐标(未乘 zoom)
       · clientWidth|Height / innerWidth|Height / scrollY|Height  → 视觉坐标(已乘 zoom)
     凡是两套坐标相遇的地方(滚动、点击取句、视口宽度)，都用下面几个换算函数过一手。 */
  const ZOOM_OK = "zoom" in document.documentElement.style;
  const GZ = () => (ZOOM_OK ? ZOOM() : 1);                          // 真正作用在整页上的缩放比
  const vpW = () => document.documentElement.clientWidth / GZ();    // 视口宽(布局坐标)
  const vpH = () => window.innerHeight / GZ();                      // 视口高(布局坐标)
  const scrolled = () => window.scrollY / GZ();                     // 已滚动距离(布局坐标)
  const toScrollTop = (y) => Math.max(0, Math.round(y * GZ()));     // 布局坐标 → scrollTo 的视觉坐标
  const toScrollDelta = (d) => Math.round(d * GZ());                // 布局位移 → scrollBy 的视觉位移
  function applyZoom() {
    // 不支持 CSS zoom 的老浏览器退回旧行为(只缩放正文页宽)，不至于点了按钮没反应
    if (ZOOM_OK) document.documentElement.style.zoom = String(ZOOM());
  }
  const pages = [];
  const itemById = {};
  const zh = {};            // 段落(块)id -> 中文
  const noZh = new Set();
  let anno = { highlights: {}, notes: [] };
  let sel = null;           // 选中：{ sents: [{item, si}, …] }，按阅读顺序(支持拖拽多选)
  let ordSeq = 0;           // 文字块全页连续序号(重置于 render)，用于比较先后
  let showZhOn = false;
  let showNotesOn = true;   // 笔记默认展开，和正文一起看
  let editingKey = null;    // 正在内嵌编辑的句子 key("p0b3#1")
  let noteDraft = "";       // 编辑中的草稿，重绘时不丢字
  let selTransList = [];    // 多选“只翻译所选句”的卡片列表：{ key, itemId, siFrom, text }
  // 笔记卡位置：默认贴在页面右侧批注栏、与句子同高；放不下则退回句子下方内嵌
  let noteSide = true, noteW = 300, noteLeft = 0;
  let curPageW = 0;         // 当前渲染的页面宽度(px)：变化需要整体 render
  const NOTE_GAP = 10;      // 右侧相邻笔记卡的最小垂直间距
  let busy = false, cancelAll = false;
  const ZH_STORE_OLD = "paperReader.showZh";   // 旧版本的浏览器本地开关（升级后不再使用）

  const $ = (id) => document.getElementById(id);

  /* ================= 句子分割(学术英文) ================= */
  const _ABBR = new Set(["fig", "figs", "eq", "eqs", "sec", "sect", "ref", "refs", "no", "nos",
    "tab", "tabs", "ch", "chaps", "app", "pp", "vol", "vols", "approx", "et", "al", "dept", "univ"]);
  const _TITLE = new Set(["dr", "prof", "mr", "mrs", "ms", "st", "mt", "fr", "jr", "sr",
    "rev", "gen", "col", "lt", "capt", "sgt", "adm", "rep", "sen", "gov", "corp", "inc", "ltd", "co"]);

  function splitSentences(text) {
    const n = text.length, ends = [];
    let i = 0;
    while (i < n) {
      const c = text[i];
      if (c === "." || c === "!" || c === "?") {
        let j = i + 1;
        while (j < n && /[.!?…]/.test(text[j])) j++;
        while (j < n && /[)\]}”’»"'’]/.test(text[j])) j++;
        const ws = j;
        while (j < n && /\s/.test(text[j])) j++;
        if (j >= n) { ends.push(ws); break; }
        const next = text[j];
        const mm = text.slice(0, i).match(/([A-Za-z]+)\s*$/);
        const tok = mm ? mm[1].toLowerCase() : "";
        let boundary = false;
        if (/[A-Z0-9"'“’(]/.test(next)) {
          const numeral = /^\d/.test(next) && _ABBR.has(tok);
          const nameLike = _TITLE.has(tok);
          const etAl = tok === "al" && /(^|\s)et\s*$/i.test(text.slice(0, ws));
          boundary = !numeral && !nameLike && !etAl;
        }
        // 小数点/编号不切（`1.1`、`3.14`）：后端 `pdf_parser.sentences.split_sentences`
        // 是同一口径 —— 数据层现在已经「一个块 = 恰好一句」，两边不一致的话会把一个块
        // 在界面上切回两句（笔记/高亮的句号就对不上了）。
        if (boundary && i > 0 && /\d/.test(text[i - 1]) && /\d/.test(next)) {
          boundary = false;
        }
        if (boundary) ends.push(ws);
        i = j;
      } else { i++; }
    }
    const sents = [];
    let start = 0;
    for (const e of ends) {
      if (e <= start) continue;
      sents.push({ s: start, e });
      start = e;
    }
    // ⚠️ 句子区间必须**铺满整块文本**：句末标点后的空白、结尾的换行、以及最后一行
    // 被擦成空格的公式空位，都归**前一句**。否则这些字符「既不是上一句的也不是下一句的」——
    // 永远选不中(sentOf 只能回落到相邻句)，选中高亮/笔记下划线会在行中间留出空洞，
    // 看上去像排版坏了。文本抽取侧不用管：sentenceText/selFragText 都会 trim+压缩空白。
    if (start < n) {
      if (/\S/.test(text.slice(start))) sents.push({ s: start, e: n });
      else if (sents.length) sents[sents.length - 1].e = n;   // 尾巴整段是空白(含换行) → 并进前一句
    }
    return sents;
  }
  function sentOf(item, idx) {
    const L = item.canonical.length;
    let i = idx == null ? 0 : idx;
    i = Math.max(0, Math.min(L - 1, i));
    const ss = item.sents;
    if (!ss.length) return 0;
    if (i < ss[0].s) return 0;
    for (let k = 0; k < ss.length; k++) if (i >= ss[k].s && i < ss[k].e) return k;
    return ss.length - 1;
  }
  function sentRange(item, si) {
    const ss = item.sents[si];
    return ss ? { a: ss.s, b: ss.e } : { a: 0, b: Math.max(0, item.canonical.length - 1) };
  }

  /* ================= 基础查询 ================= */
  function findItem(id) { return itemById[id] || null; }
  function sentenceText(item, si) {
    const ss = item.sents[si];
    return ss ? item.canonical.slice(ss.s, ss.e).replace(/\n+/g, " ").trim() : "";
  }
  function sentKey(item, si) { return item.id + "#" + si; }
  function copySentenceText(item, si) {
    const ss = item.sents[si];
    if (!ss) return "";
    let text = item.canonical.slice(ss.s, ss.e);
    const formulas = (item.formulaBoxes || [])
      .filter(box => box._copyRange && box._copyRange.start < ss.e && box._copyRange.end > ss.s)
      .sort((a, b) => b._copyRange.start - a._copyRange.start);
    for (const box of formulas) {
      const a = Math.max(0, box._copyRange.start - ss.s);
      const b = Math.min(text.length, box._copyRange.end - ss.s);
      text = text.slice(0, a) + "$" + box.dataset.latex + "$" + text.slice(b);
    }
    return text.replace(/\n+/g, " ").trim();
  }
  function keyParts(key) {
    const k = String(key || "");
    const p = k.lastIndexOf("#");
    return p > 0 ? { id: k.slice(0, p), si: parseInt(k.slice(p + 1), 10) || 0 } : { id: k, si: -1 };
  }
  function noteOf(key) { return (anno.notes || []).find(x => x.block_id === key) || null; }

  /* ---- 选中：句子级(可多句) ---- */
  function selCount() { return sel ? sel.sents.length : 0; }
  function selAnchor() { return sel && sel.sents.length ? sel.sents[0] : null; }   // 工具条/笔记卡锚定的那一句
  function selItems() {                                                            // 去重后的文字块(保持顺序)
    const out = [];
    if (sel) for (const r of sel.sents) if (out.indexOf(r.item) < 0) out.push(r.item);
    return out;
  }
  function selKeys() { return sel ? sel.sents.map(r => sentKey(r.item, r.si)) : []; }
  function isSelRef(item, si) {
    return !!sel && sel.sents.some(r => r.item === item && r.si === si);
  }
  function ordOf(ref) { return ref.item.ord * 1e6 + ref.si; }
  // 阅读顺序里 a→b 之间(含两端)的**整句**，跨块、跨页都支持
  function refsBetween(a, b) {
    const lo = ordOf(a) <= ordOf(b) ? a : b;
    const hi = lo === a ? b : a;
    const out = [];
    for (const pg of pages) {
      for (const it of pg.items) {
        if (it.kind !== "blk" || !it.sents.length) continue;
        if (it.ord < lo.item.ord || it.ord > hi.item.ord) continue;
        const from = it === lo.item ? lo.si : 0;
        const to = it === hi.item ? hi.si : it.sents.length - 1;
        for (let si = from; si <= to; si++) out.push({ item: it, si });
      }
    }
    return out.length ? out : [{ item: lo.item, si: lo.si }];
  }
  function pageOfItem(item) {
    for (let pi = 0; pi < pages.length; pi++)
      if (pages[pi].items.indexOf(item) >= 0) return pi + 1;
    return "";
  }

  /* ================= 渲染 ================= */
  /* ---- 主题：暗夜下正文颜色“保留色相、反转明度” ----
     PDF 里的字色是写死的（大多是黑），直接黑字配枪灰底就看不见了；
     所以按 HSL 把明度翻过来并夹到可读区间：黑字→近白、彩色字→同色系提亮。
     换主题时清缓存重算（见文件末尾的 themechange 监听）。 */
  const inkCache = new Map();
  function hex2rgb(hex) {
    let s = String(hex || "").replace("#", "").trim();
    if (s.length === 3) s = s.split("").map(ch => ch + ch).join("");
    if (s.length !== 6) return null;
    const n = parseInt(s, 16);
    return isNaN(n) ? null : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgb2hsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
    if (mx === mn) return [0, 0, l];
    const d = mx - mn;
    const s = l > .5 ? d / (2 - mx - mn) : d / (mx + mn);
    let h;
    if (mx === r) h = (g - b) / d + (g < b ? 6 : 0);
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    return [h / 6, s, l];
  }
  const _h2 = x => Math.round(x * 255).toString(16).padStart(2, "0");
  function hsl2hex(h, s, l) {
    if (s <= 0) { const v = _h2(l); return "#" + v + v + v; }
    const q = l < .5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const f = t => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    return "#" + _h2(f(h + 1 / 3)) + _h2(f(h)) + _h2(f(h - 1 / 3));
  }
  function invertInk(hex) {
    const rgb = hex2rgb(hex);
    if (!rgb) return hex;
    const [h, s, l] = rgb2hsl(rgb[0], rgb[1], rgb[2]);
    const nl = Math.min(.98, Math.max(.62, 1 - l));    // 夹住，别变成看不见的灰
    return hsl2hex(h, s <= 0 ? 0 : Math.min(1, s * .95 + .05), nl);
  }
  function pageInk(c) {
    const key = c || "#000";
    if (!isDark()) return key;
    if (!inkCache.has(key)) inkCache.set(key, invertInk(key));
    return inkCache.get(key);
  }

  function styleRun(sp, r, bodyS) {
    sp.style.fontFamily = FONT[r.fam] || FONT.serif;
    sp.style.fontSize = Math.max(r.s * r.S, 1) + "px";
    sp.style.color = pageInk(r.c);
    if (r.b) sp.style.fontWeight = "700";
    if (r.i) sp.style.fontStyle = "italic";
    if (r.up) {
      sp.style.verticalAlign = "super";
      // 上标字号：PDF 有两种排法 —— TeX 直接把字排小(7pt)，旧排版/Word 用 Ts
      // 只抬升、字号不变。统一取「本身字号」与「正文×0.7」的**较小值**：
      // 前者不会被缩两次，后者能补出上标该有的视觉大小。
      const s = bodyS > 0 ? Math.min(r.s, bodyS * 0.7) : r.s;
      sp.style.fontSize = Math.max(s * r.S, 1) + "px";
    }
  }

  /* 高亮底色/荧光：日间是半透明底色，暗夜换成“荧光笔”观感 ——
     很淡的同色底 + 高亮度霓虹字 + 外发光（不用 box-shadow，那句子的笔记下划线要留着）。 */
  function hlPaint(sp, key) {
    const c = hlColor(key);
    if (!isDark()) {
      sp.style.background = hexToRgba(c.color, c.alpha);
      return;
    }
    const rgb = hex2rgb(c.color) || [255, 214, 74];
    const [h, s] = rgb2hsl(rgb[0], rgb[1], rgb[2]);
    sp.style.background = hexToRgba(c.color, .2);
    sp.style.color = hsl2hex(h, Math.min(1, s * 1.1 + .15), .76);
    sp.style.textShadow = "0 0 8px " + hexToRgba(c.color, .95) +
                          ", 0 0 2px " + hexToRgba(c.color, .9);
  }

  /* ---- 公式框的“跟随高亮” ----
     页面级公式覆盖层（.fbox）是绝对定位的独立元素：盖在公式空位上方、底色是页面底色
     （要挡住尚未擦除的原字形）。高亮只画在文字 span 上，公式就成了高亮区域里一个
     “洞”。这里按几何把**落在高亮句子里**的公式框也刷上同一套高亮样式：与某个高亮
     span（含公式空位 span —— 它在句子区间里、同样带 .hl）有实质重叠即算命中，颜色
     沿用那个 span 的 data-hl（与 hlPaint 同一套调色盘）。 */
  function paintFormulaHl(fb, key) {
    const c = hlColor(key);
    const tint = hexToRgba(c.color, isDark() ? .2 : c.alpha);
    // 高亮色叠在页面底色上（公式框底色必须仍然不透明，继续挡原字形）
    fb.style.background = "linear-gradient(" + tint + "," + tint + "), var(--page-bg)";
    if (!isDark()) { fb.style.color = ""; fb.style.textShadow = ""; return; }
    const rgb = hex2rgb(c.color) || [255, 214, 74];
    const [h, s] = rgb2hsl(rgb[0], rgb[1], rgb[2]);
    fb.style.color = hsl2hex(h, Math.min(1, s * 1.1 + .15), .76);   // KaTeX 文本继承
    fb.style.textShadow = "0 0 8px " + hexToRgba(c.color, .95) +
                          ", 0 0 2px " + hexToRgba(c.color, .9);
  }
  function paintFormulaMarks() {
    for (const pg of pages) {
      if (!pg.el) continue;
      for (const fb of pg.el.querySelectorAll(".fbox.hltint")) {   // 先清旧色，再重算
        fb.classList.remove("hltint");
        fb.style.background = "";
        fb.style.color = "";
        fb.style.textShadow = "";
      }
      const marks = [...pg.el.querySelectorAll(".rn.hl[data-hl]")]
        .map(sp => ({ r: sp.getBoundingClientRect(), key: sp.dataset.hl }))
        .filter(m => m.r.width > 0 && m.r.height > 0);
      if (!marks.length) continue;
      for (const fb of pg.el.querySelectorAll(".fbox")) {
        const r = fb.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        let best = null, bestArea = 0;
        for (const m of marks) {
          const ox = Math.min(r.right, m.r.right) - Math.max(r.left, m.r.left);
          const oy = Math.min(r.bottom, m.r.bottom) - Math.max(r.top, m.r.top);
          if (ox <= 0 || oy <= 0) continue;
          const area = ox * oy;
          if (area > bestArea) { bestArea = area; best = m; }
        }
        if (!best) continue;
        const cx = (r.left + r.right) / 2, cy = (r.top + r.bottom) / 2;
        const inside = cx >= best.r.left - 4 && cx <= best.r.right + 4 &&
                       cy >= best.r.top - 4 && cy <= best.r.bottom + 4;
        // 中心不在高亮 span 里时，要求重叠面积够大 —— 否则贴着高亮句边缘的别的公式会被误染
        if (!inside && bestArea < r.width * r.height * 0.3) continue;
        fb.classList.add("hltint");
        paintFormulaHl(fb, best.key);
      }
    }
  }

  /* ---- 公式空位（被擦成空格的那截）与行宽适配 ----
     解析侧把公式字形换成空格（见 pdf_parser/math_font._blank_math_spans），并且**已经
     用行框宽度 `ln.w` 反推出每条空位的原宽**、随行级 `ln.gaps` 一起下发（解析侧是
     唯一实现）。空格个数也由那个宽度反推，所以自然排出来就接近原宽；前端只做两件事：

       1. 按 `data-gw`(原宽) 用 `word-spacing` **只加在空位 span** 上，把那段空格精确
          撑/收到原宽 —— 公式后面的正文因此落回原 x（加在 .ln 上会连正文词间格一起变）；
       2. 剩下的差只可能是**字体度量**（浏览器用 Georgia 替 PDF 内嵌字体），均摊到普通
          词间格上，让整行宽度与行框一致。

     前端**不再自己反推空位宽度**：`ln.gaps` 没带宽度（旧数据）时就不给空位定宽，
     那些空格按普通空格参与第 2 步。 */
  /* 本行的公式空位：`[[起, 止, 原宽 pt], …]` —— **只认解析侧下发的 `ln.gaps`**。
     空位原宽由后端「行框宽度反推」算好(见 backend/pdf_parser/math_font._line_gap_widths)，
     前端不自己反推：没有宽度(旧数据)就不把这段当空位，按普通空格排。 */
  function lineGaps(ln, text) {
    const out = [];
    for (const g of (ln && ln.gaps) || []) {
      const a = g[0] | 0, n = g[1] | 0, w = g[2];
      if (n > 0 && a >= 0 && a + n <= text.length && typeof w === "number" && w > 0)
        out.push([a, a + n, w]);
    }
    return out;
  }
  function gapAt(ld, x, y) {               // 块内区间 [x,y) 整段落在哪个空位里
    for (const g of ld.gaps || [])
      if (x >= ld.gs + g[0] && y <= ld.gs + g[1]) return g;
    return null;
  }
  function normalSpaceCount(el) {           // 行内「普通词间格」个数（空位 span 不算）
    let n = 0;
    for (const sp of el.children) {
      if (sp.classList.contains("rngap")) continue;
      const m = sp.textContent.match(/ /g);
      if (m) n += m.length;
    }
    return n;
  }
  /* 一行文字的**自然宽度**：用 Range 量(分数精度)。不要用 scrollWidth —— 它是整数、
     而且会把绝对定位的行内公式覆盖层算进去。 */
  function lineTextWidth(el) {
    const r = document.createRange();
    r.selectNodeContents(el);
    return r.getBoundingClientRect().width;
  }
  function fitLineEl(ld) {
    const el = ld.el;
    if (ld.free) return;      // Surya 块：流式排版（自然宽度），不按 PDF 行框拉伸/压缩
    const S = (ld.runs[0] && ld.runs[0].S) || 1;         // PDF pt → 布局 px
    el.style.transformOrigin = "left center";
    el.style.transform = "";
    el.style.wordSpacing = "";
    const gaps = [...el.querySelectorAll(".rngap")];
    for (const sp of gaps) sp.style.wordSpacing = "";   // 幂等：先复原上一轮的补偿再量
    const boxW = el.getBoundingClientRect().width;       // PDF 行框宽 = 原稿这一行排到哪
    let textW = lineTextWidth(el);
    for (const sp of gaps) {
      // 空位：原宽已由解析侧算好(行框宽度反推)，这里只把它撑/收到正好 gw
      // —— 空格的字宽 ≠ 原公式宽度，不撑的话公式后面的正文会整体左移。
      const n = sp.textContent.length;
      const gw = parseFloat(sp.dataset.gw) * S;          // 原公式宽度(px)
      if (!n || !(gw > 0)) continue;
      const nat = sp.getBoundingClientRect().width;
      const adv = nat / n;                               // 空格字宽
      const ws = Math.max((gw - nat) / n, -0.9 * adv);    // 别把空格压成负宽
      sp.style.wordSpacing = ws.toFixed(3) + "px";
      textW += ws * n;
    }
    // 行里剩下的差只可能是**字体度量**（浏览器用 Georgia 替 PDF 内嵌字体）：均摊到
    // 普通词间格上，让整行宽度与行框一致。空位有自己的内联值，不受行级影响。
    const rest = boxW - textW;
    const wn = normalSpaceCount(el);
    if (wn && Math.abs(rest) > 0.5) {
      const ws = clamp(rest / wn, -1.5, 3);
      el.style.wordSpacing = ws.toFixed(3) + "px";
      textW += ws * wn;
    }
    // 还有溢出的（空位收窄被下限拦住 / 整行真排不下）→ 整体压缩兜底
    if (textW > boxW + 2) el.style.transform = "scaleX(" + (boxW / textW).toFixed(4) + ")";
  }
  /* 把块内每一行的行宽适配跑一遍。
     ⚠️ 时机很重要：**必须在所有会改行宽的东西之后**。
     行内公式注入进来的是 `\(…\)` **源码**文本，比渲染出的 KaTeX 宽得多
     （`\(\sigma_p\)` 11 个字符 vs KaTeX 约 28px），先量会把整行压过头 ——
     实测本样本 25 行含公式的行全部被多压，平均 0.839 vs 应有 0.921，
     最狠的 `return of the portfolio, …` 压到 0.68（应为 0.911），
     正文跟着缩到 68% 宽，看着就是「这几行字号比别的小」；
     727 行不含公式的行「施加值 vs 应有值」完全相等，可作对照。 */
  function fitItemLines(item) {
    for (const ld of item.linesData) fitLineEl(ld);
  }

  /* ---- 重排时的“视觉锚点” ----
     缩放 / 窗口变化 / 目录与笔记栏开关都会改变页面尺寸，进而整页重建；
     重建后如果不按**同一处内容**还原滚动位置，用户就会莫名其妙地“跳页”
     （尤其是目录跳转之后 —— 旧行为是每次重排都滚回读到过的最后一页）。
     锚点记成「第几页 + 页内比例」，重排完再换算回新的像素位置。
     注意：getBoundingClientRect() 是**视口坐标**，探针也必须用视口坐标量，
     否则会算出“文档坐标减视口坐标”这种毫无意义的比值。 */
  function anchorProbe() {
    // 顶栏是 sticky 的，量它的底边：正文里第一行「看得见」的内容就在这下面一点
    const bar = document.querySelector(".topbar");
    return (bar ? bar.getBoundingClientRect().bottom : 0) + 8;
  }
  function pageAnchor() {
    if (!pages.length) return null;
    const probe = anchorProbe();
    const last = pages[pages.length - 1].el.getBoundingClientRect();
    if (last.bottom <= probe) {
      // 视口已经在正文下方（正在看文末的 AI 面板）：按“离文档底部多远”记，
      // 重排后照样停在 AI 面板那一段
      return { pi: -1, fromBottom: (document.documentElement.scrollHeight - window.scrollY) / GZ() };
    }
    let pi = 0;
    for (let i = 0; i < pages.length; i++) {
      if (pages[i].el.getBoundingClientRect().top <= probe) pi = i;
    }
    const r = pages[pi].el.getBoundingClientRect();
    return { pi, ratio: clamp((probe - r.top) / (r.height || 1), 0, 1) };
  }
  function restoreAnchor(a) {
    if (!a) return;
    if (a.pi < 0) {
      const top = document.documentElement.scrollHeight / GZ() - a.fromBottom;
      window.scrollTo({ top: toScrollTop(top), behavior: "auto" });
      return;
    }
    if (!pages[a.pi]) return;
    const r = pages[a.pi].el.getBoundingClientRect();
    const top = scrolled() + r.top + a.ratio * r.height - anchorProbe();
    window.scrollTo({ top: toScrollTop(top), behavior: "auto" });
  }

  function render(preAnchor) {
    if (!DOC) return;              // 文档还没到（缺 ?id= / 数据未加载）：别让缩放之类把这里点到抛错
    const anchor = preAnchor || pageAnchor();   // 首次渲染（还没有页面）为 null → 走“上次读到哪一页”
    const keepSel = sel ? sel.sents.map(r => ({ id: r.item.id, si: r.si })) : null;  // 重排后保留选中
    // 重排(缩放/窗口变化/批注栏开关)后保留已展开的译文卡，别让用户重新点一遍
    const keepTrans = Object.keys(itemById).filter(id => !itemById[id].tsec.hidden);
    const t = computePageTarget();
    curPageW = t.pageW;            // 先记下，避免 relayoutAll() 重复触发 render
    applyNoteChrome(t);

    pagesEl.textContent = "";
    pages.length = 0;
    Object.keys(itemById).forEach(k => delete itemById[k]);
    ordSeq = 0;
    sel = null; hideBar();

    // surya 解析的文档：块级数据（label/bbox/html），走独立的流式渲染分支
    // （见 buildSuryaItem）；其余后端仍是 PyMuPDF 的“逐行绝对定位”渲染。
    const surya = DOC.parser === "surya";
    DOC.pages.forEach((pg, pi) => {
      const S = t.pageW / pg.w;
      const pageEl = el("div", "page");
      pageEl.style.width = pg.w * S + "px";
      pageEl.style.height = pg.h * S + "px";
      pageEl.appendChild(el("div", "pagenum", "Page " + (pi + 1)));
      const pgObj = { el: pageEl, w: pg.w * S, h: pg.h * S, items: [] };

      for (const im of pg.images || []) {
        const div = el("div", "img");
        Object.assign(div.style, {
          left: im.x * S + "px", top: im.y * S + "px",
          width: im.w * S + "px", height: im.h * S + "px",
        });
        const img = el("img");
        const imageUrl = "/api/doc/" + docId + "/img/" + im.file;
        img.src = im.cache_key ? imageUrl + "?v=" + encodeURIComponent(im.cache_key) : imageUrl;
        img.alt = im.class_name || "Figure";
        div.appendChild(img);
        pageEl.appendChild(div);
        pgObj.items.push({ kind: "img", el: div, x: im.x * S, y: im.y * S, w: im.w * S, h: im.h * S });
      }

      for (const t of pg.texts || []) {
        const item = surya ? buildSuryaItem(t, S, pageEl, pi)
                           : buildTextItem(t, S, pageEl, pi);
        pgObj.items.push(item);
      }

      // detection-service-group 公式框覆盖层：按 bbox_norm × 页面显示宽高定位在页面上
      // （内容来自服务组回包的 latex，见 r2-math.js 的 buildFormulaBoxes）。
      // surya 文档没有这一层（公式来自块里的 <math>/latex），调用是空转。
      buildFormulaBoxes(pg, pageEl, S, pi);

      pagesEl.appendChild(pageEl);
      pages.push(pgObj);
    });

    // 套用标注并重画
    for (const id in itemById) { const it = itemById[id]; setAnnoRanges(it); rebuildItem(it); }
    for (const page of pages) attachFormulaBoxes(page, page.el);
    finalizeDisplayMath();     // 公式覆盖层已挂到文档上 → 量真实高度并回填块高
    finalizeInlineMath();      // 行内公式：量原字形矩形 → 顶上 KaTeX
    finalizeFormulaBoxes();    // 页面公式框：量自然尺寸 → 等比缩放到框内（等字体就绪）
    finalizeSuryaBlocks();     // surya 块：量流式内容的真实高度 → 回填 contentH / 卡片位置
    settleSuryaVertical();     // surya 块：算内容纵向避让量 syShift（relayoutPage 应用）
    const paraMode = !selTransList.length;      // 有“所选句译文”时不展开整段译文(两者互斥)
    if (showZhOn && paraMode) openAllTrans();
    if (paraMode) {
      for (const id of keepTrans) { const it = findItem(id); if (it && zh[id]) openTrans(it); }
    }
    renderAllNotes();          // 笔记：默认贴页面右侧批注栏(窗口不够宽时回到句子下方)
    renderSelTrans();          // “所选句译文”卡片(可多张；重排后保留)
    if (keepSel) {
      const refs = keepSel.map(k => ({ item: findItem(k.id), si: k.si })).filter(r => r.item);
      if (refs.length) setSelection(refs, true);
    }
    relayoutAll();
    refreshZoomLabel();
    requestAnimationFrame(() => {
      // 读完的文章等会儿会直接落到文末的 AI 面板，这里就别多滚一次了
      const toAi = META && META.status === "done" && !anchor;
      if (anchor) {
        restoreAnchor(anchor);      // 重排：停在原来那处内容上，不跳页
      } else if (!toAi && META && META.last_page) {
        // META.last_page 存的是“页码”(从 1 开始)，scrollToPage 收的是“下标”，
        // 这里必须减 1 并夹紧，否则读到最后一页时 pages[n] 为 undefined 会抛异常，
        // 连带后面的 updatePageInd() 也不执行(页码指示器一直停在 “-”)。
        scrollToPage(clamp(META.last_page - 1, 0, Math.max(0, pages.length - 1)), "instant");
      }
      if (sel) placeBar(true);
      updatePageInd();
    });
  }

  /* ============ 公式渲染：KaTeX（代码已拆到 r2-math.js） ============
     这里只做「装配」：把跨模块要共享的东西交出去，再把模块接口接回本作用域。
       · docId / findItem / itemById 是稳定引用，装配时取一次即可；
       · DOC 必须用 getter：换论文时它会被整体替换，模块里要每次取最新值；
       · katexReady 是模块内部的 let（加载完才置 true），只能按函数取。
     原来的 KATEX_* 常量、_katexP / _katexReady、各类对齐与宽度缓存都随模块走了。 */
  const {
    ensureKatex, docNeedsMath, renderInlineMath, inlineMathOf, _mathIdxAt, _rightAfterMath,
    finalizeInlineMath, _inlineMathRoom, finalizeDisplayMath, formulaBodyWidth, mathSizePx,
    katexRender, mathTexOf, mathBlockOf, formulaBaseSize, _katexBoxes,
    buildFormulaBoxes, finalizeFormulaBoxes,
    katexReady,
  } = R2Math({
    docId, findItem, itemById, fitItemLines,
    get DOC() { return DOC; },
  });

  function buildTextItem(t, S, pageEl, pi) {
    const bx = t.x * S, by = t.y * S;
    const blk = el("div", "blk");
    Object.assign(blk.style, { left: bx + "px", top: by + "px", width: t.w * S + "px" });
    blk.dataset.id = t.id;
    // 公式块整块内容都在 `latex` 里 —— **不建任何行元素**：后端也不再下发原字形
    // (旧数据里的 `lines[*].runs` 同样不渲染)。于是「公式覆盖掉原文字形」是结构上的
    // 事实：不依赖 `.has-math .ln{visibility:hidden}`，KaTeX 没渲染成也不会露出
    // 乱码碎片(CMEX 的大括号被读成 n/o、拆散的上下标)。`lines` 只用来摆覆盖层与
    // 按原文主体宽度缩放。
    const mb = mathBlockOf(t);

    const linesData = [];
    let contentH = t.h * S;                  // 公式块没有行元素 → 块高直接用外框
    if (!mb) {
      // ⚠️ 新数据里一行**只有一个 run**（后端 `text_layer._merge_line_runs`），
      // 所以下面循环里每行只会切出一个 `<span class="rn">`。
      for (const ln of (t.lines || [])) {
        const dy = (ln.y - t.y) * S;         // 该行相对文字块顶部的偏移(px)
        const ld = {
          ln, dy, runs: (ln.runs || []).map(r => Object.assign({}, r, { S, g: 0 })),
          el: null, text: "", gs: 0,
        };
        ld.text = ld.runs.map(r => r.t).join("");
        const lineEl = el("div", "ln");
        Object.assign(lineEl.style, {
          left: (ln.x - t.x) * S + "px", top: dy + "px",
          width: ln.w * S + "px",
        });
        ld.el = lineEl;
        blk.appendChild(lineEl);
        contentH = Math.max(contentH, (ln.y - t.y + ln.h) * S);
        linesData.push(ld);
      }
    }

    // ---------- 公式块：KaTeX 同步渲染 LaTeX ----------
    // 内容当场就渲好(不用等挂载)；块高要等挂到文档上才量得准，所以先记下来，
    // 由 render() 末尾的 finalizeDisplayMath() 回填。
    if (mb) {
      const box = el("div", "mathbox");
      box.dataset.latex = mb.latex;
      box.style.fontSize = mathSizePx(formulaBaseSize(linesData, pi, t.lines) * S) + "px";
      if (katexReady() && katexRender(box, mb.latex)) {
        // 正常路径
      } else {
        // KaTeX 没加载 / 这条 LaTeX 有问题 → 显示 LaTeX 源码兜底。
        // **不再回退显示 PDF 原字形**：那些字形本身就是碎片与错字，比源码更难读，
        // 而且"有的公式遮住了、有的露出字形"正是这条回退造成的。
        box.textContent = mb.latex;
        box.classList.add("failed");
      }
      box._blk = blk;
      box._contentH = contentH;
      box._origW = formulaBodyWidth(t) * S;    // 主体宽度(不含公式编号)，适配用
      if (box._hasTag) box.style.width = "100%";   // 编号贴到原框右边缘(与 PDF 一致)
      blk.appendChild(box);
      _katexBoxes.add(box);
    }
    blk.style.height = contentH + "px";

    // canonical：行文本原样拼接，行间一个 '\n'（与前端渲染的字符一一对应）
    let g = 0;
    for (const ld of linesData) {
      ld.gs = g;
      for (const r of ld.runs) { r.g = g; g += r.t.length; }
      g += 1; // '\n'
      // 公式空位：解析侧下发的精确区间与**原宽**（见 lineGaps）；行宽适配
      // 按那个原宽单独把空位撑开，让公式后面的正文落回原 x。
      // ⚠️ 用 `ld.ln`（建行时存下的原始行数据）—— `ln` 只是上面那个 for...of 的循环变量，
      // 出了那个循环就没了。
      ld.gaps = lineGaps(ld.ln, ld.text);
    }
    const item = {
      kind: "blk", id: t.id, pi, el: blk, text: t.text,
      x: bx, y: by, w: t.w * S, h: t.h * S, contentH,
      latex: (mb && mb.latex) || "",          // 公式源码：复制时给这个，不给字形文本
      linesData, canonical: "", sents: [],
      markerRanges: [], selRanges: [], shift: 0, expH: 0,
      ord: ordSeq++,
    };
    item.canonical = linesData.map(ld => ld.text + "\n").join("");
    item.bodyRun = bodyRunOf(linesData);       // 块内主样式（修正跨公式 run 的尾部正文）
    // 行内公式：现在是**旧数据的兼容路径** —— 旧版解析器会把匹配到的 LaTeX 内联进
    // 文本(`\(…\)` / `$$…$$`)，由 `rebuildItem` 末尾的 `renderInlineMath` 就地换
    // KaTeX。新数据（detection-service-group）没有内联：公式是**页面级公式框**，由
    // `buildFormulaBoxes` 按 bbox_norm 叠覆盖层。
    // 更旧的 Surya 路径靠块 HTML 里的 `<math>` + 覆盖层，新数据没有 `t.html`，
    // `inlineMathOf` 直接返回 null（变成空转，保留只为兼容旧 doc.json）。
    // 行内公式对齐：公式块不走这条（它的 html 就是 `<math display="block">…` 一整块数学，
    // 已经由行外覆盖层渲染了；再当行内渲染一遍会叠成**重影**）。
    item.inlineMath = mb ? null : inlineMathOf(t, item.canonical);
    // 公式块没有可读文本 → 没有句子可切/可翻译/可标注，整段跳过。
    item.sents = mb ? [] : splitSentences(item.canonical).map((sg, si) =>
      ({ si, s: sg.s, e: sg.e, key: item.id + "#" + si,
        text: item.canonical.slice(sg.s, sg.e).replace(/\n+/g, " ").trim() }));

    // 译文 + 内嵌笔记展开区(都在句子下方，同一套“顶开版式”)：
    // 译文卡片在上、笔记卡片在下，多句笔记按句序依次排列
    const exp = el("div", "exp");
    exp.style.top = contentH + "px";
    const tsec = el("div", "card-in tsec");
    tsec.hidden = true;
    exp.appendChild(tsec);
    // 多选句子时的“只翻译所选句”卡片列表：与整段译文卡互斥，但彼此可以同时存在
    const stbox = el("div", "stbox");
    stbox.hidden = true;
    exp.appendChild(stbox);
    const nbox = el("div", "nbox");
    nbox.hidden = true;
    // 卡片内的点击(编辑/删除/选中文字)不应被当成“点了正文”
    nbox.addEventListener("click", e => e.stopPropagation());
    stbox.addEventListener("click", e => e.stopPropagation());
    exp.appendChild(nbox);
    blk.appendChild(exp);
    item.exp = exp; item.tsec = tsec; item.stbox = stbox; item.nbox = nbox;

    pageEl.appendChild(blk);
    itemById[t.id] = item;
    return item;
  }

  /* ============ Surya 2 的块级渲染（与 PyMuPDF 的“逐行绝对定位”是两套） ============
     surya 后端下发的版面数据是**块级**的：每块只有 label / bbox / html（块内没有行几何）。
     所以这里不做逐行还原，而是：
       · `.blk` 仍按 bbox 绝对定位（阅读顺序与版面关系不变）；
       · 块内文字**正常流式排版**：字号/行高由解析侧估算（`t.size` / `t.line_h`，单位 pt），
         标题加粗放大、图注/脚注小一号（CSS `.sy-*`），长段落自动换行；
       · 行内公式：解析侧给了精确的字符区间（`t.math`，含 LaTeX 源码）—— 原字形
         就地换成 KaTeX（suryaInlineMath），正文围绕公式自然重排；
       · 独立公式（kind="formula" + latex）：KaTeX 块渲染（与 PyMuPDF 公式块同一套收尾）；
       · 表格（kind="table"）：白名单渲染 Surya 的 `<table>` HTML，不参与句子选中。
     文本块的内容仍由通用 `rebuildItem` 生成 —— 选中/高亮/笔记重建后样式与公式都不丢。 */
  function buildSuryaItem(t, S, pageEl, pi) {
    const bx = t.x * S, by = t.y * S;
    const blk = el("div", "blk sy");
    if (t.kind) blk.classList.add("sy-" + t.kind);
    if (t.kind === "heading") blk.classList.add("sy-l" + (t.level || 2));
    Object.assign(blk.style, { left: bx + "px", top: by + "px", width: t.w * S + "px" });
    blk.dataset.id = t.id;
    if (t.label) blk.dataset.label = t.label;
    // 对齐方式：解析侧下发 t.align（新数据）；旧数据缺字段时按同一规则本地兜底 ——
    // 正文类（text/list/caption/footnote）两端对齐（CSS `[data-align="justify"]`），
    // 让行右缘像原 PDF（LaTeX justified 排版）一样齐平到栏边界；标题/页眉页脚等
    // 保持左对齐（单行，两端对齐无意义）。末行不拉伸，与原 PDF 一致。
    blk.dataset.align = (t.align === "justify" || t.align === "left"
                         || t.align === "center" || t.align === "right")
      ? t.align
      : ((t.kind === "text" || t.kind === "list" ||
          t.kind === "caption" || t.kind === "footnote") ? "justify" : "left");

    const size = t.size > 0 ? t.size : 10;        // 解析侧估的字号（pt）
    blk.style.fontSize = size * S + "px";
    // 竖排文本（旋转 90° 的页眉/水印，如 arXiv 侧边戳）：Surya 不给旋转信息，用
    // “又高又窄 + 短文本”判 —— 内容整体旋转、沿条带自下而上排（见 CSS .sy-vertical）
    if (t.h > t.w * 3 && (t.text || "").length >= 2 && (t.text || "").length <= 80) {
      blk.classList.add("sy-vertical");
    }

    const mb = mathBlockOf(t);                     // kind="formula" + latex → KaTeX 块
    const linesData = [];
    let contentH = t.h * S;        // 初值=bbox 高；挂到文档后由 finalizeSuryaBlocks 回填实测高
    if (mb) {
      const box = el("div", "mathbox");
      box.dataset.latex = mb.latex;
      box.style.fontSize = mathSizePx(size * S) + "px";
      if (katexReady() && katexRender(box, mb.latex)) {
        // 正常路径
      } else {
        box.textContent = mb.latex;                // KaTeX 未就绪/LaTeX 有问题 → 源码兜底
        box.classList.add("failed");
      }
      box._blk = blk;
      box._contentH = contentH;
      box._origW = t.w * S;                        // 公式主体宽 = Surya 给的外框宽
      if (box._hasTag) box.style.width = "100%";
      blk.appendChild(box);
      _katexBoxes.add(box);
    } else if (t.kind === "table") {
      const tb = suryaTableEl(t.html);
      const title = suryaTableTitle(t);
      if (title) blk.appendChild(title);
      blk.appendChild(tb || el("div", "ln syln", t.text || ""));
    } else {
      // 文本块：唯一一条“流式行”。类名要含 ln（行内公式的让位机制按 .ln 找行），
      // `ld.free` 让 fitLineEl 跳过（这里的宽度是排版自然宽，不是 PDF 行框）。
      const lineEl = el("div", "ln syln");
      const lh = t.line_h > 0 ? t.line_h : size * 1.42;
      lineEl.style.lineHeight = lh * S + "px";
      blk.appendChild(lineEl);
      const ln = (t.lines && t.lines[0]) || { runs: [] };
      const ld = {
        ln, dy: 0, free: true, gaps: [],
        // 补上 run 级的字号/字体（后端 run 只有 t/b/i/up；styleRun 与公式基准字号都按 r.s 算）
        runs: (ln.runs || []).map(r => Object.assign({ s: size, fam: "serif" }, r, { S, g: 0 })),
        el: lineEl, text: "", gs: 0,
      };
      ld.text = ld.runs.map(r => r.t).join("");
      linesData.push(ld);
    }
    blk.style.height = contentH + "px";

    // canonical：行文本原样拼接，行末一个 '\n'（与 PyMuPDF 路径同口径）
    let g = 0;
    for (const ld of linesData) {
      ld.gs = g;
      for (const r of ld.runs) { r.g = g; g += r.t.length; }
      g += 1; // '\n'
    }
    const item = {
      kind: "blk", id: t.id, pi, el: blk, text: t.text || "",
      x: bx, y: by, w: t.w * S, h: t.h * S, contentH,
      latex: (mb && mb.latex) || "",
      linesData, canonical: "", sents: [],
      markerRanges: [], selRanges: [], shift: 0, expH: 0,
      ord: ordSeq++, surya: true,
      syHeading: t.kind === "heading",      // 标题：finalizeSuryaBlocks 里按需放宽不折行
    };
    item.canonical = linesData.map(ld => ld.text + "\n").join("");
    item.bodyRun = bodyRunOf(linesData)
      || { s: size, fam: "serif", i: false, b: false, up: false };
    // 行内公式：解析侧给的是 text 里的**精确字符区间**（比旧路径的“HTML 对齐”稳）。
    // 渲染交给 suryaInlineMath：`.mspan` 原字形就地换成 KaTeX（见 rebuildItem 末尾）；
    // 公式块不重复渲染（会叠重影）。
    item.inlineMath = (!mb && linesData.length && t.math && t.math.length)
      ? suryaMathRanges(t, item.canonical) : null;
    item.sents = mb ? [] : splitSentences(item.canonical).map((sg, si) =>
      ({ si, s: sg.s, e: sg.e, key: item.id + "#" + si,
        text: item.canonical.slice(sg.s, sg.e).replace(/\n+/g, " ").trim() }));
    // 表格没有可切句的 .rn 结构：点它不该弹“翻译/标注”工具条（点击分支按这个清选中）
    if (t.kind === "table") item.noSelect = true;

    // 译文 + 内嵌笔记展开区（与 buildTextItem 同一套约定）
    const exp = el("div", "exp");
    exp.style.top = contentH + "px";
    const tsec = el("div", "card-in tsec");
    tsec.hidden = true;
    exp.appendChild(tsec);
    const stbox = el("div", "stbox");
    stbox.hidden = true;
    exp.appendChild(stbox);
    const nbox = el("div", "nbox");
    nbox.hidden = true;
    nbox.addEventListener("click", e => e.stopPropagation());
    stbox.addEventListener("click", e => e.stopPropagation());
    exp.appendChild(nbox);
    blk.appendChild(exp);
    item.exp = exp; item.tsec = tsec; item.stbox = stbox; item.nbox = nbox;

    pageEl.appendChild(blk);
    itemById[t.id] = item;
    return item;
  }

  /* Surya 块的行内数学区间（解析侧给的 text 偏移）→ 覆盖层用的 {a,b,latex}。
     越界/空的丢掉；区间相对 `t.text`，canonical 只在行末多一个 '\n'，不影响行内区间。 */
  function suryaMathRanges(t, canonical) {
    const out = [];
    for (const m of (t.math || [])) {
      const a = m && m.a | 0, b = m && m.b | 0;
      const latex = String((m && m.latex) || "").trim();
      if (!latex || !(b > a) || a < 0 || b > canonical.length) continue;
      out.push({ a, b, latex });
    }
    return out.length ? out : null;
  }

  /* Surya 块的行内公式：把 `.mspan` 原字形**就地换成 KaTeX**。
     不走 .mathbox.inline 覆盖层 —— 那套是给“逐行绝对定位、行宽固定”的 PyMuPDF 版面
     设计的（覆盖层 + mspacer 让位）；流式排版里行会重排，覆盖层定位不稳（实测公式会
     压到隔壁词上）。就地渲染后正文围绕真公式自然重排，宽度与基线都交给浏览器。
     渲染失败（KaTeX 未就绪 / LaTeX 有错）就保持原字形 —— 至少可读，不会整段空白。
     `rebuildItem` 每次重建都会重新走一遍（幂等）。 */
  function suryaInlineMath(item) {
    const list = item.inlineMath;
    if (!list || !item.el) return;
    const byMi = new Map();
    for (const sp of item.el.querySelectorAll(".mspan[data-mi]")) {
      const mi = sp.dataset.mi;
      if (!byMi.has(mi)) byMi.set(mi, []);
      byMi.get(mi).push(sp);
    }
    if (!byMi.size) return;
    const S = (item.linesData[0] && item.linesData[0].runs[0] &&
               item.linesData[0].runs[0].S) || 1;
    const sizePt = (item.bodyRun && item.bodyRun.s) || 10;
    for (const [miKey, spans] of byMi) {
      const f = list[parseInt(miKey, 10)];
      if (!f) continue;
      const host = document.createElement("span");
      host.className = "katex-inline";
      host.dataset.mi = miKey;
      host.dataset.latex = f.latex;
      host.style.fontSize = mathSizePx(sizePt * S) + "px";
      let ok = false;
      try { ok = katexReady() && katexRender(host, f.latex, false); }
      catch (e) { ok = false; }
      if (!ok) continue;                       // 保底：原字形仍在（没被删）
      const first = spans[0];
      // 高亮/选中/笔记下划线是加在 span 上的：跟着公式一起搬到 KaTeX 宿主上
      for (const cls of ["hl", "hasnote", "sel"])
        if (first.classList.contains(cls)) host.classList.add(cls);
      if (first.dataset.hl) host.dataset.hl = first.dataset.hl;
      for (const st of ["background", "color", "textShadow"])
        if (first.style[st]) host.style[st] = first.style[st];
      first.parentNode.insertBefore(host, first);
      for (const sp of spans) sp.remove();     // 原字形不再占位（宽度由 KaTeX 决定）
    }
  }

  /* Surya 块的块内换行：canonical 里的 '\n'（`</li>`、`<br>` 等）在 HTML 流里会被
     折叠成空格，列表项/表格行会连成一行。把 `\n` 拆成 <br> —— 只动文本节点，
     span 上的 data-g / 高亮 class 都保持原样（选句/高亮仍按字符区间走）。
     KaTeX 子树跳过（MathML 文本节点里的换行不动它）。 */
  function suryaLineBreaks(item) {
    const line = item.el && item.el.querySelector(".syln");
    if (!line) return;
    const walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT, null);
    const jobs = [];
    let node;
    while ((node = walker.nextNode()))
      if (node.nodeValue.indexOf("\n") >= 0) jobs.push(node);
    for (const tn of jobs) {
      if (tn.parentNode && tn.parentNode.closest &&
          tn.parentNode.closest(".katex-inline, .katex")) continue;
      const parts = tn.nodeValue.split("\n");
      const frag = document.createDocumentFragment();
      parts.forEach((p, i) => {
        if (i) frag.appendChild(document.createElement("br"));
        if (p) frag.appendChild(document.createTextNode(p));
      });
      tn.parentNode.replaceChild(frag, tn);
    }
  }

  /* Surya 的表格 HTML → 安全 DOM（白名单）。
     HTML 来自外部推理服务，绝不直接 innerHTML：只放行表格结构与基础行内标签，
     属性只留 colspan/rowspan（表格布局必需）。拿不到 <table> 时返回 null。 */
  const SY_TABLE_TAGS = { TABLE: 1, THEAD: 1, TBODY: 1, TFOOT: 1, TR: 1, TD: 1, TH: 1,
                          CAPTION: 1, B: 1, I: 1, EM: 1, STRONG: 1, SUP: 1, SUB: 1,
                          BR: 1, SPAN: 1 };
  function suryaTableTitle(t) {
    const html = String(t && t.html || "");
    const label = String(t && (t.label || t.raw_label) || "").trim();
    if (!label || /<caption\b/i.test(html)) return null;
    return el("div", "sytable-title", label);
  }
  function suryaTableEl(html) {
    const s = String(html || "");
    if (!s || s.indexOf("<table") < 0) return null;
    let doc = null;
    try { doc = new DOMParser().parseFromString(s, "text/html"); }
    catch (e) { return null; }
    const src = doc && doc.body && doc.body.querySelector("table");
    if (!src) return null;
    const wrap = el("div", "sytable");
    wrap.appendChild(syCleanNode(src));
    return wrap;
  }
  function syCleanNode(node) {
    if (node.nodeType === 3) return document.createTextNode(node.nodeValue);
    if (node.nodeType !== 1) return document.createTextNode("");
    const tag = node.tagName.toUpperCase();
    if (!SY_TABLE_TAGS[tag]) {          // 不认识的标签：剥壳、保留里面的内容
      const frag = document.createDocumentFragment();
      for (const c of node.childNodes) frag.appendChild(syCleanNode(c));
      return frag;
    }
    const out = document.createElement(tag.toLowerCase());
    if (tag === "TD" || tag === "TH") {
      for (const name of ["colspan", "rowspan"]) {
        const v = parseInt(node.getAttribute(name), 10);
        if (v > 1 && v <= 100) out.setAttribute(name, String(v));
      }
    }
    for (const c of node.childNodes) out.appendChild(syCleanNode(c));
    return out;
  }

  /* Surya 块（流式排版）收尾：内容高度要等挂到文档上才量得准 —— 回填
     item.contentH / `.blk` 高度 / 译文卡位置（relayout 的避让都按 contentH 算）。
     **只放大不缩小**：文本量比原框多时按实际高度长出去（解析侧已按块框做过
     缩字适配，溢出很小）。公式/表格块不走这里（高度由各自机制管）。 */
  function finalizeSuryaBlocks() {
    for (const id in itemById) {
      const it = itemById[id];
      if (!it.surya || it.kind !== "blk" || !it.linesData.length) continue;
      if (it.syHeading) {
        // 标题不折行：量内容自然宽度，需要时把块放宽（最多到原框的 2.4 倍，长标题
        // 顶多折一行）。⚠️ CSS 的 `min(max-content, X)` 不合法（关键词不能进 min()），
        // 只能挂载后量。窄框标题（居中的 "Abstract" 之类）折行会压到下面正文上。
        const lineEl = it.linesData[0].el;
        const prevW = lineEl.style.width;
        lineEl.style.width = "max-content";     // 量单行自然宽（块级盒 max-content 有效）
        const natural = lineEl.offsetWidth;
        lineEl.style.width = prevW;
        if (natural > it.w + 1) {
          // +4px 余量：自然宽是整数，差零点几个像素就会在词中间折行（"Abstract" 实测）
          it.el.style.width = Math.min(natural + 4, it.w * 2.4).toFixed(1) + "px";
        }
      }
      let h = 0;
      for (const ld of it.linesData) h = Math.max(h, ld.el.offsetHeight + (ld.dy || 0));
      h = Math.max(h, it.h);
      it.contentH = h;
      it.el.style.height = h + "px";
      if (it.exp) it.exp.style.top = h + "px";
    }
  }

  /* Surya 块的**纵向避让**：检测框纵向膨胀 + 行内 KaTeX 撑高行盒，实测内容
     常常比块框高 —— “文本满框”时上一块的文字会与下一块叠在一起。这里在
     finalizeSuryaBlocks 量完真实高度后，按「同栏、y 升序」逐块算一个**只下移**
     的位移量 `it.syShift`：本块顶不得高于同栏任一已排块的内容底 + SY_VGAP。
     同栏 = x 区间重叠 ≥ 两者较窄宽的一半（双栏左右栏互不影响；跨栏块是
     下方两栏共同的前邻）；级联自然收敛 —— 按 y 序处理，前块用的都是避让后的底。
     ⚠️ 位移量每轮都**基于原始 it.y 重算**（且不写回数据）：缩放/窗口变化会
     整页重建，每次都以原始几何为基准，不会累积漂移；应用交给 relayoutPage
     的位移链（syShift 是它的起点，与“卡片顶开”量叠加，transform / 页面高度 /
     笔记卡位置因此都自动跟随）。 */
  const SY_VGAP = 2;      // px：避让后相邻块内容之间至少保留的空隙
  function settleSuryaVertical() {
    const byPage = new Map();
    for (const id in itemById) {
      const it = itemById[id];
      if (!it.surya || it.kind !== "blk") continue;
      if (it.el.classList.contains("sy-vertical")) continue;   // 竖排条带不参与
      let arr = byPage.get(it.pi);
      if (!arr) byPage.set(it.pi, arr = []);
      arr.push(it);
    }
    for (const list of byPage.values()) {
      list.sort((a, b) => (a.y - b.y) || (a.x - b.x));
      const settled = [];           // [{x, w, bottom}]：已排块（bottom 含避让量）
      for (const it of list) {
        let need = -Infinity;
        for (const s of settled) {
          const ov = Math.min(s.x + s.w, it.x + it.w) - Math.max(s.x, it.x);
          if (ov < Math.min(s.w, it.w) * 0.5) continue;        // 不同栏
          need = Math.max(need, s.bottom + SY_VGAP);
        }
        it.syShift = Math.max(0, need - it.y);      // 只下移，绝不上移
        settled.push({ x: it.x, w: it.w,
                       bottom: it.y + it.syShift + it.el.offsetHeight });
      }
    }
  }

  /* ============ 高亮调色盘：3 个默认色 + 3 个自定义色 ============
     自定义色**随账号存在服务端**（/api/prefs）—— 换账号会重新加载，不会互相串；
     文档里存的是色位（c1/c2/c3），颜色本身只是当前账号的偏好。 */
  const HL_STORE_OLD = "paperReader.hlColors";   // 旧版本的浏览器本地存储（升级后不再使用）
  const hlPalette = [
    { key: "yellow", color: "#ffd64a", alpha: 0.55, custom: false },
    { key: "green",  color: "#60c994", alpha: 0.50, custom: false },
    { key: "pink",   color: "#ff8a9e", alpha: 0.55, custom: false },
    { key: "c1", color: "#7ec8f2", alpha: 0.50, custom: true },
    { key: "c2", color: "#c4a3f0", alpha: 0.50, custom: true },
    { key: "c3", color: "#ffb26b", alpha: 0.50, custom: true },
  ];
  const HL_CUSTOM_AT = 3;                       // 从第几个开始是自定义色
  function hlColor(key) {                       // 色键 -> 调色盘项(未知键回退黄色)
    return hlPalette.find(c => c.key === key) || hlPalette[0];
  }
  function hexToRgba(hex, a) {
    let s = String(hex || "").replace("#", "");
    if (s.length === 3) s = s.split("").map(ch => ch + ch).join("");
    const n = parseInt(s, 16);
    if (s.length !== 6 || isNaN(n)) return "rgba(255,214,74," + a + ")";
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }
  function applyHlColors(list) {
    if (!Array.isArray(list)) return;
    list.slice(0, 3).forEach((hex, i) => {
      if (/^#[0-9a-f]{6}$/i.test(hex || "")) hlPalette[HL_CUSTOM_AT + i].color = hex;
    });
  }
  // 偏好（自定义高亮色 / 显示译文开关）随账号存在服务端：进页时拉一次
  async function loadPrefs() {
    try {
      const p = await api("/api/prefs");
      applyHlColors(p && p.hl_colors);
      if (p && typeof p.show_zh === "boolean") {
        showZhOn = p.show_zh;
        $("chkShowZh").checked = p.show_zh;
      }
    } catch (e) { /* 用默认值 */ }
    // 清掉旧版本的浏览器本地值，避免“看起来还在用本地颜色”的误会
    try { localStorage.removeItem(HL_STORE_OLD); localStorage.removeItem(ZH_STORE_OLD); } catch (e) { /* ignore */ }
  }

  function savePrefs(patch) {
    api("/api/prefs", { method: "POST", json: patch })
      .catch(err => showNotice("偏好设置没能保存到服务端：" + err.message, true));
  }

  let hlSaveTimer = null;
  function saveHlColors() {
    // 存到服务端（跟着账号走）。取色器拖动会连续触发 input，这里防抖一下。
    clearTimeout(hlSaveTimer);
    hlSaveTimer = setTimeout(() => {
      savePrefs({ hl_colors: hlPalette.slice(HL_CUSTOM_AT).map(c => c.color) });
    }, 400);
  }
  // 渲染色块 + 同步 3 个取色器（自定义色改完立即生效并记住）
  function renderHmenuSwatches() {
    const wrap = $("hmenu").querySelector(".swatches");
    wrap.textContent = "";
    for (const c of hlPalette) {
      const s = el("span", "sw" + (c.custom ? " custom" : ""));
      s.dataset.c = c.key;
      s.style.background = c.color;
      if (isDark()) s.style.boxShadow = "0 0 9px " + c.color;   // 暗夜下色块也发点荧光，方便选
      s.title = c.custom ? "自定义高亮色（用下方颜色块修改）" : c.key;
      wrap.appendChild(s);
    }
    document.querySelectorAll("#hmenu .swedit input").forEach((inp, i) => {
      inp.value = hlPalette[HL_CUSTOM_AT + i].color;
    });
    markCurSwatch();
  }
  function bindHlColorInputs() {
    document.querySelectorAll("#hmenu .swedit input").forEach((inp, i) => {
      inp.addEventListener("input", () => {
        hlPalette[HL_CUSTOM_AT + i].color = inp.value;
        saveHlColors();
        renderHmenuSwatches();
      });
    });
  }

  /* ============ 标注标记(高亮/下划线/选中) ============ */
  function setAnnoRanges(item) {
    const ranges = [];
    // 高亮(句子级)：颜色从调色盘取(支持 3 个自定义色)，用 inline 底色渲染
    for (const [key, c] of Object.entries(anno.highlights || {})) {
      const p = keyParts(key);
      if (p.id !== item.id || p.si < 0 || p.si >= item.sents.length) continue;
      const rg = sentRange(item, p.si);
      ranges.push({ a: rg.a, b: rg.b, cls: ["hl"], hl: c && c.color });
    }
    // 笔记句子加下划线
    const noteKeys = new Set();
    for (const x of anno.notes || []) noteKeys.add(x.block_id);
    for (const key of noteKeys) {
      const p = keyParts(key);
      if (p.id !== item.id || p.si < 0 || p.si >= item.sents.length) continue;
      const rg = sentRange(item, p.si);
      ranges.push({ a: rg.a, b: rg.b, cls: ["hasnote"] });
    }
    item.markerRanges = ranges;
  }

  /* 块内“主字号/主样式”：取该块里字符数最多的那个 run。
     用途：修正被 PyMuPDF 标错大小的正文片段——跨公式的 run 会整条带上标/下标字号
     （例：`t,n) is the value vector of formu-` 整条是 7pt 斜体），切走公式后
     剩下的正文就会明显变小。 */
  function bodyRunOf(linesData) {
    const cnt = new Map();
    for (const ld of linesData)
      for (const r of ld.runs) {
        const k = r.s + "|" + r.fam + "|" + (r.i ? 1 : 0) + "|" + (r.b ? 1 : 0);
        cnt.set(k, (cnt.get(k) || 0) + Math.max(1, (r.t || "").length));
      }
    let best = null, n = 0;
    for (const [k, v] of cnt) if (v > n) { n = v; best = k; }
    if (!best) return null;
    const p = best.split("|");
    return { s: parseFloat(p[0]), fam: p[1], i: p[2] === "1", b: p[3] === "1", up: false };
  }

  function rebuildItem(item) {
    if (item.kind !== "blk") return;
    const marks = item.markerRanges || [];
    const selRs = item.selRanges || [];
    const bodyS = item.bodyRun ? item.bodyRun.s : 0;   // 块内主字号，上标/小字修正都按它算
    for (const ld of item.linesData) {
      const elEl = ld.el;
      elEl.textContent = "";
      for (const run of ld.runs) {
        const gs = run.g, ge = run.g + run.t.length;
        const cuts = new Set([gs, ge]);
        const hit = [];
        for (const m of marks) {
          const a = Math.max(gs, m.a), b = Math.min(ge, m.b);
          if (a < b) { cuts.add(a); cuts.add(b); hit.push({ a, b, cls: m.cls, hl: m.hl }); }
        }
        for (const sr of selRs) {
          const a = Math.max(gs, sr.a), b = Math.min(ge, sr.b);
          if (a < b) { cuts.add(a); cuts.add(b); hit.push({ a, b, cls: ["sel"] }); }
        }
        // 行内公式的边界也切成独立 span，才能单独隐藏/测量
        for (const im of (item.inlineMath || [])) {
          const a = Math.max(gs, im.a), b = Math.min(ge, im.b);
          if (a < b) { cuts.add(a); cuts.add(b); }
        }
        // 公式空位（连续空格）也切出来：fitLineEl 要单独给这些 span 补宽度
        for (const g of (ld.gaps || [])) {
          const a = Math.max(gs, ld.gs + g[0]), b = Math.min(ge, ld.gs + g[1]);
          if (a < b) { cuts.add(a); cuts.add(b); }
        }
        const pts = [...cuts].sort((x, y) => x - y);
        for (let k = 0; k < pts.length - 1; k++) {
          const x = pts[k], y = pts[k + 1];
          if (y <= x) continue;
          const cls = new Set();
          let hl = null;
          for (const m of hit) {
            if (m.a <= x && y <= m.b) { m.cls.forEach(c => cls.add(c)); if (m.hl) hl = m.hl; }
          }
          const sp = el("span", "rn" + (cls.size ? " " + [...cls].join(" ") : ""));
          sp.dataset.g = x;
          const gp = gapAt(ld, x, y);
          if (gp) {                                          // 空位：行宽适配时按原宽单独补宽
            sp.classList.add("rngap");
            if (gp[2] > 0) sp.dataset.gw = gp[2];
          }
          styleRun(sp, run, bodyS);
          // 公式之后的正文被 PyMuPDF 误标成上标（0.7 倍）→ 按块内主样式重绘。
          // 只改“写着像单词”的片段，真正的上标（`2`、`(i)`、`*`）不动。
          // 注：解析侧现在会先校验基线再落 up（见 pdf_parser._is_super），这里主要
          // 兼容**重解析前**的旧页面数据。
          const piece = run.t.slice(x - gs, y - gs);
          const smallish = run.up || (bodyS > 0 && run.s < bodyS * 0.85);
          if (item.bodyRun && smallish && (/[A-Za-z]{2,}/.test(piece) || /^[.,;:]+$/.test(piece)) &&
              _rightAfterMath(item.inlineMath, gs, ge, x))
            styleRun(sp, Object.assign({}, run, item.bodyRun), bodyS);
          if (hl) { hlPaint(sp, hl); sp.dataset.hl = hl; }   // 公式框高亮按 data-hl 取同一色
          const mi = _mathIdxAt(item.inlineMath, x, y);
          if (mi >= 0) {
            sp.classList.add("mspan");
            sp.dataset.mi = mi;
            // 只隐“已经真渲染出来”的那几条（item.mathDone），否则渲染失败的那条
            // 会连原字形一起藏掉 → 公式直接消失。
            if (item.mathDone && item.mathDone.has(mi)) sp.classList.add("mspan-hidden");
          }
          sp.textContent = piece;
          elEl.appendChild(sp);
        }
      }
    }
    renderInlineMath(item.el);          // 行内数学：\(…\) / $$…$$ 就地换成 KaTeX
    if (item.surya) {
      suryaInlineMath(item);            // Surya 块：.mspan 原字形 → 就地 KaTeX
      suryaLineBreaks(item);            // Surya 块：canonical 里的 '\n' → <br>（列表/表格行）
    }
    // 重建会把 .ln 里的内容（连同上一轮插的 spacer）清掉 → 让位要重算，
    // 否则被推开的正文会“缩”回公式底下。没渲染过公式的块里没有 .mathbox.inline，是空转。
    _inlineMathRoom(item);
    // 行宽适配放最后：此时行内已经是 KaTeX 成品（不是 `\(…\)` 源码），量出来的才准。
    // 公式覆盖层是绝对定位的，不占行宽，所以这一步之后行宽不会再变。
    fitItemLines(item);
  }

  /* ================= 选中句子 ================= */
  function caretInfo() {
    const s = window.getSelection();
    if (!s || !s.anchorNode) return null;
    const node = s.anchorNode.nodeType === 3 ? s.anchorNode.parentElement : s.anchorNode;
    const gEl = node && node.closest ? node.closest("[data-g]") : null;
    if (!gEl) return null;
    const blkEl = gEl.closest(".blk");
    const item = blkEl && findItem(blkEl.dataset.id);
    if (!item) return null;
    let off = 0;
    if (s.anchorNode.nodeType === 3) off = s.anchorOffset || 0;
    return { item, idx: (parseInt(gEl.dataset.g, 10) || 0) + off };
  }

  function fallbackSentence(item, y) {
    // 依据点击的纵向位置，估算句子
    if (!item.sents.length) return 0;
    let best = 0, bestD = 1e9;
    for (let si = 0; si < item.sents.length; si++) {
      const r = approxSentenceRect(item, si);
      const d = y < r.top ? r.top - y : (y > r.bottom ? y - r.bottom : 0);
      if (d < bestD) { bestD = d; best = si; }
    }
    return best;
  }
  function approxSentenceRect(item, si) {
    const rg = sentRange(item, si);
    const l0 = item.linesData[lineOfChar(item, rg.a)].el;
    const l1 = item.linesData[lineOfChar(item, Math.max(rg.a, rg.b - 1))].el;
    const r0 = l0.getBoundingClientRect(), r1 = l1.getBoundingClientRect();
    return { top: r0.top, bottom: r1.bottom };
  }
  function lineOfChar(item, idx) {
    let li = 0;
    for (let i = 0; i < item.linesData.length; i++) {
      const ld = item.linesData[i];
      if (idx >= ld.gs) li = i;
      else break;
    }
    return li;
  }

  /* 点在「行与行之间的空隙 / 行尾空白 / 块框里没画字的地方」时 e.target 是 .page
     ——`.blk` 容器不吃指针事件（见 style.css 的说明），所以这里按几何位置找回最近的一行。
     入参是**布局坐标**（页内 px）；返回 { item, y }，离得太远(超出 HIT_SLOP)则 null。 */
  const HIT_SLOP = 12;                 // 容差：约一行内距（正文 12pt），超过就算点了空白
  function nearestLineAt(x, y) {
    let best = null, bestD = Infinity;
    for (const id in itemById) {
      const it = itemById[id];
      if (it.kind !== "blk" || !it.linesData || !it.linesData.length) continue;
      for (const ld of it.linesData) {
        const r = ld.el.getBoundingClientRect();
        if (!r.width) continue;
        // ⚠️ 两个方向都要在容差内。只给纵向容差的话，「横向恰好落在远处那一行上」
        // （比如页脚那块只有一两个字的块）会被当成最近的一行选中。
        if (x < r.left - HIT_SLOP || x > r.right + HIT_SLOP ||
            y < r.top - HIT_SLOP || y > r.bottom + HIT_SLOP) continue;
        const dx = Math.max(r.left - x, 0, x - r.right);       // 框内算 0
        const dy = Math.max(r.top - y, 0, y - r.bottom);
        const d = dx + dy * 2;                                 // 纵向更敏感：宁可换行也别换块
        if (d < bestD) { bestD = d; best = { item: it, y }; }
      }
    }
    return best;
  }

  function attachFormulaBoxes(page, pageEl) {
    for (const box of pageEl.querySelectorAll(".fbox")) {
      const rect = box.getBoundingClientRect();
      const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
      let bestGap = null, bestOverlap = 0;
      if (box.dataset.cls === "InlineFormula") {
        for (const item of page.items) {
          if (item.kind !== "blk" || !item.linesData.length) continue;
          for (const line of item.linesData) {
            for (const gap of line.el.querySelectorAll(".rngap")) {
              const r = gap.getBoundingClientRect();
              const overlapX = Math.max(0, Math.min(rect.right, r.right) - Math.max(rect.left, r.left));
              const overlapY = Math.max(0, Math.min(rect.bottom, r.bottom) - Math.max(rect.top, r.top));
              const overlap = overlapX * overlapY;
              if (overlap > bestOverlap) {
                const start = parseInt(gap.dataset.g, 10);
                bestOverlap = overlap;
                bestGap = { item, start, end: start + gap.textContent.length };
              }
            }
          }
        }
      }
      let item, si;
      if (bestGap && Number.isFinite(bestGap.start) && bestGap.item.sents.length) {
        item = bestGap.item;
        si = sentOf(item, bestGap.start);
        box._copyRange = bestGap;
      } else {
        let nearest = null, nearestD = Infinity;
        for (const candidate of page.items) {
          if (candidate.kind !== "blk" || !candidate.sents.length) continue;
          for (let lineIndex = 0; lineIndex < candidate.linesData.length; lineIndex++) {
            const r = candidate.linesData[lineIndex].el.getBoundingClientRect();
            const dx = Math.max(r.left - cx, 0, cx - r.right);
            const dy = Math.max(r.top - cy, 0, cy - r.bottom);
            const d = dx + dy * 2;
            if (d < nearestD) {
              nearestD = d;
              nearest = { item: candidate, si: sentOf(candidate, candidate.linesData[lineIndex].gs) };
            }
          }
        }
        if (!nearest) continue;
        item = nearest.item;
        si = nearest.si;
        if (box.dataset.cls === "InlineFormula") {
          let closest = null, closestD = Infinity;
          for (const line of item.linesData) {
            for (const gap of line.el.querySelectorAll(".rngap")) {
              const r = gap.getBoundingClientRect();
              const dx = Math.max(r.left - cx, 0, cx - r.right);
              const dy = Math.max(r.top - cy, 0, cy - r.bottom);
              const d = dx + dy * 2;
              if (d < closestD) {
                closestD = d;
                closest = { item, start: parseInt(gap.dataset.g, 10),
                  end: parseInt(gap.dataset.g, 10) + gap.textContent.length };
              }
            }
          }
          if (closest && closestD <= 120 && Number.isFinite(closest.start)) {
            box._copyRange = closest;
            si = sentOf(item, closest.start);
          }
        }
      }
      box._selectRef = { item, si };
      if (!item.formulaBoxes) item.formulaBoxes = [];
      item.formulaBoxes.push(box);
    }
  }

  function formulaSelectionTarget(target) {
    if (!target) return null;
    if (target._selectRef) return { host: target, ref: target._selectRef };
    const blk = target.closest(".blk");
    const item = blk && findItem(blk.dataset.id);
    if (!item) return null;
    if (target.matches(".mathbox.inline")) {
      const formula = item.inlineMath && item.inlineMath[parseInt(target.dataset.mi, 10)];
      if (formula && item.sents.length)
        return { host: target, ref: { item, si: sentOf(item, formula.a) }, inline: true };
    }
    if (item.sents.length)
      return { host: target, ref: { item, si: fallbackSentence(item, item.y + item.h / 2) } };
    const page = pages[item.pi];
    if (!page) return null;
    const r = target.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    let nearest = null, nearestD = Infinity;
    for (const candidate of page.items) {
      if (candidate.kind !== "blk" || !candidate.sents.length) continue;
      for (const line of candidate.linesData) {
        const lr = line.el.getBoundingClientRect();
        const dx = Math.max(lr.left - cx, 0, cx - lr.right);
        const dy = Math.max(lr.top - cy, 0, cy - lr.bottom);
        const d = dx + dy * 2;
        if (d < nearestD) {
          nearestD = d;
          nearest = { item: candidate, si: sentOf(candidate, line.gs) };
        }
      }
    }
    return nearest ? { host: target, ref: nearest } : null;
  }

  /* ---- 应用/清除选中高亮（可多句、跨块跨页） ---- */
  function applySelRanges() {
    document.querySelectorAll(".fbox.formula-sel").forEach(box => box.classList.remove("formula-sel"));
    for (const id in itemById) {
      const it = itemById[id];
      if (it.selRanges && it.selRanges.length) { it.selRanges = []; rebuildItem(it); }
    }
    if (!sel) return;
    const byItem = new Map();
    if (!sel.formulaOnly) {
      for (const r of sel.sents) {
        if (!byItem.has(r.item)) byItem.set(r.item, []);
        byItem.get(r.item).push(sentRange(r.item, r.si));
      }
    }
    for (const [it, ranges] of byItem) { it.selRanges = ranges; rebuildItem(it); }
    if (sel && !sel.formulaOnly) {
      for (const ref of sel.sents) {
        const range = sentRange(ref.item, ref.si);
        for (const box of ref.item.formulaBoxes || []) {
          const inSentence = box._copyRange &&
            box._copyRange.start < range.b && box._copyRange.end > range.a;
          if (inSentence || box === sel.formulaBox) box.classList.add("formula-sel");
        }
      }
    }
    if (sel.formulaBox && sel.formulaBox.classList.contains("fbox"))
      sel.formulaBox.classList.add("formula-sel");
  }
  function setSelection(refs, silent, formulaBox) {
    if (!refs || !refs.length) { clearSelection(); return; }
    const formulaOnly = !!formulaBox && !formulaBox.matches(".mathbox.inline") &&
      formulaBox.dataset.cls !== "InlineFormula";
    sel = { sents: refs.slice(), formulaBox: formulaBox || null, formulaOnly };
    applySelRanges();
    if (!silent) { updateBar(); placeBar(); }
  }
  function clearSelection() {
    sel = null;
    applySelRanges();
    hideBar();
  }
  // 重新套用标注(高亮/笔记下划线)并保留选中高亮
  function refreshSelectionVisual() {
    for (const it of selItems()) { setAnnoRanges(it); rebuildItem(it); }
    paintFormulaMarks();        // 高亮刚改过：公式框也要跟着变（新高亮/取消高亮）
  }
  function selectSentence(item, si) {
    if (!item) return;
    setSelection([{ item, si }]);
  }

  /* ---- 鼠标拖拽多选：拖过连续句子 → 整段（句子级）选中 ---- */
  function refFromNode(node, offset, isEnd) {
    if (!node) return null;
    const el = node.nodeType === 3 ? node.parentElement : node;
    if (!el || !el.closest) return null;
    const blk = el.closest(".blk");
    if (!blk || !pagesEl.contains(blk)) return null;
    const item = findItem(blk.dataset.id);
    const span = el.closest(".rn");
    if (!item || !span || span.dataset.g == null) return null;
    const g = parseInt(span.dataset.g, 10) + (offset || 0);
    return { item, si: sentOf(item, isEnd ? Math.max(0, g - 1) : g) };
  }
  function refsFromNativeSelection() {
    const s = window.getSelection();
    if (!s || s.isCollapsed || !s.rangeCount) return null;
    const r = s.getRangeAt(0);
    const a = refFromNode(r.startContainer, r.startOffset, false);
    const b = refFromNode(r.endContainer, r.endOffset, true);
    if (!a || !b) return null;
    return refsBetween(a, b);
  }
  /* 按点取字符位置：浏览器自带的「坐标 → 最近的字符」。一次只看一个点，
     不受拖拽路径影响（原生选区会被途中的空白重置，见 mouseup 的说明）。
     入参是**视觉坐标**(clientX/clientY)，与 elementFromPoint 同口径。 */
  function caretAtPoint(cx, cy) {
    if (document.caretRangeFromPoint) {
      const r = document.caretRangeFromPoint(cx, cy);
      return r ? { node: r.startContainer, off: r.startOffset } : null;
    }
    if (document.caretPositionFromPoint) {          // Firefox
      const p = document.caretPositionFromPoint(cx, cy);
      return p ? { node: p.offsetNode, off: p.offset } : null;
    }
    return null;
  }
  function refAtPoint(cx, cy, isEnd) {
    const c = caretAtPoint(cx, cy);
    const r = c && refFromNode(c.node, c.off, isEnd);
    if (r) return r;
    // 没命中到字符（点在空白/行尾外）→ 按几何位置找最近的一行
    const near = nearestLineAt(cx / GZ(), cy / GZ());
    return near ? { item: near.item, si: fallbackSentence(near.item, near.y) } : null;
  }
  let dragging = false, formulaSelecting = false, dragFrom = null, dragAppliedAt = 0;
  pagesEl.addEventListener("mousedown", e => {
    if (e.button !== 0) return;
    if (e.target.closest(".fbox, .mathbox.inline, .blk.has-math .mathbox")) {
      formulaSelecting = true;
      dragging = false;
      dragFrom = null;
      return;
    }
    formulaSelecting = false;
    dragging = true;
    // 拖拽起点：mouseup 时按几何位置自己算范围（不信任原生选区，见下）
    dragFrom = e.target.closest(".img, .pagenum") ? null : { x: e.clientX, y: e.clientY };
  });
  document.addEventListener("mouseup", e => {
    if (formulaSelecting) {
      formulaSelecting = false;
      return;
    }
    if (!dragging) return;
    dragging = false;
    const from = dragFrom; dragFrom = null;
    const moved = !!from && Math.abs(e.clientX - from.x) + Math.abs(e.clientY - from.y) > 3;
    // ⚠️ 不能只信原生选区：拖拽途中一旦指针扫过「没有文字的空白」（行间空隙、公式空位、
    // 相邻块的空白框），浏览器会把原生选区的锚点重置到那一块 —— 起点所在的块就丢了
    // （复现：从 p1b45 拖到 p1b47，最后只选中 p1b47）。所以拖动过就按两端的**坐标**
    // 各取一句，再取阅读顺序里两者之间的整句区间。
    let refs = null;
    if (moved && !(e.target && e.target.closest && e.target.closest(".img, .pagenum"))) {
      const a = refAtPoint(from.x, from.y, false);
      const b = refAtPoint(e.clientX, e.clientY, true);
      if (a && b) refs = refsBetween(a, b);
    }
    if (!refs) refs = refsFromNativeSelection();   // 没拖动 / 取不到端点 → 退回原生选区
    if (!refs) return;                             // 没拖出句子范围 → 交给 click 当单句选中
    setSelection(refs);                            // 首尾 → 吸附到整句
    dragAppliedAt = Date.now();
    const s = window.getSelection(); if (s && s.removeAllRanges) s.removeAllRanges();
  });

  pagesEl.addEventListener("click", e => {
    if (Date.now() - dragAppliedAt < 350) return;   // 刚用拖拽选完，别被 click 收成单句
    const formula = e.target.closest(".fbox, .mathbox.inline, .blk.has-math .mathbox");
    if (formula) {
      const target = formulaSelectionTarget(formula);
      if (target) setSelection([target.ref], false, target.host);
      else clearSelection();
      return;
    }
    // 图 / 页码不是文字：点它们照旧清掉选中
    if (e.target.closest(".img, .pagenum")) { clearSelection(); return; }
    const blk = e.target.closest(".blk");
    let item = null, hitY = e.clientY / GZ();       // hitY 是布局坐标(见文件顶部坐标说明)
    if (blk) {
      item = findItem(blk.dataset.id);
    } else {
      // 块框里没画字的地方（行间空隙、行尾空白、公式空位）target 是 .page：
      // 按几何位置找回最近的一行继续选，别平白把已有的选中清掉。
      const near = nearestLineAt(e.clientX / GZ(), hitY);
      if (near) { item = near.item; hitY = near.y; }
    }
    if (!item || item.noSelect) { clearSelection(); return; }   // noSelect：表格等无句子块
    const ci = caretInfo();
    const si = (ci && ci.item === item) ? sentOf(item, ci.idx) : fallbackSentence(item, hitY);
    // Shift+点击：从当前选区首句扩展到点中的这句
    if (e.shiftKey && selCount()) setSelection(refsBetween(selAnchor(), { item, si }));
    else selectSentence(item, si);
    const s = window.getSelection(); if (s && s.removeAllRanges) s.removeAllRanges();
  });
  let relayoutTimer = null;
  window.addEventListener("resize", () => {
    if (sel) placeBar(true);
    // 视口宽度决定笔记卡是“贴右侧”还是“回到句子下方”，需要重新排版
    clearTimeout(relayoutTimer);
    relayoutTimer = setTimeout(relayoutAll, 150);
  });
  window.addEventListener("scroll", () => { if (sel) placeBar(true); updatePageInd(); scheduleSave(); }, { passive: true });

  /* ================= 浮动工具条 ================= */
  const fbar = $("fbar");
  function hideBar() { if (fbar) fbar.hidden = true; hideHmenu(); }
  // keepMenu=true 时只挪位置、不动调色盘(调色盘就挂在工具条里，会跟着走)；
  // 换选中句子时用默认的 false，顺手收起调色盘。
  function placeBar(keepMenu) {
    if (!sel) return;
    const anchor = selAnchor();
    if (!anchor) return;
    fbar.hidden = false;
    if (!keepMenu) hideHmenu();
    const formula = sel.formulaBox && sel.formulaBox.isConnected ? sel.formulaBox : null;
    const r = formula ? formula.getBoundingClientRect() : anchor.item.el.getBoundingClientRect();
    let top = r.top - 48;
    if (top < 60) top = r.bottom + 8;
    fbar.style.top = Math.max(60, top) + "px";
    fbar.style.left = clamp(r.left, 8, Math.max(8, vpW() - fbar.offsetWidth - 8)) + "px";
  }
  // 工具条只放操作按钮（翻译/标注/笔记/复制/关闭）——
  // 原来最左侧那个显示选中内容的预览（#fbT）已去掉，不再回填。
  function updateBar() {
    const bt = $("fbTrans");
    if (!sel) return;
    const anchor = selAnchor();
    const n = selCount();
    const blocks = selItems();
    if (n > 1) {
      bt.textContent = selTransList.some(t => t.key === selFragKey())
        ? "收起所选句译文" : "翻译所选句子";
    } else {
      const id = blocks[0].id;
      if (noZh.has(id)) bt.textContent = "本段无译文";
      else if (zh[id]) bt.textContent = isTransOpen(id) ? "收起本段译文" : "显示本段译文";
      else bt.textContent = "译本段";
    }
    const nb = $("fbNote");
    if (nb) nb.textContent = noteOf(sentKey(anchor.item, anchor.si)) ? "改笔记" : "笔记";
  }
  function isTransOpen(id) {
    const it = findItem(id);
    return !!(it && !it.tsec.hidden);
  }

  fbar.addEventListener("click", e => {
    const sw = e.target.closest(".sw");
    if (sw) { if (sel) setHighlight(sw.dataset.c); hideHmenu(); return; }
    const a = e.target.closest("[data-a]") && e.target.closest("[data-a]").dataset.a;
    if (!a || !sel) return;
    if (a === "close") clearSelection();
    else if (a === "copy") {
      const n = selCount();
      const formula = sel.formulaBox;
      const formulaOnly = formula && !formula.matches(".mathbox.inline") &&
        formula.dataset.cls !== "InlineFormula";
      const mathItem = formulaOnly && formula.closest(".blk")
        ? findItem(formula.closest(".blk").dataset.id) : null;
      const s = formulaOnly
        ? (formula.dataset.latex || mathTexOf(mathItem))
        : sel.sents.map(r => mathTexOf(r.item) || copySentenceText(r.item, r.si)).join(" ");
      navigator.clipboard.writeText(s).then(() => toast(n > 1 ? "已复制 " + n + " 句" : "已复制该句"));
    } else if (a === "trans") {
      if (selCount() > 1) translateSelected();
      else onTranslateClick(selItems()[0].id);
    }
    else if (a === "hl") toggleHmenu();
    else if (a === "note") openNoteEditor(true);
    else if (a === "hlclear") { setHighlight(null); hideHmenu(); }
  });

  /* ---- 多选句子 → 只翻译所选句 ----
     把选区按文字块切成若干“连续句片段”，各自送翻译(后端按文本哈希缓存，和整段译文互不干扰)，
     结果只显示所选句的译文，不影响原来的整段翻译功能。 */
  function selGroups() {
    const groups = [];
    if (!sel) return groups;
    for (const r of sel.sents) {
      const last = groups[groups.length - 1];
      if (last && last.item === r.item) last.siTo = r.si;
      else groups.push({ item: r.item, siFrom: r.si, siTo: r.si });
    }
    return groups;
  }
  function selFragId(g) { return g.item.id + "#" + g.siFrom + "-" + g.siTo; }
  function selFragKey() { return selGroups().map(selFragId).join("|"); }
  function selFragText(g) {
    const a = sentRange(g.item, g.siFrom).a;
    const b = sentRange(g.item, g.siTo).b;
    return g.item.canonical.slice(a, b).replace(/\s*\n\s*/g, " ").replace(/\s{2,}/g, " ").trim();
  }

  async function translateSelected() {
    const groups = selGroups();
    const frags = groups.map(g => ({ id: selFragId(g), text: selFragText(g), item: g.item, si: g.siFrom }))
                       .filter(f => f.text.length > 1);
    if (!frags.length) { toast("所选句子没有可翻译的内容"); return; }
    const key = frags.map(f => f.id).join("|");
    const idx = selTransList.findIndex(t => t.key === key);
    if (idx >= 0) {                              // 同一选区再点一次 = 收起这张
      selTransList.splice(idx, 1);
      renderSelTrans(); relayoutAll(); updateBar();
      return;
    }
    try {
      const res = await api("/api/translate", {
        method: "POST",
        json: { doc_id: docId, blocks: frags.map(f => ({ id: f.id, text: f.text })) },
      });
      const parts = frags.map(f => (res.results || {})[f.id]).filter(z => z);
      if (!parts.length) {
        const skipped = new Set(res.skipped || []);
        toast(skipped.size === frags.length
          ? "所选句子无需翻译(中文/公式/符号)"
          : "没有拿到译文：翻译服务没有返回结果，请再点一次重试");
        return;
      }
      // 与整段译文互斥：切换成“所选句译文”模式，收起所有整段译文卡（译文仍在缓存里，可随时再开）
      closeAllTrans(false);
      // 卡片挂到**最后一个被选句子**的块下（不是第一句）：连选多句时若挂第一句，
      // 卡片会插在所选句中间，后面的正文看起来像被错了位。
      const last = frags[frags.length - 1];
      const card = { key, itemId: last.item.id, siFrom: last.si,
                     text: parts.join("\n\n") };
      selTransList.push(card);
      renderSelTrans();
      relayoutAll();
      updateBar(); placeBar(true);
      toast("已翻译所选 " + selCount() + " 句");
      // 存到服务端，重登/刷新后这张卡还在（失败只提示，不影响阅读）
      api("/api/doc/" + docId + "/seltrans", {
        method: "POST",
        json: { key, item_id: card.itemId, si_from: card.siFrom, text: card.text },
      }).catch(err => showNotice("所选句译文未能保存到服务端：" + err.message, true));
    } catch (err) { showNotice("翻译失败：" + err.message, true); }
  }

  /* 所选句译文卡：可以同时留多张（不同选区）；与“整段译文”互斥，二者只保留一种 */
  function clearSelTrans() {
    if (!selTransList.length) return;
    selTransList = [];
    renderSelTrans();
    updateBar();          // 卡片没了，工具条上“收起所选句译文”要跟着变回“翻译所选句子”
    api("/api/doc/" + docId + "/seltrans", { method: "DELETE" }).catch(() => {});
  }
  // 切回“整段译文”模式：把所选句译文卡全部收掉（它们在缓存里，可随时重新翻译出来）
  function enterParaTransMode() {
    if (!selTransList.length) return;
    clearSelTrans();
    relayoutAll();
  }
  function renderSelTrans() {
    for (const id in itemById) {
      const it = itemById[id];
      if (!it.stbox.hidden || it.stbox.childElementCount) {
        it.stbox.textContent = "";
        it.stbox.hidden = true;
        syncExp(it);
      }
    }
    const byItem = new Map();
    for (const t of selTransList) {
      const it = selTransHost(t);
      if (!it) continue;
      if (!byItem.has(it)) byItem.set(it, []);
      byItem.get(it).push(t);
    }
    for (const [it, list] of byItem) {
      list.sort((a, b) => (a.siFrom || 0) - (b.siFrom || 0));
      for (const t of list) it.stbox.appendChild(selTransCard(it, t));
      it.stbox.hidden = false;
      syncExp(it);
    }
  }
  function selTransHost(t) {
    /* 卡片挂在哪个块下：**最后一个被选句子**的块。
       `key` 里记着全部片段（`<块id>#<起>-<止>` 用 `|` 相连），所以连早期存的
       卡片（item_id 还是第一句的块）也能落到正确位置；解析不出来再退回 item_id。 */
    const frag = String(t.key || "").split("|").pop() || "";
    const id = frag.split("#")[0];
    return (id && findItem(id)) || findItem(t.itemId);
  }
  function selTransCard(item, t) {
    const card = el("div", "card-in stsec");
    const head = el("div", "sthead");
    head.appendChild(el("span", "", "所选句译文"));
    const bDel = el("button", "stclear", "清除");
    bDel.addEventListener("click", () => {
      selTransList = selTransList.filter(x => x.key !== t.key);
      renderSelTrans(); relayoutAll(); updateBar();
      api("/api/doc/" + docId + "/seltrans?key=" + encodeURIComponent(t.key),
          { method: "DELETE" }).catch(() => {});
    });
    head.appendChild(bDel);
    card.appendChild(head);
    card.appendChild(el("div", "stbody", t.text));
    return card;
  }
  function hideHmenu() { const m = $("hmenu"); if (m) m.classList.remove("show"); }
  function toggleHmenu() {
    const m = $("hmenu");
    m.classList.toggle("show");
    // 先定位(keepMenu=true，否则刚展开又被 hideHmenu 收掉)，再标出当前高亮色
    placeBar(true);
    if (m.classList.contains("show")) markCurSwatch();
  }
  function curHighlightColor() {
    if (!sel) return null;
    // 选中的句子全部同色才把该色块标出来
    const cur = anno.highlights || {};
    const colors = selKeys().map(k => (cur[k] && cur[k].color) || null);
    return colors.length && colors.every(c => c && c === colors[0]) ? colors[0] : null;
  }
  function markCurSwatch() {
    const cur = curHighlightColor();
    document.querySelectorAll("#hmenu .sw").forEach(s =>
      s.classList.toggle("cur", s.dataset.c === cur));
  }
  async function setHighlight(color) {
    if (!sel) return;
    if (!needWrite()) return;
    const keys = selKeys();
    try {
      const cur = anno.highlights || {};
      // 选中的句子全是这个颜色 → 再点一次 = 整段取消
      if (color && keys.every(k => cur[k] && cur[k].color === color)) color = null;
      for (const k of keys) {
        await api(`/api/doc/${docId}/anno`, {
          method: "POST",
          json: color
            ? { action: "add_highlight", block_id: k, color }
            : { action: "remove_highlight", block_id: k },
        });
      }
      anno = await api(`/api/doc/${docId}/anno`);   // 批量写入后取一次权威状态
      refreshSelectionVisual();
      const n = keys.length;
      toast(color ? (n > 1 ? `已标注 ${n} 句` : "已标注该句")
                  : (n > 1 ? `已清除 ${n} 句标注` : "已清除标注"));
    } catch (err) { toast(err.message); }
  }

  /* ================= 翻译(段落级) ================= */
  async function onTranslateClick(id) {
    const it = findItem(id); if (!it) return;
    enterParaTransMode();                       // 与“所选句译文”互斥
    if (noZh.has(id)) { toast("该段无需翻译(中文/公式/符号)"); return; }
    if (zh[id]) { toggleTransOpen(id); return; }
    try {
      const { done, failed } = await translateItems([it], { open: true });
      if (done) { updateBar(); placeBar(); }
      else if (failed) showNotice("这一段的译文没有拿到：翻译服务没有返回结果，可以再点一次重试。", true);
      else if (noZh.has(id)) toast("该段无需翻译(中文/公式/符号)");
    } catch (err) { showNotice("翻译失败：" + err.message, true); }
  }
  function toggleTransOpen(id) {
    const it = findItem(id); if (!it) return;
    enterParaTransMode();                       // 与“所选句译文”互斥
    it.tsec.hidden = !it.tsec.hidden;
    // “翻译全文”在“显示译文”未勾选时只把译文存进 zh、不渲染卡片，
    // 这里展开必须补写内容，否则弹出来的是**空卡片**（踩过）
    if (!it.tsec.hidden) fillTrans(it);
    syncExp(it);
    relayoutAll();
    updateBar(); if (sel) placeBar();
  }
  async function translateItems(items, opt) {
    const pend = items.filter(it => it.kind === "blk" && !zh[it.id] && !noZh.has(it.id) && it.text.trim().length > 1);
    if (!pend.length) {
      if (opt && opt.open) for (const it of items) if (it.kind === "blk" && zh[it.id]) openTrans(it);
      return { done: 0, failed: 0 };
    }
    const res = await api("/api/translate", {
      method: "POST",
      json: { doc_id: docId, blocks: pend.map(it => ({ id: it.id, text: it.text })) },
    });
    // 服务端把“无需翻译(skipped)”和“这次没拿到(failed)”分开返回：
    // 只有 skipped 才能记进 noZh 永久跳过，failed 必须留着让用户重试。
    const skipped = new Set(res.skipped || []);
    let done = 0, failed = 0;
    for (const it of pend) {
      const z = (res.results || {})[it.id];
      if (z) { zh[it.id] = z; done++; if (opt && opt.open) openTrans(it); }
      else if (skipped.has(it.id)) noZh.add(it.id);
      else failed++;
    }
    relayoutAll();
    return { done, failed };
  }
  // 把译文写进卡片（幂等）。译文统一从 zh[] 取，避免出现“zh 里有、卡片是空的”
  function fillTrans(item) {
    if (!item || item.kind !== "blk") return;
    const z = zh[item.id] || "";
    if (item.tsec.textContent !== z) item.tsec.textContent = z;
  }
  function openTrans(item) {
    fillTrans(item);
    item.tsec.hidden = false;
    syncExp(item);
  }
  function openAllTrans() {
    let c = 0;
    for (const id in zh) { const it = findItem(id); if (it) { openTrans(it); c++; } }
    relayoutAll();
    return c;
  }
  function closeAllTrans(doLayout) {
    for (const id in itemById) { const it = itemById[id]; it.tsec.hidden = true; syncExp(it); }
    if (doLayout !== false) relayoutAll();
  }
  // 展开区(译文/笔记)的显隐：任一卡片可见就展开 .exp
  function syncExp(item) {
    if (!item || !item.exp) return;
    item.exp.classList.toggle("open",
      !item.tsec.hidden || !item.stbox.hidden || !item.nbox.hidden);
  }
  $("chkShowZh").addEventListener("change", e => {
    showZhOn = e.target.checked;
    savePrefs({ show_zh: showZhOn });      // 这个开关也跟账号走
    if (showZhOn) { enterParaTransMode(); openAllTrans(); }
    else closeAllTrans();
  });
  /* 设置里换了“翻译目标语言”：服务端的译文缓存与“所选句译文”卡都按语言分开存，
     这里把上一个语言的译文清掉、按新语言重新拉一次，页面上不留旧语言的译文 */
  document.addEventListener("paper-target-lang", async (e) => {
    for (const id in zh) delete zh[id];
    noZh.clear();
    for (const id in itemById) { const it = itemById[id]; if (it && it.tsec) it.tsec.textContent = ""; }
    try {
      Object.assign(zh, await api("/api/doc/" + docId + "/zh_all").catch(() => ({})));
      selTransList = await loadSelTrans();
    } catch (err) { /* 拉不到就当还没翻译，不影响阅读 */ }
    renderSelTrans();
    if (showZhOn) openAllTrans(); else closeAllTrans();
    relayoutAll();
    updateBar();
    const name = (e.detail && e.detail.name) || "";
    toast(name ? ("翻译目标语言已改为 " + name + "，本页译文已按新语言刷新")
               : "翻译目标语言已切换，本页译文已刷新");
  });
  $("btnPageTrans").addEventListener("click", async () => {
    const pi = currentPageIndex();
    const items = pages[pi] ? pages[pi].items.filter(i => i.kind === "blk") : [];
    enterParaTransMode();                       // 与“所选句译文”互斥
    try {
      const { done, failed } = await translateItems(items, { open: true });
      if (done && failed) toast(`本页翻译完成 ${done} 段，另有 ${failed} 段没有拿到译文（可再点一次重试）`);
      else if (done) toast(`本页翻译完成 ${done} 段`);
      else if (failed) showNotice(`本页有 ${failed} 段没有拿到译文：翻译服务没有返回结果，可稍后再点「翻译本页」重试（这些段落仍然可以单独翻译）。`, true);
      else toast("本页没有需要翻译的英文段落");
    } catch (err) { showNotice("翻译失败：" + err.message, true); }
  });
  $("btnAllTrans").addEventListener("click", runAllTrans);
  async function runAllTrans() {
    if (busy) return;
    busy = true; cancelAll = false;
    enterParaTransMode();                       // 与“所选句译文”互斥
    const btn = $("btnAllTrans");
    btn.disabled = true;
    const prog = $("transProgress"), bar = $("tpBar"), txt = $("tpText");
    prog.hidden = false;
    const all = [];
    pages.forEach(p => p.items.forEach(it => { if (it.kind === "blk" && !zh[it.id] && !noZh.has(it.id)) all.push(it); }));
    if (!all.length) { toast("全部段落均已翻译"); prog.hidden = true; btn.disabled = false; busy = false; return; }
    const total = all.length;
    let doneN = 0, failedN = 0;
    try {
      for (let k = 0; k < all.length && !cancelAll; k += 20) {
        const chunk = all.slice(k, k + 20);
        const res = await api("/api/translate", {
          method: "POST", json: { doc_id: docId, blocks: chunk.map(i => ({ id: i.id, text: i.text })) },
        });
        const skipped = new Set(res.skipped || []);
        chunk.forEach(i => {
          const z = (res.results || {})[i.id];
          if (z) { zh[i.id] = z; if (showZhOn) openTrans(i); }
          else if (skipped.has(i.id)) noZh.add(i.id);   // 真·无需翻译才跳过
          else failedN++;                               // 没拿到 → 留着可重试
          doneN++;
        });
        bar.style.width = Math.round(doneN / total * 100) + "%";
        txt.textContent = doneN + " / " + total;
      }
      relayoutAll();
      if (failedN && !cancelAll) {
        showNotice(`全文翻译完成，但有 ${failedN} 段没有拿到译文（翻译服务没返回结果）。再点一次「翻译全文」即可只重试这些段落。`, true);
      } else if (cancelAll) {
        toast("已暂停翻译");
      } else if (!showZhOn) {
        // 没勾“显示译文”时卡片都是收着的，提示一下怎么看，别让人以为没翻译
        showNotice("全文翻译完成。勾选顶部的「显示译文」可展开全篇译文，或点任意句子后选「显示本段译文」逐段查看。", false);
      } else {
        toast("全文翻译完成 🎉");
      }
    } catch (err) {
      showNotice("翻译失败：" + err.message + "（可能未配置翻译服务或余额不足）", true);
    } finally {
      prog.hidden = true; btn.disabled = false; busy = false;
    }
  }
  $("tpCancel").addEventListener("click", () => { cancelAll = true; });

  /* ================= 布局 =================
     两种展开形式：
       1) 译文卡：留在句子下方，把后续版式内容顶开(“顶开版式”)；
       2) 笔记卡：绝对定位在**页面右侧的批注栏**里，纵向对齐笔记对应句子所在行
          —— 与正文同高、不占版面；同页多张卡自上而下依次避让。
     有笔记时会给页面右侧“预留”批注栏宽度(见 --note-reserve)：
       窗口够宽 → 页面保持原尺寸并整体左移，右边让出批注栏；
       窗口不够宽 → 页面按可用宽度等比缩小(fit)，保证能并排；
       实在放不下(可用宽度 < 520px，或右侧面板展开中) → 笔记回到句子下方内嵌。 */
  function xOverlap(a, b, aw) {
    const w = aw || a.w;
    return Math.max(0, Math.min(a.x + w, b.x + b.w) - Math.max(a.x, b.x)) > 1;
  }
  // 页面尺寸与批注栏宽度只跟窗口/缩放/有无笔记有关，不依赖已渲染的 DOM
  function computePageTarget() {
    // 整体缩放时视口在布局坐标里只有 clientWidth/ZOOM 这么宽；页面宽度保持 CSS_W，
    // 由 <html> 的 zoom 把它放大(不支持 zoom 的浏览器才退回“直接算宽页面”的老算法)
    const clientW = vpW();
    const want = ZOOM_OK ? CSS_W : CSS_W * ZOOM();
    const drawerOpen = !!($("side") && $("side").classList.contains("open"));
    // 左侧目录展开时也要给它让出宽度，否则页面会被盖住
    const tocW = tocPanelOpen() ? TOC_W : 0;
    const need = notesCount() > 0 || !!editingKey;      // 有笔记(或正在写)才预留
    const reserve = clamp(Math.round(clientW * 0.25), 210, 330);
    const room = clientW - reserve - tocW - 36;         // 让出批注栏/目录后页面能用多宽
    const side = !drawerOpen && showNotesOn && need && room >= 520;
    return {
      side, reserve, tocW,
      noteW: Math.max(190, reserve - 28),
      pageW: side ? Math.min(want, room)
                  : (tocW ? Math.min(want, Math.max(320, clientW - tocW - 36)) : want),
    };
  }
  function applyNoteChrome(t) {
    document.body.classList.toggle("notes-side", t.side);
    document.body.style.setProperty("--note-reserve", t.reserve + "px");
    // 卡片(译文/笔记)字号跟着页面的实际渲染比例走，和正文一起缩放
    // (夹紧到 0.78~1.2，避免页面很小时字小到看不清)
    document.body.style.setProperty(
      "--page-scale", clamp(t.pageW / CSS_W, 0.78, 1.2).toFixed(3));
    noteSide = t.side;
    noteW = t.noteW;
    noteLeft = t.pageW + 14;                            // 页面坐标：贴着页面右边线外侧
  }
  function relayoutAll() {
    if (!DOC || !pages.length) return;      // 论文还没加载好（或加载失败）：别去碰 DOM
    const t = computePageTarget();
    if (Math.abs(t.pageW - curPageW) > 0.5) { render(); return; }   // 页面尺寸变了 → 整体重排
    applyNoteChrome(t);
    layoutAi();
    pages.forEach((pg, pi) => relayoutPage(pi));
    // 版式（含公式框位移）刚定下来：高亮公式框的“命中”是按几何算的，要重新对一遍
    paintFormulaMarks();
  }

  /* 文末面板：宽度不够时从“左右”变“上下”（.ai-stacked），上下结构下两块等宽。
     判断方式是先把类去掉、量一下两块是否真的落在同一行 —— 比猜一个阈值可靠
     （换行临界是两块 flex-basis 之和，改 CSS 后这里不用跟着改）。
     量完立刻恢复，整个过程在同一个任务里完成，不会闪。 */
  function layoutAi() {
    const box = $("aiBody"), chat = $("aiChat"), sum = $("aiSum");
    if (!box || !chat || !sum) return;
    box.classList.remove("ai-stacked");
    const wrapped = Math.abs(chat.offsetTop - sum.offsetTop) > 2;
    if (wrapped) box.classList.add("ai-stacked");
  }
  // 右侧模式：把笔记容器绝对定位到页面右侧；否则清掉定位、回到句子下方内嵌
  function applyNoteSide(it) {
    const box = it.nbox;
    if (!box) return;
    const side = noteSide && !box.hidden;
    box.style.position = side ? "absolute" : "";
    box.style.minWidth = side ? "0" : "";
    box.style.width = side ? noteW + "px" : "";
    box.style.left = side ? (noteLeft - it.x) + "px" : "";
    if (!side) box.style.top = "";
  }
  // 右侧笔记卡：先按“句子首行”定高，再自上而下依次避让；返回最底部(页面坐标)
  function layoutSideNotes(items) {
    const boxes = items.filter(it => it.kind === "blk" && it.nbox && !it.nbox.hidden);
    for (const it of boxes) {
      const first = notesOfBlock(it)[0];
      const si = first ? keyParts(first.block_id).si : 0;
      const ld = it.linesData[lineOfChar(it, sentRange(it, si).a)] || it.linesData[0];
      it.noteDy = ld ? ld.dy : 0;                    // 句子首行相对块顶的偏移
      it.noteY = it.y + it.noteDy + (it.syShift || 0) + (it.shift || 0);  // 目标页面 y（含内容避让 + 译文顶开）
    }
    boxes.sort((a, b) => a.noteY - b.noteY);
    let cursor = -Infinity;
    for (const it of boxes) {
      const h = it.nbox.offsetHeight;
      const y = Math.max(it.noteY, cursor + NOTE_GAP);
      it.nbox.style.top = (y - it.y - it.contentH) + "px";   // 相对 .exp(块顶+contentH)
      cursor = y + h;
    }
    return boxes.length ? cursor : 0;
  }
  function relayoutPage(pi) {
    const pg = pages[pi];
    if (!pg) return;
    const items = pg.items;

    for (const it of items) if (it.kind === "blk") applyNoteSide(it);

    for (const it of items) {
      it.expH = 0; it.expW = 0;
      if (it.kind !== "blk" || !it.exp.classList.contains("open")) continue;
      const inlineNotes = !noteSide && !it.nbox.hidden;   // 右侧模式下笔记不占版面
      if (it.tsec.hidden && it.stbox.hidden && !inlineNotes) continue;
      it.expH = it.exp.offsetHeight;
      it.expW = Math.max(it.w, inlineNotes ? it.nbox.offsetWidth : 0);
    }
    // ⓪ Surya 块的内容避让（settleSuryaVertical 精确计算）：直接改 style.top，
    //    与卡片顶开（transform）互不干扰，幂等（同一布局内 syShift 不变）。
    //    ⚠️ 不走 ①/② 的位移链：那条链是给卡片设计的近似“同化”传播，会把
    //    精确的避让量再放大，反而让下方块被多推、造成新的重叠（第 2 页实测）。
    for (const it of items) {
      if (it.kind !== "blk" || it.syShift === undefined) continue;
      it.el.style.top = (it.y + it.syShift).toFixed(1) + "px";
    }
    // ① 直接位移：每张展开卡把**同栏**（x 重叠）、位于其下方的内容顶开
    for (const it of items) {
      let shift = 0;
      const bottom = it.kind === "blk" ? it.y + it.contentH : it.y + it.h;
      for (const o of items) {
        if (o === it || !o.expH || o.kind !== "blk") continue;
        const ob = o.y + o.contentH;
        if (ob <= bottom + 0.5 && xOverlap(o, it, o.expW)) shift += o.expH;
      }
      it.shift = shift;
    }
    // ② 传递位移：被顶开的**宽元素**（典型是跨栏插图）会把压在它正下方的内容也带下去。
    //    例：右栏的卡片把下面的插图顶下去 154px，插图下方那半栏的标题/页码 x 不与
    //    卡片栏重叠，①不会动它们 —— 结果就被下移的插图盖住。这里让「某个已位移元素
    //    下方 + 横向相交」的元素至少取那个位移；取 max（①里已计入的卡片高度不重复
    //    叠加），迭代到收敛（位移只增不减且有上界，必然收敛）。
    let grew = true;
    while (grew) {
      grew = false;
      for (const it of items) {
        const bottom = it.kind === "blk" ? it.y + it.contentH : it.y + it.h;
        for (const m of items) {
          if (m === it || m.shift <= it.shift) continue;
          const mb = m.kind === "blk" ? m.y + m.contentH : m.y + m.h;
          if (mb <= bottom + 0.5 && xOverlap(m, it)) { it.shift = m.shift; grew = true; }
        }
      }
    }
    for (const it of items) {
      it.el.style.transform = it.shift ? "translateY(" + it.shift + "px)" : "";
    }

    /* 页面级公式框覆盖层（`.fbox`，见 r2-math.js）不在 `items` 里，但也得跟着
       “顶开版式”走：否则译文/笔记卡把正文顶下去之后，公式还钉在原坐标 ——
       表现就是公式叠在卡片上、和下面的文字错位。判据与块一致，只是拿**公式中心**
       比展开块底边（块末行的行内公式常常比行框高，底边会探到块底下面一点点，
       用底边会把锚点块自己的公式也算进去）。
       ⚠️ **没有展开卡时不能整个跳过**：必须把上一次的位移清掉 —— 否则卡片收起/
       删除后正文弹回原位、公式却留在下方。`anyExp` 只用来省掉“本来就没有位移”
       时的量测。 */
    const anyExp = items.some(o => o.expH);
    for (const fb of pg.el.querySelectorAll(".fbox")) {
      if (!anyExp && !fb.style.transform) continue;
      const top = parseFloat(fb.style.top) || 0;
      const fh = parseFloat(fb.style.height) || 0;
      const cy = top + fh / 2;
      const fx = parseFloat(fb.style.left) || 0;
      const fw = parseFloat(fb.style.width) || 0;
      let shift = 0;
      for (const o of items) {
        if (!o.expH || o.kind !== "blk") continue;
        if (o.y + o.contentH <= cy + 0.5 && xOverlap(o, { x: fx, w: fw }, o.expW))
          shift += o.expH;
      }
      for (const m of items) {                    // ② 传递位移（与 items 同一套规则）
        if (m.shift <= shift) continue;
        const mb = m.kind === "blk" ? m.y + m.contentH : m.y + m.h;
        if (mb <= cy + 0.5 && xOverlap(m, { x: fx, w: fw })) shift = m.shift;
      }
      fb.style.transform = shift ? "translateY(" + shift + "px)" : "";
    }

    const noteBottom = noteSide ? layoutSideNotes(items) : 0;

    let maxB = pg.h;
    for (const it of items) {
      const base = it.kind === "blk" ? it.y + it.contentH : it.y + it.h;
      const sy = it.syShift || 0;         // Surya 内容避让量（走 style.top，不在 shift 里）
      let b = base + sy + (it.shift || 0);
      if (it.expH) b = Math.max(b, it.y + it.contentH + sy + (it.shift || 0) + it.expH);
      if (b > maxB) maxB = b;
    }
    if (noteBottom > maxB) maxB = noteBottom;   // 右侧笔记卡也不能压到下一页
    pg.el.style.height = Math.max(pg.h, maxB + 22) + "px";
  }

  /* ================= 句子笔记 / 全部笔记抽屉：已拆到 r2-notes.js =================
     这里只做装配。$ / itemById / inkCache / docId 等 const、以及 26 个函数声明
     装配时取一次即可；下面这些要特殊处理：
     · anno / editingKey / noteDraft / showNotesOn —— 段内既读又写（anno 甚至会整体
       重新赋值），所以 getter + setter 都要给；
     · sel / DOC —— 段内只读，给 getter；
     · canWrite / renderTocAi 声明在**本段之后**（共享权限段 / AI 段装配处），
       装配这一刻还在 TDZ，所以包一层转发，等真正调用时再去读。 */
  const { notesOfBlock, notesCount, renderAllNotes, openNoteEditor, refreshSideList } = R2Notes({
    $, itemById, docId, toScrollDelta, inkCache,
    keyParts, relayoutAll, placeBar, updateBar, relayoutPage, needWrite, findItem,
    sentKey, sentenceText, noteOf, selCount, selAnchor, isSelRef, pageOfItem,
    render, renderHmenuSwatches, setAnnoRanges, rebuildItem, selectSentence, syncExp,
    canWrite: (...a) => canWrite(...a),
    renderTocAi: (...a) => renderTocAi(...a),
    get anno() { return anno; }, set anno(v) { anno = v; },
    get editingKey() { return editingKey; }, set editingKey(v) { editingKey = v; },
    get noteDraft() { return noteDraft; }, set noteDraft(v) { noteDraft = v; },
    get showNotesOn() { return showNotesOn; }, set showNotesOn(v) { showNotesOn = v; },
    get sel() { return sel; },
    get DOC() { return DOC; },
    get myName() { return myName; },      // 共享论文里把“我写的笔记”标出来
  });

  /* ================= 滚动/页码/目录/进度/缩放：已拆到 r2-nav.js =================
     这里只做装配。稳定引用（$ / pages / docId / 各坐标换算 / ZOOM / 渲染函数）
     装配时取一次即可；下面这几个要特殊处理：
     · META 用 getter —— 换论文会整体替换，而且段内要改它的属性（META.progress = …）；
     · zoomIdx 用 getter + setter —— 段内会赋值（换缩放档位）；
     · canWrite / maybeAutoSummarize 声明在**本段之后**（共享权限段 / AI 段装配处），
       装配这一刻还在 TDZ，所以包一层转发，等真正调用时再去读；
     · saveTimer 已随模块搬走（外部没有引用），原来那行 `let saveTimer = null;` 已删。 */
  const { currentPageIndex, updatePageInd, scrollToPage, TOC_W, tocPanelOpen,
          scheduleSave, updateFinishBtn, refreshZoomLabel } = R2Nav({
    $, pages, docId, CSS_W, vpH, ZOOMS, pagesEl, ZOOM, GZ, toScrollDelta,
    relayoutAll, applyZoom, pageAnchor, render,
    canWrite: (...a) => canWrite(...a),
    maybeAutoSummarize: (...a) => maybeAutoSummarize(...a),
    get META() { return META; },
    get zoomIdx() { return zoomIdx; },
    set zoomIdx(v) { zoomIdx = v; },
  });

  /* ================= 文末 · AI 辅助阅读：已拆到 r2-ai.js =================
     这里只做装配：把跨模块共享的东西交出去，再把模块接口接回本作用域。
     · $ / docId / findItem / keyParts / pageOfItem / scrolled / toScrollTop / needWrite /
       anchorProbe 都是稳定引用（函数声明会提升），装配时取一次即可；
     · META / anno 用 getter —— 换论文 / 重拉批注时会整体替换，模块里要每次取最新值；
     · canWrite 声明在**本段之后**（共享权限那一段），装配这一刻还在 TDZ，
       所以包一层转发，等真正调用时再去读它。
     · renderInlineMath / katexReady / ensureKatex 来自 r2-math（本段之前已装配）——
       AI 气泡与整理条目按 Markdown 渲染（r2-md.js）后，里面的 $…$ / \(…\) 数学
       交给它们就地换成 KaTeX（没加载过则懒加载）。
     原来那段里 AI 自己的状态（aiSummary / aiChat / aiBusyWhat / aiEdit / …）
     随模块一起搬走了，不再占用本文件的作用域。 */
  const { maybeAutoSummarize, syncAiButtons, jumpToAi, renderTocAi, loadAi } = R2Ai({
    $, docId, findItem, keyParts, pageOfItem, scrolled, toScrollTop, needWrite, anchorProbe,
    canWrite: (...a) => canWrite(...a),
    renderInlineMath, katexReady, ensureKatex,
    get META() { return META; },
    get anno() { return anno; },
  });

  /* ================= 初始化 ================= */
  bindHlColorInputs();     // 先绑定取色器（不依赖数据），颜色稍后按账号加载
  renderHmenuSwatches();   // 先用默认色渲染 6 个色块

  /* ================= 共享权限 =================
     `owner` / `write` / `read` 由后端在 /progress 里给出。
     「只读」只是把写入口藏起来，服务端每个写接口也会再抦一次（403）。 */
  let perm = "owner";
  let docOwner = "";
  let myName = "";         // 当前登录账号：共享论文里用来把“我写的笔记”区分出来
  const canWrite = () => perm === "owner" || perm === "write";

  function shareNoteText() {
    return "这篇论文是 " + (docOwner || "别人") + " 共享给你的（"
      + (perm === "write" ? "可写" : "只读") + "）：" + (perm === "write"
        ? "你可以和对方一起改笔记、高亮与 AI 整理。"
        : "你可以看笔记/高亮/译文、自己翻译、记自己的阅读进度，但不能改笔记。");
  }

  function applyPerm() {
    const badge = $("shareBadge");
    if (badge) {
      if (perm === "owner") {
        badge.className = "share-badge";
        const n = (META && META.shared_count) || 0;
        badge.hidden = !n;
        badge.textContent = n ? "已共享给 " + n + " 人" : "";
      } else {
        badge.hidden = false;
        badge.className = "share-badge" + (perm === "write" ? " write" : "");
        badge.textContent = "来自 " + (docOwner || "别人") + " · "
          + (perm === "write" ? "可写共享" : "只读共享");
      }
    }
    const btn = $("btnShare");
    if (btn) btn.hidden = perm !== "owner";
    syncAiButtons();            // 只读时把 AI 那边的按钮也置灰
  }

  function needWrite(tip) {
    if (canWrite()) return true;
    toast(tip || "这是只读共享的论文，不能修改（可让作者把权限改成「可写」）", 3600);
    return false;
  }

  async function openShare() {
    if (!needWrite("只有论文所有者能管理共享")) return;
    openShareModal(docId, DOC.title || "", async () => {
      try {
        META = await api("/api/doc/" + docId + "/progress");
        applyPerm();
      } catch (e) { /* ignore */ }
    });
  }
  $("btnShare").addEventListener("click", openShare);

  /* 用原始 PDF 原地重解析版式：解析器升级（换 OCR 后端 / 补公式 / 调 DPI）后，
     库里存的还是旧版式，这里重新解析一遍 —— 笔记/高亮/译文/进度都不动。 */
  const btnReparse = $("btnReparse");
  const reparseFile = $("reparseFile");
  if (btnReparse && reparseFile) {
    btnReparse.addEventListener("click", () => {
      if (!needWrite("只有论文所有者能重新解析版式")) return;
      reparseFile.value = "";
      reparseFile.click();
    });
    reparseFile.addEventListener("change", async () => {
      const f = reparseFile.files && reparseFile.files[0];
      if (!f) return;
      const old = btnReparse.textContent;
      btnReparse.disabled = true;
      btnReparse.textContent = "解析中…";
      showNotice("正在用「" + f.name + "」重新解析版式，请稍候（页数多时需要一会儿）…", false);
      try {
        const fd = new FormData();
        fd.append("file", f);
        const r = await api("/api/doc/" + docId + "/reparse", { method: "POST", body: fd });
        const info = r.parser_info || {};
        const lb = r.latex_blocks || [0, 0];
        toast("版式已重新解析（" + (r.parser || "?") + "）");
        showNotice("重解析完成：后端 " + (r.parser || "?") + "，页数 " + r.pages[0]
          + "→" + r.pages[1] + "，文字块改动 " + r.changed + " 处"
          // 公式：本流程把公式字形擦成等长空格（页面上那段地方就是空白）。
          // `formula_chars` = 被擦掉的字形数，`formula_spans` = 擦除的 run 段数。
          + (info.formula_chars
            ? ("，擦除公式字形 " + info.formula_chars + " 字 / "
               + (info.formula_spans || 0) + " 段")
            : "")
          + (lb[1] ? ("，带 latex 的块 " + lb[0] + "→" + lb[1]) : "")
          + "；笔记与译文都保留。正在刷新…", false);
        setTimeout(() => location.reload(), 1200);
        setTimeout(() => location.reload(), 1200);
      } catch (err) {
        showNotice("重新解析失败：" + err.message, true);
      } finally {
        btnReparse.disabled = false;
        btnReparse.textContent = old;
      }
    });
  }

  /* “所选句译文”卡存在服务端（按目标语言分开存）：拉下来映射成前端结构 */
  async function loadSelTrans() {
    const cards = await api("/api/doc/" + docId + "/seltrans").catch(() => []);
    return (cards || []).map(c => ({
      key: c.key, itemId: c.item_id || String(c.key || "").split("#")[0],
      siFrom: c.si_from || 0, text: c.text || "",
    })).filter(c => c.key && c.text);
  }

  async function init() {
    if (!docId) { showNotice("缺少论文 id", true); return; }
    await loadPrefs();             // 自定义高亮色 / 显示译文开关：随账号存在服务端
    renderHmenuSwatches();
    try {
      [META, DOC, anno] = await Promise.all([
        api("/api/doc/" + docId + "/progress"),
        api("/api/doc/" + docId),
        api("/api/doc/" + docId + "/anno"),
      ]);
      perm = META.perm || "owner";
      docOwner = META.owner || "";
      myName = ((await authStatus()) || {}).username || "";   // 笔记作者标识用
      const zhMap = await api("/api/doc/" + docId + "/zh_all").catch(() => ({}));
      Object.assign(zh, zhMap);
      // “所选句译文”卡是存在服务端的：重新登录/刷新后原样恢复（片段译文不在 zh 里，靠它回来）
      selTransList = await loadSelTrans();
      // 两种译文互斥：恢复出来的“所选句译文”会占坑，就把“显示译文”勾去掉，
      // 让勾选状态和实际显示一致（服务端存的偏好不动，清掉卡片后再勾就行）
      if (selTransList.length) { $("chkShowZh").checked = false; showZhOn = false; }
      $("docTitle").textContent = DOC.title;
      document.title = DOC.title;
      updateFinishBtn();
      $("notesCount").textContent = (anno.notes || []).length;
      const cfg = await appConfig();
      applyPerm();
      const tips = [];
      if (perm !== "owner") tips.push(shareNoteText());
      if (!cfg.ready) {
        tips.push("⚠️ 未配置翻译服务：译文功能不可用。点右上角「⚙ 设置 → LLM 服务」填写服务地址与 API Key（也可在 config.json 里配），或安装 deep-translator 用免费兜底。");
      }
      // 图片/公式检测（detection-service-group）与所选后端不匹配时提示一声。
      const pr = cfg && cfg.parser;
      if (pr && pr.backend === "surya" && pr.surya_ready !== true) {
        // surya 后端是“全有或全无”：服务/客户端任一没就绪，上传与重解析都会失败
        tips.push("⚠️ 解析后端现为 surya，但 Surya 2 还未就绪：" + (pr.surya_message || "") +
          "（上传 / 重解析会失败；点右上角「⚙ 设置 → 存储 / OCR」查看）");
      } else if (pr && pr.backend === "pymupdf" && pr.detection_service_group_ready) {
        tips.push("⚠️ detection-service-group 已就绪，但解析后端选的是 pymupdf（不跑图片/公式检测）——" +
          "要显示 Figure 图片和公式，请在「⚙ 设置 → 存储 / OCR」里选 auto 或 detection_service_group。");
      } else if (pr && (pr.backend === "auto" || pr.backend === "detection_service_group") &&
                 pr.detection_service_group_ready === false) {
        tips.push("⚠️ detection-service-group 不可用：解析时会跳过 Figure 图片和公式框/LaTeX 增强。" +
          (pr.message || "") + "（点右上角「⚙ 设置 → 存储 / OCR」查看）");
      }
      showNotice(tips.join("　"), !!tips.length);
      // 公式用 KaTeX 渲染：脚本懒加载，先画一次(公式先按 PDF 原文字显示)，
      // 加载完再按当前阅读位置重画一遍把公式换成真渲染；加载失败就保持原文字。
      if (docNeedsMath(DOC) && !katexReady()) {
        ensureKatex().then(ok => { if (ok) render(pageAnchor()); });
      }
      render();
      // 笔记已内嵌在正文里（render 时自动展开），这里只把“全部笔记”总览的口径刷新一下，
      // 不再自动弹右侧面板——面板只在你主动点顶栏「📝 笔记」时打开。
      refreshSideList();
      // 文末的 AI 面板：先拿回上次的整理/对话；若之前已读到 95% 且还没整理，这里补一次
      await loadAi();
      maybeAutoSummarize();
      // 已经读完的文章：进来直接停在文末的 AI 面板（整理与对话都在那儿），
      // 不必先落在最后一页再自己往下滚。点「已完成 ✓」改成在读可取消这个行为。
      if (META.status === "done") jumpToAi(true);
    } catch (err) {
      showNotice("加载失败：" + err.message, true);
    }
  }
  applyZoom();      // 首屏就把整体缩放套上（默认 100%，等价于不变）
  init();
})();
