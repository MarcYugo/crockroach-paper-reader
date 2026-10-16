/* 引用跳转（PDF 内部链接 → 参考文献卡片）
   ────────────────────────────────────────────────────────────────
   数据来自 `GET /api/doc/{id}/refs`（后端 backend/refs.py 算好，零 LLM 成本）：
     · entries[i] = { i, n, page, blocks[块id…], text, uri? }       参考文献条目
     · links[]    = { page, block, start, end, token, refs[条目下标], nums }
       start/end 是**块 canonical 文本**里的字符区间（与句子/高亮/笔记同口径）。
   本模块做两件事：
     1) 把正文里对应的 `[12]` 字符区间包成 `.cite` 标记。块会被反复重建（选句/高亮/
        缩放都会走 reader2.rebuildItem），所以标记做成**幂等**：每次重建后重打一遍；
     2) 点击 `.cite` → 弹一张卡片：条目原文 + 「定位原文」（滚到参考文献那条并闪烁）
        + 「复制」+ 条目带外链时加「打开链接」。
   交互上有一处必须抢：点击发生在**捕获阶段**（root 上）并 stopPropagation ——
   否则会被阅读器的“点句子选句”吃掉（工具条会弹出来）。拖拽多选不受影响
   （只接管 click；mousedown 仅 preventDefault 掉“从引用标记上按下”的文本选择）。
   卡片是 `position: fixed` 的单例浮层：点空白 / Esc / 滚动（“定位原文”引发的那次
   除外）自动收起；亮暗两套配色走 CSS 变量（见 style.css 的 .refpop 一段）。
   用法：const ui = R2Refs({ root: pagesEl }); ui.set(data); ui.markAll() / ui.markItem(item);
*/
window.R2Refs = function (deps) {
  "use strict";
  const root = (deps && deps.root) || document;
  let data = null;               // 最近一次 set() 的数据（无效数据 = null）
  let byBlock = new Map();       // 块 id → [锚点, …]（按 start 升序）
  let popEl = null;              // 复用的弹层（懒建）
  let lastJumpAt = 0;            // 「定位原文」引发的滚动：短窗口内不收卡片

  /* ================= 数据 ================= */
  function set(d) {
    data = d && d.ok && Array.isArray(d.links) ? d : null;
    byBlock = new Map();
    if (data) {
      for (const l of data.links) {
        if (!l || !l.block) continue;
        let arr = byBlock.get(l.block);
        if (!arr) byBlock.set(l.block, arr = []);
        arr.push(l);
      }
      for (const arr of byBlock.values())
        arr.sort((a, b) => (a.start | 0) - (b.start | 0));
    }
    hide();
  }

  function findBlockEl(bid) {
    if (!root.querySelector) return null;
    const s = String(bid).replace(/["\\]/g, "\\$&");
    return root.querySelector('.blk[data-id="' + s + '"]');
  }

  /* ================= 打标记 ================= */
  /* 摘掉某块里已打的 .cite（幂等的前半步）。只把内层文本搬回原位、不碰其它结构；
     normalize 只对 .rn 父级做，避免动到 KaTeX 子树里精心分开的文本节点。 */
  function clearBlock(blockEl) {
    const chips = blockEl.querySelectorAll(".cite");
    for (const chip of chips) {
      const parent = chip.parentNode;
      if (!parent) continue;
      while (chip.firstChild) parent.insertBefore(chip.firstChild, chip);
      parent.removeChild(chip);
      if (parent.normalize) parent.normalize();
    }
  }

  /* 在某个 `[data-g]` 的 span 里把 [local, local+len) 包成 .cite。
     文本节点是累计-offset 定位的：同一个 span 里先包出来的 chip 自身也计入长度，
     而它的文本本来就是 canonical 的一部分 —— 所以多个锚点共用一个大 span 也不会错位。
     KaTeX 替换过的区间不在文本节点里，累计长度对不上时直接失败返回（外层放弃该锚点）。 */
  function wrapInSpan(sp, local, len, anchor) {
    let acc = 0, node = null, off = 0;
    const walker = document.createTreeWalker(sp, NodeFilter.SHOW_TEXT, null);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const L = n.nodeValue.length;
      if (local >= acc && local < acc + L) { node = n; off = local - acc; break; }
      acc += L;
    }
    if (!node || off + len > node.nodeValue.length) return false;
    const right = node.splitText(off);
    right.splitText(len);
    const chip = document.createElement("span");
    chip.className = "cite";
    chip.dataset.refs = (anchor.refs || []).join(",");
    chip.dataset.token = anchor.token || "";
    chip.textContent = right.nodeValue;
    chip.title = "点击查看 " + (anchor.token || "参考文献");
    right.parentNode.replaceChild(chip, right);
    return true;
  }

  /* 把块内 [start, end) 区间包成 .cite。
     `[data-g]` span 的 canonical 区间 = [g, 下一个 span 的 g)（被 KaTeX 替换掉的
     span 不在 DOM 里，它那段区间也照跳不误）；区间中间遇到“洞”（换行符/公式）就放弃：
     宁可不标，也不把标记劈到两个不相邻的地方。 */
  function wrapRange(blockEl, start, end, anchor) {
    const spans = blockEl.querySelectorAll("[data-g]");
    let cur = start;
    for (let k = 0; k < spans.length && cur < end; k++) {
      const g = parseInt(spans[k].dataset.g, 10);
      if (isNaN(g)) continue;
      const next = k + 1 < spans.length
        ? parseInt(spans[k + 1].dataset.g, 10) : Infinity;
      if (cur < g) return false;              // 中间有洞 → 放弃
      if (cur >= next) continue;              // 这个 span 在区间之前
      const take = Math.min(end, next) - cur;
      if (!wrapInSpan(spans[k], cur - g, take, anchor)) return false;
      cur += take;
    }
    return cur >= end;
  }

  function markBlockEl(blockEl, bid) {
    const list = byBlock.get(bid);
    if (!list || !list.length) return;
    clearBlock(blockEl);
    for (const a of list) {
      const s = a.start | 0, e = a.end | 0;
      if (!(e > s)) continue;
      wrapRange(blockEl, s, e, a);
    }
  }

  /* reader2.rebuildItem 末尾调用：块重建后重打一遍标记（幂等） */
  function markItem(item) {
    if (!data || !item || !item.el) return;
    markBlockEl(item.el, item.id);
  }

  /* 数据晚于渲染到达时（一般不会：init 里在 render 之前 set）：全量补一遍 */
  function markAll() {
    if (!data) return;
    for (const bid of byBlock.keys()) {
      const el = findBlockEl(bid);
      if (el) markBlockEl(el, bid);
    }
  }

  /* ================= 弹层 ================= */
  function pop() {
    if (!popEl) {
      popEl = document.createElement("div");
      popEl.className = "refpop";
      popEl.hidden = true;
      document.body.appendChild(popEl);
      popEl.addEventListener("click", onPopClick);
    }
    return popEl;
  }
  function hide() { if (popEl) popEl.hidden = true; }

  function onPopClick(e) {
    const t = e.target;
    const jump = t.closest ? t.closest("[data-jump]") : null;
    if (jump) {
      const entry = data && data.entries && data.entries[parseInt(jump.dataset.jump, 10)];
      if (entry) jumpTo(entry);
      return;
    }
    const copy = t.closest ? t.closest("[data-copy]") : null;
    if (copy) {
      const entry = data && data.entries && data.entries[parseInt(copy.dataset.copy, 10)];
      if (entry && navigator.clipboard) {
        navigator.clipboard.writeText(entry.text || "").then(
          () => { if (window.toast) toast("已复制该条参考文献"); },
          () => {});
      }
      return;
    }
    if (t.closest && t.closest(".rp-x")) hide();     // ✕ 关闭按钮
  }

  function jumpTo(entry) {
    let first = null;
    for (const bid of (entry.blocks || [])) {
      const el = findBlockEl(bid);
      if (!el) continue;
      if (!first) first = el;
      el.classList.add("ref-flash");
      setTimeout(() => el.classList.remove("ref-flash"), 1700);
    }
    if (first) {
      lastJumpAt = Date.now();
      first.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  function mk(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text != null) el.textContent = text;
    return el;
  }

  function openFor(chip) {
    if (!data) return;
    const idxs = String(chip.dataset.refs || "").split(",")
      .map(s => parseInt(s, 10)).filter(v => !isNaN(v));
    const list = [];
    for (const i of idxs) {
      const e = data.entries && data.entries[i];
      if (e) list.push([i, e]);
    }
    if (!list.length) return;
    const el = pop();
    el.textContent = "";
    const head = mk("div", "rp-head");
    head.appendChild(mk("b", "", "参考文献"));
    head.appendChild(mk("span", "rp-nums",
      list.map(([, e]) => (e.n ? "[" + e.n + "]" : "•")).join(" ")));
    const close = mk("button", "rp-x", "✕");
    close.title = "关闭";
    head.appendChild(close);
    el.appendChild(head);

    const box = mk("div", "rp-list");
    for (const [i, e] of list) {
      const item = mk("div", "rp-item");
      item.appendChild(mk("span", "rp-badge", e.n ? "[" + e.n + "]" : "•"));
      const main = mk("div", "rp-main");
      main.appendChild(mk("div", "rp-text",
        e.text || "（这一条没有可显示的文本）"));
      const acts = mk("div", "rp-acts");
      const jb = mk("button", "", "定位原文 ↧");
      jb.dataset.jump = String(i);
      acts.appendChild(jb);
      const cb = mk("button", "", "复制");
      cb.dataset.copy = String(i);
      acts.appendChild(cb);
      if (e.uri) {
        const a = mk("a", "", "打开链接 ↗");
        a.href = e.uri;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        acts.appendChild(a);
      }
      main.appendChild(acts);
      item.appendChild(main);
      box.appendChild(item);
    }
    el.appendChild(box);
    place(chip);
  }

  function place(chip) {
    const el = pop();
    const r = chip.getBoundingClientRect();
    el.hidden = false;
    el.style.visibility = "hidden";
    el.style.left = "0px";
    el.style.top = "0px";
    const pw = el.offsetWidth, ph = el.offsetHeight;
    let left = Math.max(8, Math.min(r.left, window.innerWidth - pw - 8));
    let top = r.bottom + 6;
    if (top + ph > window.innerHeight - 8) {
      const above = r.top - ph - 6;
      top = above >= 56 ? above : Math.max(56, window.innerHeight - ph - 8);
    }
    el.style.left = left + "px";
    el.style.top = top + "px";
    el.style.visibility = "";
  }

  /* ================= 事件 =================
     ⚠️ click 用**捕获**：抢在阅读器（pagesEl 的冒泡监听）之前接管并拦住，
     否则点引用会变成“选中该句 / 弹工具条”。 */
  function closestCite(t) {
    return t && t.closest ? t.closest(".cite") : null;
  }
  function onPageDown(e) {
    if (closestCite(e.target)) e.preventDefault();   // 别从引用标记上拉出文本选择
  }
  function onPageClick(e) {
    const chip = closestCite(e.target);
    if (!chip || !(root === document || root.contains(chip))) return;
    e.preventDefault();
    e.stopPropagation();
    openFor(chip);
  }
  function onDocClick(e) {
    if (!popEl || popEl.hidden) return;
    if (popEl.contains(e.target)) return;
    if (closestCite(e.target)) return;
    hide();
  }
  function onScroll() {
    // 刚点过「定位原文」：平滑滚动本身要几百毫秒，余波也别把卡片收掉
    if (Date.now() - lastJumpAt < 1500) return;
    hide();
  }
  function onWheel() {
    // 用户主动滚轮 → 立即收掉（“定位原文”平滑滚动的余波不算用户滚动）
    if (Date.now() - lastJumpAt > 300) hide();
  }
  function onKey(e) { if (e.key === "Escape") hide(); }

  root.addEventListener("mousedown", onPageDown, true);
  root.addEventListener("click", onPageClick, true);
  document.addEventListener("click", onDocClick, true);
  window.addEventListener("scroll", onScroll, { passive: true, capture: true });
  window.addEventListener("wheel", onWheel, { passive: true });
  window.addEventListener("resize", hide);
  document.addEventListener("keydown", onKey);

  return { set, markAll, markItem, hide };
};
