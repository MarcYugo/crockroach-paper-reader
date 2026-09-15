"""模块八 · AI 辅助阅读（文末：把笔记整理清楚 + 就这篇聊下去）

定位：一篇文章读到结尾（进度 ≥ 95%）时，把它上面的**笔记**整理成结构化摘要，
并在同一屏提供对话 —— 目标是「帮用户想清楚」，不是「替用户读书」。

只依赖 `translate.Translator`（OpenAI 兼容后端）：Google 免费兜底只会逐句翻译，
这类任务会直接抛 `TranslationUnavailable`，前端提示去「⚙ 设置 → LLM 服务」配 Key。

两个入口：

| 函数 | 作用 |
| --- | --- |
| `build_summary()` | 把笔记整理成 JSON 摘要（总览 / 主题 / 关联 / 疑问 / 下一步） |
| `reply_chat()` | 带着笔记与摘要聊一轮，返回助手正文（整段） |
| `reply_chat_stream()` | 同上，但**逐段 yield**，给前端做流式显示 |

`build_summary()` 除了笔记，还能带上**上一版整理**（`previous`，可能被用户手改过）与
**已发生的对话**（`chat`）—— 这就是「重新整理」把讨论成果并进来的方式。
`clean_summary(loose=True)` 给手工编辑用：上限放宽，不会把用户写的内容默默截掉。

`reply_chat_stream()` 与 `reply_chat()` 走同一个提示词（`chat_payload()` 共用），
只是把 `Translator.chat()` 换成 `Translator.chat_stream()`。

摘要为什么是 **JSON** 而不是 Markdown：前端没有构建步骤、也不想引第三方渲染库，
结构化数据可以直接渲染成卡片；而且模型偶尔字段缺失/类型不对，`clean_summary()`
能逐字段兜住，不会把页面弄花。

笔记上下文 `[{sent, note, page}]` 由前端提供 —— 只有前端知道「这条笔记挂在哪句话上」
（句子切分在 `reader2.js` 里做），后端负责收敛长度并拼成提示词。
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Iterator

from .translate import Translator

MAX_NOTES = 200            # 送进模型的笔记条数上限
MAX_SENT_CHARS = 600       # 单条「原句」截断
MAX_NOTE_CHARS = 1500      # 单条笔记截断
MAX_CONTEXT_CHARS = 12000  # 拼给模型的笔记上下文总长上限
MAX_HISTORY = 20           # 聊天回灌的历史条数（不含本次提问）
MAX_REPLY_CHARS = 6000     # 助手回复保留长度上限（防模型跑飞）
MAX_CHAT_TURNS = 24        # 「重新整理」时把对话当素材：最多带几轮
MAX_CHAT_CHARS = 8000      # 「重新整理」时对话素材的总长上限

# 手工编辑整理结果时的上限（比模型输出宽松，避免把用户写的东西默默截掉）
EDIT_THEMES = 12
EDIT_POINTS = 20
EDIT_ITEMS = 20
EDIT_TITLE_CHARS = 200
EDIT_ITEM_CHARS = 1200
EDIT_OVERVIEW_CHARS = 4000

_SUMMARY_SYSTEM = (
    "你是学术阅读助手。用户读一篇论文时做了若干条笔记，你的任务是**把他的笔记整理清楚**，"
    "让他看清自己记了什么、这些点之间有什么关系、还有什么没想清楚。\n"
    "规则：\n"
    "1. 只依据用户给的「笔记 + 它挂在哪句话上」，**不要引入论文之外的资料**，也不要编造论文内容。\n"
    "2. 用**中文**输出；关键术语可以保留英文原词。\n"
    "3. 忠实于笔记：不要替用户下他还没下的结论；笔记之间若互相矛盾，指出来，而不是抹平。\n"
    "4. `overview`：3~5 句，概括**这些笔记整体在关心什么**（不是概括整篇论文）。\n"
    "5. `themes`：把笔记按主题分成 1~5 组，每组一个短标题 + 若干条要点；"
    "每条要点尽量带上页码（如「P3」）。\n"
    "6. `connections`：笔记之间的关联 / 递进 / 冲突（没有就给空数组）。\n"
    "7. `questions`：从笔记里能看出、但还没被回答的问题（目的是帮他想下去，不是给答案）。\n"
    "8. `next`：接下来可以做的事（回去读哪一节、验证什么、补什么笔记）。\n"
    "9. 没有内容的字段返回空数组，**不要**写「暂无」「无」之类的占位符。\n"
    "10. 如果给了「上一版整理」：以它为基础**增量更新** —— 我自己改过、又没被"
    "笔记或对话推翻的内容要原样保留，别因为换了措辞就把内容丢掉。\n"
    "11. 如果给了「我们之后的对话」：把我已经聊清楚/确认过的结论并进 `themes` 与 `next`，"
    "把新冒出来的疑问并进 `questions`。\n"
    "12. 不要在结果里出现「上一版」「对话中提到」「用户改成」这类元话术，"
    "直接写成结论本身。\n"
    "只输出 JSON：{\"overview\": \"...\", \"themes\": [{\"title\": \"...\", \"points\": [\"...\"]}],"
    " \"connections\": [\"...\"], \"questions\": [\"...\"], \"next\": [\"...\"]}"
)

_CHAT_SYSTEM = (
    "你是学术阅读助手，正在和用户讨论他正在读的一篇论文。下面是这篇论文的标题、"
    "他做的笔记、以及此前生成的笔记整理。\n"
    "原则：\n"
    "1. 以**帮他想清楚**为目标：先给结构和思路，再给结论；需要时反问一句帮助他往下想，"
    "但一次最多问一个问题。\n"
    "2. 只依据下面提供的内容回答；信息不够就直说「你的笔记里没有提到」，并指出还需要看"
    "论文的哪一部分，**不要编造论文内容**。\n"
    "3. 用中文回答，简洁（一般 150~400 字）；必要时用小标题或短列表，不要空话、不要复述问题。\n"
    "4. 引用笔记时带上页码（如 P4）。"
)


def clean_notes(raw) -> list[dict]:
    """收敛前端传来的笔记上下文：条数、单条长度、总长度都设上限。"""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    used = 0
    for r in raw[:MAX_NOTES]:
        if not isinstance(r, dict):
            continue
        note = " ".join(str(r.get("note") or "").split())[:MAX_NOTE_CHARS]
        if not note:
            continue                                  # 空笔记不占位置
        sent = " ".join(str(r.get("sent") or "").split())[:MAX_SENT_CHARS]
        try:
            page = int(r.get("page") or 0)
        except Exception:
            page = 0
        size = len(sent) + len(note) + 24
        if used + size > MAX_CONTEXT_CHARS:
            break
        out.append({"sent": sent, "note": note, "page": max(0, page)})
        used += size
    return out


def notes_text(notes: list[dict]) -> str:
    """把笔记拼成给模型看的文本：一条一段，带页码与它挂在哪句话上。"""
    if not notes:
        return "（还没有笔记）"
    blocks: list[str] = []
    for i, n in enumerate(notes, 1):
        lines = [f"{i}. " + (f"【第 {n['page']} 页】" if n.get("page") else "")]
        if n.get("sent"):
            lines.append(f"   原文句：{n['sent']}")
        lines.append(f"   我的笔记：{n['note']}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def notes_fingerprint(rows) -> str:
    """笔记指纹：只看「哪句话 + 笔记内容」，与时间无关。

    用它判断已生成的整理是否过期 —— 比时间戳可靠：同一秒内改笔记也看得出来，
    换台机器/时区不一致也不会误判。`rows` 是存储层的笔记（含 `block_id` 与 `text`）。
    """
    parts = sorted(f"{str(r.get('block_id') or '')}\x01{str(r.get('text') or '').strip()}"
                   for r in (rows or []) if isinstance(r, dict))
    return hashlib.sha1("\x02".join(parts).encode("utf-8")).hexdigest()[:16]


def _str_list(value, limit: int, item_chars: int) -> list[str]:
    """把「可能是字符串、可能是数组」的字段收敛成干净的字符串列表。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        s = " ".join(str(v or "").split())
        if s:
            out.append(s[:item_chars])
        if len(out) >= limit:
            break
    return out


def clean_summary(data, *, loose: bool = False) -> dict:
    """把（模型返回或用户手改的）JSON 收敛成固定结构。

    `loose=True` 给**手工编辑**用：条数与字数上限放宽，免得把用户写的内容默默截掉；
    模型输出用默认上限，防它跑飞。字段缺失/类型不对在这两种情况下都能兜住。
    """
    max_themes = EDIT_THEMES if loose else 5
    max_points = EDIT_POINTS if loose else 8
    max_items = EDIT_ITEMS if loose else 8
    max_title = EDIT_TITLE_CHARS if loose else 80
    item_chars = EDIT_ITEM_CHARS if loose else 300
    over_chars = EDIT_OVERVIEW_CHARS if loose else 1500
    data = data if isinstance(data, dict) else {}
    themes: list[dict] = []
    raw_themes = data.get("themes")
    if isinstance(raw_themes, dict):                 # 有的模型会包一层 {1: {...}}
        raw_themes = list(raw_themes.values())
    for t in (raw_themes or [])[:max_themes] if isinstance(raw_themes, list) else []:
        if isinstance(t, dict):
            title = " ".join(str(t.get("title") or "").split())[:max_title]
            points = _str_list(t.get("points"), max_points, item_chars)
        else:
            title = " ".join(str(t or "").split())[:max_title]
            points = []
        if title or points:
            themes.append({"title": title or "要点", "points": points})
    return {
        "overview": " ".join(str(data.get("overview") or "").split())[:over_chars],
        "themes": themes,
        "connections": _str_list(data.get("connections"), max_items, item_chars),
        "questions": _str_list(data.get("questions"), max_items, item_chars),
        "next": _str_list(data.get("next"), max_items, item_chars),
    }


def summary_empty(summary: dict | None) -> bool:
    """摘要里一条可用内容都没有（模型偶尔会返回空壳）。"""
    if not isinstance(summary, dict):
        return True
    if (summary.get("overview") or "").strip():
        return False
    return not any(summary.get(k) for k in ("themes", "connections", "questions", "next"))


def summary_text(summary: dict | None) -> str:
    """把摘要压成文本，给对话当上下文（也用于回灌给模型）。"""
    if not isinstance(summary, dict):
        return ""
    lines: list[str] = []
    if summary.get("overview"):
        lines.append("【总览】" + summary["overview"])
    for t in summary.get("themes") or []:
        lines.append("【主题】" + (t.get("title") or ""))
        lines.extend("  - " + p for p in (t.get("points") or []))
    for key, label in (("connections", "关联"), ("questions", "待想清楚"),
                       ("next", "下一步")):
        rows = summary.get(key) or []
        if rows:
            lines.append(f"【{label}】")
            lines.extend("  - " + r for r in rows)
    return "\n".join(lines)


def history_text(chat) -> str:
    """把已发生的对话压成给模型看的文本（「重新整理」时当素材用）。

    只保留 `role` + `content`，并按轮数与总长度收敛 —— 整理任务不需要逐字精确。
    """
    rows: list[str] = []
    used = 0
    for m in (chat or [])[-MAX_CHAT_TURNS:]:
        if not isinstance(m, dict):
            continue
        content = " ".join(str(m.get("content") or "").split())
        if not content:
            continue
        line = ("我：" if m.get("role") == "user" else "AI：") + content[:MAX_NOTE_CHARS]
        if used + len(line) > MAX_CHAT_CHARS:
            break
        rows.append(line)
        used += len(line)
    return "\n".join(rows)


def build_summary(translator: Translator, *, title: str, notes: list[dict],
                  pages: int = 0, progress: float = 0.0,
                  previous: dict | None = None, chat=None) -> dict:
    """把笔记整理成结构化摘要。LLM 不可用时抛 `TranslationUnavailable`。

    `previous` 是上一版整理（**可能被用户手改过**），`chat` 是已发生的对话：
    两者都作为素材一起喂进去 —— 所以「重新整理」是把讨论成果并进来，而不是推倒重来。
    """
    if not notes:
        raise TranslationUnavailable("还没有笔记可以整理：先在正文里选中句子写几条笔记")
    head = [f"论文标题：{title or '（未知）'}"]
    if pages:
        pct = round(max(0.0, min(1.0, float(progress or 0))) * 100)
        head.append(f"论文共 {pages} 页，我读到 {pct}%")
    parts = ["\n".join(head) + f"\n\n我的笔记（共 {len(notes)} 条）：\n" + notes_text(notes)]
    prev = summary_text(previous)
    if prev:
        parts.append("===== 上一版整理（可能被我改过） =====\n" + prev)
    convo = history_text(chat)
    if convo:
        parts.append("===== 我们之后的对话 =====\n" + convo)
    parts.append("请按要求输出这一版的 JSON。")
    return clean_summary(translator.chat_json(_SUMMARY_SYSTEM, "\n\n".join(parts),
                                              temperature=0.2))


def reply_chat(translator: Translator, *, title: str, notes: list[dict],
               summary: dict | None, history: list[dict], message: str) -> str:
    """带着这篇的笔记与摘要回答一轮，返回助手正文。"""
    system, msgs = chat_payload(title=title, notes=notes, summary=summary,
                                history=history, message=message)
    reply = (translator.chat(system, msgs, temperature=0.5) or "").strip()
    if not reply:
        raise TranslationUnavailable("模型没有返回内容，请重试")
    return reply[:MAX_REPLY_CHARS]


def reply_chat_stream(translator: Translator, *, title: str, notes: list[dict],
                      summary: dict | None, history: list[dict], message: str
                      ) -> Iterator[str]:
    """流式版 `reply_chat()`：返回一个**逐段 yield 正文增量**的生成器。

    与 `reply_chat()` 一样，参数不对/没配 LLM 会在**调用时**立即抛
    `TranslationUnavailable`（方便接口层先回 503）；真正的网络请求等开始迭代才发。
    总长仍然招在 `MAX_REPLY_CHARS` 以内 —— 超了就停，免得模型跑飞。
    """
    system, msgs = chat_payload(title=title, notes=notes, summary=summary,
                                history=history, message=message)
    return _cap_chunks(translator.chat_stream(system, msgs, temperature=0.5),
                       MAX_REPLY_CHARS)


def _cap_chunks(chunks: Iterable[str], limit: int) -> Iterator[str]:
    """把增量流的总长度招到 `limit`：到顶就结束，不会多吐一个字。"""
    used = 0
    for c in chunks:
        text = c or ""
        if not text:
            continue
        room = limit - used
        if room <= 0:
            return
        text = text[:room]
        used += len(text)
        yield text


def chat_payload(*, title: str, notes: list[dict], summary: dict | None,
                 history: list[dict], message: str) -> tuple[str, list[dict]]:
    """拼出对话要发给模型的 `(system, messages)`。两个对话入口共用。"""
    message = (message or "").strip()
    if not message:
        raise TranslationUnavailable("请输入内容")
    ctx = [
        f"===== 论文标题 =====\n{title or '（未知）'}",
        f"===== 我的笔记（共 {len(notes)} 条）=====\n{notes_text(notes)}",
        "===== 已生成的笔记整理 =====\n" + (summary_text(summary) or "（还没生成整理）"),
    ]
    msgs: list[dict] = []
    for m in (history or [])[-MAX_HISTORY:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": message})
    return _CHAT_SYSTEM + "\n\n" + "\n\n".join(ctx), msgs
