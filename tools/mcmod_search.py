# -*- coding: utf-8 -*-
"""MC 模组百科检索工具（mcmod.cn 搜索 + 详情 + 评论）。

依赖: pip install requests beautifulsoup4 lxml
"""
import json
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.tools import tool
from infrastructure.models import load_embeddings, load_reranker

# ---------- 常量 ----------
MC_SEARCH_URL = "https://search.mcmod.cn/s"
MC_DETAIL_BASE = "https://www.mcmod.cn"
MC_COMMENT_ROW_URL = "https://www.mcmod.cn/frame/comment/CommentRow/"
MC_COMMENT_REPLY_URL = "https://www.mcmod.cn/frame/comment/CommentReply/"
MC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 板块标题启发式判定参数：mcmod 的板块名几乎随机（如 "1.18之前"/"两种途径获得"/"获取途径："），
# 不依赖固定词典，改用结构特征：纯文本长度 ≤ HEADING_MAX_LEN 且不以句末标点结尾。
MC_HEADING_MAX_LEN = 15
MC_SENTENCE_END_PUNCT = "。！？.!?…"
MC_COMMENT_CANDIDATE_NUM = 20
MC_RELEVANT_COMMENT_NUM = 3
MC_RELEVANT_REPLY_NUM = 3


# ---------- 内部辅助函数 ----------
def _mc_parse_query(raw: str) -> Tuple[str, str]:
    parts = re.split(r"[\\/]", raw)
    if len(parts) < 2 or not parts[0].strip() or not parts[-1].strip():
        raise ValueError("输入格式应为 '模组名\\物品名' 或 '模组名/物品名'")
    return parts[0].strip(), parts[-1].strip()


def _mc_absolute_href(href: str) -> str:
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return MC_DETAIL_BASE + href
    return href


def _mc_get(session, url: str, params: Optional[dict] = None) -> str:
    response = session.get(url, params=params, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def _mc_search_items(session, item_name: str) -> List[Dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        _mc_get(session, MC_SEARCH_URL, params={"key": item_name}), "lxml"
    )
    results = []
    for item in soup.select(".search-result-list .result-item"):
        record = {"title": None, "url": None, "summary": None, "mod_source": None}
        head = item.select_one(".head")
        if head:
            link = head.find("a", href=True)
            if link:
                record["title"] = link.get_text(strip=True)
                record["url"] = _mc_absolute_href(link["href"])
            record["mod_source"] = head.get_text(" ", strip=True)
        body = item.select_one(".body")
        if body:
            record["summary"] = body.get_text(" ", strip=True)
        if record["url"]:
            results.append(record)
    return results


def _mc_pick_item_url(results: List[Dict], mod_name: str, item_name: str) -> Optional[str]:
    for record in results:
        if item_name in (record.get("title") or "") and mod_name in (record.get("mod_source") or ""):
            return record["url"]
    return None


def _mc_item_id(url: str) -> str:
    match = re.search(r"/item/(\d+)\.html", url)
    if not match:
        raise RuntimeError(f"无法从详情页 URL 提取物品 ID: {url}")
    return match.group(1)


def _mc_html_text(html: str) -> str:
    from bs4 import BeautifulSoup

    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for image in soup.find_all("img"):
        image.replace_with("[表情]")
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))


def _mc_is_heading(text: str) -> bool:
    """启发式判断一段纯文本是否为板块标题。

    不依赖固定词典（mcmod 的板块名几乎随机，如 "1.18之前"/"两种途径获得"/
    "获取途径："/"用途：" 等），改用结构特征：
      - 纯文本长度 ≤ MC_HEADING_MAX_LEN
      - 不以句末标点结尾（标题不是完整句子）
    """
    if not text:
        return False
    if len(text) > MC_HEADING_MAX_LEN:
        return False
    return text[-1] not in MC_SENTENCE_END_PUNCT


def _mc_extract_sections(soup) -> List[Dict]:
    """把 .item-content 容器按板块标题切成多段。

    动态适配：板块名无固定规律，用 _mc_is_heading() 启发式识别标题边界；
    没有标题的物品全部归入 "描述" 板块。

    返回: [{"title": "1.18之前", "content": "纯文本"}, ...]
    板块按文档顺序返回；首段内容前无标题时用 "描述" 作默认标题。
    """
    node = (soup.select_one(".item-content.common-text.font14")
            or soup.select_one(".item-content"))
    if not node:
        return []

    sections: List[Dict] = []
    current_title = "描述"
    current_nodes: List = []

    def _flush():
        if not current_nodes:
            return
        text_parts = []
        for n in current_nodes:
            t = n.get_text(" ", strip=True)
            if t:
                text_parts.append(t)
        if text_parts:
            sections.append({
                "title": current_title,
                "content": "\n".join(text_parts),
            })

    for child in node.find_all(recursive=False):
        if child.name == "p":
            text = child.get_text(strip=True)
            if _mc_is_heading(text):
                _flush()
                current_title = text
                current_nodes = []
                continue
        current_nodes.append(child)

    _flush()
    return sections


def _mc_fetch_comments(session, item_id: str, max_pages: int) -> List[Dict]:
    session.headers.update({
        "Referer": f"{MC_DETAIL_BASE}/item/{item_id}.html",
        "Origin": MC_DETAIL_BASE,
        "X-Requested-With": "XMLHttpRequest",
    })
    rows = []
    for page in range(1, max_pages + 1):
        payload = {"type": "item", "channel": "1", "doid": item_id, "page": page, "selfonly": 0}
        try:
            response = session.post(
                MC_COMMENT_ROW_URL,
                data={"data": json.dumps(payload, ensure_ascii=False)},
                timeout=15,
            )
            data = response.json().get("data") or {}
        except Exception:
            break
        if data.get("is_empty") or not data.get("row"):
            break
        rows.extend(data["row"])

    comments = []
    for row in rows:
        user = row.get("user") or {}
        time_info = row.get("time") or {}
        attitude = row.get("attitude") or {}
        quote = row.get("quote")
        record = {
            "floor": row.get("floor", ""),
            "author": user.get("name", ""),
            "user_id": user.get("id", ""),
            "content": _mc_html_text(row.get("content", "")),
            "time": time_info.get("source", ""),
            "time_relative": time_info.get("range", ""),
            "reply_count": row.get("reply_count", "0"),
            "like": attitude.get("up", 0),
            "comment_id": row.get("id", ""),
            "quote": None,
        }
        if isinstance(quote, dict) and quote.get("content") is not None:
            record["quote"] = {
                "floor": quote.get("floor", ""),
                "author": quote.get("name", ""),
                "content": _mc_html_text(quote.get("content", "")),
            }
        comments.append(record)

    def floor_number(comment: Dict) -> int:
        match = re.search(r"(\d+)", comment.get("floor", ""))
        return int(match.group(1)) if match else 0

    comments.sort(key=floor_number)
    return comments


def _mc_fetch_replies(session, parent_id: str, max_pages: int) -> List[Dict]:
    session.headers.update({
        "Referer": MC_COMMENT_REPLY_URL,
        "Origin": MC_DETAIL_BASE,
        "X-Requested-With": "XMLHttpRequest",
    })
    rows = []
    for page in range(1, max_pages + 1):
        try:
            response = session.post(
                MC_COMMENT_REPLY_URL,
                data={"data": json.dumps({"replyID": str(parent_id), "page": page})},
                timeout=15,
            )
            if not response.text.strip():
                break
            data = response.json().get("data") or {}
        except Exception:
            break
        if data.get("is_empty") or not data.get("row"):
            break
        rows.extend(data["row"])

    replies = []
    for row in rows:
        user = row.get("user") or {}
        reply_user = row.get("reply_user") or {}
        time_info = row.get("time") or {}
        attitude = row.get("attitude") or {}
        replies.append({
            "floor": "",
            "author": user.get("name", ""),
            "user_id": user.get("id", ""),
            "reply_to": reply_user.get("name", ""),
            "reply_to_id": reply_user.get("id", ""),
            "content": _mc_html_text(row.get("content", "")),
            "time": time_info.get("source", ""),
            "time_relative": time_info.get("range", ""),
            "like": attitude.get("up", 0),
            "comment_id": row.get("id", ""),
        })
    return replies


def _mc_fetch_item(
    raw_query: str,
    question: str,
    max_comment_pages: int,
    max_reply_comments: int,
) -> Dict:
    import requests
    from bs4 import BeautifulSoup

    mod_name, item_name = _mc_parse_query(raw_query)
    session = requests.Session()
    session.headers.update(MC_HEADERS)
    results = _mc_search_items(session, item_name)
    url = _mc_pick_item_url(results, mod_name, item_name)
    if not url:
        # 未精确匹配：返回候选列表，让模型友善地提示用户
        candidates = [
            {"title": r.get("title"), "mod_source": r.get("mod_source"), "summary": r.get("summary")}
            for r in results[:10]
        ]
        return {
            "query": raw_query,
            "mod_name": mod_name,
            "item_name": item_name,
            "matched": False,
            "candidates": candidates,
            "message": f"未找到模组'{mod_name}'中名为'{item_name}'的物品，以下是搜索到的相似结果，请友善提示用户确认",
        }
    item_id = _mc_item_id(url)
    soup = BeautifulSoup(_mc_get(session, url), "lxml")
    all_comments = _mc_fetch_comments(session, item_id, max_comment_pages)
    comments = _mc_select_relevant_records(
        question,
        all_comments,
        _mc_comment_text,
        MC_COMMENT_CANDIDATE_NUM,
        MC_RELEVANT_COMMENT_NUM,
    )

    # 只为有回复的前 N 条评论抓子回复，避免一次工具调用产生过大的返回。
    replies_by_comment = {}
    reply_targets = [c for c in comments if c.get("reply_count", "0") not in (0, "0", "")]
    for comment in reply_targets[:max_reply_comments]:
        comment_id = str(comment.get("comment_id") or "")
        if comment_id:
            replies = _mc_fetch_replies(session, comment_id, max_comment_pages)
            replies_by_comment[comment_id] = _mc_select_relevant_records(
                question,
                replies,
                _mc_reply_text,
                MC_COMMENT_CANDIDATE_NUM,
                MC_RELEVANT_REPLY_NUM,
            )

    return {
        "query": raw_query,
        "mod_name": mod_name,
        "item_name": item_name,
        "matched": True,
        "item_id": item_id,
        "item_url": url,
        "search_results_count": len(results),
        "comments_total_count": len(all_comments),
        "sections": _mc_extract_sections(soup),
        "comments": comments,
        "replies": replies_by_comment,
    }


# ---------- 工具入口 ----------
@tool
def mcmod_search(
    query: str,
    question: str,
    max_comment_pages: int = 5,
    max_reply_comments: int = 3,
) -> str:
    """从 mcmod.cn 查询 Minecraft 模组物品的完整百科信息。

    返回内容包括：
    - sections: 所有板块（物品简介、详细过程、合成方法、更新日志等），每段含 title/content
    - comments: 与 question 最相关的最多 3 条结构化评论
    - replies: 与 question 最相关的评论下，筛选后的子回复

    输入格式必须是"模组名/物品名"或"模组名\\物品名"，例如"工业先锋/火力发电机"。
    question 应传入用户关于该物品的原始问题，用于评论和子回复的语义筛选。
    当用户询问 Minecraft 模组物品的属性、合成、用途、制作过程、使用经验、评价或评论内容时调用。
    max_comment_pages 默认只抓 5 页，max_reply_comments 默认只抓 3 条有回复的评论，
    以控制网站请求数量和返回内容大小；需要更多内容时可以调整参数。
    """
    if not query or not query.strip():
        return "错误: mc百科查询不能为空，格式应为 '模组名/物品名'。"
    if not question or not question.strip():
        return "错误: question 不能为空，必须传入用户关于该物品的完整问题。"
    try:
        result = _mc_fetch_item(
            query.strip(),
            question.strip(),
            max(1, min(max_comment_pages, 50)),
            max(0, min(max_reply_comments, 20)),
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except ImportError:
        return "错误: mcmod_search 需要依赖 requests、beautifulsoup4、lxml，请先安装。"
    except Exception as error:
        return f"mc百科检索失败: {error}"


def _mc_comments_to_texts(comments: List[Dict]) -> List[str]:
    """每条评论拼成一段纯文本，用于嵌入 / 重排序模型。"""
    texts: List[str] = []
    for c in comments:
        parts = []
        if c.get("floor"):
            parts.append(f"[{c['floor']}]")
        if c.get("author"):
            parts.append(f"{c['author']}:")
        parts.append(c.get("content", ""))
        q = c.get("quote")
        if q:
            parts.append(
                f"  (引用 {q.get('floor', '')} {q.get('author', '')}: "
                f"{q.get('content', '')})"
            )
        texts.append(" ".join(p for p in parts if p).strip())
    return texts


def _mc_replies_to_texts(replies: List[Dict]) -> List[str]:
    """把子回复拼成纯文本列表，便于直接传入嵌入/重排序模型。"""
    texts: List[str] = []
    for r in replies:
        parts = []
        if r.get("author"):
            parts.append(f"{r['author']}:")
        if r.get("reply_to"):
            parts.append(f"回复 {r['reply_to']}:")
        parts.append(r.get("content", ""))
        texts.append(" ".join(p for p in parts if p).strip())
    return texts


def _mc_select_relevant_records(
    query: str,
    records: List[Dict],
    text_builder,
    candidate_num: int,
    top_k: int,
) -> List[Dict]:
    """先用 embedding 初筛，再用 CrossEncoder 重排记录。"""
    if not query or not records:
        return records[:top_k]

    texts = [text_builder(record) for record in records]
    valid = [(record, text) for record, text in zip(records, texts) if text.strip()]
    if not valid:
        return records[:top_k]

    try:
        embeddings = load_embeddings()
        query_vector = embeddings.embed_query(query)
        record_vectors = embeddings.embed_documents([text for _, text in valid])

        def cosine(vector):
            numerator = sum(a * b for a, b in zip(query_vector, vector))
            query_norm = sum(value * value for value in query_vector) ** 0.5
            vector_norm = sum(value * value for value in vector) ** 0.5
            return numerator / (query_norm * vector_norm) if query_norm and vector_norm else 0.0

        candidates = sorted(
            ((record, text, cosine(vector))
             for (record, text), vector in zip(valid, record_vectors)),
            key=lambda item: item[2],
            reverse=True,
        )[:candidate_num]

        reranker = load_reranker()
        scores = reranker.predict(
            [(query, text) for _, text, _ in candidates],
            batch_size=8,
            max_length=2048,
            show_progress_bar=False,
        )
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        selected = []
        for (record, _, _), score in ranked:
            if score < 0:
                continue
            selected_record = dict(record)
            selected_record["relevance_score"] = float(score)
            selected.append(selected_record)
        return selected
    except Exception:
        # 模型服务暂不可用时不返回任何结果
        return []


def _mc_comment_text(comment: Dict) -> str:
    return " ".join(_mc_comments_to_texts([comment]))


def _mc_reply_text(reply: Dict) -> str:
    return " ".join(_mc_replies_to_texts([reply]))
