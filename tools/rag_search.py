# -*- coding: utf-8 -*-
"""RAG 本地知识库检索工具（Chroma + CrossEncoder 重排序）。"""
from langchain_core.tools import tool

from config import DEFAULT_SEARCH_NUM, DEFAULT_RERANK_NUM
from infrastructure.models import load_chroma, load_reranker


def _rerank_documents(query, documents, top_k=DEFAULT_RERANK_NUM):
    """用 CrossEncoder 对初检文档重排序，返回 [(doc, score)]（按分数降序）。"""
    if not documents:
        return []
    model = load_reranker()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs, batch_size=8, max_length=2048, show_progress_bar=False)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [(doc, score) for doc, score in doc_score_pairs[:top_k]]


def retrieve_context(query, search_num=DEFAULT_SEARCH_NUM, rerank_num=DEFAULT_RERANK_NUM):
    """检索本地知识库，返回重排后且分数达标的文档内容列表。"""
    chroma = load_chroma()
    initial_results = chroma.similarity_search_with_relevance_scores(query, k=search_num)
    docs = [doc for doc, _ in initial_results]
    reranked_docs = _rerank_documents(query, docs, top_k=rerank_num)
    # 过滤低分（可根据实际情况调整阈值）
    return [doc.page_content for doc, score in reranked_docs if score > 1.0]


@tool
def rag_search(query: str) -> str:
    """检索本地知识库（RAG），获取与问题相关的参考材料。

    当用户的问题涉及本地知识库中已有的资料（如角色设定、剧情、攻略等
    预先入库的内容）时调用此工具，把检索到的参考材料作为回答依据。
    若返回"未找到相关资料"，说明知识库中没有匹配内容，可改用 web_search
    联网搜索，或基于已有信息回答。

    Args:
        query: 用于检索的关键词或自然语言问题（中文或英文均可）。
    """
    if not query or not query.strip():
        return "错误: 检索关键词不能为空。"
    print(f"📚 正在执行 [RAG] 本地知识库检索: {query}")
    try:
        docs = retrieve_context(query.strip())
    except Exception as e:
        return f"检索时发生错误: {e}"
    if not docs:
        return "未找到相关资料。"
    return "\n\n".join(f"[参考材料 {i + 1}]\n{doc}" for i, doc in enumerate(docs))
