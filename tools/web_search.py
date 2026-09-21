# -*- coding: utf-8 -*-
"""联网搜索工具（SerpApi Google 搜索）。

需要配置环境变量 SERPAPI_API_KEY（可写入 .env 或启动前 export）。
依赖: pip install serpapi
"""
from langchain_core.tools import tool

from config import SERPAPI_API_KEY


@tool
def web_search(query: str) -> str:
    """联网搜索实时信息（通过 SerpApi 调用 Google 搜索）。

    当用户询问时事、最新动态，或本地参考材料未覆盖的内容时使用。
    CRITICAL RULE: When the user's question contains '最新' / '最近' / '当前' (latest/recent),
    you MUST NOT add any specific year (like 2025) to the query.
    Instead, the query should be a natural language phrase as similar as possible to the user's original wording.
    Example: user asks "华为最新款的手机" -> query = "华为最新款手机" (do NOT change to "华为 Mate 80 2025").

    Use this tool when the user asks about real-time events, latest news, or when local reference materials are insufficient.
    Args:
        query: 搜索关键词（中文或英文均可）。
    """
    if not SERPAPI_API_KEY:
        return "错误: SERPAPI_API_KEY 未配置，请先设置环境变量。"

    # 延迟导入：未安装 serpapi 时不影响服务启动
    import serpapi

    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        client = serpapi.Client(api_key=SERPAPI_API_KEY)
        results = client.search({
            "engine": "google",
            "q": query,
            "gl": "cn",      # 国家代码
            "hl": "zh-cn",   # 语言代码
        })

        # 智能解析: 优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 没有直接答案时，返回前三个有机结果的摘要
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"
