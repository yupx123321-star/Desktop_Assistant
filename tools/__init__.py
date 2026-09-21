# -*- coding: utf-8 -*-
"""服务端 Agent 工具集统一导出。

每个工具独立一个文件，便于维护和扩展。
新增工具时：
1. 在 tools/ 下创建新文件（如 tools/my_tool.py）
2. 在此 __init__.py 中导入并加入 __all__
"""
from tools.web_search import web_search
from tools.rag_search import rag_search, retrieve_context, _rerank_documents
from tools.mcmod_search import mcmod_search
from tools.screenshot import (
    Screenshot,
    save_screenshot,
    get_next_screenshot,
    analyze_screenshot,
    make_request_screenshot_tool,
    cleanup_old_screenshots,
)

__all__ = [
    "web_search",
    "rag_search",
    "retrieve_context",
    "_rerank_documents",
    "mcmod_search",
    "Screenshot",
    "save_screenshot",
    "get_next_screenshot",
    "analyze_screenshot",
    "make_request_screenshot_tool",
    "cleanup_old_screenshots",
]
