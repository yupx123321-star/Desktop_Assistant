# -*- coding: utf-8 -*-
"""Agent 调试测试脚本：打印 RAG 检索详情 + 工具调用过程 + 最终回复

用法（在 vllm 环境、服务端所在机器上运行）:
    python test_agent.py "关于茶会礼仪"
    python test_agent.py "今天有什么新闻" "介绍一下枫丹"   # 多个问题依次测试
    python test_agent.py                                    # 交互模式
"""
import sys
import uuid

from config import *
from infrastructure import ensure_all_loaded, load_chroma
from agent.core import build_agent, clean_response
from tools.rag_search import _rerank_documents
from langchain.messages import HumanMessage, AIMessage, ToolMessage

SEP = "=" * 70


def print_rag_details(query: str):
    """打印 RAG 检索全过程：初检 top-k → 重排分数 → 过滤后结果"""
    print(f"\n{SEP}\n📚 [RAG] 查询: {query}\n{SEP}")
    chroma = load_chroma()

    # 1. 相似度初检
    initial = chroma.similarity_search_with_relevance_scores(query, k=DEFAULT_SEARCH_NUM)
    print(f"\n[1] 相似度初检 (top {len(initial)}):")
    for i, (doc, score) in enumerate(initial[:5]):  # 只打印前5条，避免刷屏
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  #{i+1} score={score:.4f} | {preview}...")
    if len(initial) > 5:
        print(f"  ... 共 {len(initial)} 条")

    # 2. 重排序
    docs = [doc for doc, _ in initial]
    reranked = _rerank_documents(query, docs, top_k=DEFAULT_RERANK_NUM)
    print(f"\n[2] 重排序结果 (top {len(reranked)}, 阈值 {1.0}):")
    for doc, score in reranked:
        preview = doc.page_content[:80].replace("\n", " ")
        mark = "✅" if score > 1.0 else "❌(被过滤)"
        print(f"  {mark} rerank={score:.4f} | {preview}...")

    # 3. 最终进入 prompt 的参考材料
    final_docs = [doc.page_content for doc, score in reranked if score > 1.0]
    print(f"\n[3] 最终参考材料: {len(final_docs)} 条")
    if not final_docs:
        print("  (空 —— 所有文档 rerank 分数均 ≤ 1.0，Agent 将无参考材料)")
    return final_docs


def run_agent(query: str, thread_id: str):
    """调用 Agent 并逐步打印工具调用过程（输入消息与生产流程 get_response 保持一致）

    RAG 现在是 Agent 的工具，不再硬编码注入参考材料，
    由 Agent 根据 role_description 中描述的工作流程自主决定是否调用 rag_search。
    """
    agent = build_agent()
    input_msg = HumanMessage(content=query)
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{SEP}\n🤖 [Agent] 开始推理 (thread_id={thread_id})\n{SEP}")

    # stream_mode="updates" 逐节点输出，便于观察 模型→工具→模型 的循环
    for update in agent.stream({"messages": [input_msg]}, config=config, stream_mode="updates"):
        for node_name, node_output in update.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                # 模型发起工具调用
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n🔧 [工具调用] 节点={node_name}")
                        print(f"   工具: {tc['name']}")
                        print(f"   参数: {tc['args']}")
                # 工具返回结果
                elif isinstance(msg, ToolMessage):
                    content = msg.content
                    preview = content if len(content) <= 500 else content[:500] + f"... (共{len(content)}字符)"
                    print(f"📥 [工具返回] {msg.name}:\n{preview}")
                # 模型中间思考（无工具调用的 AI 消息）
                elif isinstance(msg, AIMessage) and msg.content:
                    preview = msg.content if len(msg.content) <= 300 else msg.content[:300] + "..."
                    print(f"💬 [模型输出] 节点={node_name}:\n{preview}")

    # 最终回复（从 checkpoint 读取完整消息历史，取最后一条 AI 消息）
    state = agent.get_state(config)
    raw = state.values["messages"][-1].content
    print(f"\n{SEP}\n✅ [最终回复]\n{SEP}\n{clean_response(raw)}\n")


if __name__ == "__main__":
    print("加载基础设施（模型/向量库/重排器）...")
    ensure_all_loaded()
    print("加载完成。\n")

    if len(sys.argv) > 1:
        queries = sys.argv[1:]
    else:
        queries = None  # 交互模式

    if queries:
        for q in queries:
            print_rag_details(q)  # 仅打印检索详情供参考，不再注入 prompt
            run_agent(q, thread_id=f"test-{uuid.uuid4().hex[:8]}")
    else:
        print("交互模式（输入 exit 退出）")
        while True:
            q = input("请输入测试问题: ").strip()
            if q.lower() == "exit":
                break
            if not q:
                continue
            print_rag_details(q)  # 仅打印检索详情供参考，不再注入 prompt
            run_agent(q, thread_id=f"test-{uuid.uuid4().hex[:8]}")
