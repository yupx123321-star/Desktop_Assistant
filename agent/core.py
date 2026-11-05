import re
from typing import Callable, Optional

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agent.agent_call import make_chat_agent_call, should_go
from config import get_settings
from infrastructure.models import load_chat_model
from infrastructure.schema import ChatResponse, chat_llm_state
from tools import (
    SCREENSHOT_UNAVAILABLE_MARKER,
    mcmod_search,
    make_request_screenshot_tool,
    rag_search,
    web_search,
)

settings = get_settings()


# ---------- 工具函数 ----------
def clean_response(content) -> str:
    """清理模型输出中的思考段与多余空行，兼容多模态 content 列表。"""
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    cleaned = re.sub(r"^.*?</think>", "", str(content), flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def _unavailable_tool_message(messages: list) -> Optional[ToolMessage]:
    """从最近一批 ToolMessage 中找出截图不可用的标记（同一轮可能并发多个工具）。"""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        if (
            message.name == "request_screenshot"
            and str(message.content).startswith(SCREENSHOT_UNAVAILABLE_MARKER)
        ):
            return message
    return None


def route_after_tools(state: dict) -> str:
    """工具执行后的路由：截图不可用时跳过 LLM，直接走规则化回复。"""
    if _unavailable_tool_message(state.get("messages") or []):
        return "screenshot_unavailable"
    return "chat_agent_call"


def screenshot_unavailable(state: dict) -> dict:
    """不调用 LLM，按固定规则生成截图失败回复。"""
    message = _unavailable_tool_message(state.get("messages") or [])
    reason = ""
    if message is not None:
        reason = str(message.content).removeprefix(SCREENSHOT_UNAVAILABLE_MARKER).strip()[:200]
    reply = "抱歉，我没有拿到可用的屏幕画面，所以无法描述你屏幕上的内容。"
    if reason:
        reply += f"\n\n原因：{reason}"
    reply += "\n\n你可以重新发起一次截图请求，或直接把屏幕上的内容描述给我，我再帮你分析。"
    return {"messages": [AIMessage(content=reply)]}


chat_agent_tools = [
    web_search,
    rag_search,
    mcmod_search,
    make_request_screenshot_tool(
        thread_id="",
        screenshot_uploaded=False,
        screenshot_round=0,
    ),
]


# ---------- Agent 构建 ----------
_checkpointer = MemorySaver()

# 构建 LLM 实例（全局共享，避免重复加载）
llm = load_chat_model()

chat_agent_graph = StateGraph(chat_llm_state)

# 添加节点
chat_agent_graph.add_node(
    "chat_agent_call", make_chat_agent_call(tool_list=chat_agent_tools, llm=llm)
)
chat_agent_graph.add_node("tools", ToolNode(chat_agent_tools))
chat_agent_graph.add_node("screenshot_unavailable", screenshot_unavailable)

# 添加边
chat_agent_graph.add_edge(START, "chat_agent_call")
chat_agent_graph.add_conditional_edges(
    "chat_agent_call", should_go, {"tools": "tools", END: END}
)
# 工具执行后先判断截图是否可用：不可用就跳过 LLM 直接回复
chat_agent_graph.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "chat_agent_call": "chat_agent_call",
        "screenshot_unavailable": "screenshot_unavailable",
    },
)
chat_agent_graph.add_edge("screenshot_unavailable", END)

# 为了能够实现图的持久化和恢复，必须在图编译前设置 checkpointer
chat_agent = chat_agent_graph.compile(checkpointer=_checkpointer)


# ---------- 完整的对话响应（核心入口） ----------
def get_response(
    user_query: str,
    thread_id: str = "user-session-1",
    request_type: str = "query",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ChatResponse:
    """调用 Agent，返回 ChatResponse。

    - request_type="query"：新问题
    - request_type="screenshot_interrupted_response"：user_query 为客户端上传的截图路径，
      用于恢复被 request_screenshot 工具 interrupt 的会话。
    """
    config = {"configurable": {"thread_id": thread_id}}

    if request_type == "query":
        # 新问题，清空该会话的旧状态
        _checkpointer.delete_thread(thread_id)
        result = chat_agent.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config,
        )
    elif request_type == "screenshot_interrupted_response":
        # 续接问题：继续上次被 interrupt 的会话，此时 user_query 是截图路径
        result = chat_agent.invoke(Command(resume=user_query), config=config)
    else:
        raise ValueError(f"未知的请求类型: {request_type}")

    # 执行到需要 interrupt 时，invoke 返回 {"__interrupt__": [Interrupt(value=payload)], ...}
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return ChatResponse(
            status="interrupted",
            response="",
            interrupt=payload,
        )

    return ChatResponse(
        status="completed",
        response=clean_response(result["messages"][-1].content),
        interrupt=None,
    )
