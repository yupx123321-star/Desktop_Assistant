from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain.messages import SystemMessage
from langgraph.graph import END

from config import get_settings

settings = get_settings()

DEFAULT_ROLE_DESC = (
    "你是一个乐于助人的桌面智能体助手。请优先理解用户的真实需求，"
    "在需要实时信息、本地知识库、MC 百科或用户屏幕内容时主动调用相应工具；"
    "如果工具没有返回有效信息，请如实说明，不要编造。"
)


def _load_role_desc() -> str:
    path = Path(settings.ROLE_DESC_PATH)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return DEFAULT_ROLE_DESC


role_desc = _load_role_desc()


def make_chat_agent_call(tool_list: list, llm: BaseChatModel):
    """构造 chat_agent_call 节点。

    通过闭包绑定工具列表和 LLM 实例，因为 LangGraph 节点只能接收 state。
    """

    def chat_agent_call(state: dict) -> dict:
        """LLM 调用节点，返回增量 state。"""
        llm_with_tool = llm.bind_tools(tools=tool_list)

        respond = llm_with_tool.invoke(
            [SystemMessage(content=role_desc)] + state["messages"]
        )

        return {
            "messages": [respond],
            "llm_call": state.get("llm_call", 0) + 1,
        }

    return chat_agent_call


def should_go(state: dict) -> str:
    """条件边：模型发起工具调用则进入 tools 节点，否则结束。"""
    messages = state.get("messages") or []
    if messages and getattr(messages[-1], "tool_calls", None):
        return "tools"
    return END
