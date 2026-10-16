from langchain_core.language_models import BaseChatModel
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from config import get_settings

settings = get_settings()
role_desc = open(settings.ROLE_DESC_PATH, encoding="utf-8").read().strip()

def make_chat_agent_call(tool_list: list, llm: BaseChatModel):
    """构造一个 chat_agent_call 函数，绑定工具列表和 LLM 实例
        用于灵活的构造不同的 chat_agent_call 函数，避免全局共享工具列表和 LLM 实例
        本质上是因为langgraph的节点仅能传入state，无法传入额外参数，所以需要通过闭包的方式传入工具列表和 LLM 实例    
    """
    def chat_agent_call(state: dict) -> dict:
        """LLM 的调用函数，返回一个增量的 state"""

        llm_with_tool = llm.bind_tools(tools=tool_list)
    
        respond = llm_with_tool.invoke(
            [SystemMessage(content=role_desc)] + state["messages"])

        return {
            "messages": [respond],
            "llm_call": state.get("llm_call", 0) + 1,
        }
    
    return chat_agent_call

def should_go(state: dict) -> Command:
    """判断下一步应该执行哪个节点"""
    if state["get_screenshot"]:
        if state["screenshot_instruction"]:
            return Command(goto = "screenshot_subagent", 
                           update = {}) 
        else:
            return Command(goto = "chat_agent_call", 
                            update = {"messages": [HumanMessage(content="请提供截图指导")]})
    tool_calls = state["messages"][-1].toolcalls
    if tool_calls:
       return  Command(goto = tool_calls["name"], update = {})
    else:
        return Command(goto = END, update = {})

