import re
from typing import Callable, Optional
from .agent_call import *
from config import get_settings
from infrastructure import load_chat_model
from tools import (
    web_search,
    rag_search,
    mcmod_search,
    make_request_screenshot_tool,
)
from logging_config import (
    get_logger,
    log_conversation_start,
    log_conversation_end,
    log_thinking,
    log_tool_call,
    log_tool_result,
)
from infrastructure.schema import chat_llm_state, screenshot_llm_state
from infrastructure.schema import screenshot_respond
from infrastructure.schema import ChatResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain.messages import HumanMessage, AIMessage, ToolMessage

# 截图工具"无新鲜截图"时返回的特殊标记（服务端据此触发客户端截图流程）
SCREENSHOT_MARKER = "__SCREENSHOT_REQUESTED__"

settings = get_settings()
# ---------- 工具函数 ----------
def clean_response(content: str) -> str:
    cleaned = re.sub(r'^.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    return cleaned.strip()

chat_agent_tools = [web_search, 
                    rag_search, 
                    mcmod_search,
                    make_request_screenshot_tool(thread_id="", 
                                                 screenshot_uploaded=False, 
                                                 screenshot_round=0)]


# ---------- Agent 构建 ----------
_checkpointer = MemorySaver()

# 构建LLM实例（全局共享，避免重复加载）
llm = load_chat_model()

chat_agent_graph = StateGraph(chat_llm_state)
screenshot_subagent_graph = StateGraph(screenshot_llm_state)

# 添加节点
chat_agent_graph.add_node("chat_agent_call", make_chat_agent_call(tool_list=chat_agent_tools, llm=llm))
chat_agent_graph.add_node("screenshot_subagent", screenshot_subagent_graph)
chat_agent_graph.add_node("should_go", should_go)

# 添加边
chat_agent_graph.add_edge(START, "chat_agent_call")
screenshot_subagent_graph.add_edge(START, )
chat_agent_graph.add_edge("chat_agent_call", "should_go")

# 为了能够实现图的持久化和恢复，必须在图编译前设置 checkpointer
chat_agent_graph.compile(checkpointer=_checkpointer)

# ---------- 完整的对话响应（核心入口） ----------
def get_response(
    user_query: str,
    thread_id: str = "user-session-1",
    request_type: str = "query",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ChatResponse:
    """调用 Agent，返回 (回复, 是否需要截图, 给用户的截图指导)。

    使用 stream 逐步执行，以便把模型的思考过程、工具调用与返回、
    最终输出都写入日志，方便 debug。
    """
    if request_type == "query":
        # 新问题，清空会话状态
        _checkpointer.delete_thread(thread_id)
        result = chat_agent_graph.invoke({"messages": [HumanMessage(content=user_query)],}, 
                config={"configurable": {"thread_id": thread_id}},
        )
    elif request_type == "screenshot_interrupted_response":
        # 续接问题，继续上次的会话状态, 此时的 user_query 是客户端上传的截图路径
        result = chat_agent_graph.invoke(Command(resume=user_query),
                config={"configurable": {"thread_id": thread_id}},
        )

    # 执行到需要interrupt时，invoke 返回：{"__interrupt__": [Interrupt(value=payload)], ...}
    if "__interrupt__" in result:
        # payload = {"type": "screenshot_request", "instruction": guidance}
        # 也就是自定义的 interrupt 的 value字典
        payload = result["__interrupt__"][0].value
        return ChatResponse(
            status="interrupted",
            response="",
            interrupt=payload,
        )

    return ChatResponse(
                status="completed",
                response=clean_response(result["messages"][-1].content),
                interrupt="",
            )


    logger = get_logger("agent")

    def report(message: str) -> None:
        if progress_callback:
            try:
                progress_callback(message)
            except Exception:
                logger.exception("进度回调失败")

    # 1. 标记一次用户对话的开始（日志分界）
    log_conversation_start(thread_id, user_query, screenshot_uploaded, screenshot_round)

    # RAG 不再硬编码注入，而是由 Agent 按需调用 rag_search 工具获取参考材料。
    # 截图也由 Agent 按需调用 request_screenshot 工具。
    if user_query.strip():
        input_msg = HumanMessage(content=user_query)
    else:
        # 续接截图：必须明确指令 Agent 调用 request_screenshot 工具去"取"截图，
        # 否则 Agent 不会主动读取已上传的截图，会误以为没收到。
        input_msg = HumanMessage(
            content=(
                "用户刚刚已上传了屏幕截图。请立即调用 request_screenshot 工具读取这张截图，"
                "查看画面内容后，结合之前的问题给出回答。"
                "不要在没有调用 request_screenshot 工具的情况下声称'没有收到截图'。"
            )
        )

    # 2. 调用 Agent（截图工具绑定到当前会话的 thread_id），逐步记录推理过程
    agent = build_agent(thread_id, screenshot_uploaded, screenshot_round=screenshot_round)
    config = {"configurable": {"thread_id": thread_id}}
    report("智能体已开始分析问题")

    # 用 stream 逐节点执行，捕获 模型→工具→模型 的完整循环
    for update in agent.stream({"messages": [input_msg]}, config=config, stream_mode="updates"):
        for node_name, node_output in update.items():
            for msg in node_output.get("messages", []):
                # 模型发起工具调用
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        log_tool_call(thread_id, tc["name"], tc["args"])
                        tool_labels = {
                            "web_search": "正在联网搜索相关资料",
                            "mcmod_search": "正在查询 MC 百科并筛选相关评论",
                            "rag_search": "正在检索本地知识库",
                            "request_screenshot": "正在处理屏幕截图请求",
                        }
                        report(tool_labels.get(tc["name"], f"正在调用工具：{tc['name']}"))
                # 工具返回结果
                elif isinstance(msg, ToolMessage):
                    log_tool_result(thread_id, msg.name, str(msg.content))
                    report(f"{msg.name} 已返回结果，正在继续分析")
                # 模型思考/中间输出（无工具调用的 AI 消息）
                elif isinstance(msg, AIMessage) and msg.content:
                    log_thinking(thread_id, str(msg.content))

    # 3. 从 checkpoint 读取完整消息历史，取最后一条 AI 消息作为最终回复
    state = agent.get_state(config)
    messages = state.values["messages"]
    last_human_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    current_messages = messages[last_human_index + 1 :]
    screenshot_requested = any(
        isinstance(message, ToolMessage)
        and SCREENSHOT_MARKER in (message.content or "")
        for message in current_messages
    )

    raw = messages[-1].content
    instruction = ""
    if screenshot_requested:
        for message in current_messages:
            if isinstance(message, ToolMessage) and SCREENSHOT_MARKER in (message.content or ""):
                instruction = str(message.content).split("\n", 1)[-1].replace(
                    "给用户的截图指导：", "", 1
                ).strip()
                break

    final_response = clean_response(raw)
    report("回答生成完成")

    # 4. 标记一次用户对话的结束，记录最终输出
    log_conversation_end(thread_id, final_response, screenshot_requested, instruction)

    return final_response, screenshot_requested, instruction