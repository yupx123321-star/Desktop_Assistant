import re
from typing import Callable, Optional
from config import *
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
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage, AIMessage, ToolMessage

# 截图工具"无新鲜截图"时返回的特殊标记（服务端据此触发客户端截图流程）
SCREENSHOT_MARKER = "__SCREENSHOT_REQUESTED__"

# ---------- 工具函数 ----------
def clean_response(content: str) -> str:
    cleaned = re.sub(r'^.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    return cleaned.strip()

# ---------- Agent 构建 ----------
_checkpointer = MemorySaver()


def build_agent(
    thread_id: str = "user-session-1",
    screenshot_uploaded: bool = False,
    screenshot_round: int = 0,
):
    """构建 Agent 实例（使用基础设施中的 ChatModel）

    截图工具绑定到指定会话：Agent 判断用户询问屏幕内容时才会调用。
    截图指导由模型在调用 request_screenshot 时通过 instruction 参数生成。
    """
    chat = load_chat_model()
    with open(ROLE_DESC_PATH, encoding="utf-8") as f:
        role_desc = f.read().strip()
    # role_description.txt 已包含各工具的使用说明和结果解读策略（含 mcmod_search 的
    # matched/candidates 处理），这里只补充截图多轮交互的程序性约束。
    system_prompt = (
        f"{role_desc}\n\n"
        "补充约束：调用 request_screenshot 时，必须在 instruction 参数中给出具体的截图指导，"
        "告诉用户截图前应该准备什么画面（例如'请打开代码编辑器并确保报错信息完整可见'），"
        "不要只写'请截图'这类空泛的话。"
        "若截图信息不足以回答问题，可以再次调用 request_screenshot 请求新的截图，"
        "并更新 instruction 说明这次需要看到什么。"
        "当提示'截图已上传'时，必须调用 request_screenshot 工具读取截图后再回答，"
        "绝不能在没有调用该工具的情况下声称'没有收到截图'。"
        "不要编造屏幕内容。"
    )

    return create_agent(
        model=chat,
        tools=[
            rag_search,
            web_search,
            mcmod_search,
            make_request_screenshot_tool(
                thread_id,
                screenshot_uploaded,
                screenshot_round,
            ),
        ],
        system_prompt=system_prompt,
        checkpointer=_checkpointer,
    )

# ---------- 完整的对话响应（核心入口） ----------
def get_response(
    user_query: str,
    thread_id: str = "user-session-1",
    screenshot_uploaded: bool = False,
    screenshot_round: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[str, bool, str]:
    """调用 Agent，返回 (回复, 是否需要截图, 给用户的截图指导)。

    使用 stream 逐步执行，以便把模型的思考过程、工具调用与返回、
    最终输出都写入日志，方便 debug。
    """
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