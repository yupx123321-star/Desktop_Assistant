from typing import Any, List, Literal, Optional

from langchain.messages import AnyMessage
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

import operator


# 客户端收发信息结构
class ChatRequest(BaseModel):
    """客户端发送过来的请求 schema。

    status 用于区分两种请求：
      - query：新的用户问题
      - screenshot_interrupted_response：截图上传后续接被 interrupt 的会话
    """

    query: str
    thread_id: str = "user-session-1"
    status: Literal["query", "screenshot_interrupted_response"] = "query"
    screenshot_uploaded: bool = False
    screenshot_round: int = 0


class ChatResponse(BaseModel):
    """服务端返回给客户端的响应 schema"""

    status: Literal["completed", "interrupted"]
    response: str = ""                                  # status=completed 时有值
    interrupt: Optional[dict[str, Any]] = None          # status=interrupted 时是 payload


# 图状态
class chat_llm_state(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], operator.add]  # 对话消息
    llm_call: int                                        # LLM 调用次数


class screenshot_llm_state(TypedDict, total=False):
    # 截图子图预留状态（当前主流程通过 request_screenshot 工具的 interrupt 实现）
    user_query: str
    screenshot_round: int
    screenshot_url: List[str]
    task_types: List[Optional[Literal["easy", "hard"]]]
    screenshot_description: List[str]
    messages: Annotated[List[AnyMessage], operator.add]
    llm_call: int
