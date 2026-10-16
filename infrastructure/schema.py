from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any
from typing_extensions import Annotated, TypedDict
from langchain.messages import AnyMessage
import operator

# 工具返回结构
class screenshot_respond(BaseModel):
    """截图请求回复schema"""
    screenshot_uploaded: bool = Field(default=False, description="是否已上传截图",)
    screenshot_round: int = Field(default=0, description="截图轮次",)
    screenshot_inner:List[str] = Field(default_factory=list, description="截图内部信息",)

def merge_screenshot_respond(old: screenshot_respond, new: screenshot_respond) -> screenshot_respond:
    """合并多个截图请求回复"""
    if old is None:
        return new
    return screenshot_respond(
        screenshot_uploaded=new.screenshot_uploaded,
        screenshot_round=new.screenshot_round + old.screenshot_round,
        screenshot_inner=old.screenshot_inner + new.screenshot_inner,
    )

# 客户端收发信息结构
class ChatRequest(BaseModel):
    """客户端发送过来的请求schema
    使用状态机的形式进行设计,因为interrupt的原因可能有多种"""
    query_type: Literal["query", "screenshot_interrupted_response"]
    query: str
    thread_id: str = "user-session-1"

class ChatResponse(BaseModel):
    """服务端返回给客户端的响应schema
    使用状态机的形式进行设计，因为interrupt的原因可能有多种"""
    status: Literal["completed", "interrupted"]
    response: str = ""                                  # status=completed 时有值
    interrupt: Optional[dict[str, Any]] = None          # status=interrupted 时是 payload

# 图状态
class chat_llm_state(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]  # 对话消息

    get_screenshot: bool    # 是否需要截图
    screenshot_instruction: str  # 截图指导
    screenshot_respond: Annotated[screenshot_respond, merge_screenshot_respond] # 截图请求回复

    llm_call: int  # LLM调用次数

class screenshot_llm_state(TypedDict):
    screenshot_respond: Annotated[screenshot_respond, merge_screenshot_respond] # 截图请求回复
    screenshot_url: Annotated[List[str], operator.add]  # 截图保存的位置

