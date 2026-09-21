import hmac
import asyncio
import json
import queue
import threading
import time
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from config import AGENT_API_KEY
from infrastructure import ensure_all_loaded
from agent.core import get_response
from tools import save_screenshot, cleanup_old_screenshots
from logging_config import get_logger, setup_logging

# ---------- 生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化日志系统（控制台 + 滚动文件）
    setup_logging()
    app_logger = get_logger("app")
    app_logger.info("服务启动，开始加载基础设施...")
    # 启动时清理过期截图，防止磁盘无限累积
    removed = cleanup_old_screenshots()
    app_logger.info("启动清理：删除 %d 张过期截图。", removed)
    # 启动时加载所有模型到显存
    ensure_all_loaded()
    app_logger.info("基础设施加载完成，服务就绪。")
    yield
    # 关闭时可做清理（如释放显存，但通常无需手动）
    app_logger.info("服务关闭。")

app = FastAPI(lifespan=lifespan)

# ---------- 认证 ----------
def require_api_key(x_api_key: str = Header(default="")):
    """校验请求头 X-API-Key；未配置 AGENT_API_KEY 时跳过校验（仅本地开发）"""
    if not AGENT_API_KEY:
        return
    if not hmac.compare_digest(x_api_key, AGENT_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ---------- 数据模型 ----------
class ChatRequest(BaseModel):
    query: str
    thread_id: str = "user-session-1"
    screenshot_uploaded: bool = False
    screenshot_round: int = 0

class ChatResponse(BaseModel):
    response: str
    screenshot_requested: bool = False  # True 表示 Agent 需要客户端截图上传后重发原问题
    screenshot_instruction: str = ""

class ScreenshotRequest(BaseModel):
    image: str          # Base64 编码的 JPEG 图片
    thread_id: str = "user-session-1"

# ---------- 路由 ----------
# 把 chat_endpoint 这个函数登记到 FastAPI 内部的路由表里，路径为 POST /chat，由FastAPI自动调用
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat_endpoint(req: ChatRequest):
    app_logger = get_logger("app")
    start = time.time()
    app_logger.info(
        "📨 [收到 /chat 请求] thread_id=%s | 截图已上传=%s | 截图轮次=%d | 问题=%r",
        req.thread_id, req.screenshot_uploaded, req.screenshot_round, req.query,
    )
    try:
        resp, screenshot_requested, screenshot_instruction = get_response(
            req.query,
            req.thread_id,
            req.screenshot_uploaded,
            req.screenshot_round,
        )
        elapsed = time.time() - start
        app_logger.info(
            "✅ [/chat 完成] thread_id=%s | 耗时=%.2fs | 需要截图=%s",
            req.thread_id, elapsed, screenshot_requested,
        )
        return ChatResponse(
            response=resp,
            screenshot_requested=screenshot_requested,
            screenshot_instruction=screenshot_instruction,
        )
    except Exception as e:
        app_logger.exception("❌ [/chat 异常] thread_id=%s | 错误=%s", req.thread_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream_endpoint(req: ChatRequest):
    """以 NDJSON 推送 Agent 进度，最后推送完整 ChatResponse。"""
    events: queue.Queue = queue.Queue()

    def progress(message: str) -> None:
        events.put({"type": "progress", "message": message})

    def run() -> None:
        try:
            response, screenshot_requested, screenshot_instruction = get_response(
                req.query,
                req.thread_id,
                req.screenshot_uploaded,
                req.screenshot_round,
                progress_callback=progress,
            )
            events.put({
                "type": "result",
                "response": response,
                "screenshot_requested": screenshot_requested,
                "screenshot_instruction": screenshot_instruction,
            })
        except Exception as error:
            events.put({"type": "error", "message": str(error)})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    async def generate():
        yield json.dumps({"type": "progress", "message": "请求已接收，智能体正在分析"}, ensure_ascii=False) + "\n"
        while True:
            try:
                event = await asyncio.to_thread(events.get, True, 5)
            except queue.Empty:
                yield json.dumps({"type": "progress", "message": "智能体仍在处理中，请稍候"}, ensure_ascii=False) + "\n"
                continue
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/screenshot", dependencies=[Depends(require_api_key)])
async def screenshot_endpoint(req: ScreenshotRequest):
    """接收客户端上传的屏幕截图，存入该会话，供 Agent 对话时参考"""
    app_logger = get_logger("app")
    try:
        shot = save_screenshot(req.thread_id, req.image)
        app_logger.info(
            "🖼️ [收到 /screenshot] thread_id=%s | 尺寸=%dx%d | 存储=%s",
            req.thread_id, shot.width, shot.height, shot.path,
        )
        return {
            "status": "success",
            "width": shot.width,
            "height": shot.height,
            "path": shot.path,
        }
    except ValueError as e:
        app_logger.warning("⚠️ [/screenshot 参数错误] thread_id=%s | 错误=%s", req.thread_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        app_logger.exception("❌ [/screenshot 异常] thread_id=%s | 错误=%s", req.thread_id, e)
        raise HTTPException(status_code=500, detail=f"截图处理失败: {e}")

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)