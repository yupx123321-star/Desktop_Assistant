# -*- coding: utf-8 -*-
"""日志配置：为每次用户对话留下详细、可追溯的日志记录。

设计要点：
- 同时输出到控制台和滚动日志文件（logs/agent.log），便于 debug。
- 以"一次用户对话"为分界，用醒目的分隔线标记，方便在日志中定位。
- 记录内容尽量详细：用户问题、模型思考过程、工具调用（名称+参数）、
  工具返回、最终智能体输出，以及截图相关状态。
- 提供 conversation_logger（对话级）与 app_logger（应用/请求级）两个 logger。
"""
import logging
import os
from logging.handlers import RotatingFileHandler

# 日志目录（与截图目录同级，便于统一管理）
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "agent.log")

# 统一格式：时间 | 级别 | logger名 | 消息
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 避免向 root 重复传播

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    # 控制台 handler（INFO 及以上，避免刷屏）
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # 文件 handler（DEBUG 全量，滚动 10MB x 5 份）
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=1, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def setup_logging() -> None:
    """初始化日志系统（幂等，重复调用不会叠加 handler）。"""
    global _configured
    if _configured:
        return
    _build_logger("agent")
    _build_logger("app")
    _configured = True


def get_logger(name: str = "agent") -> logging.Logger:
    """获取已配置的 logger；若尚未初始化则先初始化。"""
    setup_logging()
    return logging.getLogger(name)


# ---------- 对话级日志辅助 ----------
# 一次用户对话的分隔线（醒目，便于在日志中切分）
CONV_SEPARATOR = "=" * 78


def log_conversation_start(thread_id: str, query: str,
                           screenshot_uploaded: bool, screenshot_round: int) -> None:
    """标记一次用户对话的开始（作为日志分界）。"""
    logger = get_logger("agent")
    logger.info("\n%s\n📥 [对话开始] thread_id=%s | 截图已上传=%s | 截图轮次=%d\n"
                "   用户问题: %s\n%s",
                CONV_SEPARATOR, thread_id, screenshot_uploaded, screenshot_round,
                query if query.strip() else "(空，续接截图)", CONV_SEPARATOR)


def log_thinking(thread_id: str, content: str) -> None:
    """记录模型思考/中间输出。"""
    logger = get_logger("agent")
    preview = content if len(content) <= 2000 else content[:2000] + f"... (共{len(content)}字符)"
    logger.debug("[思考] thread_id=%s\n%s", thread_id, preview)


def log_tool_call(thread_id: str, tool_name: str, args: dict) -> None:
    """记录模型发起的工具调用。"""
    logger = get_logger("agent")
    logger.info("🔧 [工具调用] thread_id=%s | 工具=%s | 参数=%s",
                thread_id, tool_name, args)


def log_tool_result(thread_id: str, tool_name: str, content: str) -> None:
    """记录工具返回结果。"""
    logger = get_logger("agent")
    preview = content if len(content) <= 2000 else content[:2000] + f"... (共{len(content)}字符)"
    logger.info("📥 [工具返回] thread_id=%s | 工具=%s\n%s", thread_id, tool_name, preview)


def log_conversation_end(thread_id: str, response: str,
                         screenshot_requested: bool, screenshot_instruction: str) -> None:
    """标记一次用户对话的结束，记录最终输出。"""
    logger = get_logger("agent")
    preview = response if len(response) <= 3000 else response[:3000] + f"... (共{len(response)}字符)"
    logger.info(
        "%s\n✅ [对话结束] thread_id=%s | 需要截图=%s\n"
        "   截图指导: %s\n   最终输出:\n%s\n%s",
        CONV_SEPARATOR, thread_id, screenshot_requested,
        screenshot_instruction or "(无)", preview, CONV_SEPARATOR,
    )
