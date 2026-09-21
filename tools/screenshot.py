# -*- coding: utf-8 -*-
"""屏幕截图工具：截图存取 + 多模态分析 + Agent 工具构造。

服务端负责：
- 接收客户端上传的 base64 截图并落盘
- 按会话消费截图（每张只消费一次）
- 用多模态 chat 模型分析截图内容
- 构造绑定到指定会话的 request_screenshot 工具
"""
import base64
import io
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from langchain_core.tools import tool

from config import (
    SCREENSHOT_DIR,
    SCREENSHOT_MAX_AGE,
    SCREENSHOT_MAX_ROUNDS,
    SCREENSHOT_RETENTION,
)
from infrastructure.models import load_chat_model

# 限制最大 base64 长度（约 10MB 原始数据），防止恶意超大请求
MAX_B64_LEN = 10 * 1024 * 1024
_consumed_screenshot_times: dict[str, float] = {}
_screenshot_state_lock = threading.Lock()


@dataclass
class Screenshot:
    thread_id: str
    path: str
    width: int
    height: int
    timestamp: float  # 接收时间（epoch 秒）


def cleanup_old_screenshots(retention: float = SCREENSHOT_RETENTION) -> int:
    """删除 SCREENSHOT_DIR 中超过保留期（秒）的截图文件，返回删除数量。

    策略：按文件 mtime 判断，超过 retention 的 .jpg 一律删除。
    在服务启动时调用一次，防止截图目录无限累积。
    日志文件由 RotatingFileHandler 自动滚动，无需在此处理。
    """
    if not os.path.isdir(SCREENSHOT_DIR):
        return 0
    now = time.time()
    removed = 0
    for name in os.listdir(SCREENSHOT_DIR):
        if not name.endswith(".jpg"):
            continue
        full = os.path.join(SCREENSHOT_DIR, name)
        try:
            if now - os.path.getmtime(full) > retention:
                os.remove(full)
                removed += 1
        except OSError:
            # 文件可能正被读取或已被删除，忽略
            continue
    return removed


def save_screenshot(thread_id: str, image_b64: str) -> Screenshot:
    """解码 base64 图片并落盘，返回截图元信息。失败抛 ValueError。"""
    if len(image_b64) > MAX_B64_LEN:
        raise ValueError("图片数据过大（超过 10MB 限制）")
    try:
        img_data = base64.b64decode(image_b64, validate=True)
    except Exception:
        raise ValueError("base64 解码失败，请确认图片编码正确")

    from PIL import Image  # 延迟导入
    try:
        img = Image.open(io.BytesIO(img_data))
        img.load()  # 强制完整解码，校验图片完整性
    except Exception:
        raise ValueError("图片格式无效或已损坏")

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = time.time()
    safe_thread = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)[:64]
    path = os.path.join(SCREENSHOT_DIR, f"{safe_thread}_{int(ts * 1000)}.jpg")

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(path, format="JPEG", quality=85)

    return Screenshot(thread_id=thread_id, path=path, width=img.width, height=img.height, timestamp=ts)


def get_next_screenshot(thread_id: str) -> Optional[Screenshot]:
    """获取该会话下一张未消费且在有效期内的截图，并将其标记为已消费。

    核心逻辑：
      1. 按 thread_id 前缀筛选该会话的截图文件
      2. 过滤掉超过 SCREENSHOT_MAX_AGE 的过期截图
      3. 过滤掉已消费过的截图（mtime <= 上次消费时间戳）
      4. 取剩余中最新的一张，更新消费时间戳，返回其元信息

    每张截图只会被消费一次，保证多轮截图中 Agent 不会重复使用旧图。
    """
    if not os.path.isdir(SCREENSHOT_DIR):
        return None

    # 文件名格式: {safe_thread}_{毫秒时间戳}.jpg，用前缀匹配该会话
    prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)[:64]

    # 第一步：收集该会话所有未过期的截图
    candidates = []
    for name in os.listdir(SCREENSHOT_DIR):
        if name.startswith(prefix + "_") and name.endswith(".jpg"):
            full = os.path.join(SCREENSHOT_DIR, name)
            mtime = os.path.getmtime(full)
            if time.time() - mtime <= SCREENSHOT_MAX_AGE:
                candidates.append((mtime, full))
    if not candidates:
        return None

    # 第二步：加锁，排除已消费的截图，取最新一张并标记为已消费
    with _screenshot_state_lock:
        consumed_at = _consumed_screenshot_times.get(thread_id, 0.0)
        candidates = [(mtime, path) for mtime, path in candidates if mtime > consumed_at]
        if not candidates:
            return None
        mtime, path = max(candidates)
        _consumed_screenshot_times[thread_id] = mtime

    # 第三步：读取图片尺寸，返回元信息
    from PIL import Image
    with Image.open(path) as img:
        return Screenshot(thread_id=thread_id, path=path, width=img.width, height=img.height, timestamp=mtime)


def analyze_screenshot(screenshot: Screenshot) -> str:
    """用多模态 chat 模型（Qwen3.8 自带视觉能力）识别截图，返回文本描述。

    直接把截图编码为 base64 data URL，作为 image_url 内容块发给
    现有的 chat 模型（vLLM 的 OpenAI 兼容接口支持多模态），
    让模型描述画面内容，供 Agent 据此回答用户问题。
    """
    age = int(time.time() - screenshot.timestamp)
    try:
        with open(screenshot.path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{image_b64}"

        chat = load_chat_model()
        resp = chat.invoke([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "这是一张用户电脑屏幕的截图。请客观、详细地描述画面内容，"
                            "重点说明：当前打开的是哪个软件/应用（尽量给出软件名称）、"
                            "窗口标题、界面主要区域与可见的文字信息。"
                            "只描述你确实看到的内容，不要推测或编造。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ])
        description = str(resp.content).strip()
        if not description:
            raise ValueError("视觉模型返回空内容")
        return f"[用户屏幕截图] 分辨率 {screenshot.width}x{screenshot.height}，{age} 秒前上传。\n画面内容：{description}"
    except Exception as e:
        return (
            f"[用户屏幕截图] 分辨率 {screenshot.width}x{screenshot.height}，"
            f"{age} 秒前上传（截图已收到，但视觉识别失败：{e}）"
        )


def make_request_screenshot_tool(
    thread_id: str,
    screenshot_uploaded: bool = False,
    screenshot_round: int = 0,
):
    """构造绑定到指定会话的"请求屏幕截图"工具，供 Agent 按需调用。

    截图请求流程：
      没有新截图时返回特殊标记 + 模型生成的截图指导 → 服务端转发给客户端
      有新截图时只消费一次，返回分析结果
    """

    @tool
    def request_screenshot(instruction: str = "") -> str:
        """请求查看用户当前屏幕截图，用于回答涉及用户本地屏幕内容的问题。

        仅当用户询问"我屏幕上/我电脑里/我桌面上/我打开的页面/帮我看看"等
        需要查看其本地屏幕内容的问题时才调用此工具。

        Args:
            instruction: 给用户的截图指导，告诉用户截图前应该准备什么画面。
                必须具体、有针对性，例如"请打开代码编辑器并确保报错信息完整可见"，
                而不是泛泛地说"请截图"。
        """
        # 客户端明确上传后才读取下一张未消费截图，避免重复使用旧图。
        shot = get_next_screenshot(thread_id) if screenshot_uploaded else None
        if shot is None:
            if screenshot_round >= SCREENSHOT_MAX_ROUNDS:
                return "截图请求次数已达到上限，请基于目前已有信息回答，并明确说明信息可能不完整。"
            guidance = instruction.strip() or "请整理并展示与问题相关的窗口，然后等待倒计时截图。"
            return f"__SCREENSHOT_REQUESTED__\n给用户的截图指导：{guidance}"
        return analyze_screenshot(shot)

    return request_screenshot
