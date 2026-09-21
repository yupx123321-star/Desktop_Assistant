# -*- coding: utf-8 -*-
"""客户端工具：聊天、倒计时截图和截图上传。

运行在用户电脑上；服务端只接收 HTTP 请求，不需要安装 mss。
依赖: pip install requests mss pillow
"""
import argparse
import base64
import io
import os
import sys
import time
import uuid

from dotenv import load_dotenv
import requests


load_dotenv()

MAX_SCREENSHOT_ROUNDS = 3

# 状态滚动显示保留的行数（只显示最近 N 条，新状态到来时向上滚动覆盖）
_STATUS_VISIBLE_LINES = 3


class _RollingStatus:
    """覆盖式滚动状态显示。

    只保留最近 N 条状态，新状态到来时清除已打印的行并重绘，
    避免长任务（如多次 web_search）期间进度信息刷屏。
    非 TTY（输出被重定向到文件/管道）时退化为普通打印，避免 ANSI 乱码。
    """

    def __init__(self, visible_lines: int = _STATUS_VISIBLE_LINES):
        self.visible_lines = visible_lines
        self.lines: list[str] = []
        self.supports_ansi = sys.stdout.isatty()
        self._printed = 0

    def _erase_printed(self) -> None:
        """清除上一次绘制的行，并把光标移回这些行的起始位置。

        打印 N 行后光标停在第 N+1 行行首；上移 N 行到第 1 行，
        逐行清除并下移，最终光标回到第 N+1 行行首。
        """
        if not self._printed:
            return
        sys.stdout.write("\033[F" * self._printed)
        for _ in range(self._printed):
            sys.stdout.write("\033[2K\033[E")
        sys.stdout.flush()

    def update(self, message: str) -> None:
        if not self.supports_ansi:
            print(f"[状态] {message}", flush=True)
            return
        self.lines.append(message)
        if len(self.lines) > self.visible_lines:
            self.lines = self.lines[-self.visible_lines:]
        self._erase_printed()
        for line in self.lines:
            sys.stdout.write(f"[状态] {line}\n")
        self._printed = len(self.lines)
        sys.stdout.flush()

    def clear(self) -> None:
        """任务结束（或出错）时清掉所有状态行，让最终回复从干净位置开始。"""
        if not self.supports_ansi:
            return
        self._erase_printed()
        self._printed = 0
        self.lines = []


class AgentClient:
    def __init__(self, base_url: str, api_key: str, thread_id: str, quality: int = 70):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self.thread_id = thread_id
        self.quality = quality

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200 and response.json().get("status") == "ok"
        except requests.RequestException as error:
            print(f"[错误] 无法连接服务 {self.base_url}: {error}")
            return False

    def _post_chat(
        self,
        query: str,
        screenshot_uploaded: bool = False,
        screenshot_round: int = 0,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "query": query,
                "thread_id": self.thread_id,
                "screenshot_uploaded": screenshot_uploaded,
                "screenshot_round": screenshot_round,
            },
            headers=self.headers,
            timeout=300,
        )
        if response.status_code == 401:
            raise PermissionError("API Key 无效或缺失")
        response.raise_for_status()
        return response.json()

    def _post_chat_stream(
        self,
        query: str,
        screenshot_uploaded: bool = False,
        screenshot_round: int = 0,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/chat/stream",
            json={
                "query": query,
                "thread_id": self.thread_id,
                "screenshot_uploaded": screenshot_uploaded,
                "screenshot_round": screenshot_round,
            },
            headers=self.headers,
            timeout=300,
            stream=True,
        )
        if response.status_code == 401:
            raise PermissionError("API Key 无效或缺失")
        response.raise_for_status()
        import json

        status = _RollingStatus()
        result = None
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                if event.get("type") == "progress":
                    status.update(event["message"])
                elif event.get("type") == "error":
                    status.clear()
                    raise RuntimeError(event.get("message", "服务端执行失败"))
                elif event.get("type") == "result":
                    result = event
        finally:
            # 无论成功还是异常，都清掉滚动状态行，避免残留
            status.clear()
        if result is None:
            raise RuntimeError("服务端未返回最终结果")
        return result

    def capture_screen(self) -> bytes:
        """捕获主显示器并压缩为 JPEG。"""
        import mss
        from PIL import Image

        with mss.mss() as screen:
            monitor = screen.monitors[1]
            shot = screen.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        return buffer.getvalue()

    def upload_screenshot(self, jpeg_bytes: bytes) -> dict:
        """上传截图到服务端。"""
        image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        response = requests.post(
            f"{self.base_url}/screenshot",
            json={"image": image_b64, "thread_id": self.thread_id},
            headers=self.headers,
            timeout=30,
        )
        if response.status_code == 401:
            raise PermissionError("API Key 无效或缺失")
        response.raise_for_status()
        return response.json()

    def capture_after_countdown(self, seconds: int = 3) -> dict:
        """倒计时后截图并上传，让用户有时间整理屏幕。"""
        for remaining in range(seconds, 0, -1):
            print(f"即将截图，{remaining}...", flush=True)
            time.sleep(1)
        data = self.upload_screenshot(self.capture_screen())
        print(f"截图已上传 ({data['width']}x{data['height']})")
        return data

    def chat(self, query: str) -> str:
        """发送问题，并按 Agent 请求循环完成截图与续接。"""
        result = self._post_chat_stream(query)
        screenshot_round = 0

        while result.get("screenshot_requested", False):
            if screenshot_round >= MAX_SCREENSHOT_ROUNDS:
                return (
                    f"{result['response']}\n"
                    "已达到本次问题的截图次数上限，暂时无法继续获取屏幕信息。"
                )

            screenshot_round += 1
            instruction = result.get("screenshot_instruction", "")
            print(
                "Agent 请求查看屏幕。"
                + (f" 截图指导：{instruction}" if instruction else "")
            )
            try:
                self.capture_after_countdown()
            except ImportError:
                return "无法截图：客户端缺少依赖，请运行 pip install mss pillow"

            # 依靠服务端 MemorySaver 继续之前的问题，不重复发送原问题。
            result = self._post_chat_stream(
                "",
                screenshot_uploaded=True,
                screenshot_round=screenshot_round,
            )

        return result["response"]


def main() -> None:
    parser = argparse.ArgumentParser(description="智能体客户端")
    parser.add_argument("--host", default="localhost", help="服务器地址")
    parser.add_argument("--port", type=int, default=8002, help="服务端口")
    # 默认每次启动生成唯一会话ID，避免新会话误继承服务端 MemorySaver 中
    # 上一轮对话的上下文（如旧的截图轮次、无关历史）。
    # 如需延续某次会话，可显式传入 --thread <之前的会话ID>。
    parser.add_argument("--thread", default=None, help="会话ID（默认每次启动自动生成唯一ID）")
    parser.add_argument(
        "--api-key",
        default=os.getenv("AGENT_API_KEY", ""),
        help="服务访问密钥（默认读环境变量 AGENT_API_KEY）",
    )
    parser.add_argument("--quality", type=int, default=70, help="JPEG 压缩质量 (1-95)")
    args = parser.parse_args()

    thread_id = args.thread or f"session-{uuid.uuid4().hex[:12]}"

    client = AgentClient(
        base_url=f"http://{args.host}:{args.port}",
        api_key=args.api_key,
        thread_id=thread_id,
        quality=args.quality,
    )
    if not client.health_check():
        raise SystemExit("服务未就绪，请确认服务端已启动完成。")

    print(f"已连接 {client.base_url}，会话ID: {thread_id}（输入 exit 退出）")
    while True:
        query = input("请输入你的问题: ").strip()
        if query.lower() == "exit":
            print("退出对话。")
            break
        if not query:
            continue
        try:
            print(f"furina: {client.chat(query)}")
        except PermissionError as error:
            print(f"[认证失败] {error}")
            break
        except requests.exceptions.HTTPError as error:
            print(f"[服务端错误] {error.response.status_code}: {error.response.text}")
            break
        except Exception as error:
            print(f"[通信错误] {error}")
        print("-" * 60)


if __name__ == "__main__":
    main()
