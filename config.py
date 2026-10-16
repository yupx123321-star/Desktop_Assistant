from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# SerpApi 和 Agent 服务密钥从环境变量读取，真实值不要提交到 Git。
class Settings(BaseSettings):

    SERPAPI_API_KEY: str = ""
    AGENT_API_KEY: str = ""
    LLM_API_KEY: str = ""

    CHAT_MODEL: str = ""
    CHAT_BASE_URL: str = ""

    EMBED_MODEL: str = ""
    EMBED_BASE_URL: str = ""

    RERANKER_MODEL: str = ""
    RERANKER_DEVICE: str = "cuda:0"

    COLLECTION: str = "furina_rag"
    PERSIST_DIR: str = "/data1/ypx/vllm/dataset/furina/furina_chroma"

    ROLE_DESC_PATH: str = "/data1/ypx/vllm/dataset/mc_agent/role_description.txt"

    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 2048
    DEFAULT_SEARCH_NUM: int = 20
    DEFAULT_RERANK_NUM: int = 3

    # 截图功能：客户端截屏上传后的存储目录 + 截图有效期（秒，超时的截图不再注入对话上下文）
    SCREENSHOT_DIR: str = "/data1/ypx/vllm/dataset/screenshots"
    SCREENSHOT_MAX_AGE: int = 300
    # 单个问题中 Agent 最多可请求截图的次数（防止无限循环）
    SCREENSHOT_MAX_ROUNDS: int = 3
    # 截图文件保留期（秒）：服务启动时删除超过该时长的截图，防止磁盘无限累积
    SCREENSHOT_RETENTION: int = 24 * 3600  # 默认保留 24 小时

    class Config:
        env_file = Path(__file__).resolve().parent/ ".env"
        case_sensitive = False
        extra = "ignore"  # 忽略额外的环境变量

# 创建全局配置实例
settings = Settings()

def get_settings() -> Settings:
    """获取配置实例"""
    return settings