from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# 密钥等敏感配置从环境变量 / .env 读取，真实值不要提交到 Git。
class Settings(BaseSettings):

    SERPAPI_API_KEY: str = ""
    AGENT_API_KEY: str = ""
    LLM_API_KEY: str = ""

    # LLM（OpenAI 兼容接口，例如 DashScope、vLLM）
    LLM_MODEL: str = ""
    LLM_BASE_URL: str = ""
    # 兼容旧的 CHAT_* 配置名
    CHAT_MODEL: str = ""
    CHAT_BASE_URL: str = ""

    EMBED_MODEL: str = ""
    EMBED_BASE_URL: str = ""

    # 兼容 RERANK_MODEL 与 RERANKER_MODEL 两种写法
    RERANK_MODEL: str = ""
    RERANK_BASE_URL: str = ""
    RERANKER_MODEL: str = ""
    RERANKER_DEVICE: str = "cpu"

    COLLECTION: str = "furina_rag"
    PERSIST_DIR: str = str(BASE_DIR / "data" / "furina_chroma")

    ROLE_DESC_PATH: str = str(BASE_DIR / "data" / "role_description.txt")

    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 2048
    DEFAULT_SEARCH_NUM: int = 20
    DEFAULT_RERANK_NUM: int = 3

    # 截图功能：客户端截屏上传后的存储目录 + 截图有效期（秒，超时的截图不再注入对话上下文）
    SCREENSHOT_DIR: str = str(BASE_DIR / "data" / "screenshots")
    SCREENSHOT_MAX_AGE: int = 300
    # 单个问题中 Agent 最多可请求截图的次数（防止无限循环）
    SCREENSHOT_MAX_ROUNDS: int = 3
    # 截图文件保留期（秒）：服务启动时删除超过该时长的截图，防止磁盘无限累积
    SCREENSHOT_RETENTION: int = 24 * 3600  # 默认保留 24 小时

    class Config:
        env_file = BASE_DIR / ".env"
        case_sensitive = False
        extra = "ignore"  # 忽略额外的环境变量


# 创建全局配置实例
settings = Settings()

# 统一新旧配置名：CHAT_* 未配置时回退到 LLM_*
if not settings.CHAT_MODEL:
    settings.CHAT_MODEL = settings.LLM_MODEL
if not settings.CHAT_BASE_URL:
    settings.CHAT_BASE_URL = settings.LLM_BASE_URL
if not settings.RERANKER_MODEL:
    settings.RERANKER_MODEL = settings.RERANK_MODEL
if not settings.EMBED_BASE_URL:
    settings.EMBED_BASE_URL = settings.LLM_BASE_URL


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


# 兼容旧代码的模块级配置项（tools 中有直接 from config import xxx 的写法）
SERPAPI_API_KEY = settings.SERPAPI_API_KEY
AGENT_API_KEY = settings.AGENT_API_KEY
LLM_API_KEY = settings.LLM_API_KEY
DEFAULT_SEARCH_NUM = settings.DEFAULT_SEARCH_NUM
DEFAULT_RERANK_NUM = settings.DEFAULT_RERANK_NUM
