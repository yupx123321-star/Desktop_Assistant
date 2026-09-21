
import os
from dotenv import load_dotenv

load_dotenv()

# SerpApi 和 Agent 服务密钥从环境变量读取，真实值不要提交到 Git。
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
CHAT_MODEL = "/data1/ypx/vllm/Qwen3.8-BF16"
CHAT_BASE_URL = "http://localhost:8000/v1"
EMBED_MODEL = "/data1/ypx/vllm/Qwen3-Embedding"
EMBED_BASE_URL = "http://localhost:8001/v1"
RERANKER_MODEL = "/data1/ypx/vllm/Qwen3-Reranker"
RERANKER_DEVICE = "cuda:2"
COLLECTION = "furina_rag"
PERSIST_DIR = "/data1/ypx/vllm/dataset/furina/furina_chroma"
ROLE_DESC_PATH = "/data1/ypx/vllm/dataset/mc_agent/role_description.txt"
API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
TEMPERATURE = 0.8
MAX_TOKENS = 2048
DEFAULT_SEARCH_NUM = 20
DEFAULT_RERANK_NUM = 3

# 截图功能：客户端截屏上传后的存储目录 + 截图有效期（秒，超时的截图不再注入对话上下文）
SCREENSHOT_DIR = "/data1/ypx/vllm/dataset/screenshots"
SCREENSHOT_MAX_AGE = 300
# 单个问题中 Agent 最多可请求截图的次数（防止无限循环）
SCREENSHOT_MAX_ROUNDS = 3
# 截图文件保留期（秒）：服务启动时删除超过该时长的截图，防止磁盘无限累积
SCREENSHOT_RETENTION = 24 * 3600  # 默认保留 24 小时