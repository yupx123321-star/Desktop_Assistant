import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from config import get_settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# ---------- 全局单例变量 ----------
_chat_model = None
_vision_model = None
_embeddings = None
_chroma = None
_reranker = None
settings = get_settings()


# ---------- 各模块加载函数 ----------
def load_chat_model():
    """加载 Chat 模型（OpenAI 兼容接口，支持本地 vLLM 或云端 API Key）。"""
    global _chat_model
    if _chat_model is None:
        if not settings.CHAT_MODEL:
            raise RuntimeError(
                "CHAT_MODEL / LLM_MODEL 未配置，请在 .env 中设置 LLM_MODEL 与 LLM_BASE_URL"
            )
        print(f"Loading ChatModel from {settings.CHAT_BASE_URL or 'default'} ({settings.CHAT_MODEL}) ...")
        _chat_model = ChatOpenAI(
            model=settings.CHAT_MODEL,
            base_url=settings.CHAT_BASE_URL or None,
            api_key=settings.LLM_API_KEY or "EMPTY",
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
    return _chat_model


def load_vision_model():
    """加载截图分析用的多模态模型（必须支持图片输入）。

    未配置 VISION_MODEL 时抛 RuntimeError，由截图工具转成规则化提示，
    避免把图片发给纯文本模型后得到“没收到图片”这类无效回答。
    """
    global _vision_model
    if _vision_model is None:
        if not settings.VISION_MODEL:
            raise RuntimeError(
                "VISION_MODEL 未配置，请在 .env 中设置支持图片输入的视觉模型"
            )
        print(
            f"Loading VisionModel from {settings.VISION_BASE_URL or 'default'} "
            f"({settings.VISION_MODEL}) ..."
        )
        _vision_model = ChatOpenAI(
            model=settings.VISION_MODEL,
            base_url=settings.VISION_BASE_URL or None,
            api_key=settings.VISION_API_KEY or "EMPTY",
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
    return _vision_model


def load_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.EMBED_MODEL:
            raise RuntimeError("EMBED_MODEL 未配置，请在 .env 中设置嵌入模型")
        print(f"Loading Embeddings from {settings.EMBED_BASE_URL or 'default'} ({settings.EMBED_MODEL}) ...")
        _embeddings = OpenAIEmbeddings(
            model=settings.EMBED_MODEL,
            base_url=settings.EMBED_BASE_URL or None,
            api_key=settings.LLM_API_KEY or "EMPTY",
        )
    return _embeddings


def load_chroma():
    global _chroma
    if _chroma is None:
        print(f"Loading Chroma from {settings.PERSIST_DIR} ...")
        # 嵌入模型必须先加载
        emb = load_embeddings()
        _chroma = Chroma(
            collection_name=settings.COLLECTION,
            embedding_function=emb,
            persist_directory=settings.PERSIST_DIR,
        )
    return _chroma


def load_reranker():
    """加载本地 CrossEncoder 重排序模型（可选依赖）。

    未安装 sentence-transformers 或模型不可用时由调用方降级处理。
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        print(f"Loading Reranker from {settings.RERANKER_MODEL} on {settings.RERANKER_DEVICE} ...")
        _reranker = CrossEncoder(
            settings.RERANKER_MODEL,
            trust_remote_code=True,
            device=settings.RERANKER_DEVICE,
            max_length=2048,
        )
    return _reranker


# ---------- 一键加载（用于服务启动） ----------
def ensure_all_loaded():
    """服务启动时调用。

    只预加载 Chat 模型；Embedding / Chroma / Reranker 采用懒加载，
    避免本地缺少向量库或可选依赖时整个服务无法启动。
    """
    load_chat_model()
    print("Chat model is ready. Embedding / Chroma / Reranker will be loaded on demand.")
