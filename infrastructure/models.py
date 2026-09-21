import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from config import *  # noqa: F403
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

# ---------- 全局单例变量 ----------
_chat_model = None
_embeddings = None
_chroma = None
_reranker = None

# ---------- 各模块加载函数 ----------
def load_chat_model():
    global _chat_model
    if _chat_model is None:
        print(f"Loading ChatModel from {CHAT_MODEL} ...")
        _chat_model = ChatOpenAI(
            model=CHAT_MODEL,
            base_url=CHAT_BASE_URL,
            api_key=API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    return _chat_model

def load_embeddings():
    global _embeddings
    if _embeddings is None:
        print(f"Loading Embeddings from {EMBED_MODEL} ...")
        _embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
            api_key=API_KEY,
        )
    return _embeddings

def load_chroma():
    global _chroma
    if _chroma is None:
        print(f"Loading Chroma from {PERSIST_DIR} ...")
        # 嵌入模型必须先加载
        emb = load_embeddings()
        _chroma = Chroma(
            collection_name=COLLECTION,
            embedding_function=emb,
            persist_directory=PERSIST_DIR,
        )
    return _chroma

def load_reranker():
    global _reranker
    if _reranker is None:
        print(f"Loading Reranker from {RERANKER_MODEL} on {RERANKER_DEVICE} ...")
        _reranker = CrossEncoder(
            RERANKER_MODEL,
            trust_remote_code=True,
            device=RERANKER_DEVICE,
            max_length=2048
        )
    return _reranker

# ---------- 一键加载所有（用于服务启动） ----------
def ensure_all_loaded():
    """服务启动时调用，确保所有模型常驻内存"""
    load_chat_model()
    load_embeddings()
    load_chroma()
    load_reranker()
    print("All infrastructure components are ready.")