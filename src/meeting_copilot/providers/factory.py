"""Factory: ritorna concrete provider in base a config.

In `config.py` esiste `PROVIDERS` dataclass con stringhe tipo:
    PROVIDERS.stt = "local"          (oppure "remote" Fase 12)
    PROVIDERS.embedding = "local"
    PROVIDERS.llm = "local"
    PROVIDERS.vector_store = "local"
    PROVIDERS.rerank = "local"
    PROVIDERS.sparse_store = "local"

Le singleton istanze sono cachate per evitare di re-istanziare ad ogni chiamata.
"""
from __future__ import annotations

from threading import Lock

from ..config import PROVIDERS

_instances: dict[str, object] = {}
_lock = Lock()


def _singleton(key: str, factory):
    if key in _instances:
        return _instances[key]
    with _lock:
        if key in _instances:
            return _instances[key]
        _instances[key] = factory()
        return _instances[key]


# ──────────────────────────────────────────────────────────────
# STT
# ──────────────────────────────────────────────────────────────
def get_stt_provider():
    backend = PROVIDERS.stt
    if backend == "local":
        # Sub-backend (Fase 9.1.B): mlx / faster-whisper / voxtral / deepgram
        from ..config import STT
        sub = STT.backend
        if sub == "voxtral":
            from .local.stt_voxtral import LocalVoxtralSTTProvider
            return _singleton("stt-local-voxtral", LocalVoxtralSTTProvider)
        if sub == "deepgram":
            # Cloud STT (Deepgram REST batch): qualità top, latency 150-400ms,
            # zero CPU sul Mac. Richiede env DEEPGRAM_API_KEY.
            from .cloud.deepgram_stt import DeepgramSTTProvider
            return _singleton("stt-cloud-deepgram", DeepgramSTTProvider)
        # default: mlx-whisper o faster-whisper (gestiti dallo stesso provider)
        from .local.stt import LocalSTTProvider
        return _singleton("stt-local-whisper", LocalSTTProvider)
    if backend == "remote":
        # STT remote (via FastAPI server) non implementato per privacy.
        # Per cloud STT, usa PROVIDERS.stt='local' + STT.backend='deepgram'.
        raise NotImplementedError(
            "Remote STT non supportato by design (audio resta on-device). "
            "Per cloud STT usa STT.backend='deepgram' invece."
        )
    raise ValueError(f"Unknown STT provider: {backend}")


# ──────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────
def get_embedding_provider():
    backend = PROVIDERS.embedding
    if backend == "local":
        from .local.embedding import LocalEmbeddingProvider
        return _singleton("emb-local", LocalEmbeddingProvider)
    if backend == "remote":
        from .remote.embedding import RemoteEmbeddingProvider
        return _singleton("emb-remote", RemoteEmbeddingProvider)
    raise ValueError(f"Unknown embedding provider: {backend}")


# ──────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────
def get_llm_provider():
    backend = PROVIDERS.llm
    if backend == "local":
        from .local.llm import LocalLLMProvider
        return _singleton("llm-local", LocalLLMProvider)
    if backend == "remote":
        from .remote.llm import RemoteLLMProvider
        return _singleton("llm-remote", RemoteLLMProvider)
    if backend == "cloud":
        # Fase 22: provider cloud always-on (Anthropic/OpenAI/DeepSeek)
        from ..config import CLOUD
        if CLOUD.provider == "anthropic":
            from .cloud.anthropic import AnthropicLLMProvider
            return _singleton("llm-cloud-anthropic", AnthropicLLMProvider)
        raise ValueError(
            f"Cloud provider '{CLOUD.provider}' non ancora supportato "
            f"(per ora solo 'anthropic')"
        )
    raise ValueError(f"Unknown LLM provider: {backend}")


# ──────────────────────────────────────────────────────────────
# Vector store
# ──────────────────────────────────────────────────────────────
def get_vector_store_provider():
    backend = PROVIDERS.vector_store
    if backend == "local":
        # Sub-backend (Fase 9.7): chromadb (default) o lancedb
        from ..config import RAG
        sub = RAG.local_vector_backend
        if sub == "lancedb":
            from .local.vector_store_lance import LocalLanceDBVectorStoreProvider
            return _singleton("vec-local-lance", LocalLanceDBVectorStoreProvider)
        # default: chromadb
        from .local.vector_store import LocalVectorStoreProvider
        return _singleton("vec-local-chroma", LocalVectorStoreProvider)
    if backend == "remote":
        from .remote.vector_store import RemoteVectorStoreProvider
        return _singleton("vec-remote", RemoteVectorStoreProvider)
    raise ValueError(f"Unknown vector store provider: {backend}")


# ──────────────────────────────────────────────────────────────
# Rerank
# ──────────────────────────────────────────────────────────────
def get_rerank_provider():
    backend = PROVIDERS.rerank
    if backend == "local":
        from .local.rerank import LocalRerankProvider
        return _singleton("rerank-local", LocalRerankProvider)
    if backend == "remote":
        from .remote.rerank import RemoteRerankProvider
        return _singleton("rerank-remote", RemoteRerankProvider)
    raise ValueError(f"Unknown rerank provider: {backend}")


# ──────────────────────────────────────────────────────────────
# Sparse store (BM25)
# ──────────────────────────────────────────────────────────────
def get_sparse_store_provider():
    backend = PROVIDERS.sparse_store
    if backend == "local":
        from .local.sparse_store import LocalBM25Provider
        return _singleton("sparse-local", LocalBM25Provider)
    if backend == "remote":
        from .remote.sparse_store import RemoteSparseStoreProvider
        return _singleton("sparse-remote", RemoteSparseStoreProvider)
    raise ValueError(f"Unknown sparse store provider: {backend}")
