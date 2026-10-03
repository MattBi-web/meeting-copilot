"""Remote providers (Fase 12) — HTTP/SSE clients che parlano col server FastAPI.

Tutti riusano lo stesso client HTTP (httpx.Client) con keep-alive e retry.
Quando PROVIDERS.X = "remote", la factory istanzia uno di questi al posto del local.

Compatibilità: la stessa codebase del client (overlay Mac) può puntare a un
server remoto cambiando solo config.PROVIDERS.* + SERVER.client_base_url + env API key.
"""
from .embedding import RemoteEmbeddingProvider
from .llm import RemoteLLMProvider
from .vector_store import RemoteVectorStoreProvider
from .sparse_store import RemoteSparseStoreProvider
from .rerank import RemoteRerankProvider

__all__ = [
    "RemoteEmbeddingProvider",
    "RemoteLLMProvider",
    "RemoteVectorStoreProvider",
    "RemoteSparseStoreProvider",
    "RemoteRerankProvider",
]
