"""Provider layer — ports & adapters per cloud-readiness.

L'idea: tutto il codice business (STT, RAG, LLM) lavora con INTERFACCE astratte
(Protocol). I provider concreti (local vs remote HTTP) implementano queste
interfacce. Switching di backend = 1 flag in config.

Layout:
    providers/
    ├── base.py             ← Protocol/interfaces
    ├── factory.py          ← get_*_provider() returns concrete impl
    ├── local/              ← implementazioni che girano sulla macchina utente
    │   ├── stt.py
    │   ├── embedding.py
    │   ├── llm.py
    │   ├── vector_store.py
    │   └── rerank.py
    └── remote/             ← (stub Fase 12) HTTP client che parla con server
        └── ...
"""
from .factory import (
    get_embedding_provider,
    get_llm_provider,
    get_rerank_provider,
    get_stt_provider,
    get_vector_store_provider,
)

__all__ = [
    "get_embedding_provider",
    "get_llm_provider",
    "get_rerank_provider",
    "get_stt_provider",
    "get_vector_store_provider",
]
