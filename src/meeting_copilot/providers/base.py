"""Provider interfaces (Protocol-based).

Tutte le classi/funzioni business chiamano questi protocolli, MAI
direttamente mlx_whisper/ollama/chromadb/etc.

Per aggiungere un nuovo backend:
1. Crea una nuova classe che implementa il Protocol giusto
2. Registrala in factory.py
3. Cambia config.PROVIDERS.* a "remote" o nome custom
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


# ──────────────────────────────────────────────────────────────
# Common dataclasses (provider-agnostic)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Transcription:
    """Output di un STTProvider."""
    text: str
    language: str
    duration_s: float


@dataclass(frozen=True)
class VectorHit:
    """Output di un VectorStoreProvider.search()."""
    text: str
    source: str
    source_type: str
    section: str
    score: float           # 0..1 (1 = match perfetto)
    metadata: dict[str, Any]


# ──────────────────────────────────────────────────────────────
# STT
# ──────────────────────────────────────────────────────────────
@runtime_checkable
class STTProvider(Protocol):
    """Speech-to-text. Trascrive un segmento audio mono float32 a 16 kHz."""

    def warmup(self) -> None:
        """Pre-carica il modello in RAM. Chiamato all'avvio."""
        ...

    def transcribe(self, segment: np.ndarray) -> Transcription:
        """Trascrive un segmento. Bloccante."""
        ...


# ──────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embeddings densi (per RAG retrieval)."""

    @property
    def dimension(self) -> int:
        """Dimensione del vettore output (es. 768, 1024)."""
        ...

    @property
    def model_name(self) -> str:
        ...

    def embed(self, text: str) -> list[float]:
        """Embed di un singolo testo."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed di una batch (più efficiente quando supportato)."""
        ...


# ──────────────────────────────────────────────────────────────
# LLM (chat completion)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMMessage:
    role: str        # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """Chat completion. Sia sync che streaming."""

    def warmup(self, model: str) -> None:
        """Pre-carica un modello in RAM."""
        ...

    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        format_json: bool = False,
    ) -> str:
        """Genera risposta sincrona completa."""
        ...

    def complete_stream(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Stream token-by-token. Chiama `on_token` per ogni delta.
        Ritorna il testo completo a fine generazione."""
        ...


# ──────────────────────────────────────────────────────────────
# Vector store (dense ANN search)
# ──────────────────────────────────────────────────────────────
@runtime_checkable
class VectorStoreProvider(Protocol):
    """Astrazione su ChromaDB / LanceDB / Qdrant / pgvector."""

    def upsert(
        self,
        section: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Inserisce/aggiorna chunks nella collection di una sezione."""
        ...

    def delete_section(self, section: str) -> None:
        """Cancella tutti i chunks di una sezione (per rebuild)."""
        ...

    def search(
        self,
        section: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[VectorHit]:
        """Cerca top_k chunks più simili in una sezione."""
        ...

    def count(self, section: str) -> int:
        """Quanti chunks ha la collection. 0 se non esiste."""
        ...


# ──────────────────────────────────────────────────────────────
# Reranker (cross-encoder)
# ──────────────────────────────────────────────────────────────
@runtime_checkable
class RerankProvider(Protocol):
    """Cross-encoder reranker."""

    def warmup(self) -> None:
        ...

    def rerank(
        self,
        query: str,
        hits: list[VectorHit],
        top_k: int | None = None,
    ) -> list[VectorHit]:
        """Riordina hits via cross-encoder. Ritorna top_k (o tutti)."""
        ...


# ──────────────────────────────────────────────────────────────
# Optional: BM25 sparse retrieval (per hybrid search Fase 9.6)
# ──────────────────────────────────────────────────────────────
@runtime_checkable
class SparseStoreProvider(Protocol):
    """Inverted index (BM25) per keyword/lexical retrieval."""

    def upsert(
        self,
        section: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Indicizza chunks per BM25."""
        ...

    def delete_section(self, section: str) -> None:
        ...

    def search(
        self,
        section: str,
        query: str,
        top_k: int = 10,
    ) -> list[VectorHit]:
        """Cerca top_k chunks per BM25 score."""
        ...

    def count(self, section: str) -> int:
        ...
