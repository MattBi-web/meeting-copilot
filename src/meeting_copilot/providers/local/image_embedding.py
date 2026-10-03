"""CLIP/SigLIP multimodal embedding — Fase 20.2.

Provider che produce embedding allineati testo↔immagine.
Usa sentence-transformers per semplicità (supporta CLIP nativamente).

Modello default: clip-ViT-B-32-multilingual-v1 (multilingue IT/EN, ~860MB).
Caricato lazy, mantenuto in RAM (singleton).

API:
    embed_image(path) → list[float]      embed visivo
    embed_text(query) → list[float]      embed testuale (stesso spazio)
    dimension → int                      es. 512 per ViT-B/32

L'allineamento testo↔immagine permette di cercare slide via query naturale.
"""
from __future__ import annotations

import threading
from pathlib import Path

from ...config import IMAGE


class LocalImageEmbeddingProvider:
    """Singleton: il modello CLIP viene caricato al primo uso e mantenuto in RAM."""

    _instance_lock = threading.Lock()
    _model = None
    _dimension: int | None = None

    def __init__(self):
        # Lazy load — l'init solo in _ensure_loaded()
        pass

    @classmethod
    def _ensure_loaded(cls):
        if cls._model is not None:
            return cls._model
        with cls._instance_lock:
            if cls._model is not None:
                return cls._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers non installato. Installa:\n"
                    "  uv pip install sentence-transformers pillow"
                ) from e

            print(f"[image_embed] caricando modello {IMAGE.embed_model}…")
            cls._model = SentenceTransformer(IMAGE.embed_model)
            # Dimension: prova encode di una stringa fake per scoprirla
            try:
                v = cls._model.encode("test", convert_to_numpy=True)
                cls._dimension = int(v.shape[-1])
            except Exception:
                cls._dimension = 512  # CLIP ViT-B/32 default
            print(f"[image_embed] ✓ modello caricato (dim={cls._dimension})")
            return cls._model

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension or 512

    @property
    def model_name(self) -> str:
        return IMAGE.embed_model

    def embed_text(self, text: str) -> list[float]:
        """Embed di testo nello spazio multimodale."""
        if not text or not text.strip():
            return []
        model = self._ensure_loaded()
        v = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return v.tolist()

    def embed_image(self, image_path: Path | str) -> list[float]:
        """Embed di un'immagine on-disk (.png / .jpg).

        Si appoggia a PIL per il loading.
        """
        p = Path(image_path)
        if not p.exists():
            return []
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "Pillow non installato. Installa: uv pip install pillow"
            ) from e
        model = self._ensure_loaded()
        img = Image.open(str(p)).convert("RGB")
        v = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        return v.tolist()

    def embed_images_batch(self, image_paths: list[Path | str]) -> list[list[float]]:
        """Batch embed più immagini (più efficiente per indexing iniziale)."""
        if not image_paths:
            return []
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError("Pillow non installato") from e
        model = self._ensure_loaded()
        images = []
        for p in image_paths:
            try:
                images.append(Image.open(str(p)).convert("RGB"))
            except Exception as e:
                print(f"[image_embed] skip {p}: {e}")
                images.append(None)
        # filter None — manteniamo l'allineamento indici
        valid_pairs = [(i, im) for i, im in enumerate(images) if im is not None]
        if not valid_pairs:
            return [[] for _ in image_paths]
        idxs, valid_imgs = zip(*valid_pairs)
        embs = model.encode(
            list(valid_imgs), convert_to_numpy=True, normalize_embeddings=True,
            batch_size=8, show_progress_bar=False,
        )
        # ricomponi nell'ordine originale
        out: list[list[float]] = [[] for _ in image_paths]
        for i, vec in zip(idxs, embs):
            out[i] = vec.tolist()
        return out
