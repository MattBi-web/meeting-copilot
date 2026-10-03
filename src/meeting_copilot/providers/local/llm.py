"""Local LLM provider via Ollama."""
from __future__ import annotations

from collections.abc import Callable

from ...config import LLM
from ..base import LLMMessage


class LocalLLMProvider:
    """Implementa LLMProvider via Ollama chat endpoint."""

    def __init__(self):
        self._warmed_models: set[str] = set()

    def warmup(self, model: str) -> None:
        if model in self._warmed_models:
            return
        try:
            import ollama
            print(f"[llm-local] warming up {model}…")
            ollama.chat(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                options={"num_predict": 1},
                keep_alive=LLM.ollama_keep_alive,
                stream=False,
            )
            self._warmed_models.add(model)
        except Exception as e:
            print(f"[llm-local] warmup error: {e}")

    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        format_json: bool = False,
    ) -> str:
        import ollama
        kwargs = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "keep_alive": LLM.ollama_keep_alive,
            "stream": False,
        }
        if format_json:
            kwargs["format"] = "json"
        resp = ollama.chat(**kwargs)
        return (resp.get("message", {}).get("content") or "").strip()

    def complete_stream(
        self,
        model: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int = 256,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        import ollama
        stream = ollama.chat(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            keep_alive=LLM.ollama_keep_alive,
            stream=True,
        )
        parts: list[str] = []
        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                parts.append(piece)
                if on_token:
                    on_token(piece)
        return "".join(parts)
