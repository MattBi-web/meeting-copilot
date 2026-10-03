"""Cloud LLM providers — Fase 22.

Implementano il Protocol LLMProvider parlando direttamente con le API cloud
(Anthropic, OpenAI, DeepSeek), bypassando Ollama VM/locale.

Si attivano impostando PROVIDERS.llm = "cloud" in config.py.

Pattern: usa il mapping in CloudConfig per rimappare il model string passato
dal caller (es. "llama3.2:3b" → "claude-haiku-4-5"). In questo modo zero
modifiche ai chiamanti (analyzer/composer continuano a passare i loro model
"locali", il provider sceglie il modello cloud corrispondente).
"""
from .anthropic import AnthropicLLMProvider
from .deepgram_stt import DeepgramSTTProvider

__all__ = ["AnthropicLLMProvider", "DeepgramSTTProvider"]
