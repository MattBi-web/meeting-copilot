"""LLM layer: Analyzer (classifier) + Composer (response generator).

Architettura two-stage:
1. Analyzer (modello piccolo, veloce): legge le ultime parole della trascrizione
   e decide se è successo qualcosa di interessante (domanda, obiezione, pitch
   opportunity) o se è "rumore" da ignorare.
2. Composer (modello grosso): solo se l'analyzer triggera, prende i chunks
   recuperati e produce una risposta narrativa in markdown, in streaming.

Entrambi gli agent girano su Ollama locale — zero costi API.
"""
from .analyzer import Analyzer, AnalysisResult
from .composer import Composer
from .summarizer import summarize_session
from .planner import Plan, Planner
from .agent_orchestrator import AgentOrchestrator, AgentRunResult, AgentState
from .cloud_burst import cloud_burst

__all__ = [
    "Analyzer",
    "AnalysisResult",
    "Composer",
    "summarize_session",
    "Plan",
    "Planner",
    "AgentOrchestrator",
    "AgentRunResult",
    "AgentState",
    "cloud_burst",
]
