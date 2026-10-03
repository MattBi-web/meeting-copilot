"""Suppression manager: regole anti-spam per il Watcher Agent.

Tre meccanismi:
1. **Self-suppression**: se l'ultima riga è stata classificata come 'self' dallo
   speaker diarizer, suprimi (sei TU che parli, niente suggerimenti).
2. **Topic cooldown**: dopo che il composer ha generato una risposta su topic X,
   X va in cooldown N secondi. Se nuove righe matchano semanticamente X, score
   ridotto del fattore di cooldown.
3. **Frequency limit**: hard cap N trigger a piena potenza per minuto. Se
   superato, il sistema scende a "gancio cliccabile" only.

Tutto in RAM, persistente solo dentro la stessa sessione.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CooldownTopic:
    embedding: list[float]      # centroid del topic
    expires_at: float           # epoch
    label: str                  # breve descrizione (es. "prezzo voicebot")


@dataclass
class SuppressionState:
    cooldown_topics: list[CooldownTopic] = field(default_factory=list)
    # Timestamp dei trigger "pieni" (composer eseguito) per frequency limit
    full_triggers_at: list[float] = field(default_factory=list)
    # Self-speech run: durata recente di "tu stai parlando"
    last_self_until: float = 0.0


# Defaults
COOLDOWN_S = 75.0                # quanto dura un topic cooldown
FULL_TRIGGER_WINDOW_S = 60.0     # finestra per il rate limit
MAX_FULL_TRIGGERS_PER_WINDOW = 3 # max risposte composer per minuto
SELF_SPEECH_GRACE_S = 1.5        # dopo che parli, attendi N sec prima di riconsiderare


class SuppressionManager:
    def __init__(self):
        self.state = SuppressionState()

    # ── Self-speech ──
    def note_self_speech(self) -> None:
        """Chiama questo quando il diarizer dice che TU stai parlando."""
        self.state.last_self_until = time.time() + SELF_SPEECH_GRACE_S

    def is_in_self_grace(self) -> bool:
        return time.time() < self.state.last_self_until

    # ── Topic cooldown ──
    def add_cooldown_topic(self, embedding: list[float], label: str) -> None:
        """Aggiungi un topic appena risposto al cooldown."""
        self.state.cooldown_topics.append(CooldownTopic(
            embedding=list(embedding),
            expires_at=time.time() + COOLDOWN_S,
            label=label,
        ))
        self._prune_expired_topics()

    def topic_cooldown_penalty(self, embedding: list[float]) -> tuple[float, str | None]:
        """Confronta `embedding` con i topic in cooldown.
        Ritorna (penalty 0..1, matched_label) — più alta = più suppress.
        Una penalty di 0.8 significa: questo score sarà moltiplicato per 0.2."""
        self._prune_expired_topics()
        if not self.state.cooldown_topics:
            return (0.0, None)
        if not embedding:
            return (0.0, None)

        q = np.asarray(embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q) + 1e-9
        max_sim = 0.0
        matched_label = None
        for topic in self.state.cooldown_topics:
            t = np.asarray(topic.embedding, dtype=np.float32)
            sim = float(np.dot(q, t) / (q_norm * (np.linalg.norm(t) + 1e-9)))
            if sim > max_sim:
                max_sim = sim
                matched_label = topic.label

        # Penalty 0..1 in base alla similarity (>0.6 inizia a penalizzare)
        if max_sim < 0.6:
            return (0.0, None)
        penalty = min(0.9, (max_sim - 0.6) / 0.35 * 0.9)
        return (penalty, matched_label)

    def _prune_expired_topics(self) -> None:
        now = time.time()
        self.state.cooldown_topics = [
            t for t in self.state.cooldown_topics if t.expires_at > now
        ]

    # ── Frequency limit ──
    def note_full_trigger(self) -> None:
        """Chiama quando il composer ha generato risposta completa."""
        now = time.time()
        self.state.full_triggers_at.append(now)
        self._prune_old_triggers()

    def is_rate_limited(self) -> bool:
        self._prune_old_triggers()
        return len(self.state.full_triggers_at) >= MAX_FULL_TRIGGERS_PER_WINDOW

    def trigger_rate_per_min(self) -> int:
        self._prune_old_triggers()
        return len(self.state.full_triggers_at)

    def _prune_old_triggers(self) -> None:
        cutoff = time.time() - FULL_TRIGGER_WINDOW_S
        self.state.full_triggers_at = [t for t in self.state.full_triggers_at if t > cutoff]
