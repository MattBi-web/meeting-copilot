# Meeting Copilot

A local-first, real-time AI assistant for macOS that listens to live video calls (Google Meet, Teams, Zoom, Webex) and surfaces grounded, relevant suggestions from your own knowledge base while you talk — without recording or storing audio.

## What it does

- **Live transcription** of both sides of a call (system audio + mic), speaker-attributed, with no audio ever written to disk.
- **Hybrid retrieval-augmented generation**: dense vector search (BGE-M3 embeddings + ChromaDB/LanceDB) combined with sparse BM25 retrieval and a GraphRAG layer over entities extracted from your documents, plus a cross-encoder reranker to cut noise before anything reaches the LLM.
- **Episodic memory**: a separate store that remembers facts and decisions across sessions, not just within a single call.
- **A watcher/planner/composer LLM pipeline**: a lightweight model continuously watches the live transcript for signals worth acting on, a planner decides *if* and *how* to respond, and a grounded composer drafts the suggestion — so the assistant doesn't talk over you with noise on every sentence.
- **Local/cloud model portability**: every provider (STT, embeddings, reranking, vector store, LLM) sits behind a common interface, with a local backend (Ollama + MLX, fully on-device) and a cloud backend (Anthropic) implementing the same contract — swap one for the other without touching the pipeline logic.
- **An invisible-to-screenshare overlay** (PyQt6) so the assistant is visible to you but never appears if you share your screen.
- **An MCP server** exposing the same knowledge base and coordinator to any MCP client (e.g. Claude Desktop), so the assistant isn't locked into its own UI.

## Architecture

```
src/meeting_copilot/
  audio_capture.py        System + mic audio capture (BlackHole aggregate device)
  vad.py                  Silero VAD — speech segmentation
  stt.py                  MLX Whisper Turbo (Apple GPU/Neural Engine), faster-whisper fallback
  overlay.py               PyQt6 overlay window, invisible to screenshare
  providers/                Common interface + local (on-device) and remote (cloud) backends
    base.py                 for STT / embeddings / reranking / vector store / LLM
    local/  remote/
  kb/                      Hybrid retrieval: dense + sparse (BM25) + GraphRAG + reranker
  memory/                  Episodic memory across sessions (SQLite + embeddings)
  brain/                   Entity/knowledge consolidation layer
  llm/
    watcher.py             Lightweight signal detector on the live transcript
    planner.py             Decides whether/how to act on a signal
    composer.py            Drafts the grounded suggestion from retrieved context
    agent_orchestrator.py   Multi-step tool-using agent mode
  mcp_server.py            Exposes the knowledge base + coordinator over MCP
```

## Requirements

- macOS, Apple Silicon (M-series) for the on-device STT/embedding path
- Python 3.11+
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) as an aggregate audio device (system audio capture)
- [Ollama](https://ollama.com) for local LLM/embedding inference, or an Anthropic API key for the cloud path

## Setup

```bash
# Install with the overlay UI and MCP server
pip install -e ".[ui,mcp]"

# Copy and fill in your own .env (STT/LLM provider keys as needed)
cp .env.example .env

# Run
meeting-copilot                 # overlay mode
meeting-copilot --console       # headless/console mode
meeting-copilot kb ingest ./docs   # index a knowledge base folder
```

## Why this exists

Most "AI notetaker" tools either ship a black-box SaaS, or bolt a summarizer onto a transcript after the fact. This project was built to explore the harder, more interesting version of the problem: a system that reasons about a conversation *while it is happening*, retrieves the right grounding material in under a second, and decides on its own whether a suggestion is worth surfacing at all — entirely on-device when it needs to be.

## License

MIT — see [LICENSE](LICENSE).
