"""Preload the Ollama models used by the guideline assistant."""

import os
import time

import ollama

from bounded_composer import KEEP_ALIVE
from vector_store import DEFAULT_EMBED_MODEL


def warm_models(chat_model: str, embed_model: str | None = None) -> float:
    """Load both models, leaving the chat model as the most recently used one."""
    started = time.perf_counter()
    embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    ollama.embed(model=embed_model, input="warmup", keep_alive=KEEP_ALIVE)
    ollama.generate(model=chat_model, prompt="", keep_alive=KEEP_ALIVE)
    return round(time.perf_counter() - started, 2)
