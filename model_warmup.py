"""Preload the Ollama models used by the WHO guidelines assistant."""

import time

from config import OLLAMA_EMBED_MODEL, OLLAMA_KEEP_ALIVE
import ollama

KEEP_ALIVE = OLLAMA_KEEP_ALIVE

def warm_models(chat_model: str, embed_model: str | None = None) -> float:
    """Load both models, leaving the chat model as the most recently used one."""
    started = time.perf_counter()
    embed_model = embed_model or OLLAMA_EMBED_MODEL
    ollama.embed(model=embed_model, input="warmup", keep_alive=OLLAMA_KEEP_ALIVE)
    ollama.generate(model=chat_model, prompt="", keep_alive=OLLAMA_KEEP_ALIVE)
    return round(time.perf_counter() - started, 2)
