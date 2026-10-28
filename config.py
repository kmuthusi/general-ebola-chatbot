"""Environment-backed settings shared by the local RAG application."""

import os

from dotenv import load_dotenv


load_dotenv()

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Keep these aligned with the example-rag vector index configuration.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
