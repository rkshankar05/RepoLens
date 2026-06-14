import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")


def embedding_collection_suffix() -> str:
    return OLLAMA_EMBED_MODEL.replace(":", "_").replace("-", "_").lower()


def get_embeddings_model() -> Embeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def get_embedding(text: str) -> list[float]:
    return get_embeddings_model().embed_query(text)
