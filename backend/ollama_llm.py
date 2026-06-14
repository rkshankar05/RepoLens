import os

import requests
from dotenv import load_dotenv


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_ollama_models() -> list[str]:
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=5
    )
    response.raise_for_status()

    data = response.json()
    return [
        model.get("name", "")
        for model in data.get("models", [])
        if model.get("name")
    ]


def check_ollama() -> bool:
    try:
        get_ollama_models()
        return True
    except Exception:
        return False


def model_available(model_name: str) -> bool:
    try:
        available_models = get_ollama_models()
    except Exception:
        return False

    acceptable_names = {
        model_name,
        f"{model_name}:latest" if ":" not in model_name else model_name
    }

    return any(model in acceptable_names for model in available_models)


def ollama_status() -> dict:
    running = check_ollama()
    models = get_ollama_models() if running else []

    return {
        "running": running,
        "base_url": OLLAMA_BASE_URL,
        "llm_model": OLLAMA_MODEL,
        "embedding_model": OLLAMA_EMBED_MODEL,
        "llm_model_available": model_available(OLLAMA_MODEL),
        "embedding_model_available": model_available(OLLAMA_EMBED_MODEL),
        "available_models": models
    }


def ask_ollama(question: str, context: str):
    prompt = f"""
You are a GitHub repository assistant.

Answer the user's question using the repository context below.

If the answer is not available in the repository context, say:
"I don't know from this repository."

Repository Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    data = response.json()

    if "response" in data:
        return data["response"]

    if "error" in data:
        return "Ollama error: " + data["error"]

    return str(data)
