import os

import requests
from dotenv import load_dotenv


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def check_ollama():
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )

        if response.status_code == 200:
            return True

        return False

    except Exception:
        return False


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
