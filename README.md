# RepoLens AI

RepoLens AI is a local GitHub repository assistant. It indexes a GitHub codebase, stores local embeddings in ChromaDB, and answers questions about the repository through a Streamlit chat UI.

## Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain
- LangGraph
- Ollama
- ChromaDB
- GitHub API

## Features

- Load public GitHub repositories
- Optional GitHub token support for higher API limits and private repos
- Recursive code chunking with LangChain
- Local embeddings with Ollama `nomic-embed-text`
- Local chat generation with Ollama `llama3.1`
- ChromaDB vector storage
- Commit-aware indexing
- Skip re-indexing when the latest commit is already indexed
- Replace old embeddings when the repo commit changes
- Delete stored embeddings from the UI
- Stream answers token by token in Streamlit
- Show source files used for answers

## Project Structure

```text
backend/
  app.py             Streamlit UI
  main.py            FastAPI API
  rag.py             LangGraph RAG workflow
  github_loader.py   GitHub repository loader
  chunker.py         Recursive code splitter
  embeddings.py      Ollama embedding model
  vector_store.py    ChromaDB indexing/search/delete helpers
  ollama_llm.py      Ollama health/config helper
  requirements.txt   Python dependencies
```

## Setup

Create and activate a virtual environment:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment example:

```bash
cp .env.example .env
```

Recommended `.env` values:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
OLLAMA_EMBED_MODEL=nomic-embed-text:latest
BACKEND_API_URL=http://127.0.0.1:8000
GITHUB_TOKEN=optional_github_token
```

Pull the Ollama models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Run

Start the FastAPI backend:

```bash
cd backend
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

Start the Streamlit UI in another terminal:

```bash
cd backend
.venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

- `GET /` - health message
- `GET /health` - backend and Ollama status
- `POST /load-repo` - load repository metadata and sample files
- `POST /index-repo` - index repository chunks
- `POST /index-status` - check existing index status
- `POST /delete-index` - delete stored embeddings for a repo
- `POST /chat` - non-streaming chat
- `POST /chat-stream` - streaming chat

## Resume Bullet

Built RepoLens AI, a local RAG-based GitHub repository assistant using FastAPI, Streamlit, LangChain, LangGraph, Ollama, and ChromaDB, with commit-aware indexing, recursive code chunking, local embeddings, streaming answers, source citations, and vector deletion.
