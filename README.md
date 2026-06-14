# RepoLens AI 🔍

RepoLens AI is a local GitHub Repository RAG Assistant that indexes a GitHub codebase, stores embeddings locally in ChromaDB, and lets you ask source-grounded questions through a Streamlit chat UI.

## Problem It Solves

Understanding an unfamiliar repository usually means jumping between files, commits, docs, and search results. RepoLens AI turns a GitHub repository into a local, searchable knowledge base so developers can quickly ask questions like "Where is authentication handled?", "How does indexing work?", or "Which files should I read first?" and get answers grounded in retrieved source chunks.

## Key Features

- Index a GitHub repository from its URL.
- Load repository metadata, branch, latest commit SHA, and sample files.
- Split source files into retrieval-friendly chunks with LangChain.
- Generate local embeddings with Ollama.
- Store vectors locally in ChromaDB.
- Ask codebase questions through a Streamlit chat interface.
- Retrieve relevant source chunks and show citations/context.
- Stream model answers from the FastAPI backend.
- Check whether a repository is already indexed.
- Skip unnecessary re-indexing when the latest commit is already stored.
- Delete a repository index from the UI/API.
- Optional GitHub token support for higher rate limits and private repositories.

## Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain
- LangGraph
- Ollama
- ChromaDB
- GitHub API

## Architecture Flow

```text
User
  |
  v
Streamlit UI (backend/app.py)
  |
  v
FastAPI Backend (backend/main.py)
  |
  +--> GitHub Loader (github_loader.py)
  |      |
  |      v
  |    GitHub API
  |
  +--> Chunker (chunker.py)
  |      |
  |      v
  |    Source code chunks
  |
  +--> Embeddings (embeddings.py)
  |      |
  |      v
  |    Ollama embedding model
  |
  +--> Vector Store (vector_store.py)
  |      |
  |      v
  |    Local ChromaDB (backend/chroma_db/)
  |
  +--> RAG Workflow (rag.py + LangGraph)
         |
         v
       Ollama chat model
         |
         v
       Source-grounded answer + retrieved chunks
```

## Folder Structure

```text
github/
├── README.md
├── .gitignore
├── .env
├── .venv/
├── .agents/
├── .codex/
└── backend/
    ├── app.py
    ├── main.py
    ├── rag.py
    ├── github_loader.py
    ├── chunker.py
    ├── embeddings.py
    ├── ollama_llm.py
    ├── vector_store.py
    ├── requirements.txt
    ├── .env.example
    ├── .env
    ├── .venv/
    ├── __pycache__/
    └── chroma_db/
```

> Note: local environment files, virtual environments, caches, agent config, and ChromaDB data are ignored by Git.

## Installation

1. Clone or open the project root:

```bash
cd github
```

2. Create and activate a backend virtual environment:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your local environment file:

```bash
cp .env.example .env
```

5. Edit `backend/.env` with your local settings.

## Environment Variables

Create `backend/.env` from `backend/.env.example`.

| Variable | Required | Description | Example |
| --- | --- | --- | --- |
| `GITHUB_TOKEN` | Optional | GitHub token for higher API limits or private repositories. | `ghp_...` |
| `OLLAMA_BASE_URL` | Yes | Base URL for the local Ollama server. | `http://localhost:11434` |
| `OLLAMA_MODEL` | Yes | Ollama chat model used to answer questions. | `llama3.1:latest` |
| `OLLAMA_EMBED_MODEL` | Yes | Ollama embedding model used for code chunks. | `nomic-embed-text` |
| `CHROMA_DB_DIR` | Yes | Local directory for ChromaDB persistence. | `./chroma_db` |
| `BACKEND_API_URL` | Yes | FastAPI URL used by the Streamlit app. | `http://localhost:8000` |

## Ollama Setup

Install and start Ollama, then pull the required models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Confirm Ollama is running:

```bash
ollama list
```

By default, RepoLens AI expects Ollama at:

```text
http://localhost:11434
```

## Run The FastAPI Backend

From the `backend/` folder:

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

FastAPI will be available at:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Run The Streamlit UI

In a second terminal, from the `backend/` folder:

```bash
cd backend
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open the UI at:

```text
http://127.0.0.1:8501
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Returns a basic backend health message. |
| `GET` | `/health` | Returns backend status and whether Ollama is reachable. |
| `POST` | `/load-repo` | Loads repository metadata and sample file previews. |
| `POST` | `/index-repo` | Loads, chunks, embeds, and indexes a repository. |
| `POST` | `/index-status` | Checks whether a repository already has a stored index. |
| `POST` | `/delete-index` | Deletes stored embeddings for a repository. |
| `POST` | `/chat` | Returns a non-streaming RAG answer. |
| `POST` | `/chat-stream` | Streams a RAG answer as newline-delimited JSON. |

## Screenshots

Add screenshots to the following paths:

### Home

![RepoLens AI home screen](docs/screenshots/home.png)

### Indexing

![RepoLens AI indexing workflow](docs/screenshots/indexing.png)

### Chat

![RepoLens AI chat with retrieved source chunks](docs/screenshots/chat.png)

Create the screenshot folder with:

```bash
mkdir -p docs/screenshots
```

## Demo Workflow

1. Start Ollama and make sure the chat and embedding models are available.
2. Start the FastAPI backend from `backend/`.
3. Start the Streamlit app from `backend/`.
4. Open `http://127.0.0.1:8501`.
5. Enter a GitHub repository URL, for example `https://github.com/owner/repo`.
6. Click `Load` to preview repository metadata and sample files.
7. Click `Index` to chunk the code and store embeddings in ChromaDB.
8. Ask questions in the chat UI.
9. Review the answer and the retrieved source chunks.
10. Use `Status` to inspect the current index or `Delete` to remove stored embeddings.

## Resume Bullet

- Built RepoLens AI, a local GitHub repository RAG assistant using FastAPI, Streamlit, LangChain, LangGraph, Ollama, ChromaDB, and the GitHub API to index repositories, generate local embeddings, stream source-grounded answers, show retrieved code chunks, and manage commit-aware vector indexes.

## Future Improvements

- Add authentication for shared or team deployments.
- Support repository branch selection from the UI.
- Add file type filters before indexing.
- Persist chat history per repository.
- Add richer source citations with line ranges.
- Support local folder ingestion in addition to GitHub URLs.
- Add automated tests for API routes and RAG workflow behavior.
- Add Docker Compose for FastAPI, Streamlit, Ollama, and ChromaDB setup.
- Add evaluation scripts for answer quality and retrieval relevance.

## License

Add a license before publishing this repository. MIT is a common choice for open-source developer tools.
