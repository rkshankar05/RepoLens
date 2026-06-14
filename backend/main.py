from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from github_loader import load_repository
from chunker import chunk_files
from github_loader import canonical_repo_url, parse_github_url
from vector_store import delete_repo_index, get_collection, get_index_status, index_chunks
from rag import answer_question, stream_answer
from ollama_llm import check_ollama

app = FastAPI(
    title="GitHub Repository AI Assistant Backend"
)


class RepoRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    repo_url: str
    question: str


@app.get("/")
def home():
    return {
        "message": "GitHub Repository AI Assistant Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ollama_running": check_ollama()
    }


@app.post("/load-repo")
def load_repo(request: RepoRequest):
    try:
        data = load_repository(request.repo_url)

        return {
            "repo_url": data["repo_url"],
            "owner": data["owner"],
            "repo": data["repo"],
            "branch": data["branch"],
            "commit_sha": data["commit_sha"],
            "total_files": data["total_files"],
            "sample_files": [
                {
                    "path": file["path"],
                    "preview": file["content"][:300]
                }
                for file in data["files"][:5]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/index-repo")
def index_repo(request: RepoRequest):
    try:
        repo_data = load_repository(request.repo_url)

        chunks = chunk_files(repo_data["files"])

        collection = get_collection(
            repo_data["owner"],
            repo_data["repo"]
        )

        result = index_chunks(
            collection=collection,
            chunks=chunks,
            repo_url=repo_data["repo_url"],
            owner=repo_data["owner"],
            repo=repo_data["repo"],
            branch=repo_data["branch"],
            commit_sha=repo_data["commit_sha"]
        )

        return {
            "repo_url": repo_data["repo_url"],
            "owner": repo_data["owner"],
            "repo": repo_data["repo"],
            "branch": repo_data["branch"],
            "commit_sha": repo_data["commit_sha"],
            "total_files": repo_data["total_files"],
            "total_chunks": len(chunks),
            "index_status": result
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/index-status")
def index_status(request: RepoRequest):
    try:
        repo_url = canonical_repo_url(request.repo_url)
        owner, repo = parse_github_url(request.repo_url)
        collection = get_collection(owner, repo)

        return get_index_status(
            collection=collection,
            repo_url=repo_url
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/delete-index")
def delete_index(request: RepoRequest):
    try:
        repo_url = canonical_repo_url(request.repo_url)
        owner, repo = parse_github_url(request.repo_url)
        collection = get_collection(owner, repo)

        return delete_repo_index(
            collection=collection,
            repo_url=repo_url
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        result = answer_question(
            repo_url=request.repo_url,
            question=request.question
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat-stream")
def chat_stream(request: ChatRequest):
    try:
        return StreamingResponse(
            stream_answer(
                repo_url=request.repo_url,
                question=request.question
            ),
            media_type="application/x-ndjson"
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
