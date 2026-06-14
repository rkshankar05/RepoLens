import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from chunker import chunk_files
from github_loader import (
    GitHubLoaderError,
    canonical_repo_url,
    get_repository_metadata,
    load_repository,
    parse_github_url
)
from index_cache import delete_cached_repo, get_cached_repo, update_cached_repo
from ollama_llm import (
    OLLAMA_EMBED_MODEL,
    OLLAMA_MODEL,
    check_ollama,
    model_available,
    ollama_status
)
from rag import answer_question, run_repo_intelligence, stream_answer
from vector_store import (
    CHROMA_PATH,
    collection_name,
    collection_status,
    delete_repo_index,
    get_collection,
    get_index_status,
    index_chunks
)


load_dotenv()

app = FastAPI(
    title="RepoLens AI Backend"
)

ACTIVE_REPO = {
    "repo_url": None,
    "owner": None,
    "repo": None,
    "branch": None,
    "commit_sha": None,
    "file_count": 0,
    "chunk_count": 0,
    "indexed": False
}


class RepoRequest(BaseModel):
    repo_url: str


class IndexRepoRequest(BaseModel):
    repo_url: str
    force: bool = False


class ChatRequest(BaseModel):
    repo_url: str
    question: str


def api_response(success: bool, message: str, data=None, error=None, status_code: int = 200):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": success,
            "message": message,
            "data": data or {},
            "error": error
        }
    )


def success_response(message: str, data=None):
    return api_response(
        success=True,
        message=message,
        data=data,
        error=None
    )


def error_response(message: str, error: str, status_code: int = 400, data=None):
    return api_response(
        success=False,
        message=message,
        data=data or {},
        error=error,
        status_code=status_code
    )


def handle_error(exc: Exception):
    if isinstance(exc, GitHubLoaderError):
        return error_response(
            message=exc.message,
            error=exc.code,
            status_code=exc.status_code
        )

    return error_response(
        message="Something went wrong. Check the repository URL, Ollama, and backend logs.",
        error=str(exc),
        status_code=500
    )


def ensure_ollama_ready(require_embedding: bool = False, require_llm: bool = False):
    if not check_ollama():
        raise RuntimeError(
            "ollama_not_running|Ollama is not running. Start it with the Ollama app or `ollama serve`."
        )

    if require_embedding and not model_available(OLLAMA_EMBED_MODEL):
        raise RuntimeError(
            f"embedding_model_not_available|Embedding model `{OLLAMA_EMBED_MODEL}` is not available. Run `ollama pull {OLLAMA_EMBED_MODEL}`."
        )

    if require_llm and not model_available(OLLAMA_MODEL):
        raise RuntimeError(
            f"llm_model_not_available|LLM model `{OLLAMA_MODEL}` is not available. Run `ollama pull {OLLAMA_MODEL}`."
        )


def handle_runtime_error(exc: RuntimeError):
    raw_message = str(exc)
    if "|" in raw_message:
        error, message = raw_message.split("|", 1)
        status_code = 503 if error in ("ollama_not_running", "embedding_model_not_available", "llm_model_not_available") else 400
        return error_response(message=message, error=error, status_code=status_code)

    return handle_error(exc)


def repo_index_payload(repo_url: str):
    repo_url = canonical_repo_url(repo_url)
    owner, repo = parse_github_url(repo_url)
    collection = get_collection(owner, repo)
    index_status = get_index_status(collection, repo_url)
    db_status = collection_status(collection)
    cached_repo = get_cached_repo(repo_url)

    return {
        "repo_url": repo_url,
        "owner": owner,
        "repo": repo,
        "indexed": index_status["indexed"],
        "total_chunks": index_status["total_chunks"],
        "commit_sha": index_status["commit_sha"],
        "latest_commit_sha": cached_repo.get("latest_commit_sha") if cached_repo else index_status["commit_sha"],
        "cache": {
            "exists": cached_repo is not None,
            "up_to_date": cached_repo is not None and cached_repo.get("latest_commit_sha") == index_status["commit_sha"],
            "indexed_at": cached_repo.get("indexed_at") if cached_repo else None,
            "file_count": cached_repo.get("file_count", 0) if cached_repo else 0,
            "chunk_count": cached_repo.get("chunk_count", 0) if cached_repo else 0,
            "collection_name": cached_repo.get("collection_name") if cached_repo else None
        },
        "collection": db_status
    }


def run_intelligence_endpoint(request: RepoRequest, template_name: str, message: str):
    try:
        ensure_ollama_ready(require_llm=True)
        index_data = repo_index_payload(request.repo_url)

        if not index_data["indexed"]:
            return error_response(
                message="This repository is not indexed yet. Click Index before using Repo Intelligence.",
                error="chroma_index_missing",
                status_code=404
            )

        result = run_repo_intelligence(
            repo_url=request.repo_url,
            template_name=template_name
        )

        return success_response(
            message,
            result
        )

    except RuntimeError as exc:
        return handle_runtime_error(exc)
    except Exception as exc:
        return handle_error(exc)


@app.get("/")
def home():
    return success_response(
        "RepoLens AI backend is running.",
        {
            "service": "RepoLens AI Backend"
        }
    )


@app.get("/health")
def health():
    status = ollama_status()

    return success_response(
        "Health check completed.",
        {
            "backend_status": "ok",
            "ollama_status": "running" if status["running"] else "not_running",
            "ollama_running": status["running"],
            "selected_llm_model": OLLAMA_MODEL,
            "embedding_model": OLLAMA_EMBED_MODEL,
            "llm_model_available": status["llm_model_available"],
            "embedding_model_available": status["embedding_model_available"],
            "chroma_db_path": CHROMA_PATH,
            "github_token_configured": bool(os.getenv("GITHUB_TOKEN"))
        }
    )


@app.get("/status")
def status():
    data = {
        "active_repo_url": ACTIVE_REPO["repo_url"],
        "active_repo_name": f"{ACTIVE_REPO['owner']}/{ACTIVE_REPO['repo']}" if ACTIVE_REPO["owner"] and ACTIVE_REPO["repo"] else None,
        "indexed": ACTIVE_REPO["indexed"],
        "file_count": ACTIVE_REPO["file_count"],
        "chunk_count": ACTIVE_REPO["chunk_count"],
        "latest_commit_sha": ACTIVE_REPO["commit_sha"],
        "cache": {
            "exists": False,
            "up_to_date": False,
            "indexed_at": None,
            "file_count": 0,
            "chunk_count": 0,
            "collection_name": None
        },
        "collection": {
            "available": False,
            "total_items": 0,
            "message": "No active repository selected yet."
        }
    }

    if ACTIVE_REPO["repo_url"]:
        try:
            index_data = repo_index_payload(ACTIVE_REPO["repo_url"])
            data["indexed"] = index_data["indexed"]
            data["chunk_count"] = index_data["total_chunks"]
            data["latest_commit_sha"] = index_data["latest_commit_sha"]
            data["cache"] = index_data["cache"]
            data["collection"] = index_data["collection"]
            ACTIVE_REPO["indexed"] = index_data["indexed"]
            ACTIVE_REPO["chunk_count"] = index_data["total_chunks"]
        except Exception as exc:
            data["collection"] = {
                "available": False,
                "total_items": 0,
                "error": str(exc)
            }

    return success_response("Repository status loaded.", data)


@app.post("/load-repo")
def load_repo(request: RepoRequest):
    try:
        data = load_repository(request.repo_url)
        ACTIVE_REPO.update({
            "repo_url": data["repo_url"],
            "owner": data["owner"],
            "repo": data["repo"],
            "branch": data["branch"],
            "commit_sha": data["commit_sha"],
            "file_count": data["total_files"],
            "chunk_count": 0,
            "indexed": False
        })

        return success_response(
            "Repository loaded successfully.",
            {
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
        )

    except Exception as exc:
        return handle_error(exc)


@app.post("/index-repo")
def index_repo(request: IndexRepoRequest):
    try:
        repo_meta = get_repository_metadata(request.repo_url)
        cached_repo = get_cached_repo(repo_meta["repo_url"])
        cached_collection = collection_name(repo_meta["owner"], repo_meta["repo"])

        if (
            cached_repo
            and not request.force
            and cached_repo.get("latest_commit_sha") == repo_meta["latest_commit_sha"]
        ):
            ACTIVE_REPO.update({
                "repo_url": cached_repo["repo_url"],
                "owner": cached_repo["owner"],
                "repo": cached_repo["repo"],
                "branch": cached_repo["branch"],
                "commit_sha": cached_repo["latest_commit_sha"],
                "file_count": cached_repo["file_count"],
                "chunk_count": cached_repo["chunk_count"],
                "indexed": True
            })

            return success_response(
                "Repository already indexed and up to date",
                {
                    "repo_url": cached_repo["repo_url"],
                    "owner": cached_repo["owner"],
                    "repo": cached_repo["repo"],
                    "branch": cached_repo["branch"],
                    "latest_commit_sha": cached_repo["latest_commit_sha"],
                    "indexed": True,
                    "skipped": True,
                    "forced": False,
                    "file_count": cached_repo["file_count"],
                    "chunk_count": cached_repo["chunk_count"],
                    "collection_name": cached_repo.get("collection_name", cached_collection),
                    "cache": cached_repo
                }
            )

        ensure_ollama_ready(require_embedding=True)
        repo_data = load_repository(request.repo_url)
        chunks = chunk_files(repo_data["files"])

        if not chunks:
            return error_response(
                message="The repository loaded, but no indexable chunks were created.",
                error="empty_repository",
                status_code=400
            )

        collection = get_collection(
            repo_data["owner"],
            repo_data["repo"]
        )

        if request.force:
            delete_repo_index(collection, repo_data["repo_url"])

        result = index_chunks(
            collection=collection,
            chunks=chunks,
            repo_url=repo_data["repo_url"],
            owner=repo_data["owner"],
            repo=repo_data["repo"],
            branch=repo_data["branch"],
            commit_sha=repo_data["commit_sha"]
        )

        indexed = result.get("status") in ("indexed", "skipped")
        chunk_count = result.get("total_chunks", len(chunks))
        current_collection_name = collection_name(repo_data["owner"], repo_data["repo"])
        cache_entry = update_cached_repo(
            repo_url=repo_data["repo_url"],
            owner=repo_data["owner"],
            repo=repo_data["repo"],
            branch=repo_data["branch"],
            latest_commit_sha=repo_data["commit_sha"],
            file_count=repo_data["total_files"],
            chunk_count=chunk_count,
            collection_name=current_collection_name
        )
        ACTIVE_REPO.update({
            "repo_url": repo_data["repo_url"],
            "owner": repo_data["owner"],
            "repo": repo_data["repo"],
            "branch": repo_data["branch"],
            "commit_sha": repo_data["commit_sha"],
            "file_count": repo_data["total_files"],
            "chunk_count": chunk_count,
            "indexed": indexed
        })

        message = "Repository force re-indexed successfully" if request.force else result.get("message", "Repository index updated.")

        return success_response(
            message,
            {
                "repo_url": repo_data["repo_url"],
                "owner": repo_data["owner"],
                "repo": repo_data["repo"],
                "branch": repo_data["branch"],
                "latest_commit_sha": repo_data["commit_sha"],
                "indexed": indexed,
                "skipped": False,
                "forced": request.force,
                "file_count": repo_data["total_files"],
                "chunk_count": chunk_count,
                "collection_name": current_collection_name,
                "cache": cache_entry,
                "commit_sha": repo_data["commit_sha"],
                "total_files": repo_data["total_files"],
                "total_chunks": chunk_count,
                "index_status": result
            }
        )

    except RuntimeError as exc:
        return handle_runtime_error(exc)
    except Exception as exc:
        return handle_error(exc)


@app.post("/index-status")
def index_status(request: RepoRequest):
    try:
        data = repo_index_payload(request.repo_url)

        return success_response(
            "Repository index status loaded.",
            data
        )

    except Exception as exc:
        return handle_error(exc)


@app.post("/delete-index")
def delete_index(request: RepoRequest):
    try:
        repo_url = canonical_repo_url(request.repo_url)
        owner, repo = parse_github_url(request.repo_url)
        collection = get_collection(owner, repo)
        result = delete_repo_index(
            collection=collection,
            repo_url=repo_url
        )

        if ACTIVE_REPO["repo_url"] == repo_url:
            ACTIVE_REPO["indexed"] = False
            ACTIVE_REPO["chunk_count"] = 0

        delete_cached_repo(repo_url)

        return success_response(
            result["message"],
            {
                "repo_url": repo_url,
                "index_status": result
            }
        )

    except Exception as exc:
        return handle_error(exc)


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        ensure_ollama_ready(require_llm=True)
        index_data = repo_index_payload(request.repo_url)

        if not index_data["indexed"]:
            return error_response(
                message="This repository is not indexed yet. Click Index before chatting.",
                error="chroma_index_missing",
                status_code=404
            )

        result = answer_question(
            repo_url=request.repo_url,
            question=request.question
        )

        return success_response(
            "Answer generated.",
            result
        )

    except RuntimeError as exc:
        return handle_runtime_error(exc)
    except Exception as exc:
        return handle_error(exc)


@app.post("/chat-stream")
def chat_stream(request: ChatRequest):
    try:
        ensure_ollama_ready(require_llm=True)
        index_data = repo_index_payload(request.repo_url)

        if not index_data["indexed"]:
            return error_response(
                message="This repository is not indexed yet. Click Index before chatting.",
                error="chroma_index_missing",
                status_code=404
            )

        return StreamingResponse(
            stream_answer(
                repo_url=request.repo_url,
                question=request.question
            ),
            media_type="application/x-ndjson"
        )

    except RuntimeError as exc:
        return handle_runtime_error(exc)
    except Exception as exc:
        return handle_error(exc)


@app.post("/repo-summary")
def repo_summary(request: RepoRequest):
    return run_intelligence_endpoint(
        request=request,
        template_name="repo_summary",
        message="Repository summary generated."
    )


@app.post("/explain-architecture")
def explain_architecture(request: RepoRequest):
    return run_intelligence_endpoint(
        request=request,
        template_name="explain_architecture",
        message="Architecture explanation generated."
    )


@app.post("/code-quality-review")
def code_quality_review(request: RepoRequest):
    return run_intelligence_endpoint(
        request=request,
        template_name="code_quality_review",
        message="Code quality review generated."
    )


@app.post("/generate-readme")
def generate_readme(request: RepoRequest):
    return run_intelligence_endpoint(
        request=request,
        template_name="generate_readme",
        message="README draft generated."
    )


@app.post("/interview-questions")
def interview_questions(request: RepoRequest):
    return run_intelligence_endpoint(
        request=request,
        template_name="interview_questions",
        message="Interview questions generated."
    )
