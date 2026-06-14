import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from services.embeddings import embedding_collection_suffix, get_embeddings_model
from services.github_loader import canonical_repo_url, parse_github_url


load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_DB_DIR", "./chroma_db")


def collection_name(owner: str, repo: str):
    base_name = f"{owner}_{repo}".replace("-", "_").lower()
    return f"{base_name}_{embedding_collection_suffix()}"


def get_collection(owner: str, repo: str):
    return Chroma(
        collection_name=collection_name(owner, repo),
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings_model()
    )


def collection_count(collection) -> int:
    return collection._collection.count()


def collection_status(collection) -> dict:
    try:
        return {
            "available": True,
            "total_items": collection_count(collection)
        }
    except Exception as exc:
        return {
            "available": False,
            "total_items": 0,
            "error": str(exc)
        }


def get_indexed_file_paths(collection: Chroma, repo_url: str) -> list[str]:
    results = get_repo_documents(collection, repo_url)
    file_paths = set()

    for metadata in results.get("metadatas", []):
        if metadata and metadata.get("file_path"):
            file_paths.add(metadata["file_path"])

    return sorted(file_paths)


def filter_collection_results(results: dict, repo_url: str) -> dict:
    owner, repo = parse_github_url(repo_url)
    normalized_url = canonical_repo_url(repo_url).lower()
    normalized_owner = owner.lower()
    normalized_repo = repo.lower()
    filtered = {
        "ids": [],
        "documents": [],
        "metadatas": []
    }

    ids = results.get("ids", [])
    documents = results.get("documents") or [None] * len(ids)
    metadatas = results.get("metadatas") or [None] * len(ids)

    for item_id, document, metadata in zip(ids, documents, metadatas):
        if not metadata:
            continue

        metadata_url = str(metadata.get("repo_url", "")).lower()
        metadata_owner = str(metadata.get("owner", "")).lower()
        metadata_repo = str(metadata.get("repo", "")).lower()

        if (
            metadata_url == normalized_url
            or (
                metadata_owner == normalized_owner
                and metadata_repo == normalized_repo
            )
        ):
            filtered["ids"].append(item_id)
            filtered["documents"].append(document)
            filtered["metadatas"].append(metadata)

    return filtered


def get_repo_documents(collection: Chroma, repo_url: str):
    results = collection.get(
        where={"repo_url": canonical_repo_url(repo_url)}
    )

    if results.get("ids"):
        return results

    return filter_collection_results(
        collection.get(include=["metadatas"]),
        repo_url
    )


def get_repo_documents_with_content(collection: Chroma, repo_url: str):
    results = collection.get(
        where={"repo_url": canonical_repo_url(repo_url)},
        include=["documents", "metadatas"]
    )

    if results.get("ids"):
        return results

    return filter_collection_results(
        collection.get(include=["documents", "metadatas"]),
        repo_url
    )


def get_index_status(collection: Chroma, repo_url: str):
    results = get_repo_documents(collection, repo_url)
    ids = results.get("ids", [])
    metadatas = results.get("metadatas", [])

    if not ids:
        return {
            "indexed": False,
            "total_chunks": 0,
            "commit_sha": None
        }

    commit_sha = None

    for metadata in metadatas:
        if metadata and metadata.get("commit_sha"):
            commit_sha = metadata["commit_sha"]
            break

    return {
        "indexed": True,
        "total_chunks": len(ids),
        "commit_sha": commit_sha
    }


def repo_already_indexed(collection: Chroma, repo_url: str, commit_sha: str) -> bool:
    results = collection.get(
        where={
            "$and": [
                {"repo_url": {"$eq": repo_url}},
                {"commit_sha": {"$eq": commit_sha}}
            ]
        }
    )

    return len(results["ids"]) > 0


def delete_repo_index(collection: Chroma, repo_url: str):
    results = get_repo_documents(collection, repo_url)
    ids = results.get("ids", [])

    if ids:
        collection.delete(ids=ids)

    return {
        "status": "deleted" if ids else "not_found",
        "message": "Repository index deleted" if ids else "No index found for repository",
        "deleted_chunks": len(ids)
    }


def index_chunks(collection, chunks, repo_url, owner, repo, branch, commit_sha):
    if repo_already_indexed(collection, repo_url, commit_sha):
        return {
            "status": "skipped",
            "message": "Repository already indexed at latest commit",
            "total_chunks": get_index_status(collection, repo_url)["total_chunks"],
            "commit_sha": commit_sha
        }

    previous_index = get_index_status(collection, repo_url)

    if previous_index["indexed"]:
        delete_repo_index(collection, repo_url)

    documents = []
    ids = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{owner}_{repo}_{commit_sha[:12]}_chunk_{i}")
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "repo_url": repo_url,
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "commit_sha": commit_sha,
                    "file_path": chunk["file_path"],
                    "language": chunk["language"],
                    "chunk_no": chunk["chunk_no"]
                }
            )
        )

    if documents:
        collection.add_documents(
            documents=documents,
            ids=ids
        )

    return {
        "status": "indexed",
        "message": "Repository indexed successfully",
        "total_chunks": len(chunks),
        "commit_sha": commit_sha,
        "previous_commit_sha": previous_index["commit_sha"]
    }


def search_chunks(collection, question: str, n_results: int = 8):
    return collection.similarity_search_with_score(
        question,
        k=n_results
    )
