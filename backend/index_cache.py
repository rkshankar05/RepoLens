import json
from datetime import datetime, timezone
from pathlib import Path


CACHE_PATH = Path(__file__).with_name("index_cache.json")


def ensure_cache_file() -> None:
    if not CACHE_PATH.exists():
        CACHE_PATH.write_text("{}", encoding="utf-8")


def read_cache() -> dict:
    ensure_cache_file()

    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_cache(cache: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, sort_keys=True),
        encoding="utf-8"
    )


def get_cached_repo(repo_url: str) -> dict | None:
    return read_cache().get(repo_url)


def update_cached_repo(
    repo_url: str,
    owner: str,
    repo: str,
    branch: str,
    latest_commit_sha: str,
    file_count: int,
    chunk_count: int,
    collection_name: str
) -> dict:
    cache = read_cache()
    entry = {
        "repo_url": repo_url,
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "latest_commit_sha": latest_commit_sha,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "file_count": file_count,
        "chunk_count": chunk_count,
        "collection_name": collection_name
    }
    cache[repo_url] = entry
    write_cache(cache)
    return entry


def delete_cached_repo(repo_url: str) -> None:
    cache = read_cache()

    if repo_url in cache:
        del cache[repo_url]
        write_cache(cache)
