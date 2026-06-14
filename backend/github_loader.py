import os
import requests
from urllib.parse import urlparse

from dotenv import load_dotenv


load_dotenv()


ALLOWED_EXTENSIONS = (
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css", ".md", ".json",
    ".yml", ".yaml", ".txt"
)

IGNORE_PARTS = (
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".next"
)


def github_headers():
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        return {}

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }


def parse_github_url(repo_url: str):
    parsed = urlparse(repo_url)

    parts = parsed.path.strip("/").split("/")

    if len(parts) < 2:
        raise ValueError("Invalid GitHub repository URL")

    owner = parts[0]
    repo = parts[1].replace(".git", "")

    return owner, repo


def canonical_repo_url(repo_url: str):
    owner, repo = parse_github_url(repo_url)

    return f"https://github.com/{owner}/{repo}"


def get_default_branch(owner: str, repo: str):
    url = f"https://api.github.com/repos/{owner}/{repo}"

    response = requests.get(url, headers=github_headers(), timeout=20)

    if response.status_code != 200:
        raise Exception(f"GitHub repo not found: {response.text}")

    data = response.json()

    return data["default_branch"]


def get_latest_commit_sha(owner: str, repo: str, branch: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"

    response = requests.get(url, headers=github_headers(), timeout=20)

    if response.status_code != 200:
        raise Exception(f"Could not fetch latest commit: {response.text}")

    data = response.json()

    return data["sha"]


def should_include_file(path: str):
    lower_path = path.lower()

    if any(part in lower_path for part in IGNORE_PARTS):
        return False

    return lower_path.endswith(ALLOWED_EXTENSIONS)


def list_repo_files(owner: str, repo: str, branch: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"

    response = requests.get(url, headers=github_headers(), timeout=30)

    if response.status_code != 200:
        raise Exception(f"Could not fetch repo tree: {response.text}")

    data = response.json()

    files = []

    for item in data.get("tree", []):
        if item.get("type") == "blob":
            path = item.get("path")

            if path and should_include_file(path):
                files.append(path)

    return files


def download_file(owner: str, repo: str, branch: str, file_path: str):
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"

    response = requests.get(url, headers=github_headers(), timeout=30)

    if response.status_code != 200:
        return None

    # skip large files above 200 KB
    if len(response.text) > 200_000:
        return None

    return response.text


def load_repository(repo_url: str):
    owner, repo = parse_github_url(repo_url)
    normalized_repo_url = canonical_repo_url(repo_url)

    branch = get_default_branch(owner, repo)

    commit_sha = get_latest_commit_sha(owner, repo, branch)

    file_paths = list_repo_files(owner, repo, branch)

    loaded_files = []

    for path in file_paths:
        content = download_file(owner, repo, branch, path)

        if content:
            loaded_files.append(
                {
                    "path": path,
                    "content": content
                }
            )

    return {
        "repo_url": normalized_repo_url,
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "commit_sha": commit_sha,
        "total_files": len(loaded_files),
        "files": loaded_files
    }
