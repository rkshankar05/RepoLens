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


class GitHubLoaderError(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


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

    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != "github.com":
        raise GitHubLoaderError(
            "Enter a valid GitHub repository URL, for example https://github.com/owner/repo.",
            "invalid_github_url",
            400
        )

    parts = parsed.path.strip("/").split("/")

    if len(parts) < 2:
        raise GitHubLoaderError(
            "Enter a valid GitHub repository URL, for example https://github.com/owner/repo.",
            "invalid_github_url",
            400
        )

    owner = parts[0]
    repo = parts[1].replace(".git", "")

    if not owner or not repo:
        raise GitHubLoaderError(
            "Enter a valid GitHub repository URL, for example https://github.com/owner/repo.",
            "invalid_github_url",
            400
        )

    return owner, repo


def raise_for_github_response(response, default_message: str):
    if response.status_code == 200:
        return

    token_configured = bool(os.getenv("GITHUB_TOKEN"))

    if response.status_code == 401:
        raise GitHubLoaderError(
            "GitHub rejected the token. Check GITHUB_TOKEN in backend/.env.",
            "github_token_invalid",
            401
        )

    if response.status_code == 403:
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            raise GitHubLoaderError(
                "GitHub API rate limit exceeded. Add a GITHUB_TOKEN or wait for the limit to reset.",
                "github_rate_limit_exceeded",
                429
            )

        raise GitHubLoaderError(
            "GitHub denied access. If this is a private repo, add a token with repository access.",
            "private_repo_without_token" if not token_configured else "github_access_denied",
            403
        )

    if response.status_code == 404:
        if token_configured:
            message = "Repository not found, or your GitHub token cannot access it."
            code = "repo_not_found"
        else:
            message = "Repository not found. If it is private, add GITHUB_TOKEN in backend/.env."
            code = "private_repo_without_token"

        raise GitHubLoaderError(message, code, 404)

    raise GitHubLoaderError(
        f"{default_message}: {response.text[:300]}",
        "github_api_error",
        response.status_code
    )


def canonical_repo_url(repo_url: str):
    owner, repo = parse_github_url(repo_url)

    return f"https://github.com/{owner}/{repo}"


def get_default_branch(owner: str, repo: str):
    url = f"https://api.github.com/repos/{owner}/{repo}"

    response = requests.get(url, headers=github_headers(), timeout=20)

    raise_for_github_response(response, "Could not fetch repository metadata")

    data = response.json()

    return data["default_branch"]


def get_latest_commit_sha(owner: str, repo: str, branch: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"

    response = requests.get(url, headers=github_headers(), timeout=20)

    raise_for_github_response(response, "Could not fetch latest commit")

    data = response.json()

    return data["sha"]


def get_repository_metadata(repo_url: str):
    owner, repo = parse_github_url(repo_url)
    normalized_repo_url = canonical_repo_url(repo_url)
    branch = get_default_branch(owner, repo)
    latest_commit_sha = get_latest_commit_sha(owner, repo, branch)

    return {
        "repo_url": normalized_repo_url,
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "latest_commit_sha": latest_commit_sha
    }


def should_include_file(path: str):
    lower_path = path.lower()

    if any(part in lower_path for part in IGNORE_PARTS):
        return False

    return lower_path.endswith(ALLOWED_EXTENSIONS)


def list_repo_files(owner: str, repo: str, branch: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"

    response = requests.get(url, headers=github_headers(), timeout=30)

    raise_for_github_response(response, "Could not fetch repository file tree")

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

    if not loaded_files:
        raise GitHubLoaderError(
            "No supported code or text files were found in this repository.",
            "empty_repository",
            400
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
