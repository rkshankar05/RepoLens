import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL = os.getenv("BACKEND_API_URL", DEFAULT_API_URL).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


st.set_page_config(
    page_title="RepoLens AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(49, 51, 63, 0.12);
    }
    .source-row {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.6rem;
        background: rgba(250, 250, 250, 0.72);
    }
    .small-muted {
        color: rgba(49, 51, 63, 0.72);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def friendly_http_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "The backend returned an unexpected error."

    return (
        payload.get("message")
        or payload.get("error")
        or payload.get("detail")
        or "The backend returned an unexpected error."
    )


def unwrap_api_response(payload: dict) -> dict:
    if "success" not in payload:
        return payload

    if payload.get("success"):
        return payload.get("data", {})

    raise RuntimeError(payload.get("message") or payload.get("error") or "Request failed.")


def post_json(path: str, payload: dict, timeout: int = 120) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)

    if response.status_code >= 400:
        raise RuntimeError(friendly_http_error(response))

    return unwrap_api_response(response.json())


def get_json(path: str, timeout: int = 8) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)

    if response.status_code >= 400:
        raise RuntimeError(friendly_http_error(response))

    return unwrap_api_response(response.json())


def stream_chat(repo_url: str, question: str):
    response = requests.post(
        f"{API_URL}/chat-stream",
        json={
            "repo_url": repo_url,
            "question": question
        },
        stream=True,
        timeout=240
    )

    if response.status_code >= 400:
        raise RuntimeError(friendly_http_error(response))

    for line in response.iter_lines(decode_unicode=True):
        if line:
            yield json.loads(line)


def show_error(error: Exception) -> None:
    st.error(str(error))


def load_backend_health() -> dict:
    try:
        return {
            "connected": True,
            "data": get_json("/health", timeout=5),
            "error": None
        }
    except Exception as exc:
        return {
            "connected": False,
            "data": {},
            "error": str(exc)
        }


def load_repo_status() -> dict:
    try:
        return get_json("/status", timeout=8)
    except Exception:
        return {}


def refresh_status() -> None:
    st.session_state.backend_health = load_backend_health()
    st.session_state.repo_status = load_repo_status()


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Retrieved source chunks", expanded=False):
        for source in sources:
            score = source.get("score", source.get("distance"))
            score_text = f" | score {score:.4f}" if isinstance(score, (float, int)) else ""
            st.markdown(
                f"""
                <div class="source-row">
                    <strong>{source.get("file_path", "Unknown file")}</strong><br>
                    <span class="small-muted">
                        chunk {source.get("chunk_no", "unknown")}{score_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "repo_info" not in st.session_state:
    st.session_state.repo_info = None

if "index_info" not in st.session_state:
    st.session_state.index_info = None

if "backend_health" not in st.session_state:
    st.session_state.backend_health = load_backend_health()

if "repo_status" not in st.session_state:
    st.session_state.repo_status = load_repo_status()


health = st.session_state.backend_health
health_data = health.get("data", {})
repo_status = st.session_state.repo_status
backend_connected = health.get("connected", False)
ollama_running = health_data.get("ollama_running", False)
github_token_configured = health_data.get("github_token_configured", False)
repo_indexed = repo_status.get("indexed", False)


with st.sidebar:
    st.title("RepoLens AI")
    st.caption("Local GitHub repository RAG assistant")

    api_url = st.text_input(
        "Backend URL",
        value=API_URL,
        disabled=True
    )

    if st.button("Refresh status", use_container_width=True):
        refresh_status()
        st.rerun()

    st.divider()

    st.subheader("Status")

    if backend_connected:
        st.success("Backend connected")
    else:
        st.error("Backend not connected")
        st.caption(f"Start FastAPI at {api_url}")

    if ollama_running:
        st.success("Ollama running")
    else:
        st.warning("Ollama not running")

    if github_token_configured:
        st.success("GitHub token configured")
    else:
        st.info("GitHub token not configured")

    st.caption(f"LLM: {health_data.get('selected_llm_model', OLLAMA_MODEL)}")
    st.caption(f"Embeddings: {health_data.get('embedding_model', OLLAMA_EMBED_MODEL)}")
    st.caption(f"ChromaDB: {health_data.get('chroma_db_path', './chroma_db')}")

    st.divider()

    active_repo_name = repo_status.get("active_repo_name") or "None"
    st.subheader("Repository")
    st.caption(f"Active repo: {active_repo_name}")

    if repo_indexed:
        st.success("Indexed")
    else:
        st.warning("Not indexed")

    st.metric("Files", repo_status.get("file_count", 0))
    st.metric("Chunks", repo_status.get("chunk_count", 0))

    latest_commit_sha = repo_status.get("latest_commit_sha")
    if latest_commit_sha:
        st.caption(f"Latest commit: {latest_commit_sha[:12]}")

    cache = repo_status.get("cache", {})
    if cache.get("exists"):
        if cache.get("up_to_date"):
            st.success("Cache up to date")
        else:
            st.warning("Cache needs rebuild")
        st.caption(f"Cached chunks: {cache.get('chunk_count', 0)}")
        if cache.get("indexed_at"):
            st.caption(f"Cached at: {cache['indexed_at']}")
    else:
        st.info("No index cache yet")

    collection = repo_status.get("collection", {})
    if collection.get("available"):
        st.caption(f"Collection items: {collection.get('total_items', 0)}")
    else:
        st.caption(collection.get("message", "Collection not ready"))

    st.divider()

    repo_url = st.text_input(
        "GitHub repository URL",
        value=st.session_state.repo_url,
        placeholder="https://github.com/owner/repo"
    )
    st.session_state.repo_url = repo_url.strip()

    force_reindex = st.checkbox(
        "Force Re-index",
        value=False,
        help="Ignore the cache and rebuild embeddings for this repository."
    )

    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)

    with col_a:
        load_clicked = st.button(
            "Load",
            use_container_width=True,
            disabled=not backend_connected or not st.session_state.repo_url
        )

    with col_b:
        index_clicked = st.button(
            "Index",
            use_container_width=True,
            disabled=not backend_connected or not st.session_state.repo_url
        )

    with col_c:
        status_clicked = st.button(
            "Status",
            use_container_width=True,
            disabled=not backend_connected
        )

    with col_d:
        delete_clicked = st.button(
            "Delete",
            use_container_width=True,
            disabled=not backend_connected or not st.session_state.repo_url
        )

    if load_clicked:
        with st.spinner("Loading repository files..."):
            try:
                st.session_state.repo_info = post_json(
                    "/load-repo",
                    {"repo_url": st.session_state.repo_url},
                    timeout=180
                )
                st.session_state.index_info = None
                refresh_status()
                st.success("Repository loaded")
            except Exception as exc:
                show_error(exc)

    if status_clicked:
        refresh_status()
        st.success("Status refreshed")

    if index_clicked:
        with st.spinner("Indexing repository chunks..."):
            try:
                st.session_state.index_info = post_json(
                    "/index-repo",
                    {
                        "repo_url": st.session_state.repo_url,
                        "force": force_reindex
                    },
                    timeout=600
                )
                refresh_status()
                if st.session_state.index_info.get("skipped"):
                    st.success("Repository already indexed and up to date")
                elif st.session_state.index_info.get("forced"):
                    st.success("Repository rebuilt with force re-index")
                else:
                    st.success("Repository indexed")
            except Exception as exc:
                show_error(exc)

    if delete_clicked:
        with st.spinner("Deleting repository embeddings..."):
            try:
                st.session_state.index_info = post_json(
                    "/delete-index",
                    {"repo_url": st.session_state.repo_url},
                    timeout=60
                )
                refresh_status()
                st.success("Index deleted")
            except Exception as exc:
                show_error(exc)

    if st.session_state.repo_info:
        repo_info = st.session_state.repo_info
        st.divider()
        st.subheader(f"{repo_info['owner']}/{repo_info['repo']}")
        st.caption(f"Branch: {repo_info['branch']}")
        if repo_info.get("commit_sha"):
            st.caption(f"Commit: {repo_info['commit_sha'][:12]}")
        st.metric("Files loaded", repo_info["total_files"])

        with st.expander("Sample files", expanded=False):
            for file in repo_info.get("sample_files", []):
                st.markdown(f"**{file['path']}**")
                st.code(file["preview"], language="text")

    if st.session_state.index_info:
        index_info = st.session_state.index_info
        status = index_info.get("index_status", {})
        st.divider()
        st.subheader("Index")
        if index_info.get("skipped"):
            st.success("Skipped: repository already indexed and up to date")
        elif index_info.get("forced"):
            st.success("Rebuilt: force re-index completed")
        else:
            st.success("Rebuilt: repository index updated")

        if index_info.get("latest_commit_sha"):
            st.caption(f"Latest commit: {index_info['latest_commit_sha'][:12]}")
        st.caption(status.get("message", "Index status updated"))
        st.metric(
            "Chunks",
            index_info.get("chunk_count", index_info.get("total_chunks", status.get("total_chunks", 0)))
        )
        if status.get("commit_sha"):
            st.caption(f"Indexed commit: {status['commit_sha'][:12]}")
        if status.get("previous_commit_sha"):
            st.caption(f"Previous commit: {status['previous_commit_sha'][:12]}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


health = st.session_state.backend_health
health_data = health.get("data", {})
repo_status = st.session_state.repo_status
backend_connected = health.get("connected", False)
repo_indexed = repo_status.get("indexed", False)

st.title("Ask Your Repository")
st.caption("Index a GitHub repo, then ask questions grounded in its code.")

if not backend_connected:
    st.info("Start the FastAPI backend before loading or chatting with a repository.")
elif not st.session_state.repo_url:
    st.info("Enter a GitHub repository URL in the sidebar to begin.")
elif not repo_indexed:
    st.info("Load and index the repository before starting a chat.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))


question = st.chat_input(
    "Ask about the indexed repository",
    disabled=not backend_connected or not st.session_state.repo_url or not repo_indexed
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through the indexed code..."):
            try:
                answer = ""
                sources = []
                answer_placeholder = st.empty()

                for event in stream_chat(st.session_state.repo_url, question):
                    if event["type"] == "token":
                        answer += event["content"]
                        answer_placeholder.markdown(answer)

                    if event["type"] == "sources":
                        sources = event.get("sources", [])

                show_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as exc:
                show_error(exc)
