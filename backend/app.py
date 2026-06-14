import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL = os.getenv("BACKEND_API_URL", DEFAULT_API_URL).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")


st.set_page_config(
    page_title="Repository AI Assistant",
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


def post_json(path: str, payload: dict, timeout: int = 120) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


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
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if line:
            yield json.loads(line)


def get_json(path: str, timeout: int = 8) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def show_error(error: Exception) -> None:
    if isinstance(error, requests.HTTPError):
        try:
            detail = error.response.json().get("detail", error.response.text)
        except ValueError:
            detail = error.response.text
        st.error(detail)
        return

    st.error(str(error))


if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "repo_info" not in st.session_state:
    st.session_state.repo_info = None

if "index_info" not in st.session_state:
    st.session_state.index_info = None


with st.sidebar:
    st.title("Repository AI")
    st.caption("LangChain + LangGraph backend")
    st.caption(f"Chat model: {OLLAMA_MODEL}")
    st.caption(f"Embedding model: {OLLAMA_EMBED_MODEL}")

    api_url = st.text_input(
        "Backend URL",
        value=API_URL,
        disabled=True
    )

    try:
        get_json("/")
        st.success("Backend online")
        st.caption("Start Ollama before chatting")
    except Exception as exc:
        st.error("Backend offline")
        st.caption(f"Start FastAPI at {api_url}")

    st.divider()

    repo_url = st.text_input(
        "GitHub repository URL",
        value=st.session_state.repo_url,
        placeholder="https://github.com/owner/repo"
    )
    st.session_state.repo_url = repo_url.strip()

    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)

    with col_a:
        load_clicked = st.button(
            "Load",
            use_container_width=True,
            disabled=not st.session_state.repo_url
        )

    with col_b:
        index_clicked = st.button(
            "Index",
            use_container_width=True,
            disabled=not st.session_state.repo_url
        )

    with col_c:
        status_clicked = st.button(
            "Status",
            use_container_width=True,
            disabled=not st.session_state.repo_url
        )

    with col_d:
        delete_clicked = st.button(
            "Delete",
            use_container_width=True,
            disabled=not st.session_state.repo_url
        )

    if load_clicked:
        with st.spinner("Loading repository files..."):
            try:
                st.session_state.repo_info = post_json(
                    "/load-repo",
                    {"repo_url": st.session_state.repo_url},
                    timeout=180
                )
                st.success("Repository loaded")
            except Exception as exc:
                show_error(exc)

    if status_clicked:
        with st.spinner("Checking index..."):
            try:
                st.session_state.index_info = {
                    "index_status": post_json(
                        "/index-status",
                        {"repo_url": st.session_state.repo_url},
                        timeout=60
                    )
                }
            except Exception as exc:
                show_error(exc)

    if index_clicked:
        with st.spinner("Indexing repository chunks..."):
            try:
                st.session_state.index_info = post_json(
                    "/index-repo",
                    {"repo_url": st.session_state.repo_url},
                    timeout=600
                )
                st.success("Repository indexed")
            except Exception as exc:
                show_error(exc)

    if delete_clicked:
        with st.spinner("Deleting repository embeddings..."):
            try:
                st.session_state.index_info = {
                    "index_status": post_json(
                        "/delete-index",
                        {"repo_url": st.session_state.repo_url},
                        timeout=60
                    )
                }
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
        else:
            st.caption("Commit: click Load again to refresh")
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
        st.caption(status.get("message", "Index status updated"))
        st.metric(
            "Chunks",
            index_info.get("total_chunks", status.get("total_chunks", 0))
        )
        if status.get("commit_sha"):
            st.caption(f"Indexed commit: {status['commit_sha'][:12]}")
        if status.get("previous_commit_sha"):
            st.caption(f"Previous commit: {status['previous_commit_sha'][:12]}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("Ask Your Repository")
st.caption("Index a GitHub repo, then ask questions grounded in its code.")

if not st.session_state.repo_url:
    st.info("Enter a GitHub repository URL in the sidebar to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("Sources", expanded=False):
                for source in sources:
                    st.markdown(
                        f"""
                        <div class="source-row">
                            <strong>{source["file_path"]}</strong><br>
                            <span class="small-muted">
                                {source["language"]} | chunk {source["chunk_no"]} | distance {source["distance"]:.4f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


question = st.chat_input(
    "Ask about the repository",
    disabled=not st.session_state.repo_url
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

                if sources:
                    with st.expander("Sources", expanded=False):
                        for source in sources:
                            st.markdown(
                                f"""
                                <div class="source-row">
                                    <strong>{source["file_path"]}</strong><br>
                                    <span class="small-muted">
                                        {source["language"]} | chunk {source["chunk_no"]} | distance {source["distance"]:.4f}
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as exc:
                show_error(exc)
