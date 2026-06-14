import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from services.github_loader import parse_github_url
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from services.prompts import INTELLIGENCE_PROMPTS, INTELLIGENCE_SYSTEM_PROMPT
from storage.vector_store import (
    collection_count,
    get_collection,
    get_indexed_file_paths,
    search_chunks
)

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


class RagState(TypedDict):
    repo_url: str
    question: str
    owner: str
    repo: str
    collection: object
    documents: list[Document]
    scores: list[float]
    answer: str
    sources: list[dict]
    stop_reason: str | None


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a GitHub repository assistant.

Answer the user's question using only the repository context.
For questions about project stack, technologies, frameworks, dependencies, or architecture,
infer only from filenames, imports, dependency/config files, and code shown in the context.
If the answer is not available in the repository context, say:
"I don't know from this repository."
"""
    ),
    (
        "human",
        """Repository Context:
{context}

Question:
{question}

Answer:"""
    )
])

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_predict=350
)


def is_overview_question(question: str) -> bool:
    normalized = question.lower()
    keywords = (
        "tech stack",
        "teach stack",
        "technology stack",
        "technologies used",
        "technology used",
        "framework used",
        "frameworks used",
        "dependencies",
        "main stack",
        "architecture",
        "project flow",
        "how this project works",
        "what is used"
    )

    return any(keyword in normalized for keyword in keywords)


def retrieval_query(question: str) -> str:
    if not is_overview_question(question):
        return question

    return f"""
{question}

Find repository overview evidence from README, requirements.txt, pyproject.toml,
package.json, settings.py, manage.py, asgi.py, wsgi.py, app.py, main.py, Dockerfile,
docker-compose.yml, imports, dependencies, framework configuration, database settings,
templates, static files, and project entry points.
"""


def load_collection(state: RagState) -> RagState:
    owner, repo = parse_github_url(state["repo_url"])
    collection = get_collection(owner, repo)

    return {
        **state,
        "owner": owner,
        "repo": repo,
        "collection": collection
    }


def check_index(state: RagState) -> RagState:
    if collection_count(state["collection"]) == 0:
        return {
            **state,
            "answer": "Repository is not indexed yet. Please index it first.",
            "sources": [],
            "stop_reason": "not_indexed"
        }

    return {
        **state,
        "stop_reason": None
    }


def should_retrieve(state: RagState) -> str:
    if state.get("stop_reason"):
        return "format_response"

    return "retrieve"


def retrieve(state: RagState) -> RagState:
    results = search_chunks(
        collection=state["collection"],
        question=retrieval_query(state["question"]),
        n_results=6 if is_overview_question(state["question"]) else 8
    )

    documents = [document for document, _score in results]
    scores = [score for _document, score in results]

    if not documents:
        return {
            **state,
            "documents": [],
            "scores": [],
            "answer": "No relevant chunks found.",
            "sources": [],
            "stop_reason": "no_results"
        }

    return {
        **state,
        "documents": documents,
        "scores": scores,
        "stop_reason": None
    }


def format_context(documents: list[Document]) -> str:
    context_parts = []

    for document in documents:
        meta = document.metadata
        context_parts.append(
            f"""
File: {meta["file_path"]}
Language: {meta["language"]}
Chunk: {meta["chunk_no"]}

Code/Text:
{document.page_content}
"""
        )

    return "\n\n---\n\n".join(context_parts)


def format_indexed_files(file_paths: list[str]) -> str:
    if not file_paths:
        return "No indexed file paths found."

    return "\n".join(f"- {file_path}" for file_path in file_paths)


def search_multiple_queries(collection, queries: list[str], n_results: int = 5):
    seen = set()
    documents = []
    scores = []

    for query in queries:
        for document, score in search_chunks(
            collection=collection,
            question=query,
            n_results=n_results
        ):
            meta = document.metadata
            key = (
                meta.get("file_path"),
                meta.get("chunk_no"),
                document.page_content[:80]
            )

            if key in seen:
                continue

            seen.add(key)
            documents.append(document)
            scores.append(score)

    return documents, scores


def should_generate(state: RagState) -> str:
    if state.get("stop_reason"):
        return "format_response"

    return "generate"


def generate(state: RagState) -> RagState:
    chain = prompt | llm
    response = chain.invoke({
        "context": format_context(state["documents"]),
        "question": state["question"]
    })

    return {
        **state,
        "answer": response.content
    }


def format_response(state: RagState) -> RagState:
    if state.get("sources"):
        return state

    sources = []

    for document, score in zip(
        state.get("documents", []),
        state.get("scores", [])
    ):
        meta = document.metadata
        sources.append({
            "file_path": meta["file_path"],
            "language": meta["language"],
            "chunk_no": meta["chunk_no"],
            "distance": score,
            "score": score
        })

    return {
        **state,
        "sources": sources
    }


graph_builder = StateGraph(RagState)
graph_builder.add_node("load_collection", load_collection)
graph_builder.add_node("check_index", check_index)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("format_response", format_response)
graph_builder.set_entry_point("load_collection")
graph_builder.add_edge("load_collection", "check_index")
graph_builder.add_conditional_edges(
    "check_index",
    should_retrieve,
    {
        "retrieve": "retrieve",
        "format_response": "format_response"
    }
)
graph_builder.add_conditional_edges(
    "retrieve",
    should_generate,
    {
        "generate": "generate",
        "format_response": "format_response"
    }
)
graph_builder.add_edge("generate", "format_response")
graph_builder.add_edge("format_response", END)
rag_graph = graph_builder.compile()


def initial_state(repo_url: str, question: str):
    return {
        "repo_url": repo_url,
        "question": question,
        "owner": "",
        "repo": "",
        "collection": None,
        "documents": [],
        "scores": [],
        "answer": "",
        "sources": [],
        "stop_reason": None
    }


def answer_question(repo_url: str, question: str):
    result = rag_graph.invoke(initial_state(repo_url, question))

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


def run_repo_intelligence(repo_url: str, template_name: str):
    template = INTELLIGENCE_PROMPTS[template_name]
    queries = template["queries"]
    state = load_collection(initial_state(repo_url, queries[0]))
    state = check_index(state)

    if state.get("stop_reason"):
        return {
            "result": state["answer"],
            "sources": [],
            "indexed_files": []
        }

    indexed_files = get_indexed_file_paths(
        collection=state["collection"],
        repo_url=repo_url
    )
    documents, scores = search_multiple_queries(
        collection=state["collection"],
        queries=queries,
        n_results=5 if template_name == "code_quality_review" else 4
    )

    if not documents:
        return {
            "result": "Not enough retrieved context",
            "sources": [],
            "indexed_files": indexed_files
        }

    intelligence_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            INTELLIGENCE_SYSTEM_PROMPT
        ),
        (
            "human",
            """Indexed files:
{indexed_files}

Repository Context:
{context}

Task:
{instruction}

Result:"""
        )
    ])
    chain = intelligence_prompt | llm
    response = chain.invoke({
        "indexed_files": format_indexed_files(indexed_files),
        "context": format_context(documents),
        "instruction": template["instruction"]
    })
    state = format_response({
        **state,
        "documents": documents,
        "scores": scores,
        "answer": response.content,
        "sources": []
    })

    return {
        "result": response.content,
        "sources": state["sources"],
        "indexed_files": indexed_files
    }


def stream_answer(repo_url: str, question: str):
    state = load_collection(initial_state(repo_url, question))
    state = check_index(state)

    if state.get("stop_reason"):
        yield json.dumps({
            "type": "token",
            "content": state["answer"]
        }) + "\n"
        yield json.dumps({
            "type": "sources",
            "sources": []
        }) + "\n"
        return

    state = retrieve(state)

    if state.get("stop_reason"):
        yield json.dumps({
            "type": "token",
            "content": state["answer"]
        }) + "\n"
        yield json.dumps({
            "type": "sources",
            "sources": []
        }) + "\n"
        return

    chain = prompt | llm

    for chunk in chain.stream({
        "context": format_context(state["documents"]),
        "question": question
    }):
        if chunk.content:
            yield json.dumps({
                "type": "token",
                "content": chunk.content
            }) + "\n"

    state = format_response(state)
    yield json.dumps({
        "type": "sources",
        "sources": state["sources"]
    }) + "\n"
