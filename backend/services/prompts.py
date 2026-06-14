STRICT_GROUNDING_RULES = """
Strict grounding rules:
- Use only the retrieved repository context and indexed file list.
- Do not invent files, languages, commands, dependencies, tools, or architecture.
- If the retrieved context is insufficient for a claim, say "Not enough retrieved context".
- Always mention source file paths for important claims.
- If no Python chunks were retrieved but the indexed file list contains .py files, do not claim the repository has no Python files.
- Do not mention Java, javac, Maven, Gradle, Node, React, databases, cloud services, or commands unless they appear in the retrieved context or indexed file list.
"""


INTELLIGENCE_PROMPTS = {
    "repo_summary": {
        "queries": [
            "README requirements dependencies project overview app.py main.py rag.py",
            "FastAPI Streamlit LangGraph ChromaDB Ollama GitHub loader vector store",
            "main folders files setup run commands API endpoints"
        ],
        "instruction": """
Create a repository summary with these sections:

## Project Purpose
Explain what the project does.

## Tech Stack
List only technologies visible in the retrieved context or indexed file list.

## Main Folders and Files
Mention important source file paths.

## How The Project Works
Describe the workflow from user input to final output.

## Possible Improvements
Suggest practical improvements grounded in the retrieved code.
"""
    },
    "explain_architecture": {
        "queries": [
            "FastAPI backend main.py API endpoints request response flow",
            "Streamlit app.py frontend UI API calls status chat",
            "LangGraph rag.py workflow retrieval generation",
            "ChromaDB vector_store.py indexing search delete embeddings"
        ],
        "instruction": """
Explain the architecture with these sections:

## Data Flow
Describe how data moves through the system.

## Main Modules
Explain each important module and include source file paths.

## Frontend/Backend Connection
Explain how the UI talks to the backend.

## API Flow
Explain the main API endpoints and request/response flow.

## Vector DB Flow
Explain how documents become chunks, embeddings, and vector search results.
"""
    },
    "code_quality_review": {
        "queries": [
            "FastAPI backend main.py API endpoints error handling",
            "Streamlit app.py UI API calls status error handling",
            "LangGraph rag.py workflow retrieval generation",
            "ChromaDB vector_store.py indexing search delete",
            "GitHub loader github_loader.py repo fetching file filtering",
            "embeddings.py ollama_llm.py model configuration"
        ],
        "instruction": """
Review code quality with these sections:

## Strengths
List what is implemented well and cite source file paths.

## Weaknesses
List maintainability, reliability, or clarity issues grounded in retrieved code.

## Possible Bugs
Identify likely bugs or edge cases only from retrieved code.

## Missing Error Handling
Point out missing or incomplete error handling only when supported by source chunks.

## Refactoring Suggestions
Suggest simple refactors that would improve the project.
"""
    },
    "generate_readme": {
        "queries": [
            "README requirements setup run commands API endpoints",
            "FastAPI Streamlit LangGraph ChromaDB Ollama GitHub API",
            "app.py main.py rag.py github_loader.py vector_store.py embeddings.py"
        ],
        "instruction": """
Generate a README draft with these sections:

# Project Title

## Description

## Features

## Tech Stack

## Setup

## Run Commands

## API Endpoints

## Future Improvements

## Resume Bullet
"""
    },
    "interview_questions": {
        "queries": [
            "project architecture API routes UI code RAG workflow",
            "GitHub loading chunking embeddings vector store Ollama integration",
            "FastAPI Streamlit LangGraph ChromaDB source grounded answers"
        ],
        "instruction": """
Create 10 project-specific interview questions. For each question include:

- Question
- Simple Answer
- Technical Explanation

Focus only on this repository's actual design and implementation details.
"""
    }
}


INTELLIGENCE_SYSTEM_PROMPT = f"""
You are a senior Python AI engineer reviewing a GitHub repository.
Be specific, practical, and beginner-friendly.
{STRICT_GROUNDING_RULES}
"""
