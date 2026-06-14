from langchain_text_splitters import RecursiveCharacterTextSplitter


def detect_language(file_path: str) -> str:
    if file_path.endswith(".py"):
        return "python"
    if file_path.endswith((".js", ".jsx")):
        return "javascript"
    if file_path.endswith((".ts", ".tsx")):
        return "typescript"
    if file_path.endswith(".html"):
        return "html"
    if file_path.endswith(".css"):
        return "css"
    if file_path.endswith(".md"):
        return "markdown"
    if file_path.endswith(".json"):
        return "json"
    if file_path.endswith((".yml", ".yaml")):
        return "yaml"
    return "text"


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\nclass ",
            "\ndef ",
            "\nfunction ",
            "\nconst ",
            "\nlet ",
            "\nvar ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    return [
        chunk
        for chunk in splitter.split_text(text)
        if chunk.strip()
    ]


def chunk_files(files):
    all_chunks = []

    for file in files:
        file_path = file["path"]
        content = file["content"]
        language = detect_language(file_path)

        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "file_path": file_path,
                "language": language,
                "chunk_no": i
            })

    return all_chunks
