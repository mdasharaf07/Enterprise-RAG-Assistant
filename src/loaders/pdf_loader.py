from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.utils import file_sha256


def load_pdf(path: Path, knowledge_base: str) -> list[Document]:
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    file_hash = file_sha256(path)

    for index, document in enumerate(documents, start=1):
        page_number = int(document.metadata.get("page", index - 1)) + 1
        document.metadata.update(
            {
                "source": path.name,
                "source_file": path.name,
                "source_path": str(path),
                "page": page_number,
                "page_number": page_number,
                "file_type": "pdf",
                "file_hash": file_hash,
                "knowledge_base": knowledge_base,
            }
        )
    return documents
