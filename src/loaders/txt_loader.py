from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.utils import file_sha256


def load_txt(path: Path, knowledge_base: str) -> list[Document]:
    loader = TextLoader(str(path), autodetect_encoding=True)
    documents = loader.load()
    file_hash = file_sha256(path)

    for document in documents:
        document.metadata.update(
            {
                "source": path.name,
                "source_file": path.name,
                "source_path": str(path),
                "page": None,
                "page_number": None,
                "file_type": "txt",
                "file_hash": file_hash,
                "knowledge_base": knowledge_base,
            }
        )
    return documents
