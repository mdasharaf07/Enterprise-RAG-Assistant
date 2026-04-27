from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.processing.cleaner import clean_text


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    cleaned_docs = []
    for document in documents:
        text = clean_text(document.page_content)
        if text:
            cleaned_docs.append(Document(page_content=text, metadata=dict(document.metadata)))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(cleaned_docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
