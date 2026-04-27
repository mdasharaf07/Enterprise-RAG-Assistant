from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embedding_manager import get_cached_embeddings


logger = logging.getLogger(__name__)


class VectorStoreMissingError(FileNotFoundError):
    """Raised when the FAISS index does not exist yet."""


class VectorStoreManager:
    def __init__(self, index_dir: Path, cache_dir: Path):
        self.index_dir = index_dir
        self.cache_dir = cache_dir
        self._embeddings = get_cached_embeddings(cache_dir)

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir.parent / "metadata.json"

    def exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def rebuild(self, chunks: list[Document]) -> FAISS:
        if not chunks:
            raise ValueError("Cannot build a vector index from zero chunks.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building FAISS index with %s chunks", len(chunks))
        vectorstore = FAISS.from_documents(chunks, self._embeddings)
        vectorstore.save_local(str(self.index_dir))
        self._write_manifest(chunks)
        logger.info("Saved FAISS index at %s", self.index_dir)
        return vectorstore

    def add_documents(self, chunks: list[Document]) -> FAISS:
        if not chunks:
            raise ValueError("Cannot add zero chunks to a vector index.")
        if self.exists():
            vectorstore = self.load()
            vectorstore.add_documents(chunks)
        else:
            vectorstore = FAISS.from_documents(chunks, self._embeddings)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.index_dir))
        self._write_manifest(list(vectorstore.docstore._dict.values()))
        logger.info("Incrementally added %s chunks to FAISS index at %s", len(chunks), self.index_dir)
        return vectorstore

    def load(self) -> FAISS:
        if not self.exists():
            raise VectorStoreMissingError("FAISS index not found. Build the knowledge base first.")

        logger.info("Loading FAISS index from %s", self.index_dir)
        return FAISS.load_local(
            str(self.index_dir),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

    def _write_manifest(self, chunks: list[Document]) -> None:
        sources = sorted({chunk.metadata.get("source", "unknown") for chunk in chunks})
        page_values = {
            chunk.metadata.get("page_number") or chunk.metadata.get("page")
            for chunk in chunks
            if chunk.metadata.get("page_number") or chunk.metadata.get("page")
        }
        summary = self._build_summary(chunks, sources, len(page_values))
        payload = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_count": len(chunks),
            "sources": sources,
            "total_pages": len(page_values),
            "summary": summary,
            "suggested_questions": self._suggested_questions(chunks),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _build_summary(chunks: list[Document], sources: list[str], total_pages: int) -> str:
        text = " ".join(chunk.page_content[:500] for chunk in chunks[:20]).lower()
        topic_keywords = {
            "Leave Policies": ["leave", "holiday", "vacation", "absence"],
            "Payroll": ["payroll", "salary", "compensation", "bonus"],
            "Benefits": ["benefit", "insurance", "health", "retirement"],
            "Security": ["security", "incident", "authentication", "access"],
            "Compliance": ["compliance", "policy", "audit", "regulated"],
            "Vendor Management": ["vendor", "contract", "supplier", "third party"],
        }
        topics = [topic for topic, words in topic_keywords.items() if any(word in text for word in words)]
        if not topics:
            topics = ["Enterprise policies", "Operational knowledge", "Document guidance"]

        topic_lines = "\n".join(f"- {topic}" for topic in topics[:6])
        return (
            "Summary\n\n"
            f"Topics:\n{topic_lines}\n\n"
            f"Total Pages: {total_pages}\n"
            f"Total Documents: {len(sources)}"
        )

    @staticmethod
    def _suggested_questions(chunks: list[Document]) -> list[str]:
        text = " ".join(chunk.page_content[:500] for chunk in chunks[:20]).lower()
        questions = []
        if "leave" in text or "holiday" in text:
            questions.append("What is the leave policy?")
            questions.append("How many annual holidays are available?")
        if "benefit" in text or "insurance" in text:
            questions.append("What benefits are offered?")
        if "security" in text or "incident" in text:
            questions.append("What is the security incident reporting process?")
        if "vendor" in text:
            questions.append("What are the vendor access requirements?")

        defaults = [
            "What are the key policies in these documents?",
            "What actions or approvals are required?",
            "Which rules should employees follow?",
        ]
        for question in defaults:
            if len(questions) >= 5:
                break
            if question not in questions:
                questions.append(question)
        return questions[:5]
