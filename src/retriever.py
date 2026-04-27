from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    confidence: float


class RetrieverService:
    def __init__(self, vectorstore: FAISS, top_k: int = 5):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str, metadata_filter: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        logger.info("Retrieving top %s chunks for query", self.top_k)
        try:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=self.top_k,
                filter=metadata_filter,
            )
        except TypeError:
            results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
            if metadata_filter:
                results = [
                    (document, score)
                    for document, score in results
                    if all(document.metadata.get(key) == value for key, value in metadata_filter.items())
                ]
        chunks = [
            RetrievedChunk(
                document=document,
                score=float(score),
                confidence=self._score_to_confidence(float(score)),
            )
            for document, score in results
        ]
        logger.info("Retrieved %s chunks", len(chunks))
        return chunks

    @staticmethod
    def _score_to_confidence(score: float) -> float:
        return max(0.0, min(1.0, 1.0 / (1.0 + score)))
