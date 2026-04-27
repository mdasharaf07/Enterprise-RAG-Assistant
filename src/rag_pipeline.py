from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from langchain_ollama import ChatOllama

from src.prompt_template import RAG_PROMPT, build_context
from src.retriever import RetrievedChunk, RetrieverService


logger = logging.getLogger(__name__)
FALLBACK_ANSWER = "I could not find this information in the uploaded documents."


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    confidence: float = 0.0
    retrieval_time: float = 0.0
    response_time: float = 0.0


class RAGPipeline:
    def __init__(self, retriever: RetrieverService):
        self.retriever = retriever
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def answer(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        metadata_filter: dict | None = None,
    ) -> RAGResult:
        retrieval_start = time.perf_counter()
        retrieved = self.retriever.retrieve(question, metadata_filter=metadata_filter)
        retrieval_time = time.perf_counter() - retrieval_start
        if not retrieved:
            logger.info("No retrieval results for question")
            return RAGResult(answer=FALLBACK_ANSWER, sources=[], retrieval_time=retrieval_time)

        context_blocks = [self._format_chunk(chunk) for chunk in retrieved]
        context = build_context(context_blocks)
        prompt = RAG_PROMPT.format(context=context, question=self._with_history(question, chat_history or []))

        try:
            generation_start = time.perf_counter()
            llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0,
            )
            response = llm.invoke(prompt)
            response_time = time.perf_counter() - generation_start
            answer_text = getattr(response, "content", str(response)).strip()
            logger.info("Generated answer with Ollama model %s", self.model_name)
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError(
                "Ollama is not reachable. Start Ollama and run `ollama pull llama3`, "
                "then restart the Streamlit app."
            ) from exc

        confidence = sum(item.confidence for item in retrieved) / len(retrieved)
        return RAGResult(
            answer=answer_text or FALLBACK_ANSWER,
            sources=self._sources(retrieved),
            confidence=confidence,
            retrieval_time=retrieval_time,
            response_time=response_time,
        )

    @staticmethod
    def _format_chunk(chunk: RetrievedChunk) -> str:
        metadata = chunk.document.metadata
        source = metadata.get("source", "Unknown document")
        page = metadata.get("page") or "N/A"
        return f"Source: {source}\nPage: {page}\nContent:\n{chunk.document.page_content}"

    @staticmethod
    def _with_history(question: str, chat_history: list[dict]) -> str:
        if not chat_history:
            return question
        recent = chat_history[-6:]
        turns = "\n".join(f"{item['role']}: {item['content']}" for item in recent)
        return f"Recent conversation:\n{turns}\n\nCurrent question: {question}"

    @staticmethod
    def _sources(retrieved: list[RetrievedChunk]) -> list[dict]:
        sources = []
        for item in retrieved:
            metadata = item.document.metadata
            sources.append(
                {
                    "source": metadata.get("source", "Unknown document"),
                    "source_file": metadata.get("source_file", metadata.get("source", "Unknown document")),
                    "page": metadata.get("page_number") or metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk": item.document.page_content,
                    "score": item.score,
                    "confidence": item.confidence,
                }
            )
        return sources
