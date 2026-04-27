from __future__ import annotations

from langchain_core.prompts import PromptTemplate


RAG_PROMPT = PromptTemplate.from_template(
    """You are an enterprise knowledge assistant.

Answer only from the provided context.

If information is unavailable, respond:

"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:"""
)


def build_context(source_blocks: list[str]) -> str:
    return "\n\n---\n\n".join(source_blocks)
