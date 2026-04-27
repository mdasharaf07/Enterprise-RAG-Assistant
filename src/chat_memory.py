from __future__ import annotations

from datetime import datetime

import streamlit as st


MEMORY_KEY = "enterprise_rag_messages"


def init_memory() -> None:
    if MEMORY_KEY not in st.session_state:
        st.session_state[MEMORY_KEY] = []


def get_messages() -> list[dict]:
    init_memory()
    return st.session_state[MEMORY_KEY]


def add_message(
    role: str,
    content: str,
    sources: list[dict] | None = None,
    metadata: dict | None = None,
) -> None:
    init_memory()
    st.session_state[MEMORY_KEY].append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )


def clear_messages() -> None:
    st.session_state[MEMORY_KEY] = []


def chat_to_markdown(messages: list[dict]) -> str:
    lines = ["# Enterprise RAG Assistant Chat History", ""]
    for message in messages:
        lines.append(f"## {message['role'].title()} - {message.get('created_at', '')}")
        lines.append(message.get("content", ""))
        if message.get("sources"):
            lines.append("")
            lines.append("Sources:")
            for source in message["sources"]:
                page = source.get("page") or "N/A"
                lines.append(f"- {source.get('source', 'Unknown')} page {page}")
        lines.append("")
    return "\n".join(lines)
