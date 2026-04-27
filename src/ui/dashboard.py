from __future__ import annotations

import streamlit as st

from src.database.feedback_db import Analytics
from src.utils import KnowledgeBaseSummary


def render_top_cards(knowledge_base: str, summary: KnowledgeBaseSummary, analytics: Analytics) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Knowledge Base", knowledge_base)
    m2.metric("Documents", summary.document_count)
    m3.metric("Chunks", summary.chunk_count)
    m4.metric("Last Updated", summary.last_updated or "Not built")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Knowledge Bases", summary.knowledge_base_count)
    a2.metric("Questions Asked", analytics.questions_asked)
    a3.metric("Avg Retrieval", f"{analytics.average_retrieval_time:.2f}s")
    a4.metric("Avg Response", f"{analytics.average_response_time:.2f}s")


def render_kb_summary(summary: KnowledgeBaseSummary) -> None:
    with st.expander("Knowledge base summary", expanded=bool(summary.auto_summary)):
        if summary.auto_summary:
            st.markdown(summary.auto_summary)
        else:
            st.caption("Build the knowledge base to generate a summary.")
        if summary.total_pages:
            st.caption(f"Total pages: {summary.total_pages}")


def render_suggested_questions(questions: list[str]) -> str | None:
    if not questions:
        return None

    st.subheader("Suggested Questions")
    columns = st.columns(min(3, len(questions)))
    selected_question = None
    for index, question in enumerate(questions[:6]):
        with columns[index % len(columns)]:
            if st.button(question, key=f"suggested-question-{index}", use_container_width=True):
                selected_question = question
    return selected_question
