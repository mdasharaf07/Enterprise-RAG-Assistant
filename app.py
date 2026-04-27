from __future__ import annotations

import logging
from html import escape

import streamlit as st

from src.chat_memory import (
    add_message,
    chat_to_markdown,
    clear_messages,
    get_messages,
    init_memory,
)
from src.database.feedback_db import get_analytics, init_db, save_feedback, save_query_metric
from src.document_loader import DocumentLoadError, load_documents
from src.rag_pipeline import RAGPipeline
from src.retriever import RetrieverService
from src.text_splitter import split_documents
from src.utils import (
    APP_ROOT,
    DEFAULT_KB_NAME,
    ensure_directories,
    export_chat_pdf,
    format_file_size,
    get_kb_data_dir,
    get_kb_vector_dir,
    knowledge_base_summary,
    list_knowledge_bases,
    remove_knowledge_base,
    save_uploaded_files,
    setup_logging,
)
from src.ui.dashboard import render_kb_summary, render_suggested_questions, render_top_cards
from src.vector_store import VectorStoreManager, VectorStoreMissingError


ensure_directories()
setup_logging()
init_db()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="EA",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_vector_manager(knowledge_base: str) -> VectorStoreManager:
    return VectorStoreManager(
        index_dir=get_kb_vector_dir(knowledge_base),
        cache_dir=APP_ROOT / "vectorstore" / knowledge_base / "embedding_cache",
    )


@st.cache_resource(show_spinner=False)
def load_vectorstore(knowledge_base: str):
    return get_vector_manager(knowledge_base).load()


def reset_vector_cache() -> None:
    get_vector_manager.clear()
    load_vectorstore.clear()


def render_css() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .block-container { padding-top: 1.6rem; max-width: 1180px; }
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid rgba(148, 163, 184, 0.22);
        }
        .stApp { background: #0b1120; color: #e5e7eb; }
        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.18);
            padding: 0.8rem 0.9rem;
            border-radius: 8px;
        }
        .source-box {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            padding: 0.85rem;
            background: #111827;
            margin-bottom: 0.65rem;
        }
        .source-title { font-weight: 700; color: #f8fafc; }
        .source-meta { color: #94a3b8; font-size: 0.86rem; margin-bottom: 0.35rem; }
        .chunk {
            color: #d1d5db;
            font-size: 0.92rem;
            white-space: pre-wrap;
            max-height: 170px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_knowledge_base(knowledge_base: str, uploaded_files: list) -> None:
    if not uploaded_files:
        st.warning("Upload one or more PDF, DOCX, or TXT files first.")
        return

    data_dir = get_kb_data_dir(knowledge_base)
    manager = get_vector_manager(knowledge_base)

    with st.status("Building knowledge base", expanded=True) as status:
        try:
            st.write("Saving uploaded files...")
            saved_paths = save_uploaded_files(uploaded_files, data_dir)
            logger.info("Saved %s uploaded files for knowledge base '%s'", len(saved_paths), knowledge_base)
            indexed_paths = [
                path
                for path in sorted(data_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"}
            ]

            st.write("Loading and cleaning documents...")
            documents = load_documents(indexed_paths, knowledge_base=knowledge_base)
            if not documents:
                st.error("No readable text was found in the uploaded files.")
                return

            st.write("Chunking documents...")
            chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
            if not chunks:
                st.error("Document text could not be chunked.")
                return

            st.write("Creating embeddings...")
            st.write("Indexing vectors in FAISS...")
            manager.rebuild(chunks)
            reset_vector_cache()
            status.update(label="Knowledge base ready", state="complete")
            st.success(f"Indexed {len(chunks)} chunks from {len(indexed_paths)} file(s).")
        except DocumentLoadError as exc:
            logger.exception("Document loading failed")
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface production-friendly error in UI
            logger.exception("Knowledge base build failed")
            st.error(f"Failed to build the knowledge base: {exc}")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No sources were returned for this answer.")
        return

    for source in sources:
        page = source.get("page")
        page_label = f"Page {page}" if page else "Page unavailable"
        score = source.get("score")
        confidence = source.get("confidence")
        score_label = f"Similarity: {score:.3f}" if isinstance(score, float) else "Similarity unavailable"
        conf_label = f"Confidence: {confidence:.0%}" if isinstance(confidence, float) else ""
        chunk_id = source.get("chunk_id")
        chunk_label = f"Chunk {chunk_id}" if chunk_id is not None else "Retrieved chunk"
        source_name = escape(str(source.get("source", "Unknown document")))
        chunk_text = escape(str(source.get("chunk", "")))
        st.markdown(
            f"""
            <div class="source-box">
              <div class="source-title">{source_name}</div>
              <div class="source-meta">{chunk_label} | {page_label} | {score_label} {conf_label}</div>
              <div class="chunk">{chunk_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feedback(message_index: int, question: str, answer: str) -> None:
    if not answer:
        return
    key = f"feedback-{message_index}"
    saved_key = f"{key}-saved"
    if st.session_state.get(saved_key):
        st.caption("Feedback saved.")
        return

    rating_value = st.feedback("thumbs", key=key)
    if rating_value is None:
        return

    rating = "helpful" if rating_value == 1 else "not_helpful"
    save_feedback(question=question, answer=answer, rating=rating)
    st.session_state[saved_key] = True
    st.toast("Feedback saved.")


def render_sidebar() -> tuple[str, str | None]:
    st.sidebar.title("Enterprise RAG")
    kb_names = list_knowledge_bases()
    if DEFAULT_KB_NAME not in kb_names:
        kb_names.insert(0, DEFAULT_KB_NAME)

    selected_kb = st.sidebar.selectbox("Knowledge base", kb_names, index=0)
    new_kb = st.sidebar.text_input("Create or switch KB", placeholder="finance-policies")
    if new_kb.strip():
        selected_kb = new_kb.strip()

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("Build KB", use_container_width=True):
            build_knowledge_base(selected_kb, uploaded_files)
    with col_b:
        if st.button("Clear DB", use_container_width=True):
            remove_knowledge_base(selected_kb)
            reset_vector_cache()
            clear_messages()
            st.success(f"Cleared '{selected_kb}'.")
            st.rerun()

    st.sidebar.divider()
    summary = knowledge_base_summary(selected_kb)
    analytics = get_analytics(selected_kb)
    st.sidebar.subheader("Knowledge Base")
    st.sidebar.metric("Documents", summary.document_count)
    st.sidebar.metric("Chunks", summary.chunk_count)
    st.sidebar.metric("Storage", format_file_size(summary.storage_bytes))
    st.sidebar.metric("Questions Asked", analytics.questions_asked)

    if summary.files:
        source_options = ["All documents"] + [file_info["name"] for file_info in summary.files]
        selected_source = st.sidebar.selectbox("Source filter", source_options)
        with st.sidebar.expander("Indexed files", expanded=False):
            for file_info in summary.files:
                st.caption(f"{file_info['name']} - {format_file_size(file_info['size'])}")
    else:
        selected_source = None

    st.sidebar.divider()
    st.sidebar.subheader("Chat")
    if st.sidebar.button("Clear chat", use_container_width=True):
        clear_messages()
        st.rerun()

    markdown_history = chat_to_markdown(get_messages())
    st.sidebar.download_button(
        "Download chat history",
        data=markdown_history,
        file_name=f"{selected_kb}-chat-history.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if st.sidebar.button("Export chat as PDF", use_container_width=True):
        pdf_path = export_chat_pdf(selected_kb, get_messages())
        st.sidebar.success(f"PDF exported: {pdf_path.name}")

    with st.sidebar.expander("Recent conversation", expanded=False):
        for msg in get_messages()[-8:]:
            st.caption(f"{msg['role'].title()}: {msg['content'][:140]}")

    return selected_kb, None if selected_source == "All documents" else selected_source


def main() -> None:
    render_css()
    init_memory()
    knowledge_base, selected_source = render_sidebar()

    st.title("Enterprise RAG Assistant")
    st.caption("Ask questions grounded in your uploaded enterprise documents.")

    summary = knowledge_base_summary(knowledge_base)
    analytics = get_analytics(knowledge_base)
    render_top_cards(knowledge_base, summary, analytics)
    render_kb_summary(summary)
    suggested_question = render_suggested_questions(summary.suggested_questions or [])

    history = get_messages()
    last_user_question = ""
    for index, message in enumerate(history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "user":
                last_user_question = message["content"]
            if message["role"] == "assistant" and message.get("sources"):
                metadata = message.get("metadata", {})
                if metadata.get("confidence"):
                    st.caption(
                        f"Confidence: {metadata['confidence']:.0%} | "
                        f"Retrieval: {metadata.get('retrieval_time', 0):.2f}s | "
                        f"Response: {metadata.get('response_time', 0):.2f}s"
                    )
                with st.expander("Sources", expanded=False):
                    render_sources(message["sources"])
                render_feedback(index, last_user_question, message["content"])

    question = suggested_question or st.chat_input("Ask a question about the uploaded documents")
    if not question:
        return

    add_message("user", question)
    with st.chat_message("user"):
        st.markdown(question)

    try:
        vectorstore = load_vectorstore(knowledge_base)
    except VectorStoreMissingError:
        logger.warning("User asked a question before a vector database existed")
        answer = "No vector database was found. Upload documents and build a knowledge base first."
        add_message("assistant", answer, [])
        with st.chat_message("assistant"):
            st.warning(answer)
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not load vector database")
        answer = f"Could not load the vector database: {exc}"
        add_message("assistant", answer, [])
        with st.chat_message("assistant"):
            st.error(answer)
        return

    with st.chat_message("assistant"):
        with st.status("Answering", expanded=True) as status:
            try:
                st.write("Retrieving relevant chunks...")
                retriever = RetrieverService(vectorstore=vectorstore, top_k=5)
                pipeline = RAGPipeline(retriever=retriever)

                st.write("Generating answer with Ollama llama3...")
                metadata_filter = {"source": selected_source} if selected_source else None
                result = pipeline.answer(
                    question=question,
                    chat_history=history,
                    metadata_filter=metadata_filter,
                )
                status.update(label="Answer ready", state="complete")

                st.markdown(result.answer)
                st.caption(
                    f"Confidence: {result.confidence:.0%} | "
                    f"Retrieval: {result.retrieval_time:.2f}s | "
                    f"Response: {result.response_time:.2f}s"
                )
                if result.sources:
                    with st.expander("Sources and retrieved chunks", expanded=True):
                        render_sources(result.sources)
                save_query_metric(
                    knowledge_base=knowledge_base,
                    question=question,
                    retrieval_time=result.retrieval_time,
                    response_time=result.response_time,
                    confidence=result.confidence,
                )
                add_message(
                    "assistant",
                    result.answer,
                    result.sources,
                    {
                        "confidence": result.confidence,
                        "retrieval_time": result.retrieval_time,
                        "response_time": result.response_time,
                    },
                )
            except ConnectionError as exc:
                logger.exception("Ollama generation failed")
                answer = str(exc)
                st.error(answer)
                add_message("assistant", answer, [])
            except Exception as exc:  # noqa: BLE001
                logger.exception("RAG pipeline failed")
                answer = f"I could not generate an answer because an error occurred: {exc}"
                st.error(answer)
                add_message("assistant", answer, [])


if __name__ == "__main__":
    main()
