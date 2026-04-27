from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_embedding_model(cache_folder: str | None = None):
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=cache_folder,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    return embeddings


def get_cached_embeddings(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = get_embedding_model(str(cache_dir / "models"))
    try:
        from langchain.embeddings import CacheBackedEmbeddings
        from langchain.storage import LocalFileStore

        store = LocalFileStore(str(cache_dir / "documents"))
        return CacheBackedEmbeddings.from_bytes_store(
            base,
            store,
            namespace=EMBEDDING_MODEL,
        )
    except Exception:  # noqa: BLE001 - optional optimization across LangChain versions
        logger.warning("Embedding cache backend unavailable; using in-memory model only", exc_info=True)
        return base
