from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.loaders.docx_loader import load_docx
from src.loaders.pdf_loader import load_pdf
from src.loaders.txt_loader import load_txt
from src.utils import DEFAULT_KB_NAME


import logging


logger = logging.getLogger(__name__)


class DocumentLoadError(RuntimeError):
    """Raised when one or more uploaded documents cannot be loaded."""


def load_documents(paths: list[Path], knowledge_base: str = DEFAULT_KB_NAME) -> list[Document]:
    documents: list[Document] = []
    errors: list[str] = []

    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{path.name}: empty or missing file")
            continue

        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                loaded = load_pdf(path, knowledge_base)
            elif suffix == ".docx":
                loaded = load_docx(path, knowledge_base)
            elif suffix == ".txt":
                loaded = load_txt(path, knowledge_base)
            else:
                errors.append(f"{path.name}: unsupported file format")
                continue

            documents.extend(loaded)
            logger.info("Loaded %s document sections from %s", len(loaded), path.name)
        except DocumentLoadError as exc:
            logger.exception("Failed loading %s", path.name)
            errors.append(str(exc))

    if errors and not documents:
        raise DocumentLoadError("No documents could be loaded. " + " | ".join(errors))
    if errors:
        logger.warning("Some documents failed to load: %s", " | ".join(errors))

    return documents
