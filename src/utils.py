from __future__ import annotations

import hashlib
import html
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
VECTORSTORE_DIR = APP_ROOT / "vectorstore"
LOG_DIR = APP_ROOT / "logs"
ASSETS_DIR = APP_ROOT / "assets"
DATABASE_DIR = APP_ROOT / "database"
DEFAULT_KB_NAME = "default"
MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class KnowledgeBaseSummary:
    document_count: int
    chunk_count: int
    storage_bytes: int
    files: list[dict]
    last_updated: str | None
    knowledge_base_count: int = 0
    total_pages: int = 0
    auto_summary: str = ""
    suggested_questions: list[str] | None = None


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=False,
    )


def ensure_directories() -> None:
    load_dotenv(APP_ROOT / ".env")
    for directory in (DATA_DIR, VECTORSTORE_DIR, LOG_DIR, ASSETS_DIR, DATABASE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name.strip()]
    sanitized = "".join(keep).strip("._")
    return sanitized or DEFAULT_KB_NAME


def get_kb_data_dir(knowledge_base: str) -> Path:
    return DATA_DIR / sanitize_name(knowledge_base)


def get_kb_vector_dir(knowledge_base: str) -> Path:
    return VECTORSTORE_DIR / sanitize_name(knowledge_base) / "faiss_index"


def list_knowledge_bases() -> list[str]:
    names = {
        path.name
        for path in DATA_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    names.update(
        path.name
        for path in VECTORSTORE_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "faiss_index"
    )
    return sorted(names)


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded in uploaded_files:
        safe_name = sanitize_name(uploaded.name)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            logging.warning("Rejected unsupported upload: %s", uploaded.name)
            continue

        file_bytes = uploaded.getvalue()
        if not file_bytes:
            logging.warning("Rejected empty upload: %s", uploaded.name)
            continue
        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            logging.warning("Rejected oversized upload: %s", uploaded.name)
            continue

        destination = target_dir / safe_name
        destination.write_bytes(file_bytes)
        saved_paths.append(destination)

    return saved_paths


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def remove_knowledge_base(knowledge_base: str) -> None:
    for path in (get_kb_data_dir(knowledge_base), VECTORSTORE_DIR / sanitize_name(knowledge_base)):
        if path.exists():
            shutil.rmtree(path)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def knowledge_base_summary(knowledge_base: str) -> KnowledgeBaseSummary:
    data_dir = get_kb_data_dir(knowledge_base)
    vector_dir = get_kb_vector_dir(knowledge_base)
    manifest_path = vector_dir / "manifest.json"
    files: list[dict] = []
    chunk_count = 0
    last_updated = None
    total_pages = 0
    auto_summary = ""
    suggested_questions: list[str] = []

    if data_dir.exists():
        for file in sorted(data_dir.iterdir()):
            if file.is_file():
                files.append({"name": file.name, "size": file.stat().st_size})

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunk_count = int(manifest.get("chunk_count", 0))
            last_updated = manifest.get("last_updated")
            total_pages = int(manifest.get("total_pages", 0))
            auto_summary = str(manifest.get("summary", ""))
            suggested_questions = list(manifest.get("suggested_questions", []))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            logging.exception("Could not read manifest at %s", manifest_path)

    return KnowledgeBaseSummary(
        document_count=len(files),
        chunk_count=chunk_count,
        storage_bytes=_directory_size(data_dir) + _directory_size(vector_dir),
        files=files,
        last_updated=last_updated,
        knowledge_base_count=len(list_knowledge_bases()) or 1,
        total_pages=total_pages,
        auto_summary=auto_summary,
        suggested_questions=suggested_questions,
    )


def format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def export_chat_pdf(knowledge_base: str, messages: list[dict]) -> Path:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError("Install reportlab to export conversations as PDF.") from exc

    export_dir = APP_ROOT / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / f"{sanitize_name(knowledge_base)}-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Chat History: {knowledge_base}", styles["Title"]), Spacer(1, 12)]

    for message in messages:
        role = message.get("role", "message").title()
        content = message.get("content", "")
        story.append(Paragraph(f"<b>{role}</b>", styles["Heading3"]))
        story.append(Paragraph(html.escape(content).replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    return output_path
