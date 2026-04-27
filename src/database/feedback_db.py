from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils import DATABASE_DIR


@dataclass(frozen=True)
class Analytics:
    feedback_count: int
    helpful_count: int
    not_helpful_count: int
    questions_asked: int
    average_retrieval_time: float
    average_response_time: float


DB_PATH = DATABASE_DIR / "enterprise_rag.db"


def get_connection() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback(
                id INTEGER PRIMARY KEY,
                question TEXT,
                answer TEXT,
                rating TEXT,
                timestamp DATETIME
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_metrics(
                id INTEGER PRIMARY KEY,
                knowledge_base TEXT,
                question TEXT,
                retrieval_time REAL,
                response_time REAL,
                confidence REAL,
                timestamp DATETIME
            )
            """
        )


def save_feedback(question: str, answer: str, rating: str) -> None:
    init_db()
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO feedback(question, answer, rating, timestamp) VALUES (?, ?, ?, ?)",
            (question, answer, rating, datetime.now().isoformat(timespec="seconds")),
        )


def save_query_metric(
    knowledge_base: str,
    question: str,
    retrieval_time: float,
    response_time: float,
    confidence: float,
) -> None:
    init_db()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO query_metrics(
                knowledge_base, question, retrieval_time, response_time, confidence, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                knowledge_base,
                question,
                retrieval_time,
                response_time,
                confidence,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def get_analytics(knowledge_base: str | None = None) -> Analytics:
    init_db()
    with get_connection() as connection:
        feedback_row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN rating = 'helpful' THEN 1 ELSE 0 END) AS helpful,
                SUM(CASE WHEN rating = 'not_helpful' THEN 1 ELSE 0 END) AS not_helpful
            FROM feedback
            """
        ).fetchone()

        if knowledge_base:
            metrics_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    AVG(retrieval_time) AS avg_retrieval,
                    AVG(response_time) AS avg_response
                FROM query_metrics
                WHERE knowledge_base = ?
                """,
                (knowledge_base,),
            ).fetchone()
        else:
            metrics_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    AVG(retrieval_time) AS avg_retrieval,
                    AVG(response_time) AS avg_response
                FROM query_metrics
                """
            ).fetchone()

    return Analytics(
        feedback_count=int(feedback_row["total"] or 0),
        helpful_count=int(feedback_row["helpful"] or 0),
        not_helpful_count=int(feedback_row["not_helpful"] or 0),
        questions_asked=int(metrics_row["total"] or 0),
        average_retrieval_time=float(metrics_row["avg_retrieval"] or 0.0),
        average_response_time=float(metrics_row["avg_response"] or 0.0),
    )
