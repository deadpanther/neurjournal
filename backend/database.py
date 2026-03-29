"""
Persistent storage layer using SQLite.

Stores sessions, subjects, activation history, and uploaded content.
Domain-agnostic — works for therapy, education, UX research, etc.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / "neurjournal.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS domains (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            config JSON DEFAULT '{}',
            created_at REAL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS subjects (
            id TEXT PRIMARY KEY,
            domain_id TEXT NOT NULL REFERENCES domains(id),
            name TEXT NOT NULL,
            metadata JSON DEFAULT '{}',
            created_at REAL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL REFERENCES subjects(id),
            domain_id TEXT NOT NULL REFERENCES domains(id),
            started_at REAL DEFAULT (unixepoch()),
            ended_at REAL,
            notes TEXT DEFAULT '',
            metadata JSON DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS activations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            subject_id TEXT NOT NULL REFERENCES subjects(id),
            query TEXT NOT NULL,
            generated_answer TEXT DEFAULT '',
            regions JSON DEFAULT '{}',
            vertex_activations BLOB,
            emotion JSON DEFAULT '{}',
            processing_time_ms REAL DEFAULT 0,
            model_mode TEXT DEFAULT 'demo',
            metadata JSON DEFAULT '{}',
            created_at REAL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS uploaded_content (
            id TEXT PRIMARY KEY,
            domain_id TEXT NOT NULL REFERENCES domains(id),
            subject_id TEXT REFERENCES subjects(id),
            filename TEXT NOT NULL,
            content_type TEXT DEFAULT 'text/plain',
            content TEXT,
            processed INTEGER DEFAULT 0,
            metadata JSON DEFAULT '{}',
            created_at REAL DEFAULT (unixepoch())
        );

        CREATE INDEX IF NOT EXISTS idx_activations_session ON activations(session_id);
        CREATE INDEX IF NOT EXISTS idx_activations_subject ON activations(subject_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_subject ON sessions(subject_id);
        CREATE INDEX IF NOT EXISTS idx_subjects_domain ON subjects(domain_id);
    """)
    conn.commit()
    conn.close()


class SessionDB:
    """High-level interface for session/activation persistence."""

    def __init__(self):
        init_db()

    def ensure_domain(self, domain_id: str, name: str, description: str = "", config: dict = None) -> dict:
        conn = get_db()
        existing = conn.execute("SELECT * FROM domains WHERE id = ?", (domain_id,)).fetchone()
        if existing:
            conn.close()
            return dict(existing)
        conn.execute(
            "INSERT INTO domains (id, name, description, config) VALUES (?, ?, ?, ?)",
            (domain_id, name, description, json.dumps(config or {})),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM domains WHERE id = ?", (domain_id,)).fetchone()
        conn.close()
        return dict(row)

    def list_domains(self) -> list[dict]:
        conn = get_db()
        rows = conn.execute("SELECT * FROM domains ORDER BY created_at").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def create_subject(self, domain_id: str, name: str, metadata: dict = None) -> dict:
        sid = str(uuid.uuid4())[:8]
        conn = get_db()
        conn.execute(
            "INSERT INTO subjects (id, domain_id, name, metadata) VALUES (?, ?, ?, ?)",
            (sid, domain_id, name, json.dumps(metadata or {})),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM subjects WHERE id = ?", (sid,)).fetchone()
        conn.close()
        return dict(row)

    def list_subjects(self, domain_id: str = None) -> list[dict]:
        conn = get_db()
        if domain_id:
            rows = conn.execute("SELECT * FROM subjects WHERE domain_id = ? ORDER BY name", (domain_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM subjects ORDER BY name").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_or_create_subject(self, domain_id: str, name: str) -> dict:
        conn = get_db()
        row = conn.execute("SELECT * FROM subjects WHERE domain_id = ? AND name = ?", (domain_id, name)).fetchone()
        conn.close()
        if row:
            return dict(row)
        return self.create_subject(domain_id, name)

    def create_session(self, subject_id: str, domain_id: str, metadata: dict = None) -> dict:
        sid = str(uuid.uuid4())[:8]
        conn = get_db()
        conn.execute(
            "INSERT INTO sessions (id, subject_id, domain_id, metadata) VALUES (?, ?, ?, ?)",
            (sid, subject_id, domain_id, json.dumps(metadata or {})),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        conn.close()
        return dict(row)

    def save_activation(self, session_id: str, subject_id: str, query: str,
                        generated_answer: str = "", regions: dict = None,
                        vertex_activations: list = None, emotion: dict = None,
                        processing_time_ms: float = 0, model_mode: str = "demo",
                        metadata: dict = None) -> dict:
        aid = str(uuid.uuid4())[:8]
        verts_blob = None
        if vertex_activations:
            import numpy as np
            verts_blob = np.array(vertex_activations, dtype=np.float32).tobytes()
        conn = get_db()
        conn.execute(
            """INSERT INTO activations
               (id, session_id, subject_id, query, generated_answer, regions,
                vertex_activations, emotion, processing_time_ms, model_mode, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (aid, session_id, subject_id, query, generated_answer,
             json.dumps(regions or {}), verts_blob, json.dumps(emotion or {}),
             processing_time_ms, model_mode, json.dumps(metadata or {})),
        )
        conn.commit()
        conn.close()
        return {"id": aid, "session_id": session_id, "subject_id": subject_id}

    def get_subject_history(self, subject_id: str, limit: int = 50) -> list[dict]:
        conn = get_db()
        rows = conn.execute(
            """SELECT id, session_id, query, generated_answer, regions, emotion,
                      processing_time_ms, model_mode, created_at
               FROM activations WHERE subject_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (subject_id, limit),
        ).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["regions"] = json.loads(d["regions"]) if d["regions"] else {}
            d["emotion"] = json.loads(d["emotion"]) if d["emotion"] else {}
            results.append(d)
        return results

    def get_subject_trends(self, subject_id: str) -> dict:
        """Compute longitudinal activation trends for a subject."""
        history = self.get_subject_history(subject_id, limit=200)
        if not history:
            return {"total_queries": 0, "regions": {}, "emotions": {}}

        region_series = {}
        emotion_counts = {}
        for h in reversed(history):
            for rk, rv in h.get("regions", {}).items():
                if rk not in region_series:
                    region_series[rk] = {"label": rv.get("label", rk), "values": []}
                region_series[rk]["values"].append(rv.get("activation", 0))
            emo = h.get("emotion", {}).get("dominant")
            if emo:
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

        trends = {}
        for rk, data in region_series.items():
            vals = data["values"]
            if len(vals) >= 2:
                half = len(vals) // 2
                early_avg = sum(vals[:half]) / half
                late_avg = sum(vals[half:]) / (len(vals) - half)
                trends[rk] = {
                    "label": data["label"],
                    "current_avg": round(late_avg, 4),
                    "previous_avg": round(early_avg, 4),
                    "change": round(late_avg - early_avg, 4),
                    "n_samples": len(vals),
                }

        return {
            "total_queries": len(history),
            "regions": trends,
            "emotions": emotion_counts,
        }

    def save_upload(self, domain_id: str, filename: str, content: str,
                    subject_id: str = None, content_type: str = "text/plain") -> dict:
        uid = str(uuid.uuid4())[:8]
        conn = get_db()
        conn.execute(
            "INSERT INTO uploaded_content (id, domain_id, subject_id, filename, content_type, content) VALUES (?, ?, ?, ?, ?, ?)",
            (uid, domain_id, subject_id, filename, content_type, content),
        )
        conn.commit()
        conn.close()
        return {"id": uid, "filename": filename}
