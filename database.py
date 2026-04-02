"""SQLite database for OpenShorts clip library.

Stores videos, clips, and job history persistently so clips survive
container restarts and can be browsed/reused from the Clip Library tab.
"""

import sqlite3
import os
import threading
from datetime import datetime, timezone

DB_PATH = os.getenv("OPENSHORTS_DB_PATH", "data/openshorts.db")
LIBRARY_DIR = os.getenv("OPENSHORTS_LIBRARY_DIR", "data/library")

_local = threading.local()


def get_db() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Create tables if they don't exist."""
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            youtube_id  TEXT,
            url         TEXT,
            title       TEXT NOT NULL,
            duration    REAL,
            thumbnail   TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_youtube_id
            ON videos(youtube_id) WHERE youtube_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS clips (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id        INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            job_id          TEXT NOT NULL,
            file_path       TEXT NOT NULL,
            file_name       TEXT NOT NULL,
            start_time      REAL,
            end_time        REAL,
            rank            INTEGER,
            title           TEXT,
            description     TEXT,
            hashtags        TEXT,
            hook_text       TEXT,
            mode            TEXT DEFAULT 'ranking',
            favorite        INTEGER DEFAULT 0,
            gemini_model    TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id);
        CREATE INDEX IF NOT EXISTS idx_clips_job_id ON clips(job_id);
        CREATE INDEX IF NOT EXISTS idx_clips_favorite ON clips(favorite);

        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            video_id        INTEGER REFERENCES videos(id) ON DELETE SET NULL,
            mode            TEXT,
            status          TEXT NOT NULL DEFAULT 'queued',
            gemini_model    TEXT,
            input_tokens    INTEGER,
            output_tokens   INTEGER,
            total_cost      REAL,
            clip_count      INTEGER DEFAULT 0,
            duration_secs   REAL,
            error           TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at    TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

        CREATE TABLE IF NOT EXISTS scenes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            scene_index INTEGER NOT NULL,
            start_time  REAL NOT NULL,
            end_time    REAL NOT NULL,
            duration    REAL,
            is_junk     INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_scenes_video_id ON scenes(video_id);

        CREATE TABLE IF NOT EXISTS gemini_analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id        INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            job_id          TEXT,
            mode            TEXT,
            gemini_model    TEXT,
            prompt_hash     TEXT,
            raw_response    TEXT,
            parsed_json     TEXT,
            scene_count     INTEGER,
            clip_count      INTEGER,
            input_tokens    INTEGER,
            output_tokens   INTEGER,
            total_cost      REAL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_analyses_video_id ON gemini_analyses(video_id);
        CREATE INDEX IF NOT EXISTS idx_analyses_prompt_hash ON gemini_analyses(prompt_hash);

        CREATE TABLE IF NOT EXISTS transcripts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            language     TEXT,
            full_text   TEXT,
            segments    TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_video_lang
            ON transcripts(video_id, language);
    """)
    conn.commit()


# ── Video Operations ──────────────────────────────────────────────────

def upsert_video(youtube_id: str = None, url: str = None, title: str = "Untitled",
                 duration: float = None, thumbnail: str = None) -> int:
    """Insert or update a video record. Returns video ID."""
    conn = get_db()
    if youtube_id:
        row = conn.execute("SELECT id FROM videos WHERE youtube_id = ?", (youtube_id,)).fetchone()
        if row:
            conn.execute("""
                UPDATE videos SET url=COALESCE(?,url), title=COALESCE(?,title),
                       duration=COALESCE(?,duration), thumbnail=COALESCE(?,thumbnail)
                WHERE id=?
            """, (url, title, duration, thumbnail, row["id"]))
            conn.commit()
            return row["id"]

    cur = conn.execute(
        "INSERT INTO videos (youtube_id, url, title, duration, thumbnail) VALUES (?,?,?,?,?)",
        (youtube_id, url, title, duration, thumbnail)
    )
    conn.commit()
    return cur.lastrowid


def list_videos(limit: int = 50, offset: int = 0):
    """List all videos with clip counts."""
    conn = get_db()
    rows = conn.execute("""
        SELECT v.*, COUNT(c.id) as clip_count
        FROM videos v LEFT JOIN clips c ON c.video_id = v.id
        GROUP BY v.id ORDER BY v.created_at DESC
        LIMIT ? OFFSET ?
    """, (limit, offset)).fetchall()
    return [dict(r) for r in rows]


def get_video(video_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    return dict(row) if row else None


# ── Clip Operations ───────────────────────────────────────────────────

def save_clip(video_id: int, job_id: str, file_path: str, file_name: str,
              start_time: float = None, end_time: float = None, rank: int = None,
              title: str = None, description: str = None, hashtags: str = None,
              hook_text: str = None, mode: str = "ranking", gemini_model: str = None) -> int:
    """Save a clip to the library. Returns clip ID."""
    conn = get_db()
    cur = conn.execute("""
        INSERT INTO clips (video_id, job_id, file_path, file_name, start_time, end_time,
                           rank, title, description, hashtags, hook_text, mode, gemini_model)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (video_id, job_id, file_path, file_name, start_time, end_time,
          rank, title, description, hashtags, hook_text, mode, gemini_model))
    conn.commit()
    return cur.lastrowid


def list_clips(video_id: int = None, favorite_only: bool = False,
               search: str = None, limit: int = 100, offset: int = 0):
    """List clips with optional filtering."""
    conn = get_db()
    query = """
        SELECT c.*, v.title as video_title, v.youtube_id, v.url as video_url
        FROM clips c JOIN videos v ON c.video_id = v.id
        WHERE 1=1
    """
    params = []

    if video_id is not None:
        query += " AND c.video_id = ?"
        params.append(video_id)
    if favorite_only:
        query += " AND c.favorite = 1"
    if search:
        query += " AND (c.title LIKE ? OR v.title LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    query += " ORDER BY c.created_at DESC, c.rank ASC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def toggle_favorite(clip_id: int) -> bool:
    """Toggle favorite status. Returns new state."""
    conn = get_db()
    conn.execute("UPDATE clips SET favorite = 1 - favorite WHERE id = ?", (clip_id,))
    conn.commit()
    row = conn.execute("SELECT favorite FROM clips WHERE id = ?", (clip_id,)).fetchone()
    return bool(row["favorite"]) if row else False


def delete_clip(clip_id: int) -> bool:
    """Delete a clip from DB and disk."""
    conn = get_db()
    row = conn.execute("SELECT file_path FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if not row:
        return False
    filepath = row["file_path"]
    if os.path.exists(filepath):
        os.remove(filepath)
    conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
    conn.commit()
    return True


def get_clip(clip_id: int):
    conn = get_db()
    row = conn.execute("""
        SELECT c.*, v.title as video_title, v.youtube_id, v.url as video_url
        FROM clips c JOIN videos v ON c.video_id = v.id
        WHERE c.id = ?
    """, (clip_id,)).fetchone()
    return dict(row) if row else None


# ── Job Operations ────────────────────────────────────────────────────

def save_job(job_id: str, video_id: int = None, mode: str = None,
             status: str = "queued", gemini_model: str = None):
    """Record a job start."""
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO jobs (id, video_id, mode, status, gemini_model)
        VALUES (?,?,?,?,?)
    """, (job_id, video_id, mode, status, gemini_model))
    conn.commit()


def complete_job(job_id: str, status: str = "completed", clip_count: int = 0,
                 input_tokens: int = None, output_tokens: int = None,
                 total_cost: float = None, duration_secs: float = None, error: str = None):
    """Update job with completion info."""
    conn = get_db()
    conn.execute("""
        UPDATE jobs SET status=?, clip_count=?, input_tokens=?, output_tokens=?,
               total_cost=?, duration_secs=?, error=?, completed_at=datetime('now')
        WHERE id=?
    """, (status, clip_count, input_tokens, output_tokens, total_cost,
          duration_secs, error, job_id))
    conn.commit()


def list_jobs(limit: int = 50, offset: int = 0):
    """List job history."""
    conn = get_db()
    rows = conn.execute("""
        SELECT j.*, v.title as video_title
        FROM jobs j LEFT JOIN videos v ON j.video_id = v.id
        ORDER BY j.created_at DESC LIMIT ? OFFSET ?
    """, (limit, offset)).fetchall()
    return [dict(r) for r in rows]


def get_library_stats():
    """Get library statistics."""
    conn = get_db()
    stats = {}
    stats["total_videos"] = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    stats["total_clips"] = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    stats["total_favorites"] = conn.execute("SELECT COUNT(*) FROM clips WHERE favorite=1").fetchone()[0]
    stats["total_jobs"] = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    stats["completed_jobs"] = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='completed'").fetchone()[0]
    stats["total_analyses"] = conn.execute("SELECT COUNT(*) FROM gemini_analyses").fetchone()[0]
    cost_row = conn.execute("SELECT SUM(total_cost) FROM jobs WHERE total_cost IS NOT NULL").fetchone()
    stats["total_cost"] = round(cost_row[0] or 0, 6)
    return stats


# ── Scene Operations ──────────────────────────────────────────────────

def save_scenes(video_id: int, scenes: list, junk_indices: set = None):
    """Save scene boundaries for a video. Replaces existing scenes."""
    conn = get_db()
    conn.execute("DELETE FROM scenes WHERE video_id = ?", (video_id,))
    junk_indices = junk_indices or set()
    for i, (start, end) in enumerate(scenes):
        conn.execute("""
            INSERT INTO scenes (video_id, scene_index, start_time, end_time, duration, is_junk)
            VALUES (?,?,?,?,?,?)
        """, (video_id, i + 1, start, end, end - start, 1 if i in junk_indices else 0))
    conn.commit()


def get_scenes(video_id: int, include_junk: bool = False):
    """Get saved scenes for a video."""
    conn = get_db()
    query = "SELECT * FROM scenes WHERE video_id = ?"
    params = [video_id]
    if not include_junk:
        query += " AND is_junk = 0"
    query += " ORDER BY scene_index"
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def has_scenes(video_id: int) -> bool:
    """Check if scenes are already cached for a video."""
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) FROM scenes WHERE video_id = ?", (video_id,)).fetchone()
    return row[0] > 0


# ── Gemini Analysis Operations ────────────────────────────────────────

def save_analysis(video_id: int, job_id: str = None, mode: str = None,
                  gemini_model: str = None, prompt_hash: str = None,
                  raw_response: str = None, parsed_json: str = None,
                  scene_count: int = None, clip_count: int = None,
                  input_tokens: int = None, output_tokens: int = None,
                  total_cost: float = None) -> int:
    """Save a Gemini analysis result."""
    conn = get_db()
    cur = conn.execute("""
        INSERT INTO gemini_analyses (video_id, job_id, mode, gemini_model, prompt_hash,
                raw_response, parsed_json, scene_count, clip_count,
                input_tokens, output_tokens, total_cost)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (video_id, job_id, mode, gemini_model, prompt_hash,
          raw_response, parsed_json, scene_count, clip_count,
          input_tokens, output_tokens, total_cost))
    conn.commit()
    return cur.lastrowid


def get_cached_analysis(video_id: int, mode: str = None, prompt_hash: str = None):
    """Find a cached Gemini analysis for a video. Returns most recent match."""
    conn = get_db()
    query = "SELECT * FROM gemini_analyses WHERE video_id = ?"
    params = [video_id]
    if mode:
        query += " AND mode = ?"
        params.append(mode)
    if prompt_hash:
        query += " AND prompt_hash = ?"
        params.append(prompt_hash)
    query += " ORDER BY created_at DESC LIMIT 1"
    row = conn.execute(query, params).fetchone()
    return dict(row) if row else None


def list_analyses(video_id: int = None, limit: int = 50):
    """List Gemini analyses."""
    conn = get_db()
    query = """
        SELECT a.*, v.title as video_title
        FROM gemini_analyses a JOIN videos v ON a.video_id = v.id
    """
    params = []
    if video_id:
        query += " WHERE a.video_id = ?"
        params.append(video_id)
    query += " ORDER BY a.created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ── Transcript Operations ─────────────────────────────────────────────

def save_transcript(video_id: int, full_text: str, segments: str, language: str = "auto"):
    """Save transcript for a video."""
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO transcripts (video_id, language, full_text, segments)
        VALUES (?,?,?,?)
    """, (video_id, language, full_text, segments))
    conn.commit()


def get_transcript(video_id: int, language: str = None):
    """Get cached transcript for a video."""
    conn = get_db()
    if language:
        row = conn.execute("SELECT * FROM transcripts WHERE video_id = ? AND language = ?",
                           (video_id, language)).fetchone()
    else:
        row = conn.execute("SELECT * FROM transcripts WHERE video_id = ? ORDER BY created_at DESC LIMIT 1",
                           (video_id,)).fetchone()
    return dict(row) if row else None
