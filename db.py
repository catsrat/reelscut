"""
SQLite store for users and the job queue.

One file at DATA_DIR/reelscut.db (default ./data). WAL mode lets the web app and
the worker process read and write at the same time. Plain sqlite3, no ORM.

On a cloud host, put DATA_DIR on a persistent disk — otherwise every redeploy
forgets users and the minutes they've used this month.
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "reelscut.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,   -- Whop user tag (user_...), or "local"
    email           TEXT,
    name            TEXT,
    plan            TEXT NOT NULL DEFAULT 'free',  -- plan key (see billing.py)
    plan_checked_at REAL NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    status       TEXT NOT NULL,         -- queued | running | done | error
    progress     INTEGER NOT NULL DEFAULT 0,
    message      TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    opts         TEXT NOT NULL,         -- JSON editing options for the worker
    result       TEXT,                  -- JSON {clips, compliance, ai}
    minutes      REAL NOT NULL DEFAULT 0,  -- video minutes charged (on success)
    created_at   REAL NOT NULL,
    started_at   REAL,
    heartbeat_at REAL,
    finished_at  REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS jobs_user ON jobs(user_id, created_at);
"""

ACTIVE = ("queued", "running")

_local = threading.local()


def conn():
    """One connection per thread (sqlite3 connections aren't thread-safe)."""
    c = getattr(_local, "conn", None)
    if c is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        # isolation_level=None: autocommit; multi-step writes use BEGIN IMMEDIATE.
        c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=30000")
        _local.conn = c
    return c


def init():
    conn().executescript(SCHEMA)


# ---------------------------------------------------------------- users

def upsert_user(user_id, email=None, name=None):
    conn().execute(
        "INSERT INTO users (id, email, name, created_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "email = COALESCE(excluded.email, email), name = COALESCE(excluded.name, name)",
        (user_id, email, name, time.time()),
    )
    return get_user(user_id)


def get_user(user_id):
    return conn().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def set_plan(user_id, plan):
    conn().execute("UPDATE users SET plan = ?, plan_checked_at = ? WHERE id = ?",
                   (plan, time.time(), user_id))


def month_start():
    """Start of the current calendar month (UTC) as a unix timestamp."""
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


def minutes_used(user_id, since=None):
    row = conn().execute(
        "SELECT COALESCE(SUM(minutes), 0) FROM jobs "
        "WHERE user_id = ? AND status = 'done' AND finished_at >= ?",
        (user_id, month_start() if since is None else since),
    ).fetchone()
    return float(row[0])


# ---------------------------------------------------------------- jobs

def create_job(job_id, user_id, opts, title=""):
    conn().execute(
        "INSERT INTO jobs (id, user_id, status, message, title, opts, created_at) "
        "VALUES (?, ?, 'queued', 'Waiting to start...', ?, ?, ?)",
        (job_id, user_id, title, json.dumps(opts), time.time()),
    )


def get_job(job_id):
    return conn().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def active_job_for_user(user_id):
    row = conn().execute(
        "SELECT id FROM jobs WHERE user_id = ? AND status IN (?, ?) LIMIT 1",
        (user_id, *ACTIVE),
    ).fetchone()
    return row["id"] if row else None


def active_job_ids():
    rows = conn().execute("SELECT id FROM jobs WHERE status IN (?, ?)", ACTIVE)
    return {r["id"] for r in rows}


def queue_length():
    return conn().execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0]


def queue_position(job):
    """How many queued jobs are ahead of this one."""
    return conn().execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND created_at < ?",
        (job["created_at"],),
    ).fetchone()[0]


def job_public(job):
    """The /status payload the UI polls (same shape as the old in-memory job)."""
    result = json.loads(job["result"]) if job["result"] else {}
    message = job["message"]
    if job["status"] == "queued":
        ahead = queue_position(job)
        message = (f"In line — {ahead} video{'s' if ahead != 1 else ''} ahead of yours..."
                   if ahead else "Starting...")
    return {
        "status": "error" if job["status"] == "error" else
                  "done" if job["status"] == "done" else "running",
        "queued": job["status"] == "queued",
        "progress": job["progress"],
        "message": message,
        "title": job["title"],
        "clips": result.get("clips", []),
        "compliance": result.get("compliance"),
        "ai": result.get("ai", False),
    }


def claim_next():
    """Atomically take the oldest queued job and mark it running. Safe with
    several worker processes: BEGIN IMMEDIATE holds the write lock."""
    c = conn()
    c.execute("BEGIN IMMEDIATE")
    try:
        job = c.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if job:
            now = time.time()
            c.execute(
                "UPDATE jobs SET status = 'running', message = 'Starting...', "
                "started_at = ?, heartbeat_at = ? WHERE id = ?",
                (now, now, job["id"]),
            )
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    return get_job(job["id"]) if job else None


def update_progress(job_id, pct, message):
    conn().execute(
        "UPDATE jobs SET progress = ?, message = ?, heartbeat_at = ? WHERE id = ?",
        (int(pct), message, time.time(), job_id),
    )


def heartbeat(job_id):
    conn().execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (time.time(), job_id))


def set_title(job_id, title):
    conn().execute("UPDATE jobs SET title = ? WHERE id = ?", (title, job_id))


def finish(job_id, result, minutes):
    conn().execute(
        "UPDATE jobs SET status = 'done', progress = 100, message = 'Done', "
        "result = ?, minutes = ?, finished_at = ? WHERE id = ?",
        (json.dumps(result), float(minutes), time.time(), job_id),
    )


def fail(job_id, message, result=None):
    conn().execute(
        "UPDATE jobs SET status = 'error', message = ?, result = ?, finished_at = ? "
        "WHERE id = ?",
        (message, json.dumps(result) if result else None, time.time(), job_id),
    )


def fail_stale_running(max_silence_secs):
    """A running job whose worker stopped heartbeating (crash, redeploy) would
    otherwise spin forever in the UI — mark it failed so the user can retry."""
    conn().execute(
        "UPDATE jobs SET status = 'error', finished_at = ?, message = "
        "'Processing was interrupted (server restart). Please try again.' "
        "WHERE status = 'running' AND heartbeat_at < ?",
        (time.time(), time.time() - max_silence_secs),
    )
