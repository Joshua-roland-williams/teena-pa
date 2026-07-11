"""
database.py — SQLite database setup for Teena Bot

This module handles all direct database interactions:
  • Creating tables (tasks, mood_logs, messages, facts)
  • Helper functions for CRUD operations

We use Python's built-in `sqlite3` module (no ORM) and context managers
to make sure connections are always properly closed, even if an error occurs.
"""

import sqlite3
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The database file will be created in the same folder as this script.
# SQLite creates the file automatically if it doesn't exist yet.
DATABASE_NAME = "teena.db"


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def init_db():
    """
    Create all required tables if they don't already exist.

    'IF NOT EXISTS' ensures this function is safe to call multiple times —
    it won't destroy data that's already there.
    """

    # `sqlite3.connect()` opens (or creates) the database file.
    # Using it as a context manager (`with`) auto-commits on success
    # and auto-rolls-back on error.
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()

        # --- tasks table ---
        # Stores to-do items for the user.
        #   • id          – unique identifier, auto-incremented by SQLite
        #   • text        – the task description (required, cannot be empty)
        #   • due_date    – optional deadline stored as text (e.g. "2026-07-15")
        #   • done        – 0 = open, 1 = completed (defaults to 0)
        #   • created_at  – timestamp auto-set to the moment the row is inserted
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT    NOT NULL,
                due_date    TEXT,
                done        INTEGER NOT NULL DEFAULT 0,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # --- mood_logs table ---
        # Tracks the user's mood over time.
        #   • score  – a numeric mood rating (e.g. 1–10)
        #   • note   – optional free-text note about the mood
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mood_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                score       INTEGER NOT NULL,
                note        TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # --- messages table ---
        # Keeps a history of the conversation between user and assistant.
        #   • role    – either 'user' or 'assistant'
        #   • content – the message text
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # --- facts table ---
        # Stores personal facts/preferences the bot learns about the user.
        #   • fact – a single piece of information (e.g. "User likes cats")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                fact        TEXT    NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Commit is handled automatically by the context manager,
        # but calling it explicitly makes the intent crystal clear.
        conn.commit()

    print("✅ Database initialised — all tables are ready.")


# ---------------------------------------------------------------------------
# Task helper functions
# ---------------------------------------------------------------------------

def add_task(text, due_date=None):
    """
    Insert a new task and return its id.

    Parameters
    ----------
    text : str
        The task description (e.g. "Buy groceries").
    due_date : str or None
        Optional deadline as a string (e.g. "2026-07-15").

    Returns
    -------
    int
        The auto-generated id of the newly created task.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()

        # Using parameterised queries (the ? placeholders) prevents
        # SQL injection — NEVER use f-strings or .format() for SQL values!
        cursor.execute(
            "INSERT INTO tasks (text, due_date) VALUES (?, ?);",
            (text, due_date),
        )
        conn.commit()

        # `lastrowid` gives us the id SQLite assigned to the new row.
        new_id = cursor.lastrowid

    return new_id


def get_open_tasks():
    """
    Fetch all tasks that haven't been completed yet (done = 0).

    Returns
    -------
    list of dict
        Each dict has keys: id, text, due_date, done, created_at.
        Results are ordered from oldest to newest.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        # `Row` factory lets us access columns by name (like a dict)
        # instead of by index — much more readable!
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM tasks WHERE done = 0 ORDER BY created_at;"
        )

        # Convert each sqlite3.Row to a plain dict for easier use elsewhere.
        tasks = [dict(row) for row in cursor.fetchall()]

    return tasks


def mark_task_done(task_id):
    """
    Mark a task as completed by setting done = 1.

    Parameters
    ----------
    task_id : int
        The id of the task to mark as done.

    Returns
    -------
    bool
        True if the task was found and updated, False if no task
        matched the given id.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE tasks SET done = 1 WHERE id = ?;",
            (task_id,),
        )
        conn.commit()

        # `rowcount` tells us how many rows the UPDATE affected.
        # If it's 0, no task with that id exists.
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Message history helper functions
# ---------------------------------------------------------------------------

def save_message(role: str, content: str) -> None:
    """
    Save a single chat message to the messages table.

    Parameters
    ----------
    role : str
        Either 'user' or 'assistant'.
    content : str
        The message text.
    """
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?);",
            (role, content),
        )
        conn.commit()


def get_recent_messages(limit: int = 10) -> list[dict]:
    """
    Fetch the most recent conversation messages, oldest-first.

    We query the last `limit` rows by id (descending), then reverse
    them so the caller gets chronological order — exactly what the
    LLM needs to understand the flow of conversation.

    Parameters
    ----------
    limit : int
        Maximum number of messages to return (default 10).

    Returns
    -------
    list of dict
        Each dict has keys: role ('user' or 'assistant'), content (str).
    """
    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Grab the N most recent rows (newest first), then reverse
        cursor.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?;",
            (limit,),
        )
        rows = [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

    # Reverse so oldest message comes first (chronological order)
    rows.reverse()
    return rows


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
# Running this file directly (python database.py) will initialise the DB
# and run a small smoke test so you can verify everything works.

if __name__ == "__main__":
    # 1. Create tables
    init_db()

    # 2. Add a couple of test tasks
    task1_id = add_task("Learn SQLite basics")
    task2_id = add_task("Build Teena Bot", due_date="2026-07-31")
    print(f"✅ Added tasks with ids: {task1_id}, {task2_id}")

    # 3. List open tasks
    open_tasks = get_open_tasks()
    print(f"📋 Open tasks ({len(open_tasks)}):")
    for t in open_tasks:
        due = f" (due {t['due_date']})" if t["due_date"] else ""
        print(f"   • [{t['id']}] {t['text']}{due}")

    # 4. Mark the first task as done
    result = mark_task_done(task1_id)
    print(f"✅ Marked task {task1_id} as done: {result}")

    # 5. Verify it's gone from the open list
    open_tasks = get_open_tasks()
    print(f"📋 Open tasks after completing one ({len(open_tasks)}):")
    for t in open_tasks:
        due = f" (due {t['due_date']})" if t["due_date"] else ""
        print(f"   • [{t['id']}] {t['text']}{due}")

    # 6. Try marking a non-existent task
    result = mark_task_done(9999)
    print(f"❌ Marked non-existent task 9999 as done: {result}")
