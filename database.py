"""
database.py — SQLite database setup for Teena Bot

This module handles all direct database interactions:
  • Creating tables (tasks, mood_logs, messages, facts)
  • Helper functions for the operations currently supported:
    - Tasks: add/read tasks, mark done, delete, view completed
    - Mood logging: log mood, retrieve recent entries, compute averages
    - Messages: save/read conversation messages

We use Python's built-in `sqlite3` module (no ORM) and context managers
to make sure connections are always properly closed, even if an error occurs.
"""

import sqlite3
from datetime import datetime, timedelta

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

        # --- tasks table migration ---
        # If the database already exists from an earlier version, it won't
        # have the newer columns (priority, category, completed_at).
        # CREATE TABLE IF NOT EXISTS won't add them retroactively, so we
        # check which columns currently exist and ALTER TABLE to add any
        # that are missing.  This is safe to run repeatedly — once a
        # column exists, it simply won't be added again.
        cursor.execute("PRAGMA table_info(tasks);")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Map of column_name -> full ALTER TABLE statement
        migrations = {
            "priority": "ALTER TABLE tasks ADD COLUMN priority TEXT DEFAULT 'medium';",
            "category": "ALTER TABLE tasks ADD COLUMN category TEXT;",
            "completed_at": "ALTER TABLE tasks ADD COLUMN completed_at TIMESTAMP;",
            # Soft-delete columns: instead of permanently removing a task,
            # we set deleted = 1 and record when.  The row stays in the DB
            # for history / potential undo, but is hidden from active views.
            "deleted": "ALTER TABLE tasks ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0;",
            "deleted_at": "ALTER TABLE tasks ADD COLUMN deleted_at TIMESTAMP;",
        }

        for col_name, alter_sql in migrations.items():
            if col_name not in existing_columns:
                cursor.execute(alter_sql)

        # --- mood_logs table ---
        # Stores mood entries for the user's mood-tracking feature.
        # See log_mood(), get_recent_moods(), get_mood_average().
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
        # Schema-only for now — reserved for the future long-term memory
        # phase.  No helper functions read/write this table yet.
        #   • fact – a single piece of information (e.g. "User likes cats")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                fact        TEXT    NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # --- bot_config table ---
        # A simple key-value store for bot-wide settings (e.g. the
        # Telegram chat ID).  Scheduled / proactive jobs need the
        # chat ID persisted here because they run without an incoming
        # Update object, so there's no update.effective_chat.id to
        # read from at send time.
        #   • key   – unique config name (e.g. "chat_id")
        #   • value – the config value stored as text
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bot_config (
                key    TEXT PRIMARY KEY,
                value  TEXT NOT NULL
            );
        """)

        # --- proactive_state table ---
        # Tracks the state of proactive multi-turn conversations.
        # Teena initiates casual conversations 2-3 times per day;
        # this table manages the session lifecycle (idle → active →
        # exiting → idle) and enforces the daily message cap.
        #
        # Only ONE row ever exists (id = 1); we use INSERT OR REPLACE
        # to upsert it.  This is simpler than a key-value approach
        # for structured state with multiple related fields.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proactive_state (
                id             INTEGER PRIMARY KEY DEFAULT 1,
                status         TEXT    NOT NULL DEFAULT 'idle',
                exchanges      INTEGER NOT NULL DEFAULT 0,
                max_exchanges  INTEGER NOT NULL DEFAULT 3,
                started_at     TIMESTAMP,
                expires_at     TIMESTAMP,
                today_count    INTEGER NOT NULL DEFAULT 0,
                last_date      TEXT
            );
        """)

        # Ensure exactly one row exists in proactive_state so all
        # helpers can read/update it without INSERT-vs-UPDATE logic.
        cursor.execute(
            "INSERT OR IGNORE INTO proactive_state (id) VALUES (1);"
        )

        # Commit is handled automatically by the context manager,
        # but calling it explicitly makes the intent crystal clear.
        conn.commit()

    print("✅ Database initialised — all tables are ready.")


# ---------------------------------------------------------------------------
# Task helper functions
# ---------------------------------------------------------------------------

def add_task(text, due_date=None, priority="medium", category=None):
    """
    Insert a new task and return its id.

    Parameters
    ----------
    text : str
        The task description (e.g. "Buy groceries").
    due_date : str or None
        Optional deadline as a string (e.g. "2026-07-15").
    priority : str
        Task priority — one of 'low', 'medium', 'high' (default 'medium').
    category : str or None
        Optional free-text category (e.g. "work", "personal", "health").

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
            "INSERT INTO tasks (text, due_date, priority, category) VALUES (?, ?, ?, ?);",
            (text, due_date, priority, category),
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
        Each dict has keys: id, text, due_date, done, created_at,
        priority, category, completed_at.
        Results are ordered from oldest to newest.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        # `Row` factory lets us access columns by name (like a dict)
        # instead of by index — much more readable!
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Exclude both completed AND soft-deleted tasks so the active
        # list only shows genuinely open, non-deleted items.
        cursor.execute(
            "SELECT * FROM tasks WHERE done = 0 AND deleted = 0 ORDER BY created_at;"
        )

        # Convert each sqlite3.Row to a plain dict for easier use elsewhere.
        tasks = [dict(row) for row in cursor.fetchall()]

    return tasks


def mark_task_done(task_id):
    """
    Mark a task as completed by setting done = 1 and recording
    the completion timestamp in completed_at.

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

        # Set done = 1 and stamp the completion time
        cursor.execute(
            "UPDATE tasks SET done = 1, completed_at = ? WHERE id = ? AND done = 0;",
            (datetime.now().isoformat(), task_id),
        )
        conn.commit()

        # `rowcount` tells us how many rows the UPDATE affected.
        # If it's 0, no open task with that id exists.
        return cursor.rowcount > 0


def delete_task(task_id):
    """
    Soft-delete a task by setting deleted = 1 and recording the
    deletion timestamp.  The task is NOT permanently removed — it
    stays in the database for history and potential undo, but will
    no longer appear in get_open_tasks().

    Only tasks that are not already deleted (deleted = 0) will be
    updated, preventing double-deletes.

    Parameters
    ----------
    task_id : int
        The id of the task to soft-delete.

    Returns
    -------
    bool
        True if the task was found and soft-deleted, False if no
        task matched the given id or it was already deleted.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()

        # Only mark as deleted if it isn't already — mirrors the
        # guard in mark_task_done (AND done = 0).
        cursor.execute(
            "UPDATE tasks SET deleted = 1, deleted_at = ? WHERE id = ? AND deleted = 0;",
            (datetime.now().isoformat(), task_id),
        )
        conn.commit()

        # rowcount == 0 means no matching non-deleted task was found.
        return cursor.rowcount > 0


def get_completed_tasks(limit: int = 10) -> list[dict]:
    """
    Fetch the most recently completed tasks (done = 1) that have NOT
    been soft-deleted (deleted = 0).

    This closes the "what did I complete?" honesty gap — instead of the
    LLM guessing from conversation history, it gets real completion data
    straight from the database.

    Parameters
    ----------
    limit : int
        Maximum number of completed tasks to return (default 10).

    Returns
    -------
    list of dict
        Each dict has all task columns (id, text, due_date, done,
        created_at, priority, category, completed_at, deleted,
        deleted_at).  Ordered by completed_at descending (most
        recently completed first).
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Only return genuinely completed tasks — exclude soft-deleted
        # ones so "what did I finish?" doesn't surface removed tasks.
        cursor.execute(
            "SELECT * FROM tasks WHERE done = 1 AND deleted = 0 "
            "ORDER BY completed_at DESC LIMIT ?;",
            (limit,),
        )

        tasks = [dict(row) for row in cursor.fetchall()]

    return tasks


def get_tasks_completed_today() -> list[dict]:
    """
    Fetch tasks that were completed today (done = 1, completed_at is today).

    Used by the evening wrap-up to show what the user accomplished during
    the current day — more focused than get_completed_tasks() which spans
    all time.  Excludes soft-deleted tasks.

    Returns
    -------
    list of dict
        Each dict has all task columns.  Ordered by completed_at ascending
        (earliest completion first) so the list reads chronologically.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # date('now', 'localtime') gives today's date in the system's
        # local timezone.  completed_at is stored as an ISO timestamp
        # via datetime.now().isoformat(), which is also local time.
        # For a single-user bot on a local machine this is consistent.
        cursor.execute(
            "SELECT * FROM tasks WHERE done = 1 AND deleted = 0 "
            "AND date(completed_at) = date('now', 'localtime') "
            "ORDER BY completed_at ASC;"
        )

        tasks = [dict(row) for row in cursor.fetchall()]

    return tasks


# ---------------------------------------------------------------------------
# Mood logging helper functions
# ---------------------------------------------------------------------------

def log_mood(score: int, note: str | None = None) -> int:
    """
    Insert a new mood entry and return its id.

    Parameters
    ----------
    score : int
        A mood rating from 1 (worst) to 10 (best).
    note : str or None
        Optional free-text note describing the mood.

    Returns
    -------
    int
        The auto-generated id of the newly created mood entry.

    Raises
    ------
    ValueError
        If score is not an integer in the 1–10 range.
    """

    # Validate score before touching the database.
    if not isinstance(score, int) or not 1 <= score <= 10:
        raise ValueError(f"score must be an integer between 1 and 10, got {score!r}")

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO mood_logs (score, note) VALUES (?, ?);",
            (score, note),
        )
        conn.commit()
        new_id = cursor.lastrowid

    return new_id


def get_recent_moods(limit: int = 7) -> list[dict]:
    """
    Fetch the most recent mood entries, newest-first.

    The default limit of 7 gives roughly a week's worth of entries
    (assuming one entry per day).

    Parameters
    ----------
    limit : int
        Maximum number of mood entries to return (default 7).

    Returns
    -------
    list of dict
        Each dict has keys: id, score, note, created_at.
        Ordered by created_at descending (newest first).
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Secondary sort by id ensures a deterministic order when multiple
        # entries share the same created_at timestamp (e.g. several moods
        # logged within the same second).
        cursor.execute(
            "SELECT id, score, note, created_at FROM mood_logs "
            "ORDER BY created_at DESC, id DESC LIMIT ?;",
            (limit,),
        )

        moods = [dict(row) for row in cursor.fetchall()]

    return moods


def get_mood_average(days: int = 7) -> float | None:
    """
    Compute the average mood score over the last N days.

    Uses created_at to determine which entries fall within the window.
    Returns None when there are no entries in that period so callers
    can distinguish "no data" from "average of 0".

    Parameters
    ----------
    days : int
        Number of days to look back (default 7).

    Returns
    -------
    float or None
        The average score rounded to 1 decimal place, or None if
        there are no mood entries in the specified window.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()

        # SQLite's datetime('now', '-N days') gives us the cutoff point.
        # We use a parameterised modifier string for safety.
        cursor.execute(
            "SELECT AVG(score) FROM mood_logs "
            "WHERE created_at >= datetime('now', ?);",
            (f"-{days} days",),
        )

        result = cursor.fetchone()[0]

    # AVG() returns NULL (→ Python None) when there are no matching rows.
    if result is None:
        return None

    return round(result, 1)


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
# Bot config helper functions (chat ID persistence)
# ---------------------------------------------------------------------------
# Scheduled (proactive) messages — like morning briefings or reminders —
# need to know which Telegram chat to send to.  Unlike command or message
# handlers, scheduled jobs don't receive an incoming Update object, so
# there's no update.effective_chat.id available at send time.  We solve
# this by capturing the chat ID when the user first interacts with the
# bot (via /start) and persisting it in the bot_config table.

def save_chat_id(chat_id: int) -> None:
    """
    Persist the Telegram chat ID so scheduled jobs can retrieve it later.

    Uses INSERT OR REPLACE so the first call creates the row and
    subsequent calls simply update the value — no need for separate
    "does the row already exist?" logic.

    Parameters
    ----------
    chat_id : int
        The Telegram chat ID to save.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO bot_config (key, value) VALUES (?, ?);",
            ("chat_id", str(chat_id)),
        )
        conn.commit()


def get_chat_id() -> int | None:
    """
    Retrieve the persisted Telegram chat ID.

    Returns
    -------
    int or None
        The saved chat ID as an integer, or None if /start has never
        been run (i.e. no chat_id row exists in bot_config yet).
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM bot_config WHERE key = ?;",
            ("chat_id",),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    return int(row[0])


# ---------------------------------------------------------------------------
# Generalized config helpers
# ---------------------------------------------------------------------------
# These extend the bot_config key-value store beyond just the chat ID.
# Any bot-wide setting can be persisted here (e.g. proactive message
# counters, feature flags, user preferences).

def save_config(key: str, value: str) -> None:
    """
    Persist a key-value pair in the bot_config table.

    Uses INSERT OR REPLACE so the first call creates the row and
    subsequent calls update it.

    Parameters
    ----------
    key : str
        The config key (e.g. "proactive_daily_max").
    value : str
        The config value (stored as text; caller handles type conversion).
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO bot_config (key, value) VALUES (?, ?);",
            (key, value),
        )
        conn.commit()


def get_config(key: str) -> str | None:
    """
    Retrieve a config value by key.

    Returns
    -------
    str or None
        The stored value as a string, or None if the key doesn't exist.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM bot_config WHERE key = ?;",
            (key,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    return row[0]


# ---------------------------------------------------------------------------
# Proactive conversation state helpers
# ---------------------------------------------------------------------------
# These manage the lifecycle of Teena's proactive multi-turn conversations.
# The proactive_state table has exactly ONE row (id=1) that tracks:
#   - status: 'idle' | 'active' | 'exiting'
#   - exchanges: how many back-and-forths in the current session
#   - max_exchanges: random 2-3, chosen when session starts
#   - started_at / expires_at: timing for the current session
#   - today_count: how many proactive messages sent today (2-3 cap)
#   - last_date: date string for auto-resetting today_count

def get_proactive_state() -> dict:
    """
    Fetch the current proactive conversation state.

    Automatically resets today_count when the date has changed
    (i.e. a new day has started since the last proactive message).

    Returns
    -------
    dict
        Keys: status, exchanges, max_exchanges, started_at,
        expires_at, today_count, last_date.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM proactive_state WHERE id = 1;")
        row = cursor.fetchone()

    if row is None:
        # Shouldn't happen (init_db seeds the row), but be safe
        return {
            "status": "idle",
            "exchanges": 0,
            "max_exchanges": 3,
            "started_at": None,
            "expires_at": None,
            "today_count": 0,
            "last_date": None,
        }

    state = dict(row)

    # Auto-reset the daily counter if the date has rolled over
    today_str = datetime.now().strftime("%Y-%m-%d")
    if state.get("last_date") != today_str:
        state["today_count"] = 0
        state["last_date"] = today_str
        # Persist the reset so it sticks
        with sqlite3.connect(DATABASE_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE proactive_state SET today_count = 0, "
                "last_date = ? WHERE id = 1;",
                (today_str,),
            )
            conn.commit()

    return state


def start_proactive_session(max_exchanges: int = 3) -> None:
    """
    Begin a new proactive conversation session.

    Sets status to 'active', records the start time, calculates
    the expiry (4 minutes from now), and increments the daily counter.

    Parameters
    ----------
    max_exchanges : int
        How many back-and-forths before Teena exits (typically 2-3).
    """
    now = datetime.now()
    expires = now + timedelta(minutes=4)

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE proactive_state SET "
            "status = 'active', "
            "exchanges = 0, "
            "max_exchanges = ?, "
            "started_at = ?, "
            "expires_at = ?, "
            "today_count = today_count + 1, "
            "last_date = ? "
            "WHERE id = 1;",
            (max_exchanges, now.isoformat(), expires.isoformat(),
             now.strftime("%Y-%m-%d")),
        )
        conn.commit()


def increment_proactive_exchanges() -> int:
    """
    Increment the exchange counter for the active proactive session.

    Returns
    -------
    int
        The NEW exchange count after incrementing.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE proactive_state SET exchanges = exchanges + 1 "
            "WHERE id = 1;"
        )
        conn.commit()

        # Fetch the updated count
        cursor.execute(
            "SELECT exchanges FROM proactive_state WHERE id = 1;"
        )
        row = cursor.fetchone()

    return row[0] if row else 0


def end_proactive_session() -> None:
    """
    End the current proactive session — reset status to 'idle'.

    Clears the session-specific fields (exchanges, timing) but
    preserves today_count and last_date for daily cap enforcement.
    """

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE proactive_state SET "
            "status = 'idle', "
            "exchanges = 0, "
            "started_at = NULL, "
            "expires_at = NULL "
            "WHERE id = 1;"
        )
        conn.commit()


def update_proactive_expiry() -> None:
    """
    Extend the proactive session expiry by 4 minutes from now.

    Called when the user replies during an active session — gives
    them another window to respond before Teena auto-exits.
    """
    new_expiry = datetime.now() + timedelta(minutes=4)

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE proactive_state SET expires_at = ? WHERE id = 1;",
            (new_expiry.isoformat(),),
        )
        conn.commit()


def get_proactive_today_count() -> int:
    """
    Get the number of proactive messages sent today.

    Handles daily reset automatically by checking the stored date.

    Returns
    -------
    int
        Number of proactive messages sent today (0 if none or new day).
    """
    state = get_proactive_state()  # auto-resets if date changed
    return state.get("today_count", 0)


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
