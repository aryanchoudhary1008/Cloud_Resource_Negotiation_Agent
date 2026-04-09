"""
database/db.py
--------------
SQLite database helpers for persisting negotiation sessions.

Uses Python's built-in `sqlite3` module — no external database
library needed. The database file is created automatically on
first run at `negotiation_history.db` in the project root.

Table: negotiations
    id           INTEGER PRIMARY KEY AUTOINCREMENT
    cpu          INTEGER  — cores requested
    ram          INTEGER  — GB requested
    storage      INTEGER  — GB requested
    budget       REAL     — user's max budget (USD)
    final_price  REAL     — agreed price (0 if no deal)
    final_cpu    INTEGER  — CPU in final deal
    final_ram    INTEGER  — RAM in final deal
    rounds       INTEGER  — number of rounds
    success      INTEGER  — 0 (fail) or 1 (success); SQLite has no bool
    total_reward REAL     — RL reward earned this episode
    timestamp    TEXT     — ISO-8601 UTC string
"""

import os
import sqlite3
from models.negotiation import NegotiationSession

# Path to the SQLite database file (relative to wherever app.py is run)
DB_PATH = "negotiation_history.db"


def get_connection():
    """
    Open and return a SQLite connection.

    Sets row_factory to sqlite3.Row so columns can be accessed
    by name (row["cpu"]) instead of just by index (row[1]).

    Returns:
        sqlite3.Connection
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create the negotiations table if it doesn't already exist.

    Safe to call on every app startup — the IF NOT EXISTS clause
    ensures existing data is never wiped.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS negotiations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            cpu          INTEGER NOT NULL,
            ram          INTEGER NOT NULL,
            storage      INTEGER NOT NULL,
            budget       REAL    NOT NULL,
            final_price  REAL    NOT NULL,
            final_cpu    INTEGER NOT NULL,
            final_ram    INTEGER NOT NULL,
            rounds       INTEGER NOT NULL,
            success      INTEGER NOT NULL,
            total_reward REAL    NOT NULL DEFAULT 0.0,
            timestamp    TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized: {os.path.abspath(DB_PATH)}")


def save_session(session: NegotiationSession) -> int:
    """
    Insert a completed negotiation session into the database.

    Args:
        session (NegotiationSession): The session to persist.

    Returns:
        int: The auto-assigned primary key (id) of the new row.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO negotiations
            (cpu, ram, storage, budget, final_price, final_cpu, final_ram,
             rounds, success, total_reward, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session.cpu,
        session.ram,
        session.storage,
        session.budget,
        session.final_price,
        session.final_cpu,
        session.final_ram,
        session.rounds,
        int(session.success),   # store Python bool as 0 / 1
        session.total_reward,
        session.timestamp,
    ))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_all_sessions() -> list:
    """
    Retrieve every negotiation session, newest first.

    Returns:
        list[NegotiationSession]: All recorded sessions.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM negotiations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        session = NegotiationSession(
            id           = row["id"],
            cpu          = row["cpu"],
            ram          = row["ram"],
            storage      = row["storage"],
            budget       = row["budget"],
            final_price  = row["final_price"],
            final_cpu    = row["final_cpu"],
            final_ram    = row["final_ram"],
            rounds       = row["rounds"],
            success      = bool(row["success"]),
            total_reward = row["total_reward"],
            timestamp    = row["timestamp"],
        )
        sessions.append(session)

    return sessions


def get_stats() -> dict:
    """
    Compute aggregate statistics across all recorded sessions.

    Metrics:
        total_sessions        — total rows in the table
        successful            — rows where success = 1
        failed                — rows where success = 0
        success_rate          — successful / total × 100  (%)
        avg_rounds            — mean number of rounds per session
        avg_cost_efficiency   — mean (budget - final_price) for successes
                                (positive = under budget, negative = over)
        resource_satisfaction — % of sessions where final_cpu == cpu
                                AND final_ram == ram (no downgrade occurred)

    Returns:
        dict: All metrics with float values rounded to 2 d.p.
              Returns all-zero dict if no sessions exist yet.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # Total sessions
    cursor.execute("SELECT COUNT(*) FROM negotiations")
    total = cursor.fetchone()[0]

    if total == 0:
        conn.close()
        return {
            "total_sessions":        0,
            "successful":            0,
            "failed":                0,
            "success_rate":          0.0,
            "avg_rounds":            0.0,
            "avg_cost_efficiency":   0.0,
            "resource_satisfaction": 0.0,
        }

    # Successful sessions
    cursor.execute("SELECT COUNT(*) FROM negotiations WHERE success = 1")
    successful = cursor.fetchone()[0]

    # Average rounds across all sessions
    cursor.execute("SELECT AVG(rounds) FROM negotiations")
    avg_rounds = cursor.fetchone()[0] or 0.0

    # Cost efficiency: how far under budget the user landed (successes only)
    cursor.execute("""
        SELECT AVG(budget - final_price)
        FROM negotiations
        WHERE success = 1
    """)
    avg_cost_eff = cursor.fetchone()[0] or 0.0

    # Resource satisfaction: no downgrade occurred
    cursor.execute("""
        SELECT COUNT(*) FROM negotiations
        WHERE final_cpu = cpu AND final_ram = ram
    """)
    full_resource_count = cursor.fetchone()[0]

    conn.close()

    return {
        "total_sessions":        total,
        "successful":            successful,
        "failed":                total - successful,
        "success_rate":          round(successful / total * 100, 1),
        "avg_rounds":            round(avg_rounds, 2),
        "avg_cost_efficiency":   round(avg_cost_eff, 2),
        "resource_satisfaction": round(full_resource_count / total * 100, 1),
    }
