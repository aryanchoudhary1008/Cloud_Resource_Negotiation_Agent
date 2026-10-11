"""
database/db.py
--------------
SQLite persistence for negotiation sessions.

Table: negotiations
    id, cpu, ram, storage, budget, final_price, final_cpu, final_ram,
    rounds, success, total_reward, timestamp           (original columns)
    provider, algorithm, tier, quantity, list_price    (added in Phase 1)
    request_id, auto_mode, selected                    (Auto provider selection:
        one Auto request stores one row per provider, sharing a request_id;
        the winner has selected=1)

MIGRATION
    init_db() upgrades an existing database in place with ALTER TABLE, so
    old negotiation_history.db files keep working. Old rows get
    provider='Legacy', algorithm='qlearning', tier='Custom'.
"""

import os
import sqlite3
from config import DB_PATH
from models.negotiation import NegotiationSession

# Columns added after the original schema: name → SQL definition
NEW_COLUMNS = {
    "provider":   "TEXT    NOT NULL DEFAULT 'Legacy'",
    "algorithm":  "TEXT    NOT NULL DEFAULT 'qlearning'",
    "tier":       "TEXT    NOT NULL DEFAULT 'Custom'",
    "quantity":   "INTEGER NOT NULL DEFAULT 1",
    "list_price": "REAL    NOT NULL DEFAULT 0.0",
    "request_id": "TEXT    NOT NULL DEFAULT ''",
    "auto_mode":  "INTEGER NOT NULL DEFAULT 0",
    "selected":   "INTEGER NOT NULL DEFAULT 1",
}


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the table if needed and add any missing Phase-1 columns."""
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

    existing = {row["name"] for row in cursor.execute("PRAGMA table_info(negotiations)")}
    for name, definition in NEW_COLUMNS.items():
        if name not in existing:
            cursor.execute(f"ALTER TABLE negotiations ADD COLUMN {name} {definition}")
            print(f"[DB] Migrated: added column '{name}'")

    conn.commit()
    conn.close()
    print(f"[DB] Initialized: {os.path.abspath(DB_PATH)}")


def save_session(session: NegotiationSession) -> int:
    """Insert a session and return its new id."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO negotiations
            (cpu, ram, storage, budget, final_price, final_cpu, final_ram,
             rounds, success, total_reward, timestamp,
             provider, algorithm, tier, quantity, list_price,
             request_id, auto_mode, selected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session.cpu, session.ram, session.storage, session.budget,
        session.final_price, session.final_cpu, session.final_ram,
        session.rounds, int(session.success), session.total_reward,
        session.timestamp,
        session.provider, session.algorithm, session.tier,
        session.quantity, session.list_price,
        session.request_id, int(session.auto_mode), int(session.selected),
    ))
    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_all_sessions() -> list:
    """All sessions, newest first."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM negotiations ORDER BY id DESC").fetchall()
    conn.close()

    return [
        NegotiationSession(
            id=row["id"], cpu=row["cpu"], ram=row["ram"], storage=row["storage"],
            budget=row["budget"], final_price=row["final_price"],
            final_cpu=row["final_cpu"], final_ram=row["final_ram"],
            rounds=row["rounds"], success=bool(row["success"]),
            total_reward=row["total_reward"], timestamp=row["timestamp"],
            provider=row["provider"], algorithm=row["algorithm"],
            tier=row["tier"], quantity=row["quantity"], list_price=row["list_price"],
            request_id=row["request_id"], auto_mode=bool(row["auto_mode"]),
            selected=bool(row["selected"]),
        )
        for row in rows
    ]


def _breakdown(cursor, column):
    """Success statistics grouped by `column` ('provider' or 'algorithm')."""
    cursor.execute(f"""
        SELECT {column}                                   AS name,
               COUNT(*)                                   AS total,
               SUM(success)                               AS successes,
               AVG(rounds)                                AS avg_rounds,
               SUM(CASE WHEN auto_mode = 1 AND selected = 1 THEN 1 ELSE 0 END) AS auto_wins,
               AVG(CASE WHEN success = 1 AND list_price > 0
                        THEN (list_price - final_price) * 100.0 / list_price END) AS avg_savings
        FROM negotiations
        GROUP BY {column}
        ORDER BY {column}
    """)
    return [
        {
            "name":            r["name"],
            "total":           r["total"],
            "successes":       r["successes"] or 0,
            "success_rate":    round((r["successes"] or 0) / r["total"] * 100, 1),
            "avg_rounds":      round(r["avg_rounds"] or 0.0, 2),
            "avg_savings_pct": round(r["avg_savings"] or 0.0, 1),
            "auto_wins":       r["auto_wins"] or 0,
        }
        for r in cursor.fetchall()
    ]


def get_stats() -> dict:
    """
    Aggregate statistics.

    Original keys: total_sessions, successful, failed, success_rate,
    avg_rounds, avg_cost_efficiency, resource_satisfaction.
    New keys     : by_provider, by_algorithm  (lists of per-group stats).
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM negotiations")
    total = cursor.fetchone()[0]

    if total == 0:
        conn.close()
        return {
            "total_sessions": 0, "successful": 0, "failed": 0,
            "success_rate": 0.0, "avg_rounds": 0.0,
            "avg_cost_efficiency": 0.0, "resource_satisfaction": 0.0,
            "by_provider": [], "by_algorithm": [],
        }

    cursor.execute("SELECT COUNT(*) FROM negotiations WHERE success = 1")
    successful = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(rounds) FROM negotiations")
    avg_rounds = cursor.fetchone()[0] or 0.0

    cursor.execute("SELECT AVG(budget - final_price) FROM negotiations WHERE success = 1")
    avg_cost_eff = cursor.fetchone()[0] or 0.0

    cursor.execute("SELECT COUNT(*) FROM negotiations WHERE final_cpu = cpu AND final_ram = ram")
    full_resource = cursor.fetchone()[0]

    by_provider  = _breakdown(cursor, "provider")
    by_algorithm = _breakdown(cursor, "algorithm")
    conn.close()

    return {
        "total_sessions":        total,
        "successful":            successful,
        "failed":                total - successful,
        "success_rate":          round(successful / total * 100, 1),
        "avg_rounds":            round(avg_rounds, 2),
        "avg_cost_efficiency":   round(avg_cost_eff, 2),
        "resource_satisfaction": round(full_resource / total * 100, 1),
        "by_provider":           by_provider,
        "by_algorithm":          by_algorithm,
    }
