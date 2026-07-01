"""
memory.py
Session persistence: save and load conversation history to/from Postgres.
This is the hand-rolled version of what LangGraph's checkpointer does
automatically in Phase 9.

A session is identified by a session_id (e.g. "session_abc123").
History is stored as JSONB -- a list of {role, content} dicts.
"""

import json
import psycopg2
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    """
    Uses Neon (hosted) if NEON_DATABASE_URL is set, otherwise
    falls back to local Docker Postgres.
    """
    neon_url = os.getenv("NEON_DATABASE_URL")
    if neon_url:
        return psycopg2.connect(neon_url)
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def save_session(session_id: str, history: list, customer_id: str = None):
    """
    Upsert conversation history for a session.
    history is a list of serializable dicts -- we strip any non-serializable
    Gemini proto objects before saving (those are rebuilt on load anyway).
    """
    # Gemini response objects aren't JSON-serializable -- we only save
    # plain user messages and text assistant responses, not tool call protos.
    serializable = []
    for msg in history:
        if isinstance(msg, dict):
            serializable.append(msg)
        # non-dict entries (Gemini proto Content objects) are skipped --
        # they get re-added naturally when the session resumes

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversations (session_id, customer_id, history, updated_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (session_id) DO UPDATE
            SET history    = EXCLUDED.history,
                customer_id = EXCLUDED.customer_id,
                updated_at  = EXCLUDED.updated_at
    """, (
        session_id,
        customer_id,
        json.dumps(serializable),
        datetime.now(timezone.utc),
    ))
    conn.commit()
    cur.close()
    conn.close()


def load_session(session_id: str) -> list:
    """
    Load conversation history for a session.
    Returns an empty list if the session doesn't exist yet.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT history FROM conversations WHERE session_id = %s",
        (session_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return []

    return row[0] if isinstance(row[0], list) else json.loads(row[0])


def list_sessions() -> list:
    """
    List all sessions -- useful for debugging and the eval suite later.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT session_id, customer_id, updated_at,
               jsonb_array_length(history) as turns
        FROM conversations
        ORDER BY updated_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "session_id":  r[0],
            "customer_id": r[1],
            "updated_at":  str(r[2]),
            "turns":       r[3],
        }
        for r in rows
    ]


if __name__ == "__main__":
    # Quick test -- save a fake session, reload it, confirm match
    TEST_ID = "test_session_001"
    fake_history = [
        {"role": "user", "content": "Where is my order?"},
        {"role": "assistant", "content": "I can help with that -- what's your order ID?"},
    ]

    save_session(TEST_ID, fake_history, customer_id="CUST-1001")
    print("Saved session.")

    loaded = load_session(TEST_ID)
    print(f"Loaded {len(loaded)} messages:")
    for msg in loaded:
        print(f"  [{msg['role']}] {msg['content']}")

    sessions = list_sessions()
    print(f"\nAll sessions in DB: {len(sessions)}")
    for s in sessions:
        print(f"  {s}")