"""
tracing.py
Langfuse v4 instrumentation helpers.
"""

import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


def log_turn(
    session_id: str,
    user_message: str,
    final_reply: str,
    tool_calls: list = None,
    escalated: bool = False,
):
    """
    Logs a complete agent turn to Langfuse as a single event.
    """
    langfuse.create_event(
        name="agent-turn",
        input={"message": user_message},
        output={"reply": final_reply},
        metadata={
            "session_id": session_id,
            "escalated": escalated,
            "tool_calls": tool_calls or [],
            "tool_call_count": len(tool_calls or []),
        },
    )
    langfuse.flush()


if __name__ == "__main__":
    print("Sending test event to Langfuse...")
    log_turn(
        session_id="test_session",
        user_message="Is this working?",
        final_reply="Yes it is!",
        tool_calls=[{"tool": "test_tool", "input": {"x": 1}, "output": {"y": 2}}],
        escalated=False,
    )
    print("✅ Done -- check your Langfuse dashboard for an event called 'agent-turn'")