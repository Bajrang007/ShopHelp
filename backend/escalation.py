"""
escalation.py
Local (no LLM) escalation signal detection.
Two detectors:
  - sentiment_check: is the customer frustrated/angry?
  - confidence_check: is the agent uncertain about its answer?

These run on raw text, zero API calls, zero quota cost.
"""

# ── Sentiment signals ─────────────────────────────────────────────────────────
# Words/phrases that strongly suggest customer frustration or anger.
# Kept deliberately simple -- a production system would use a small
# classifier here, but rule-based is transparent and debuggable.

FRUSTRATION_SIGNALS = [
    # anger
    "this is ridiculous", "this is unacceptable", "so frustrated",
    "i am angry", "i'm angry", "very angry", "extremely angry",
    "i am furious", "i'm furious", "this is outrageous",
    # disappointment
    "worst experience", "terrible service", "awful service",
    "never shopping here again", "never again", "last time",
    "completely useless", "waste of time", "wasted my time",
    # urgency/desperation
    "need this fixed now", "fix this now", "this is urgent",
    "i need a human", "speak to a human", "talk to a human",
    "speak to a person", "talk to a person", "real person",
    "talk to someone", "speak to someone", "get me a manager",
    "i want a manager", "supervisor", "escalate this",
    # profanity signals (mild, keeps this pg)
    "this is bs", "what the hell", "are you kidding",
]

# ── Confidence signals ────────────────────────────────────────────────────────
# Phrases the agent might say when it's uncertain.
# We scan the agent's OWN response for these.

UNCERTAINTY_SIGNALS = [
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "i'm unsure", "i am unsure", "not certain", "i'm uncertain",
    "i cannot confirm", "i can't confirm", "unable to confirm",
    "i cannot verify", "i can't verify", "unable to verify",
    "i don't have information", "i do not have information",
    "i cannot find", "i can't find", "no information available",
    "beyond my ability", "outside my ability", "cannot help with that",
    "can't help with that",
]


def check_frustration(user_message: str) -> dict:
    """
    Scans a user message for frustration/anger signals.
    Returns:
      {"frustrated": True, "trigger": "phrase that matched"}
      {"frustrated": False}
    """
    lowered = user_message.lower()
    for signal in FRUSTRATION_SIGNALS:
        if signal in lowered:
            return {"frustrated": True, "trigger": signal}
    return {"frustrated": False}


def check_agent_confidence(agent_reply: str) -> dict:
    """
    Scans the agent's reply for uncertainty signals.
    Returns:
      {"uncertain": True, "trigger": "phrase that matched"}
      {"uncertain": False}
    """
    lowered = agent_reply.lower()
    for signal in UNCERTAINTY_SIGNALS:
        if signal in lowered:
            return {"uncertain": True, "trigger": signal}
    return {"uncertain": False}


def should_escalate(user_message: str, agent_reply: str = "") -> dict:
    """
    Combined check -- call this after every turn.
    Returns escalation recommendation with reason.
    """
    frustration = check_frustration(user_message)
    if frustration["frustrated"]:
        return {
            "escalate": True,
            "reason": "customer_frustration",
            "detail": f"Frustration signal detected: '{frustration['trigger']}'",
        }

    if agent_reply:
        confidence = check_agent_confidence(agent_reply)
        if confidence["uncertain"]:
            return {
                "escalate": True,
                "reason": "agent_uncertainty",
                "detail": f"Uncertainty signal in agent reply: '{confidence['trigger']}'",
            }

    return {"escalate": False}


if __name__ == "__main__":
    # Quick tests
    tests = [
        ("This is ridiculous, I've been waiting 2 weeks!", ""),
        ("Where is my order?", ""),
        ("Can I return this?", "I'm not sure if that item is eligible."),
        ("What's your return policy?", "You can return items within 30 days."),
        ("I want to speak to a real person", ""),
        ("Hi, what payment methods do you accept?", ""),
    ]

    print("=== Escalation Signal Tests ===\n")
    for user_msg, agent_reply in tests:
        result = should_escalate(user_msg, agent_reply)
        status = "🚨 ESCALATE" if result["escalate"] else "✅ continue"
        reason = f" — {result['detail']}" if result["escalate"] else ""
        print(f"{status}{reason}")
        print(f"  user:  {user_msg[:60]}")
        if agent_reply:
            print(f"  agent: {agent_reply[:60]}")
        print()