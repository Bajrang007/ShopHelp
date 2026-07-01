"""
eval_suite.py
Automated test suite for the ShopFlow agent.

Two tiers:
  1. Deterministic tests (no LLM calls) -- test tools, guardrails,
     escalation detection directly. Fast, free, run anytime.
  2. Live tests (LLM calls) -- test full conversations end-to-end.
     Costs quota, run sparingly.

Run with: python backend/eval_suite.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.tools import get_order_status, check_return_eligibility, cancel_order
from backend.escalation import should_escalate
from backend.agent_lg import run_tool

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"{status}  {name}" + (f"  — {detail}" if detail and not condition else ""))


def run_deterministic_tests():
    print("\n=== Tier 1: Deterministic tests (no LLM calls) ===\n")

    # Tool correctness
    r = get_order_status("ORD-5001")
    check("get_order_status returns correct item", r.get("item") == "Wireless Headphones", str(r))

    r = get_order_status("ORD-9999")
    check("get_order_status handles missing order", "error" in r, str(r))

    r = check_return_eligibility("ORD-5002")  # shipped, not delivered
    check("eligibility rejects non-delivered order", r.get("eligible") is False, str(r))

    # Guardrail: confirmation required
    r = run_tool("cancel_order", {"order_id": "ORD-5003"}, awaiting_confirmation=False)
    check("cancel_order blocked without confirmation", "BLOCKED" in str(r.get("error", "")), str(r))

    # Guardrail: $100 refund threshold
    r = run_tool("initiate_refund", {"order_id": "ORD-5006", "reason": "test"}, awaiting_confirmation=True)
    check("refund >= $100 is blocked", "BLOCKED" in str(r.get("error", "")), str(r))

    # Escalation: frustration detection
    r = should_escalate("This is ridiculous, fix it now!")
    check("frustration triggers escalation", r["escalate"] and r["reason"] == "customer_frustration", str(r))

    r = should_escalate("What's your return policy?")
    check("normal question does not trigger escalation", not r["escalate"], str(r))

    # Escalation: agent uncertainty detection
    r = should_escalate("Can I return this?", "I'm not sure about that.")
    check("agent uncertainty triggers escalation", r["escalate"] and r["reason"] == "agent_uncertainty", str(r))


def run_live_tests():
    """
    Live tests -- makes real LLM calls. Costs quota.
    Only run this explicitly, not as part of routine testing.
    """
    print("\n=== Tier 2: Live conversation tests (uses LLM quota) ===\n")
    from backend.agent_lg import build_graph
    import uuid

    graph = build_graph()

    scenarios = [
        {
            "name": "order status lookup",
            "turns": ["What's the status of order ORD-5001?"],
            "expect_in_reply": ["delivered", "5001"],
        },
    ]

    for scenario in scenarios:
        session_id = str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": session_id}}
        messages = []
        final_reply = ""

        for turn in scenario["turns"]:
            messages.append({"role": "user", "content": turn})
            state = {
                "session_id": session_id,
                "messages": messages,
                "tool_calls_log": [],
                "last_user_message": turn,
                "last_reply": "",
                "awaiting_confirmation": False,
                "escalated": False,
                "steps": 0,
                "pending_tool_name": "",
                "pending_tool_input": {},
            }
            result = graph.invoke(state, config=config)
            final_reply = result.get("last_reply", "")
            messages.append({"role": "model", "content": final_reply})

        passed = all(kw.lower() in final_reply.lower() for kw in scenario["expect_in_reply"])
        check(f"live: {scenario['name']}", passed, final_reply)


def print_summary():
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed")
    print(f"{'=' * 50}")
    if passed < total:
        print("\nFailed tests:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")


if __name__ == "__main__":
    run_deterministic_tests()
    print_summary()

    # Uncomment to also run live LLM tests (uses quota):
    # run_live_tests()
    # print_summary()