"""
agent.py
The hand-rolled agentic loop, using Google Gemini's free tier.

Mechanics are identical to the Claude version conceptually:
  1. Send conversation + tool definitions to the model
  2. Model responds with either text, or a function_call
  3. We execute the real Python function and send the result back
  4. Loop until the model gives a final text answer

Run interactively with: python backend/agent.py
"""

import os
import sys
import uuid

# This must come before any 'from backend.*' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import google.generativeai as genai
from dotenv import load_dotenv
from backend.rag import search_policy
from backend.memory import save_session, load_session
from backend.tracing import log_turn
from backend.escalation import should_escalate
from backend.tools import (
    get_order_status,
    check_return_eligibility,
    initiate_refund,
    cancel_order,
    escalate_to_human,
)

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MAX_STEPS = 8  # guardrail: stop runaway tool-call loops

TOOL_FUNCTIONS = {
    "get_order_status": get_order_status,
    "check_return_eligibility": check_return_eligibility,
    "initiate_refund": initiate_refund,
    "cancel_order": cancel_order,
    "escalate_to_human": escalate_to_human,
    "search_policy": search_policy,
}

CONFIRMATION_REQUIRED_TOOLS = {"initiate_refund", "cancel_order"}

SYSTEM_PROMPT = """You are a helpful customer support agent for ShopFlow, an e-commerce store.

You can look up orders, check return eligibility, process refunds, cancel orders,
and escalate to a human agent when needed.

Rules you must always follow:
1. Never call initiate_refund or cancel_order without the customer explicitly
   confirming first. Ask "Just to confirm, you'd like me to [action] for
   [order/item] -- shall I proceed?" and wait for their next message before
   calling the tool.
2. If a refund amount is $100 or more, do not process it yourself -- escalate
   to a human agent instead, even if the customer confirms.
3. If the customer asks to speak to a human, or seems frustrated/angry,
   escalate immediately using escalate_to_human with a clear summary.
4. Always check return eligibility before discussing or processing a refund.
5. Be concise, warm, and clear. Don't make up policy details -- if you're
   unsure, say so and offer to escalate.
6. For general policy questions not tied to a specific order (returns policy,
   shipping times, payment methods, etc.), use search_policy to find the
   accurate answer rather than guessing.
"""

GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "get_order_status",
                "description": "Look up the current status of a customer's order by order ID. Use whenever a customer asks where their order is.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order ID, e.g. ORD-5001"},
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "check_return_eligibility",
                "description": "Check whether an order is eligible for return/refund under the 30-day policy. Use before initiating any refund.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order ID to check."},
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "initiate_refund",
                "description": "Initiate a refund for a delivered order. Only call after explicit customer confirmation. For refunds >= $100, escalate instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order ID to refund."},
                        "reason": {"type": "string", "description": "Customer's reason for the return."},
                    },
                    "required": ["order_id", "reason"],
                },
            },
            {
                "name": "cancel_order",
                "description": "Cancel an order that has not yet shipped. Only call after explicit customer confirmation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "The order ID to cancel."},
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "escalate_to_human",
                "description": "Escalate to a human agent: customer asks for one, refund >= $100, customer frustrated, or you can't resolve confidently.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why this is being escalated."},
                        "summary": {"type": "string", "description": "Full conversation summary."},
                        "order_id": {"type": "string", "description": "Related order ID if applicable."},
                        "customer_id": {"type": "string", "description": "Customer ID if known."},
                    },
                    "required": ["reason", "summary"],
                },
            },
            {
                "name": "search_policy",
                "description": "Search ShopFlow's policy documents (returns, shipping, general FAQ) for relevant information. Use this whenever a customer asks a general policy question that isn't about a specific order -- e.g. 'can I return swimwear', 'how long does shipping take', 'what payment methods do you accept'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The customer's question, used to search the policy docs."},
                    },
                    "required": ["query"],
                },
            },
        ]
    }
]


def run_tool(tool_name: str, tool_input: dict, awaiting_confirmation: bool) -> dict:
    """Executes the real tool function, enforcing guardrails."""
    if tool_name in CONFIRMATION_REQUIRED_TOOLS and not awaiting_confirmation:
        return {
            "error": "BLOCKED: This action requires explicit user confirmation "
                     "first. Ask the customer to confirm before calling this tool again."
        }

    if tool_name == "initiate_refund":
        preview = get_order_status(tool_input.get("order_id", ""))
        if "price_usd" in preview and preview["price_usd"] >= 100:
            return {
                "error": "BLOCKED: Refund amount is $100 or more. "
                         "You must escalate this to a human agent instead "
                         "using escalate_to_human."
            }

    func = TOOL_FUNCTIONS.get(tool_name)
    if not func:
        return {"error": f"Unknown tool: {tool_name}"}

    return func(**tool_input)


def agent_turn(model, chat, user_message: str, awaiting_confirmation: bool) -> tuple:
    """
    Runs one full agent turn on an existing Gemini ChatSession.
    Returns (reply_text, tool_calls_log) tuple.
    """
    steps = 0
    tool_calls_log = []
    response = chat.send_message(user_message)

    while steps < MAX_STEPS:
        steps += 1

        function_call = None
        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                function_call = part.function_call
                break

        if not function_call:
            return response.text, tool_calls_log

        tool_name = function_call.name
        tool_input = dict(function_call.args)

        print(f"   🔧 [calling {tool_name} with {tool_input}]")
        result = run_tool(tool_name, tool_input, awaiting_confirmation)
        print(f"   ✅ [result: {result}]")

        # log every tool call for tracing
        tool_calls_log.append({
            "tool": tool_name,
            "input": tool_input,
            "output": result,
        })

        response = chat.send_message(
            genai.protos.Content(
                parts=[
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": result},
                        )
                    )
                ]
            )
        )

    return "⚠️  Max steps reached -- stopping to avoid a runaway loop.", tool_calls_log


def main():
    print("=" * 60)
    print("ShopFlow Support Agent (Gemini, type 'quit' to exit)")
    print("=" * 60)

    session_input = input("\nEnter session ID to resume (or press Enter for new session): ").strip()
    if session_input:
        session_id = session_input
        prior_history = load_session(session_id)
        if prior_history:
            print(f"✅ Resumed session '{session_id}' ({len(prior_history)} prior messages)")
        else:
            print(f"No prior history found for '{session_id}' -- starting fresh")
    else:
        session_id = str(uuid.uuid4())[:8]
        prior_history = []
        print(f"🆕 New session: {session_id}")

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=GEMINI_TOOLS,
        system_instruction=SYSTEM_PROMPT,
    )

    gemini_history = []
    for msg in prior_history:
        if isinstance(msg, dict) and msg.get("role") in ("user", "model"):
            gemini_history.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })

    chat = model.start_chat(history=gemini_history)
    plain_history = list(prior_history)
    last_was_agent_question = False

    while True:
        user_input = input("\n🧑 You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            save_session(session_id, plain_history)
            print(f"💾 Session '{session_id}' saved. Use this ID to resume.")
            break

        # ── Frustration check BEFORE agent turn (free, no LLM call) ───────
        pre_check = should_escalate(user_input)
        if pre_check["escalate"] and pre_check["reason"] == "customer_frustration":
            print(f"\n⚠️  [{pre_check['detail']}]")
            print(f"   🔧 [auto-escalating before agent turn]")
            escalation_result = escalate_to_human(
                reason=pre_check["detail"],
                summary=f"Customer showed frustration. Last message: '{user_input}'. "
                        f"Session: {session_id}. Prior turns: {len(plain_history)}.",
                customer_id=None,
                order_id=None,
            )
            reply = escalation_result.get("message", "I've escalated this to a human agent.")
            print(f"\n🤖 Agent: {reply}")
            plain_history.append({"role": "user", "content": user_input})
            plain_history.append({"role": "model", "content": reply})
            save_session(session_id, plain_history)
            log_turn(
                session_id=session_id,
                user_message=user_input,
                final_reply=reply,
                tool_calls=[],
                escalated=True,
            )
            continue

        plain_history.append({"role": "user", "content": user_input})
        awaiting_confirmation = last_was_agent_question

        reply, tool_calls_log = agent_turn(model, chat, user_input, awaiting_confirmation)
        print(f"\n🤖 Agent: {reply}")

        plain_history.append({"role": "model", "content": reply})
        save_session(session_id, plain_history)

        log_turn(
            session_id=session_id,
            user_message=user_input,
            final_reply=reply,
            tool_calls=tool_calls_log,
            escalated=False,
        )

        # ── Confidence check AFTER agent reply ────────────────────────────
        post_check = should_escalate(user_input, reply)
        if post_check["escalate"] and post_check["reason"] == "agent_uncertainty":
            print(f"\n⚠️  [{post_check['detail']}]")
            print(f"   🔧 [agent uncertain -- suggesting escalation]")
            escalation_result = escalate_to_human(
                reason=post_check["detail"],
                summary=f"Agent expressed uncertainty. Last reply: '{reply[:200]}'. "
                        f"Session: {session_id}.",
                customer_id=None,
                order_id=None,
            )
            esc_reply = escalation_result.get("message", "")
            print(f"\n🤖 Agent (escalation): {esc_reply}")
            plain_history.append({"role": "model", "content": esc_reply})
            save_session(session_id, plain_history)
            log_turn(
                session_id=session_id,
                user_message=user_input,
                final_reply=esc_reply,
                tool_calls=[],
                escalated=True,
            )

        last_was_agent_question = (
            "confirm" in reply.lower() or
            "shall i proceed" in reply.lower()
        )


if __name__ == "__main__":
    main()