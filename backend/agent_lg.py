"""
agent_lg.py
LangGraph rebuild of the ShopFlow support agent.

Key differences from the hand-rolled version (agent.py):
  - State is a typed TypedDict, not scattered variables
  - Flow is an explicit graph (nodes + edges), not a while loop
  - Checkpointing is automatic after every node (not just every turn)
  - Confirmation gating is a proper graph state, not a string-match heuristic
  - Every node is independently testable
  - State contains ONLY serializable data (no live objects) so
    checkpointing actually works

Run interactively with: python backend/agent_lg.py
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import google.generativeai as genai
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.rag import search_policy
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

MAX_STEPS = 8


# ── State definition ──────────────────────────────────────────────────────────
# Every field here must be plain, serializable data -- no live objects
# (like ChatSession) -- or LangGraph's checkpointer will crash trying
# to save it to disk/memory after each node.

class AgentState(TypedDict):
    session_id: str
    messages: list
    tool_calls_log: list
    last_user_message: str
    last_reply: str
    awaiting_confirmation: bool
    escalated: bool
    steps: int
    pending_tool_name: str
    pending_tool_input: dict


# ── Tool setup ────────────────────────────────────────────────────────────────

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
6. For general policy questions not tied to a specific order, use search_policy.
"""

GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "get_order_status",
                "description": "Look up order status by order ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "e.g. ORD-5001"},
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "check_return_eligibility",
                "description": "Check if order is eligible for return under 30-day policy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "initiate_refund",
                "description": "Initiate refund. Only after explicit confirmation. Escalate if >= $100.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["order_id", "reason"],
                },
            },
            {
                "name": "cancel_order",
                "description": "Cancel unshipped order. Only after explicit confirmation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                    },
                    "required": ["order_id"],
                },
            },
            {
                "name": "escalate_to_human",
                "description": "Escalate: customer asks, refund >= $100, frustrated, or uncertain.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "summary": {"type": "string"},
                        "order_id": {"type": "string"},
                        "customer_id": {"type": "string"},
                    },
                    "required": ["reason", "summary"],
                },
            },
            {
                "name": "search_policy",
                "description": "Search policy docs for general questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        ]
    }
]


def run_tool(tool_name, tool_input, awaiting_confirmation):
    if tool_name in CONFIRMATION_REQUIRED_TOOLS and not awaiting_confirmation:
        return {"error": "BLOCKED: Requires explicit confirmation first."}
    if tool_name == "initiate_refund":
        preview = get_order_status(tool_input.get("order_id", ""))
        if "price_usd" in preview and preview["price_usd"] >= 100:
            return {"error": "BLOCKED: Refund >= $100. Escalate instead."}
    func = TOOL_FUNCTIONS.get(tool_name)
    if not func:
        return {"error": f"Unknown tool: {tool_name}"}
    return func(**tool_input)


def _build_chat():
    """Fresh model + no history -- used as a base for rebuilding context."""
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=GEMINI_TOOLS,
        system_instruction=SYSTEM_PROMPT,
    )
    return model


def _rebuild_history(messages: list) -> list:
    """Converts our plain message list into Gemini's expected format."""
    gemini_history = []
    for msg in messages[:-1]:  # exclude the current (last) message
        if msg.get("role") in ("user", "model"):
            gemini_history.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })
    return gemini_history


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_frustration_check(state: AgentState) -> dict:
    """Pre-check: detect frustration before spending an LLM call."""
    check = should_escalate(state["last_user_message"])
    if check["escalate"] and check["reason"] == "customer_frustration":
        result = escalate_to_human(
            reason=check["detail"],
            summary=f"Customer frustrated. Message: '{state['last_user_message']}'. "
                    f"Session: {state['session_id']}.",
            customer_id=None,
            order_id=None,
        )
        reply = result.get("message", "I've escalated this to a human agent.")
        print(f"   ⚠️  [frustration detected -- auto-escalating]")
        print(f"\n🤖 Agent: {reply}")
        return {
            "last_reply": reply,
            "escalated": True,
            "tool_calls_log": [],
        }
    return {"escalated": False}


def node_call_model(state: AgentState) -> dict:
    """
    Calls Gemini fresh (no stored chat object -- rebuilt from plain
    message history each time so state stays serializable).
    """
    model = _build_chat()
    chat = model.start_chat(history=_rebuild_history(state["messages"]))
    response = chat.send_message(state["last_user_message"])

    function_call = None
    for part in response.parts:
        if hasattr(part, "function_call") and part.function_call and part.function_call.name:
            function_call = part.function_call
            break

    if function_call:
        return {
            "pending_tool_name": function_call.name,
            "pending_tool_input": dict(function_call.args),
            "last_reply": "",
            "steps": state["steps"] + 1,
        }
    else:
        return {
            "last_reply": response.text,
            "pending_tool_name": "",
            "pending_tool_input": {},
            "steps": state["steps"] + 1,
        }


def node_run_tools(state: AgentState) -> dict:
    """
    Executes the pending tool call. Rebuilds chat context fresh
    from plain history + tool_calls_log so far this turn (no live
    objects stored in state).
    """
    tool_name = state.get("pending_tool_name", "")
    tool_input = state.get("pending_tool_input", {})

    print(f"   🔧 [calling {tool_name} with {tool_input}]")
    result = run_tool(tool_name, tool_input, state["awaiting_confirmation"])
    print(f"   ✅ [result: {result}]")

    log = state["tool_calls_log"] + [{"tool": tool_name, "input": tool_input, "output": result}]

    # Rebuild chat, replay the user message, then replay every tool
    # call+response made so far this turn, in order, to reach the
    # same conversational point the live chat object would have been at.
    model = _build_chat()
    chat = model.start_chat(history=_rebuild_history(state["messages"]))
    chat.send_message(state["last_user_message"])

    for prior_call in log[:-1]:
        chat.send_message(
            genai.protos.Content(parts=[genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=prior_call["tool"], response={"result": prior_call["output"]}
                )
            )])
        )

    response = chat.send_message(
        genai.protos.Content(parts=[genai.protos.Part(
            function_response=genai.protos.FunctionResponse(
                name=tool_name, response={"result": result}
            )
        )])
    )

    function_call = None
    for part in response.parts:
        if hasattr(part, "function_call") and part.function_call and part.function_call.name:
            function_call = part.function_call
            break

    if function_call:
        return {
            "tool_calls_log": log,
            "pending_tool_name": function_call.name,
            "pending_tool_input": dict(function_call.args),
        }
    else:
        return {
            "tool_calls_log": log,
            "last_reply": response.text,
            "pending_tool_name": "",
            "pending_tool_input": {},
        }


def node_post_check(state: AgentState) -> dict:
    """Post-check: detect agent uncertainty after reply."""
    check = should_escalate(state["last_user_message"], state["last_reply"])
    if check["escalate"] and check["reason"] == "agent_uncertainty":
        result = escalate_to_human(
            reason=check["detail"],
            summary=f"Agent uncertain. Reply: '{state['last_reply'][:200]}'. "
                    f"Session: {state['session_id']}.",
            customer_id=None,
            order_id=None,
        )
        esc_msg = result.get("message", "")
        print(f"   ⚠️  [agent uncertainty -- escalating]")
        return {
            "last_reply": state["last_reply"] + "\n\n" + esc_msg,
            "escalated": True,
        }

    reply_lower = state["last_reply"].lower()
    awaiting = "confirm" in reply_lower or "shall i proceed" in reply_lower

    print(f"\n🤖 Agent: {state['last_reply']}")
    return {"awaiting_confirmation": awaiting}


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_after_frustration_check(state: AgentState) -> str:
    if state.get("escalated"):
        return "done"
    return "call_model"


def route_after_model(state: AgentState) -> str:
    if state.get("steps", 0) >= MAX_STEPS:
        return "post_check"
    if state.get("pending_tool_name"):
        return "run_tools"
    return "post_check"


def route_after_tools(state: AgentState) -> str:
    if state.get("steps", 0) >= MAX_STEPS:
        return "post_check"
    if state.get("pending_tool_name"):
        return "run_tools"
    return "post_check"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("frustration_check", node_frustration_check)
    builder.add_node("call_model", node_call_model)
    builder.add_node("run_tools", node_run_tools)
    builder.add_node("post_check", node_post_check)

    builder.add_edge(START, "frustration_check")

    builder.add_conditional_edges(
        "frustration_check",
        route_after_frustration_check,
        {"done": END, "call_model": "call_model"},
    )
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {"run_tools": "run_tools", "post_check": "post_check"},
    )
    builder.add_conditional_edges(
        "run_tools",
        route_after_tools,
        {"run_tools": "run_tools", "post_check": "post_check"},
    )

    builder.add_edge("post_check", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ── Interactive runner ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ShopFlow Agent — LangGraph version")
    print("=" * 60)

    graph = build_graph()
    session_id = str(uuid.uuid4())[:8]
    print(f"🆕 Session: {session_id}")

    messages = []
    awaiting_confirmation = False

    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = input("\n🧑 You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})

        initial_state: AgentState = {
            "session_id": session_id,
            "messages": messages,
            "tool_calls_log": [],
            "last_user_message": user_input,
            "last_reply": "",
            "awaiting_confirmation": awaiting_confirmation,
            "escalated": False,
            "steps": 0,
            "pending_tool_name": "",
            "pending_tool_input": {},
        }

        result = graph.invoke(initial_state, config=config)

        reply = result.get("last_reply", "")
        tool_calls_log = result.get("tool_calls_log", [])
        escalated = result.get("escalated", False)

        messages.append({"role": "model", "content": reply})
        awaiting_confirmation = result.get("awaiting_confirmation", False)

        log_turn(
            session_id=session_id,
            user_message=user_input,
            final_reply=reply,
            tool_calls=tool_calls_log,
            escalated=escalated,
        )


if __name__ == "__main__":
    main()