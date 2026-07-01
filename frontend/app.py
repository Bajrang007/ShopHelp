"""
app.py
Streamlit chat UI for the ShopFlow support agent.
Run with: streamlit run frontend/app.py
"""

import os
import sys
import uuid
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import google.generativeai as genai
from dotenv import load_dotenv
from backend.rag import search_policy
import subprocess

# Build the ChromaDB index on first run if it doesn't exist yet
# (needed for cloud deployment, since chroma_db/ is gitignored)
_chroma_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "chroma_db")
if not os.path.exists(_chroma_path):
    subprocess.run(["python", "backend/build_index.py"], check=True)
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

MAX_STEPS = 8

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
                "description": "Look up the current status of a customer's order by order ID.",
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
                "description": "Check whether an order is eligible for return/refund under the 30-day policy.",
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
                "description": "Initiate a refund for a delivered order. Only call after explicit customer confirmation.",
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
                "description": "Escalate to a human agent.",
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
                "description": "Search ShopFlow policy documents for relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The customer's question."},
                    },
                    "required": ["query"],
                },
            },
        ]
    }
]


def run_tool(tool_name, tool_input, awaiting_confirmation):
    if tool_name in CONFIRMATION_REQUIRED_TOOLS and not awaiting_confirmation:
        return {"error": "BLOCKED: Requires explicit user confirmation first."}
    if tool_name == "initiate_refund":
        preview = get_order_status(tool_input.get("order_id", ""))
        if "price_usd" in preview and preview["price_usd"] >= 100:
            return {"error": "BLOCKED: Refund >= $100. Escalate to human instead."}
    func = TOOL_FUNCTIONS.get(tool_name)
    if not func:
        return {"error": f"Unknown tool: {tool_name}"}
    return func(**tool_input)


def agent_turn(chat, user_message, awaiting_confirmation):
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
        result = run_tool(tool_name, tool_input, awaiting_confirmation)
        tool_calls_log.append({
            "tool": tool_name,
            "input": tool_input,
            "output": result,
        })

        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": result},
                    )
                )]
            )
        )

    return "⚠️ Max steps reached.", tool_calls_log


# ── Streamlit app ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ShopFlow Support",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ ShopFlow Customer Support")
st.caption("Powered by an agentic AI — ask about your orders, returns, refunds, or policies.")

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_trace" not in st.session_state:
    st.session_state.tool_trace = []

if "chat" not in st.session_state:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=GEMINI_TOOLS,
        system_instruction=SYSTEM_PROMPT,
    )
    st.session_state.chat = model.start_chat()

if "last_was_question" not in st.session_state:
    st.session_state.last_was_question = False

# ── Layout: chat + sidebar ────────────────────────────────────────────────────
col_chat, col_trace = st.columns([2, 1])

with col_chat:
    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Frustration check
        pre_check = should_escalate(user_input)
        if pre_check["escalate"] and pre_check["reason"] == "customer_frustration":
            escalation_result = escalate_to_human(
                reason=pre_check["detail"],
                summary=f"Customer frustrated. Message: '{user_input}'. Session: {st.session_state.session_id}.",
                customer_id=None,
                order_id=None,
            )
            reply = escalation_result.get("message", "I've escalated this to a human agent.")
            tool_calls_log = []
            escalated = True
        else:
            reply, tool_calls_log = agent_turn(
                st.session_state.chat,
                user_input,
                st.session_state.last_was_question,
            )
            # Confidence check
            post_check = should_escalate(user_input, reply)
            escalated = False
            if post_check["escalate"] and post_check["reason"] == "agent_uncertainty":
                esc_result = escalate_to_human(
                    reason=post_check["detail"],
                    summary=f"Agent uncertain. Reply: '{reply[:200]}'. Session: {st.session_state.session_id}.",
                    customer_id=None,
                    order_id=None,
                )
                reply = reply + "\n\n" + esc_result.get("message", "")
                escalated = True

        # Show agent reply
        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.tool_trace = tool_calls_log
        st.session_state.last_was_question = (
            "confirm" in reply.lower() or "shall i proceed" in reply.lower()
        )

        # Save session + log trace
        plain_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        save_session(st.session_state.session_id, plain_history)
        log_turn(
            session_id=st.session_state.session_id,
            user_message=user_input,
            final_reply=reply,
            tool_calls=tool_calls_log,
            escalated=escalated,
        )

with col_trace:
    st.subheader("🔧 Tool Call Trace")
    st.caption(f"Session: `{st.session_state.session_id}`")

    if not st.session_state.tool_trace:
        st.info("No tool calls yet this turn.")
    else:
        for i, call in enumerate(st.session_state.tool_trace, 1):
            with st.expander(f"Step {i}: `{call['tool']}`", expanded=True):
                st.write("**Input:**")
                st.json(call["input"])
                st.write("**Output:**")
                st.json(call["output"])