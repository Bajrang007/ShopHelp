"""
tools.py
The 5 tools our agent can call, plus their JSON schemas.

Two layers here:
  1. The actual Python functions — these hit the database and do real work.
  2. TOOL_SCHEMAS — a list of dicts in Anthropic's tool format that we pass
     to Claude so it knows what tools exist and how to call them.

Claude never imports or calls these functions directly. It produces a
tool_use block saying which tool it wants and with what arguments.
Our agent loop (Phase 3) reads that, calls the matching function here,
and feeds the result back to Claude.
"""

import psycopg2
import os
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

# ── DB connection helper ──────────────────────────────────────────────────────
def get_conn():
    """Return a fresh psycopg2 connection. Called inside each tool so we
    never hold a connection open between agent turns."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


# ── Tool 1: get_order_status ──────────────────────────────────────────────────
def get_order_status(order_id: str) -> dict:
    """
    Look up an order by ID and return its full status.
    Safe read-only operation — no confirmation needed.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT o.order_id, c.name, o.item_name, o.price_usd,
               o.status, o.order_date, o.ship_date,
               o.delivery_date, o.tracking_no
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_id = %s
    """, (order_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return {"error": f"No order found with ID {order_id}"}

    order_id, name, item, price, status, order_date, ship_date, delivery_date, tracking = row

    # Build a human-readable status message alongside the raw data
    status_messages = {
        "placed":    "Your order has been placed and is being prepared.",
        "shipped":   f"Your order has shipped. Tracking number: {tracking}.",
        "delivered": f"Your order was delivered on {delivery_date}.",
        "cancelled": "This order has been cancelled.",
        "returned":  "This order has been returned and a refund is being processed.",
    }

    return {
        "order_id":     order_id,
        "customer":     name,
        "item":         item,
        "price_usd":    float(price),
        "status":       status,
        "order_date":   str(order_date),
        "ship_date":    str(ship_date) if ship_date else None,
        "delivery_date": str(delivery_date) if delivery_date else None,
        "tracking_no":  tracking,
        "message":      status_messages.get(status, "Status unknown."),
    }


# ── Tool 2: check_return_eligibility ─────────────────────────────────────────
def check_return_eligibility(order_id: str) -> dict:
    """
    Check whether an order is eligible for return.
    Rules (from policy_returns.md):
      - Must be 'delivered' status
      - Must be within 30 days of delivery date
      - Cannot already be 'returned' or 'cancelled'
    Safe read-only — no confirmation needed.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT order_id, item_name, price_usd, status, delivery_date
        FROM orders WHERE order_id = %s
    """, (order_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return {"error": f"No order found with ID {order_id}"}

    order_id, item, price, status, delivery_date = row

    if status == "cancelled":
        return {"eligible": False, "reason": "This order was cancelled and cannot be returned."}

    if status == "returned":
        return {"eligible": False, "reason": "This order has already been returned."}

    if status in ("placed", "shipped"):
        return {
            "eligible": False,
            "reason": "This order has not been delivered yet. "
                      "If you want to cancel it, ask about cancellation instead.",
        }

    if status == "delivered" and delivery_date:
        days_since = (date.today() - delivery_date).days
        if days_since > 30:
            return {
                "eligible": False,
                "reason": f"The 30-day return window has passed "
                          f"({days_since} days since delivery on {delivery_date}).",
            }
        return {
            "eligible":      True,
            "order_id":      order_id,
            "item":          item,
            "price_usd":     float(price),
            "delivery_date": str(delivery_date),
            "days_remaining": 30 - days_since,
            "reason": f"Eligible for return. {30 - days_since} days remaining in the return window.",
        }

    return {"eligible": False, "reason": "Unable to determine eligibility."}


# ── Tool 3: initiate_refund ───────────────────────────────────────────────────
def initiate_refund(order_id: str, reason: str) -> dict:
    """
    Mark an order as returned and log a support ticket.
    WRITE OPERATION — agent loop must get explicit user confirmation
    before calling this. Never call without confirmation.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Safety check: only refund delivered orders
    cur.execute("SELECT status, price_usd, item_name FROM orders WHERE order_id = %s", (order_id,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return {"error": f"No order found with ID {order_id}"}

    status, price, item = row

    if status != "delivered":
        conn.close()
        return {"error": f"Cannot refund order with status '{status}'. Only delivered orders can be refunded."}

    # Update order status
    cur.execute(
        "UPDATE orders SET status = 'returned' WHERE order_id = %s",
        (order_id,)
    )

    # Log a ticket for the record
    cur.execute("""
        INSERT INTO tickets (customer_id, order_id, reason, summary, status)
        SELECT customer_id, %s, %s, %s, 'resolved'
        FROM orders WHERE order_id = %s
    """, (order_id, "refund", f"Refund initiated for {item} (${price}). Reason: {reason}", order_id))

    conn.commit()
    cur.close()
    conn.close()

    return {
        "success":    True,
        "order_id":   order_id,
        "item":       item,
        "refund_usd": float(price),
        "message":    f"Refund of ${price} for '{item}' has been initiated. "
                      f"You will receive it within 5-7 business days.",
    }


# ── Tool 4: cancel_order ──────────────────────────────────────────────────────
def cancel_order(order_id: str) -> dict:
    """
    Cancel an order if it has not yet shipped.
    WRITE OPERATION — agent loop must get explicit user confirmation
    before calling this. Never call without confirmation.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT status, item_name, price_usd FROM orders WHERE order_id = %s", (order_id,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return {"error": f"No order found with ID {order_id}"}

    status, item, price = row

    if status != "placed":
        conn.close()
        return {
            "error": f"Cannot cancel order with status '{status}'. "
                     "Only orders that have not yet shipped can be cancelled."
        }

    cur.execute("UPDATE orders SET status = 'cancelled' WHERE order_id = %s", (order_id,))

    cur.execute("""
        INSERT INTO tickets (customer_id, order_id, reason, summary, status)
        SELECT customer_id, %s, %s, %s, 'resolved'
        FROM orders WHERE order_id = %s
    """, (order_id, "cancellation", f"Order cancelled for {item} (${price}).", order_id))

    conn.commit()
    cur.close()
    conn.close()

    return {
        "success":    True,
        "order_id":   order_id,
        "item":       item,
        "refund_usd": float(price),
        "message":    f"Order for '{item}' has been cancelled. "
                      f"A full refund of ${price} will be returned within 5-7 business days.",
    }


# ── Tool 5: escalate_to_human ─────────────────────────────────────────────────
def escalate_to_human(reason: str, summary: str, order_id: str = None, customer_id: str = None) -> dict:
    """
    Create an open support ticket and signal the agent to stop.
    Called when the agent can't resolve the issue, the refund is over $100,
    the customer is frustrated, or the customer asks for a human.
    WRITE OPERATION — logs a ticket, but no confirmation needed since
    escalating to a human is always a safe action.
    """
    conn = get_conn()
    cur = conn.cursor()

    # customer_id is required to create a ticket — look it up from order if needed
    if not customer_id and order_id:
        cur.execute("SELECT customer_id FROM orders WHERE order_id = %s", (order_id,))
        row = cur.fetchone()
        if row:
            customer_id = row[0]

    if not customer_id:
        customer_id = "UNKNOWN"

    cur.execute("""
        INSERT INTO tickets (customer_id, order_id, reason, summary, status)
        VALUES (%s, %s, %s, %s, 'escalated')
        RETURNING ticket_id
    """, (customer_id, order_id, reason, summary))

    ticket_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return {
        "escalated":  True,
        "ticket_id":  ticket_id,
        "message":    "I've created a support ticket and a human agent will follow up with you shortly. "
                      f"Your ticket number is #{ticket_id}.",
    }


# ── Tool schemas (sent to Claude) ─────────────────────────────────────────────
# This is the format Anthropic's API expects. Each entry tells Claude:
#   - the tool's name (must match the function name exactly)
#   - what it does (description — Claude reads this to decide when to use it)
#   - what arguments it needs (input_schema — Claude fills these in)

TOOL_SCHEMAS = [
    {
        "name": "get_order_status",
        "description": (
            "Look up the current status of a customer's order by order ID. "
            "Returns status, tracking number, dates, and item details. "
            "Use this whenever a customer asks where their order is or what its status is."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. ORD-5001",
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "check_return_eligibility",
        "description": (
            "Check whether an order is eligible for return and refund based on "
            "the 30-day return policy. Use this before initiating any refund."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to check return eligibility for.",
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "initiate_refund",
        "description": (
            "Initiate a refund for a delivered order. "
            "IMPORTANT: Only call this after the customer has explicitly confirmed "
            "they want the refund. Never call without confirmation. "
            "For refunds over $100, escalate to a human instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to refund.",
                },
                "reason": {
                    "type": "string",
                    "description": "The customer's reason for the return.",
                },
            },
            "required": ["order_id", "reason"],
        },
    },
    {
        "name": "cancel_order",
        "description": (
            "Cancel an order that has not yet shipped. "
            "IMPORTANT: Only call this after the customer has explicitly confirmed "
            "they want to cancel. Never call without confirmation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to cancel.",
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Escalate the conversation to a human support agent. Use when: "
            "the customer asks for a human, the refund amount is $100 or more, "
            "the customer is frustrated or angry, or you cannot confidently resolve the issue. "
            "Always provide a clear summary of the conversation so far."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this is being escalated.",
                },
                "summary": {
                    "type": "string",
                    "description": "Full summary of the conversation and what was attempted.",
                },
                "order_id": {
                    "type": "string",
                    "description": "Related order ID if applicable.",
                },
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID if known.",
                },
            },
            "required": ["reason", "summary"],
        },
    },
]