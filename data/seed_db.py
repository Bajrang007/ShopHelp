"""
seed_db.py
Creates and seeds the PostgreSQL database with realistic mock
e-commerce data: customers, orders, and a tickets table.

Run once with: python data/seed_db.py
Re-run any time to wipe and reset to a clean state.
"""

import psycopg2
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── connection ────────────────────────────────────────────────────────────────
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)
cur = conn.cursor()

# ── drop & recreate tables ────────────────────────────────────────────────────
cur.execute("DROP TABLE IF EXISTS tickets CASCADE")
cur.execute("DROP TABLE IF EXISTS orders CASCADE")
cur.execute("DROP TABLE IF EXISTS customers CASCADE")

cur.execute("""
CREATE TABLE customers (
    customer_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    created_at    DATE NOT NULL
)
""")

cur.execute("""
CREATE TABLE orders (
    order_id      TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
    item_name     TEXT NOT NULL,
    price_usd     NUMERIC(10,2) NOT NULL,
    status        TEXT NOT NULL,
    order_date    DATE NOT NULL,
    ship_date     DATE,
    delivery_date DATE,
    tracking_no   TEXT
)
""")

cur.execute("""
CREATE TABLE tickets (
    ticket_id     SERIAL PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
    order_id      TEXT REFERENCES orders(order_id),
    reason        TEXT NOT NULL,
    summary       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""")

# ── seed data ─────────────────────────────────────────────────────────────────
today = datetime.now()

def d(days_ago):
    return (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

customers = [
    ("CUST-1001", "Asha Mehta",   "asha.mehta@example.com",   "2024-03-12"),
    ("CUST-1002", "Daniel Cho",   "daniel.cho@example.com",   "2024-06-01"),
    ("CUST-1003", "Priya Nair",   "priya.nair@example.com",   "2025-01-20"),
]

orders = [
    # id,        customer,      item,                      price,  status,      ordered,  shipped,  delivered, tracking
    ("ORD-5001", "CUST-1001", "Wireless Headphones",       79.99, "delivered", d(10),    d(8),     d(5),      "TRK998811"),
    ("ORD-5002", "CUST-1001", "Running Shoes - Size 9",    64.50, "shipped",   d(3),     d(2),     None,      "TRK998822"),
    ("ORD-5003", "CUST-1002", "Smart Watch",              149.00, "placed",    d(1),     None,     None,      None),
    ("ORD-5004", "CUST-1002", "Yoga Mat",                  24.99, "delivered", d(20),    d(18),    d(15),     "TRK998833"),
    ("ORD-5005", "CUST-1003", "Bluetooth Speaker",         45.00, "cancelled", d(7),     None,     None,      None),
    ("ORD-5006", "CUST-1003", "Office Chair",             189.99, "delivered", d(40),    d(38),    d(33),     "TRK998844"),
]

cur.executemany(
    "INSERT INTO customers VALUES (%s, %s, %s, %s)", customers
)
cur.executemany(
    "INSERT INTO orders VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", orders
)

conn.commit()
cur.close()
conn.close()

print("✅ Database seeded successfully")
print(f"   customers : {len(customers)}")
print(f"   orders    : {len(orders)}")
print(f"   tickets   : 0 (created by the agent at runtime)")