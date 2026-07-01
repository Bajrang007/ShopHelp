"""
setup_neon.py
One-time setup: creates all tables (customers, orders, tickets,
conversations) on the hosted Neon Postgres database, and seeds it
with the same demo data as local Docker Postgres.

Run once: python backend/setup_neon.py
"""

import psycopg2
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEON_URL = os.getenv("NEON_DATABASE_URL")

conn = psycopg2.connect(NEON_URL)
cur = conn.cursor()

print("Creating tables on Neon...")

cur.execute("DROP TABLE IF EXISTS tickets CASCADE")
cur.execute("DROP TABLE IF EXISTS conversations CASCADE")
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
    customer_id   TEXT NOT NULL,
    order_id      TEXT REFERENCES orders(order_id),
    reason        TEXT NOT NULL,
    summary       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""")

cur.execute("""
CREATE TABLE conversations (
    session_id    TEXT PRIMARY KEY,
    customer_id   TEXT,
    history       JSONB NOT NULL DEFAULT '[]',
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""")

print("Seeding data...")

today = datetime.now()

def d(days_ago):
    return (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

customers = [
    ("CUST-1001", "Asha Mehta",   "asha.mehta@example.com",   "2024-03-12"),
    ("CUST-1002", "Daniel Cho",   "daniel.cho@example.com",   "2024-06-01"),
    ("CUST-1003", "Priya Nair",   "priya.nair@example.com",   "2025-01-20"),
]

orders = [
    ("ORD-5001", "CUST-1001", "Wireless Headphones",       79.99, "delivered", d(10),    d(8),     d(5),      "TRK998811"),
    ("ORD-5002", "CUST-1001", "Running Shoes - Size 9",    64.50, "shipped",   d(3),     d(2),     None,      "TRK998822"),
    ("ORD-5003", "CUST-1002", "Smart Watch",              149.00, "placed",    d(1),     None,     None,      None),
    ("ORD-5004", "CUST-1002", "Yoga Mat",                  24.99, "delivered", d(20),    d(18),    d(15),     "TRK998833"),
    ("ORD-5005", "CUST-1003", "Bluetooth Speaker",         45.00, "cancelled", d(7),     None,     None,      None),
    ("ORD-5006", "CUST-1003", "Office Chair",             189.99, "delivered", d(40),    d(38),    d(33),     "TRK998844"),
]

cur.executemany("INSERT INTO customers VALUES (%s, %s, %s, %s)", customers)
cur.executemany("INSERT INTO orders VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", orders)

conn.commit()
cur.close()
conn.close()

print("✅ Neon database ready: tables created, demo data seeded")