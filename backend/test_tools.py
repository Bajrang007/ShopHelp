"""
Quick sanity check — runs each tool once and prints the result.
Run with: python backend/test_tools.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.tools import (
    get_order_status,
    check_return_eligibility,
    initiate_refund,
    cancel_order,
)

print("=== get_order_status ===")
print(get_order_status("ORD-5001"))  # delivered
print()

print("=== check_return_eligibility (delivered, in window) ===")
print(check_return_eligibility("ORD-5001"))
print()

print("=== check_return_eligibility (shipped, not delivered) ===")
print(check_return_eligibility("ORD-5002"))
print()

print("=== cancel_order (status=placed, should succeed) ===")
print(cancel_order("ORD-5003"))
print()

print("=== get_order_status after cancel ===")
print(get_order_status("ORD-5003"))