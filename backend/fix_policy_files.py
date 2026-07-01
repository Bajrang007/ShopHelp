"""
One-time fix: rewrites the policy markdown files with clean Unix line
endings and no escaped characters.
"""

import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

FILES = {
    "policy_returns.md": """# ShopFlow Returns & Refunds Policy

## Return window
Most items can be returned within 30 days of the delivery date for a full refund.
Items must be unused, in original packaging, with tags attached.

## Non-returnable items
The following cannot be returned under any circumstances:
- Underwear and swimwear (hygiene items)
- Gift cards
- Items marked "Final Sale" at time of purchase

## Refund timeline
Once a return is received at our warehouse, refunds are processed within 5-7
business days back to the original payment method. Refunds are never issued
in cash or store credit unless the customer explicitly requests store credit.

## Refund approval thresholds
- Refunds under $100: can be auto-approved by support if return eligibility is confirmed.
- Refunds of $100 or more: require explicit customer confirmation before processing.

## Damaged or defective items
If an item arrives damaged or defective, customers can request a replacement
or refund within 90 days of delivery. Return shipping is free.

## Order not yet shipped
If an order has not yet shipped, customers can cancel it directly for a full
refund with no return process needed.
""",

    "policy_shipping.md": """# ShopFlow Shipping Policy

## Standard shipping
- Delivery in 5-7 business days.
- Free on orders over $50; otherwise $4.99.

## Express shipping
- Delivery in 1-2 business days.
- Flat rate of $14.99, available at checkout.

## International shipping
ShopFlow ships to the United States, Canada, and the United Kingdom.
International orders take 10-14 business days. Customs fees may apply.

## Order tracking
Tracking numbers are emailed automatically once an order ships, and are
visible in order history. Tracking can take up to 24 hours to update.

## Lost packages
If tracking shows delivered but the customer hasn't received the package:
- Ask customer to check with neighbors and building front desk first.
- Wait 48 hours before filing a lost package claim.
- If tracking hasn't updated for more than 7 business days past the expected
  delivery window, treat as lost and escalate to a human agent.

## Address changes
Shipping addresses can only be changed before an order has shipped. Once
shipped, the customer must redirect with the carrier directly.
""",

    "policy_general_faq.md": """# ShopFlow General FAQ

## Payment methods
We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.
We do not accept cash on delivery or bank transfers.

## Order cancellation
Orders can be cancelled free of charge as long as they have not yet shipped.
Once shipped, the customer must wait for delivery and initiate a return.

## Changing items in an order
We cannot add or swap items in a placed order. The customer must cancel
(if not yet shipped) and place a new order, or return after delivery.

## Promo codes
Promo codes must be applied at checkout and cannot be applied retroactively.

## Account and password issues
Direct customers to use the "Forgot password" link on the login page.
Support agents cannot reset passwords manually for security reasons.

## Escalation triggers
A conversation must be escalated to a human support agent when:
- The customer explicitly asks to speak to a human
- The requested refund is $100 or more
- The customer has contacted support 2 or more times about the same issue
- The agent cannot find relevant policy to answer confidently
- The customer expresses significant frustration or anger
""",
}

for filename, content in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"✅ Rewrote {filename} ({len(content)} chars, clean \\n line endings)")