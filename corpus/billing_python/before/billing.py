# billing.py – a minimal realistic SaaS billing core (std lib only)
"""Enterprise‑grade subscription billing implementation.

Features covered:
* Multiple plan tiers (free, starter, pro, enterprise)
* Seat‑based pricing with proration on upgrades/downgrades
* Promo‑code discount handling
* Regional tax rules and currency rounding (2‑decimal)
* Simple webhook event router (payloads collected in an in‑memory log)
* Audit logging (timestamped JSON lines)
* Defensive validation and error handling

The module is deliberately self‑contained – it does **not** depend on any third‑party
packages and is compatible with Python 3.10+.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging / audit
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("billing")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

def audit(event: str, data: dict) -> None:
    """Write a JSON‑line audit record.

    In a real system this would be written to a file or external store.
    For the purpose of the tests we simply log to stdout. ``datetime``
    objects are converted to ISO‑8601 strings to keep the JSON serialisable.
    """
    record = {"timestamp": datetime.utcnow().isoformat() + "Z", "event": event, "data": data}
    # Use ``default=str`` so that any non‑serialisable objects (e.g. datetime) are stringified.
    LOGGER.info(json.dumps(record, default=str))

# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    price_per_seat: Decimal  # price per seat per month
    currency: str = "USD"
    is_free: bool = False

# Pre‑defined plans (could be loaded from DB in production)
PLANS: Dict[str, Plan] = {
    "free": Plan(id="free", name="Free", price_per_seat=Decimal("0"), is_free=True),
    "starter": Plan(id="starter", name="Starter", price_per_seat=Decimal("29.99")),
    "pro": Plan(id="pro", name="Pro", price_per_seat=Decimal("79.99")),
    "enterprise": Plan(id="enterprise", name="Enterprise", price_per_seat=Decimal("199.99")),
}

@dataclass
class Coupon:
    code: str
    discount_percent: Decimal  # e.g. 15 means 15%
    max_uses: Optional[int] = None
    expires_at: Optional[datetime] = None
    used: int = 0

# Simple in‑memory coupon store for demo purposes
COUPONS: Dict[str, Coupon] = {}

@dataclass(frozen=True)
class TaxRule:
    region: str
    rate: Decimal  # e.g. 0.07 for 7%

# Very small static tax table
TAX_RULES: Dict[str, TaxRule] = {
    "US-CA": TaxRule(region="US-CA", rate=Decimal("0.075")),
    "US-NY": TaxRule(region="US-NY", rate=Decimal("0.088")),
    "EU-DE": TaxRule(region="EU-DE", rate=Decimal("0.19")),
}

# ---------------------------------------------------------------------------
# Webhook router (in‑memory for testability)
# ---------------------------------------------------------------------------
WEBHOOK_LOG: List[Tuple[str, dict]] = []  # (event_name, payload)

def dispatch_webhook(event: str, payload: dict) -> None:
    """Record a webhook payload.

    In production this would POST to registered URLs. Here we simply store it
    for verification in unit tests.
    """
    WEBHOOK_LOG.append((event, payload))
    audit("webhook.dispatch", {"event": event, "payload": payload})

# ---------------------------------------------------------------------------
# Subscription and invoicing
# ---------------------------------------------------------------------------
@dataclass
class Subscription:
    id: str
    customer_id: str
    plan_id: str
    seats: int
    start_date: datetime
    end_date: Optional[datetime] = None
    cancel_at_period_end: bool = False

    def current_plan(self) -> Plan:
        return PLANS[self.plan_id]

@dataclass
class Invoice:
    id: str
    subscription_id: str
    amount: Decimal
    currency: str
    paid: bool = False
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = datetime.utcnow()

    def to_payload(self) -> dict:
        return {
            "id": self.id,
            "subscription_id": self.subscription_id,
            "amount": str(self.amount),
            "currency": self.currency,
            "paid": self.paid,
        }

# ---------------------------------------------------------------------------
# Billing engine
# ---------------------------------------------------------------------------
class BillingError(Exception):
    pass

class BillingSystem:
    def __init__(self) -> None:
        # In‑memory stores – in real life these are persisted.
        self.subscriptions: Dict[str, Subscription] = {}
        self.invoices: Dict[str, Invoice] = {}
        self.coupons: Dict[str, Coupon] = COUPONS

    # -------------------------------------------------------------------
    # Subscription lifecycle
    # -------------------------------------------------------------------
    def create_subscription(self, customer_id: str, plan_id: str, seats: int = 1,
                            coupon_code: Optional[str] = None,
                            region: str = "US-CA") -> Subscription:
        """Create a new subscription and its initial invoice.

        If a ``coupon_code`` is supplied and valid, the coupon's ``used`` counter
        is incremented. This mirrors real‑world behaviour where each redemption
        consumes one usage slot.
        """
        if plan_id not in PLANS:
            raise BillingError(f"Unknown plan {plan_id}")
        plan = PLANS[plan_id]
        if seats < 1:
            raise BillingError("Seats must be at least 1")
        sub_id = str(uuid.uuid4())
        sub = Subscription(
            id=sub_id,
            customer_id=customer_id,
            plan_id=plan_id,
            seats=seats,
            start_date=datetime.utcnow(),
        )
        self.subscriptions[sub_id] = sub
        audit("subscription.created", asdict(sub))
        # Generate invoice, applying coupon if provided
        invoice = self._generate_invoice(sub, coupon_code, region)
        self.invoices[invoice.id] = invoice
        dispatch_webhook("invoice.created", invoice.to_payload())
        # Update coupon usage after successful invoice generation
        if coupon_code:
            coupon = self.coupons.get(coupon_code)
            if coupon:
                # Increment usage counter (mutable dataclass)
                coupon.used = (coupon.used or 0) + 1
        return sub
        if plan_id not in PLANS:
            raise BillingError(f"Unknown plan {plan_id}")
        plan = PLANS[plan_id]
        if seats < 1:
            raise BillingError("Seats must be at least 1")
        sub_id = str(uuid.uuid4())
        sub = Subscription(
            id=sub_id,
            customer_id=customer_id,
            plan_id=plan_id,
            seats=seats,
            start_date=datetime.utcnow(),
        )
        self.subscriptions[sub_id] = sub
        audit("subscription.created", asdict(sub))
        invoice = self._generate_invoice(sub, coupon_code, region)
        self.invoices[invoice.id] = invoice
        dispatch_webhook("invoice.created", invoice.to_payload())
        return sub

    # -------------------------------------------------------------------
    # Invoice generation (proration, tax, coupon)
    # -------------------------------------------------------------------
    def _generate_invoice(self, sub: Subscription, coupon_code: Optional[str], region: str) -> Invoice:
        plan = sub.current_plan()
        base = plan.price_per_seat * sub.seats
        # Apply coupon if present
        discount = Decimal("0")
        if coupon_code:
            coupon = self.coupons.get(coupon_code)
            if not coupon:
                raise BillingError("Invalid coupon code")
            if coupon.expires_at and datetime.utcnow() > coupon.expires_at:
                raise BillingError("Coupon expired")
            if coupon.max_uses is not None and coupon.used >= coupon.max_uses:
                raise BillingError("Coupon usage limit reached")
            discount = (base * coupon.discount_percent) / Decimal("100")
        subtotal = base - discount
        # Tax lookup
        tax_rate = TAX_RULES.get(region, TaxRule(region, Decimal("0"))).rate
        tax = (subtotal * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total = (subtotal + tax).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        invoice = Invoice(
            id=str(uuid.uuid4()),
            subscription_id=sub.id,
            amount=total,
            currency=plan.currency,
        )
        audit("invoice.generated", {
            "invoice_id": invoice.id,
            "subscription_id": sub.id,
            "base": str(base),
            "discount": str(discount),
            "tax": str(tax),
            "total": str(total),
        })
        return invoice

    # -------------------------------------------------------------------
    # Payment simulation & dunning
    # -------------------------------------------------------------------
    def pay_invoice(self, invoice_id: str, succeed: bool = True) -> None:
        invoice = self.invoices.get(invoice_id)
        if not invoice:
            raise BillingError("Invoice not found")
        if invoice.paid:
            return
        invoice.attempts += 1
        if succeed:
            invoice.paid = True
            dispatch_webhook("invoice.paid", invoice.to_payload())
            audit("invoice.paid", {"invoice_id": invoice.id})
        else:
            dispatch_webhook("invoice.failed", invoice.to_payload())
            audit("invoice.failed", {"invoice_id": invoice.id, "attempt": invoice.attempts})
            if invoice.attempts >= invoice.max_attempts:
                # Dunning exhausted – cancel subscription
                sub = self.subscriptions.get(invoice.subscription_id)
                if sub:
                    sub.cancel_at_period_end = True
                dispatch_webhook("subscription.canceled", {"subscription_id": invoice.subscription_id})
                audit("subscription.canceled", {"subscription_id": invoice.subscription_id})

    # -------------------------------------------------------------------
    # Upgrade / downgrade with proration (simplified)
    # -------------------------------------------------------------------
    def change_plan(self, subscription_id: str, new_plan_id: str, effective_date: Optional[datetime] = None,
                    coupon_code: Optional[str] = None, region: str = "US-CA") -> Invoice:
        sub = self.subscriptions.get(subscription_id)
        if not sub:
            raise BillingError("Subscription not found")
        if new_plan_id not in PLANS:
            raise BillingError("Unknown target plan")
        old_plan = sub.current_plan()
        new_plan = PLANS[new_plan_id]
        # Simplified proration: charge difference for remaining days in month
        today = datetime.utcnow()
        month_end = (today.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        remaining_days = (month_end - today).days + 1
        days_in_month = (month_end - today.replace(day=1)).days + 1
        factor = Decimal(remaining_days) / Decimal(days_in_month)
        price_diff = (new_plan.price_per_seat - old_plan.price_per_seat) * sub.seats * factor
        # Negative diff means credit; we still generate an invoice (could be negative)
        base = price_diff
        discount = Decimal("0")
        if coupon_code:
            coupon = self.coupons.get(coupon_code)
            if coupon:
                discount = (base * coupon.discount_percent) / Decimal("100")
        subtotal = base - discount
        tax_rate = TAX_RULES.get(region, TaxRule(region, Decimal("0"))).rate
        tax = (subtotal * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total = (subtotal + tax).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        invoice = Invoice(
            id=str(uuid.uuid4()),
            subscription_id=sub.id,
            amount=total,
            currency=new_plan.currency,
        )
        # Apply change
        sub.plan_id = new_plan_id
        self.invoices[invoice.id] = invoice
        audit("plan.changed", {
            "subscription_id": sub.id,
            "old_plan": old_plan.id,
            "new_plan": new_plan.id,
            "proration_amount": str(total),
        })
        dispatch_webhook("invoice.created", invoice.to_payload())
        return invoice

    # -------------------------------------------------------------------
    # Helpers for tests
    # -------------------------------------------------------------------
    def get_webhook_log(self) -> List[Tuple[str, dict]]:
        return list(WEBHOOK_LOG)

    def clear_webhook_log(self) -> None:
        WEBHOOK_LOG.clear()

# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
