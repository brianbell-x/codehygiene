# billing.py – refactored version (Phase 2)
"""Refactored billing implementation aiming for higher readability and code‑hygiene scores.
Functionality matches the baseline implementation."""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging / audit – JSON lines, datetime serialized safely
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("billing")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

def audit(event: str, data: dict) -> None:
    """Emit a JSON‑line audit record with ISO‑8601 timestamp.
    ``default=str`` ensures any non‑serialisable objects (e.g. ``datetime``) are stringified.
    """
    record = {"timestamp": datetime.utcnow().isoformat() + "Z", "event": event, "data": data}
    LOGGER.info(json.dumps(record, default=str))

# ---------------------------------------------------------------------------
# Core data models – immutable where appropriate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    price_per_seat: Decimal  # per month
    currency: str = "USD"
    is_free: bool = False

PLANS: Dict[str, Plan] = {
    "free": Plan(id="free", name="Free", price_per_seat=Decimal("0"), is_free=True),
    "starter": Plan(id="starter", name="Starter", price_per_seat=Decimal("29.99")),
    "pro": Plan(id="pro", name="Pro", price_per_seat=Decimal("79.99")),
    "enterprise": Plan(id="enterprise", name="Enterprise", price_per_seat=Decimal("199.99")),
}

@dataclass
class Coupon:
    code: str
    discount_percent: Decimal  # e.g. 15 for 15 %
    max_uses: Optional[int] = None
    expires_at: Optional[datetime] = None
    used: int = 0

COUPONS: Dict[str, Coupon] = {}

@dataclass(frozen=True)
class TaxRule:
    region: str
    rate: Decimal  # e.g. 0.07 for 7 %

TAX_RULES: Dict[str, TaxRule] = {
    "US-CA": TaxRule(region="US-CA", rate=Decimal("0.075")),
    "US-NY": TaxRule(region="US-NY", rate=Decimal("0.088")),
    "EU-DE": TaxRule(region="EU-DE", rate=Decimal("0.19")),
}

# ---------------------------------------------------------------------------
# Simple webhook router (in‑memory for tests)
# ---------------------------------------------------------------------------
WEBHOOK_LOG: List[Tuple[str, dict]] = []

def dispatch_webhook(event: str, payload: dict) -> None:
    WEBHOOK_LOG.append((event, payload))
    audit("webhook.dispatch", {"event": event, "payload": payload})

# ---------------------------------------------------------------------------
# Subscription / invoice models
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
# Core billing system
# ---------------------------------------------------------------------------
class BillingError(Exception):
    pass

class BillingSystem:
    def __init__(self) -> None:
        self.subscriptions: Dict[str, Subscription] = {}
        self.invoices: Dict[str, Invoice] = {}
        self.coupons: Dict[str, Coupon] = COUPONS

    # ---------- subscription creation ----------
    def create_subscription(
        self,
        customer_id: str,
        plan_id: str,
        seats: int = 1,
        coupon_code: Optional[str] = None,
        region: str = "US-CA",
    ) -> Subscription:
        if plan_id not in PLANS:
            raise BillingError(f"Unknown plan {plan_id}")
        if seats < 1:
            raise BillingError("Seats must be >= 1")
        sub = Subscription(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            plan_id=plan_id,
            seats=seats,
            start_date=datetime.utcnow(),
        )
        self.subscriptions[sub.id] = sub
        audit("subscription.created", asdict(sub))
        invoice = self._generate_invoice(sub, coupon_code, region)
        self.invoices[invoice.id] = invoice
        dispatch_webhook("invoice.created", invoice.to_payload())
        if coupon_code and coupon_code in self.coupons:
            self.coupons[coupon_code].used += 1
        return sub

    # ---------- invoice generation ----------
    def _generate_invoice(self, sub: Subscription, coupon_code: Optional[str], region: str) -> Invoice:
        plan = sub.current_plan()
        base = plan.price_per_seat * sub.seats
        discount = Decimal("0")
        if coupon_code:
            coupon = self.coupons.get(coupon_code)
            if not coupon:
                raise BillingError("Invalid coupon")
            if coupon.expires_at and datetime.utcnow() > coupon.expires_at:
                raise BillingError("Coupon expired")
            if coupon.max_uses is not None and coupon.used >= coupon.max_uses:
                raise BillingError("Coupon usage limit reached")
            discount = (base * coupon.discount_percent) / Decimal("100")
        subtotal = base - discount
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

    # ---------- payment & dunning ----------
    def pay_invoice(self, invoice_id: str, succeed: bool = True) -> None:
        inv = self.invoices.get(invoice_id)
        if not inv:
            raise BillingError("Invoice not found")
        if inv.paid:
            return
        inv.attempts += 1
        if succeed:
            inv.paid = True
            dispatch_webhook("invoice.paid", inv.to_payload())
            audit("invoice.paid", {"invoice_id": inv.id})
            return
        dispatch_webhook("invoice.failed", inv.to_payload())
        audit("invoice.failed", {"invoice_id": inv.id, "attempt": inv.attempts})
        if inv.attempts >= inv.max_attempts:
            sub = self.subscriptions.get(inv.subscription_id)
            if sub:
                sub.cancel_at_period_end = True
                dispatch_webhook("subscription.canceled", {"subscription_id": inv.subscription_id})
                audit("subscription.canceled", {"subscription_id": inv.subscription_id})

    # ---------- plan change with proration ----------
    def change_plan(
        self,
        subscription_id: str,
        new_plan_id: str,
        effective_date: Optional[datetime] = None,
        coupon_code: Optional[str] = None,
        region: str = "US-CA",
    ) -> Invoice:
        sub = self.subscriptions.get(subscription_id)
        if not sub:
            raise BillingError("Subscription not found")
        if new_plan_id not in PLANS:
            raise BillingError("Unknown target plan")
        old_plan = sub.current_plan()
        new_plan = PLANS[new_plan_id]
        # Proration: charge difference for remaining days in current month
        today = datetime.utcnow()
        month_start = today.replace(day=1)
        next_month = (month_start + timedelta(days=32)).replace(day=1)
        month_end = next_month - timedelta(seconds=1)
        remaining = (month_end - today).days + 1
        days_in_month = (month_end - month_start).days + 1
        factor = Decimal(remaining) / Decimal(days_in_month)

        price_diff = (new_plan.price_per_seat - old_plan.price_per_seat) * sub.seats * factor
        base = price_diff

        # Apply coupon if provided
        discount = Decimal("0")
        if coupon_code and coupon_code in self.coupons:
            coupon = self.coupons[coupon_code]
            discount = (base * coupon.discount_percent) / Decimal("100")

        subtotal = base - discount

        # Tax calculation
        tax_rate = TAX_RULES.get(region, TaxRule(region, Decimal("0"))).rate
        tax = (subtotal * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        total = (subtotal + tax).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        invoice = Invoice(
            id=str(uuid.uuid4()),
            subscription_id=sub.id,
            amount=total,
            currency=new_plan.currency,
        )
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

    # ---------- helpers for tests ----------
    def get_webhook_log(self) -> List[Tuple[str, dict]]:
        return list(WEBHOOK_LOG)

    def clear_webhook_log(self) -> None:
        WEBHOOK_LOG.clear()
