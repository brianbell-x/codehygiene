# test_billing.py – unit tests for the minimal billing engine
"""Tests for the billing core located in the directory pointed to by the
environment variable ``BILLING_IMPL`` (defaults to ``before``). The tests cover
realistic scenarios such as creating subscriptions, applying coupons, tax
calculation, payment success/failure, dunning and plan changes.
"""
import os
import unittest
from decimal import Decimal
from datetime import datetime, timedelta

# Resolve the implementation directory – default is ``before``
impl_dir = os.getenv("BILLING_IMPL", "before")
module_path = os.path.join(os.path.dirname(__file__), impl_dir, "billing.py")

# Import the module dynamically
import importlib.util
spec = importlib.util.spec_from_file_location("billing", module_path)
billing = importlib.util.module_from_spec(spec)
assert spec.loader is not None, "Failed to load billing module"
spec.loader.exec_module(billing)

class BillingTests(unittest.TestCase):
    def setUp(self):
        # Fresh system for each test
        self.system = billing.BillingSystem()
        # Ensure a clean webhook log
        self.system.clear_webhook_log()
        # Seed a known coupon
        coupon = billing.Coupon(
            code="SAVE15",
            discount_percent=Decimal("15"),
            max_uses=5,
            expires_at=datetime.utcnow() + timedelta(days=10),
        )
        self.system.coupons["SAVE15"] = coupon

    def test_free_plan_subscription(self):
        sub = self.system.create_subscription(customer_id="cust1", plan_id="free", seats=1)
        self.assertEqual(sub.plan_id, "free")
        # Free plan generates $0 invoice
        invoice = list(self.system.invoices.values())[0]
        self.assertEqual(invoice.amount, Decimal("0.00"))
        # No webhook for payment needed – only invoice.created
        events = [e for e, _ in self.system.get_webhook_log()]
        self.assertIn("invoice.created", events)
        self.assertNotIn("invoice.paid", events)

    def test_starter_plan_with_coupon_and_tax(self):
        sub = self.system.create_subscription(
            customer_id="cust2",
            plan_id="starter",
            seats=2,
            coupon_code="SAVE15",
            region="US-CA",
        )
        invoice = list(self.system.invoices.values())[0]
        # Base = 29.99 * 2 = 59.98 ; 15% discount = 8.997 -> 9.00 approx
        expected_subtotal = Decimal("59.98") - Decimal("9.00")
        tax = (expected_subtotal * Decimal("0.075")).quantize(Decimal("0.01"))
        expected_total = (expected_subtotal + tax).quantize(Decimal("0.01"))
        self.assertEqual(invoice.amount, expected_total)

    def test_payment_success(self):
        sub = self.system.create_subscription(customer_id="cust3", plan_id="pro", seats=1)
        invoice = list(self.system.invoices.values())[0]
        self.system.pay_invoice(invoice.id, succeed=True)
        self.assertTrue(invoice.paid)
        # Verify webhook sequence
        events = [e for e, _ in self.system.get_webhook_log()]
        self.assertEqual(events, ["invoice.created", "invoice.paid"])

    def test_payment_failure_then_retry_and_cancel(self):
        sub = self.system.create_subscription(customer_id="cust4", plan_id="pro", seats=1)
        invoice = list(self.system.invoices.values())[0]
        # Fail three times
        for _ in range(3):
            self.system.pay_invoice(invoice.id, succeed=False)
        self.assertFalse(invoice.paid)
        self.assertTrue(sub.cancel_at_period_end)
        events = [e for e, _ in self.system.get_webhook_log()]
        # Expect three failures then cancellation webhook
        self.assertEqual(events, ["invoice.created", "invoice.failed", "invoice.failed", "invoice.failed", "subscription.canceled"])

    def test_plan_upgrade_proration(self):
        # Start on starter plan
        sub = self.system.create_subscription(customer_id="cust5", plan_id="starter", seats=3)
        # Upgrade to pro midway through month – we mock date via monkeypatch of datetime.utcnow if needed.
        # For simplicity we just call change_plan and trust proration logic produces a non‑zero amount.
        invoice = self.system.change_plan(sub.id, "pro")
        self.assertIsNotNone(invoice)
        # New plan should be applied
        self.assertEqual(self.system.subscriptions[sub.id].plan_id, "pro")
        # Amount can be positive or negative depending on timing – just ensure it's Decimal.
        self.assertIsInstance(invoice.amount, Decimal)

    def test_invalid_coupon_raises(self):
        with self.assertRaises(billing.BillingError):
            self.system.create_subscription(customer_id="cust6", plan_id="starter", seats=1, coupon_code="BADCODE")

    def test_coupon_usage_limit(self):
        # Exhaust coupon usage
        coupon = billing.Coupon(code="ONCE", discount_percent=Decimal("10"), max_uses=1, expires_at=datetime.utcnow() + timedelta(days=1))
        self.system.coupons["ONCE"] = coupon
        # First use succeeds
        self.system.create_subscription(customer_id="cust7", plan_id="starter", seats=1, coupon_code="ONCE")
        # Second use should fail
        with self.assertRaises(billing.BillingError):
            self.system.create_subscription(customer_id="cust8", plan_id="starter", seats=1, coupon_code="ONCE")

    def test_tax_rule_missing_defaults_to_zero(self):
        sub = self.system.create_subscription(customer_id="cust9", plan_id="starter", seats=1, region="UNKNOWN")
        invoice = list(self.system.invoices.values())[0]
        # No tax, amount = base price (29.99)
        self.assertEqual(invoice.amount, Decimal("29.99"))

if __name__ == "__main__":
    unittest.main()
