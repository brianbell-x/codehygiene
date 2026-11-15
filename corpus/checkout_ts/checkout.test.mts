import { strict as assert } from 'assert';
const impl = process.env.CHECKOUT_IMPL || 'before';
const { processOrder } = await import(`./${impl}/checkout.ts`);
import type { Order } from './before/checkout.ts';

// Helper to clone order and override fields
function clone(base: Partial<Order> = {}): Order {
  const defaults: Order = {
    id: 'order-1',
    items: [{ sku: 'SKU-01', quantity: 1, price: 1000 }],
    shippingAddress: {
      line1: '123 Main St',
      city: 'Anytown',
      state: 'NY',
      zip: '12345',
      country: 'US',
    },
    billingAddress: {
      line1: '123 Main St',
      city: 'Anytown',
      state: 'NY',
      zip: '12345',
      country: 'US',
    },
    payment: { cardNumber: '4111111111111111', expMonth: 12, expYear: 2030, cvv: '123' },
    promos: [],
  };
  return { ...defaults, ...base } as Order;
}

// Scenario 1: simple single‑item order, stock available, domestic express carrier
const order1 = clone();
const result1 = processOrder(order1);
assert.equal(result1.success, true, 'order1 should succeed');
assert.equal(result1.shipments?.length, 1, 'order1 one shipment');
assert.equal(result1.shipments?.[0].carrier, 'EXPRESS');

// Scenario 2: multi‑warehouse allocation (SKU‑01 spread across WH1 and WH2)
const order2 = clone({
  id: 'order-2',
  items: [{ sku: 'SKU-01', quantity: 8, price: 1000 }],
});
const result2 = processOrder(order2);
assert.equal(result2.success, true);
assert.equal(result2.shipments?.length, 2, 'order2 should split across two warehouses');

// Scenario 3: promo percent discount reduces total
const order3 = clone({
  id: 'order-3',
  promos: [{ code: '10OFF', discountPct: 10 }],
});
const result3 = processOrder(order3);
assert.equal(result3.success, true);
const expectedSubtotal = order3.items.reduce((s, i) => s + i.price * i.quantity, 0);
const expectedAfterPromo = Math.round(expectedSubtotal * 0.9);
assert.equal(result3.totalAmount, expectedAfterPromo + (result3.shipments?.[0].cost ?? 0), 'promo applied');

// Scenario 4: insufficient stock triggers failure
const order4 = clone({
  id: 'order-4',
  items: [{ sku: 'SKU-03', quantity: 20, price: 500 }],
});
const result4 = processOrder(order4);
assert.equal(result4.success, false);
assert.equal(result4.error, 'Insufficient stock');

// Scenario 5: fraud detection on high value order
const order5 = clone({
  id: 'order-5',
  items: [{ sku: 'SKU-01', quantity: 100, price: 10_000 }], // $1,000.00 each, total $100,000
});
const result5 = processOrder(order5);
assert.equal(result5.success, false);
assert.equal(result5.error, 'Fraud detected');

// Scenario 6: international shipping selects appropriate carrier (CN -> FREIGHT only)
const order6 = clone({
  id: 'order-6',
  shippingAddress: { line1: '1 Beijing Rd', city: 'Beijing', state: 'BJ', zip: '100000', country: 'CN' },
  billingAddress: { line1: '1 Beijing Rd', city: 'Beijing', state: 'BJ', zip: '100000', country: 'CN' },
});
const result6 = processOrder(order6);
assert.equal(result6.success, true);
assert.equal(result6.shipments?.[0].carrier, 'FREIGHT');

// Scenario 7: mismatched billing/shipping triggers fraud
const order7 = clone({
  id: 'order-7',
  billingAddress: { line1: '999 Other St', city: 'Other', state: 'CA', zip: '99999', country: 'US' },
});
const result7 = processOrder(order7);
assert.equal(result7.success, false);
assert.equal(result7.error, 'Fraud detected');

// Scenario 8: gift‑card (fixed amount) reduces total to zero
const order8 = clone({
  id: 'order-8',
  promos: [{ code: 'FREE', discountAmt: 2000 }], // covers item price 1000 + shipping approx
});
const result8 = processOrder(order8);
assert.equal(result8.success, true);
assert.ok(result8.totalAmount >= 0, 'total non‑negative');

console.log('All checkout tests passed.');
