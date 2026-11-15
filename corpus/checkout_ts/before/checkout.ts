// checkout.ts – realistic e‑commerce order fulfillment engine (Phase 1 baseline)
// No external dependencies, pure TypeScript, Node built‑in execution.

// ---------- Types ----------
export type SKU = string;
export type WarehouseId = string;
export type CarrierId = string;
export type Currency = string;

export interface OrderItem {
  sku: SKU;
  quantity: number;
  price: number; // unit price in minor units (e.g., cents)
}

export interface Address {
  line1: string;
  line2?: string;
  city: string;
  state: string;
  zip: string;
  country: string;
}

export interface PaymentInfo {
  cardNumber: string; // plain for demo – never store this in prod
  expMonth: number;
  expYear: number;
  cvv: string;
}

export interface PromoCode {
  code: string;
  discountPct?: number; // 0‑100
  discountAmt?: number; // fixed amount in minor units
}

export interface Order {
  // Order identifier
  id: string;
  items: OrderItem[];
  shippingAddress: Address;
  billingAddress: Address;
  payment: PaymentInfo;
  promos?: PromoCode[];
}
  // Order identifier




export interface Allocation {
  warehouse: WarehouseId;
  sku: SKU;
  quantity: number;
}

export interface Shipment {
  carrier: CarrierId;
  allocations: Allocation[];
  cost: number; // shipping cost in minor units
}

export interface ProcessResult {
  orderId: string;
  success: boolean;
  error?: string;
  shipments?: Shipment[];
  totalAmount: number; // items + shipping - discounts, in minor units
}

// ---------- Mock Data (in‑memory) ----------
// Stock per warehouse: { [warehouseId]: { [sku]: quantity } }
const stock: Record<WarehouseId, Record<SKU, number>> = {
  "WH1": { "SKU-01": 5, "SKU-02": 10 },
  "WH2": { "SKU-01": 3, "SKU-03": 7 },
};

// Carrier cost matrix (simple rules)
const carrierCost = (carrier: CarrierId, weightKg: number, domestic: boolean): number => {
  // cost in cents
  const base = carrier === "EXPRESS" ? 1500 : carrier === "FREIGHT" ? 3000 : 800; // base fee
  const perKg = domestic ? 200 : 400;
  return base + perKg * weightKg;
};

// Supported carriers per country
const carriersByCountry: Record<string, CarrierId[]> = {
  "US": ["EXPRESS", "FREIGHT"],
  "CA": ["EXPRESS"],
  "CN": ["FREIGHT"],
  "DE": ["EXPRESS", "FREIGHT"],
};

// Simple fraud rule: block orders over $5000 or with mismatched billing/shipping
const FRAUD_LIMIT = 500_00; // $500.00 in cents (example)

// ---------- Helper Functions ----------
function validateAddress(addr: Address): boolean {
  // Very naive validation – just check required fields are non‑empty
  return Boolean(addr.line1 && addr.city && addr.state && addr.zip && addr.country);
}

function screenFraud(order: Order, totalBeforeDiscount: number): boolean {
  if (totalBeforeDiscount > FRAUD_LIMIT) return false;
  // Simple mismatch check
  const same = JSON.stringify(order.shippingAddress) === JSON.stringify(order.billingAddress);
  return same;
}

function applyPromos(order: Order, subtotal: number): { discounted: number; applied: PromoCode[] } {
  if (!order.promos) return { discounted: subtotal, applied: [] };
  let amount = subtotal;
  const applied: PromoCode[] = [];
  for (const promo of order.promos) {
    if (promo.discountPct) {
      const discount = Math.round(amount * (promo.discountPct / 100));
      amount -= discount;
      applied.push(promo);
    } else if (promo.discountAmt) {
      amount -= promo.discountAmt;
      applied.push(promo);
    }
  }
  if (amount < 0) amount = 0;
  return { discounted: amount, applied };
}

function allocateInventory(order: Order): Allocation[] | null {
  // Greedy multi‑warehouse allocation aiming to minimize number of warehouses used.
  const allocations: Allocation[] = [];
  const needed = new Map<SKU, number>();
  for (const it of order.items) needed.set(it.sku, (needed.get(it.sku) ?? 0) + it.quantity);

  // Sort warehouses by total available stock (descending) to prefer larger ones.
  const localStock: Record<WarehouseId, Record<SKU, number>> = JSON.parse(JSON.stringify(stock));
const warehouses = Object.entries(localStock).sort((a, b) => {
    const totalA = Object.values(a[1]).reduce((s, q) => s + q, 0);
    const totalB = Object.values(b[1]).reduce((s, q) => s + q, 0);
    return totalB - totalA;
  });

  for (const [wh, whStock] of warehouses) {
    for (const [sku, qtyNeeded] of needed) {
      if (qtyNeeded <= 0) continue;
      const available = whStock[sku] ?? 0;
      if (!available) continue;
      const allocate = Math.min(qtyNeeded, available);
      allocations.push({ warehouse: wh, sku, quantity: allocate });
      needed.set(sku, qtyNeeded - allocate);
      // decrement mock stock (mutation for subsequent calls)
      whStock[sku] = available - allocate;
    }
  }

  // If any needed qty left >0, allocation fails.
  const stillNeeded = Array.from(needed.values()).some(q => q > 0);
  return stillNeeded ? null : allocations;
}

function selectCarrier(address: Address, weightKg: number): CarrierId | null {
  const country = address.country.toUpperCase();
  const carriers = carriersByCountry[country] ?? [];
  if (carriers.length === 0) return null;
  // Prefer EXPRESS for domestic, otherwise cheapest based on cost function.
  const domestic = country === "US"; // assume US is home base
  if (domestic && carriers.includes("EXPRESS")) return "EXPRESS";
  // evaluate cost and pick cheapest
  let best: CarrierId | null = null;
  let bestCost = Infinity;
  for (const c of carriers) {
    const cost = carrierCost(c, weightKg, domestic);
    if (cost < bestCost) { bestCost = cost; best = c; }
  }
  return best;
}

// ---------- Main Engine ----------
export function processOrder(order: Order): ProcessResult {
  // 1. basic validation
  if (!validateAddress(order.shippingAddress) || !validateAddress(order.billingAddress)) {
    return { orderId: order.id, success: false, error: "Invalid address", totalAmount: 0 };
  }

  // 2. compute subtotal (price * qty)
  const subtotal = order.items.reduce((sum, it) => sum + it.price * it.quantity, 0);

  // 3. fraud screening before discounts
  if (!screenFraud(order, subtotal)) {
    return { orderId: order.id, success: false, error: "Fraud detected", totalAmount: subtotal };
  }

  // 4. apply promos
  const { discounted: afterPromo, applied: appliedPromos } = applyPromos(order, subtotal);

  // 5. allocate inventory
  const allocations = allocateInventory(order);
  if (!allocations) {
    return { orderId: order.id, success: false, error: "Insufficient stock", totalAmount: afterPromo };
  }

  // 6. split shipments per warehouse (each warehouse gets its own shipment)
  const shipmentsMap: Record<WarehouseId, Allocation[]> = {};
  for (const al of allocations) {
    if (!shipmentsMap[al.warehouse]) shipmentsMap[al.warehouse] = [];
    shipmentsMap[al.warehouse].push(al);
  }

  const shipments: Shipment[] = [];
  for (const [wh, allocs] of Object.entries(shipmentsMap)) {
    // approximate weight: 0.5kg per item unit
    const weightKg = allocs.reduce((w, a) => w + a.quantity * 0.5, 0);
    const carrier = selectCarrier(order.shippingAddress, weightKg);
    if (!carrier) {
      return { orderId: order.id, success: false, error: `No carrier for ${order.shippingAddress.country}`, totalAmount: afterPromo };
    }
    const cost = carrierCost(carrier, weightKg, order.shippingAddress.country.toUpperCase() === "US");
    shipments.push({ carrier, allocations: allocs, cost });
  }

  // 7. compute total amount (items after promo + shipping)
  const shippingTotal = shipments.reduce((s, sh) => s + sh.cost, 0);
  const total = afterPromo + shippingTotal;

  return { orderId: order.id, success: true, shipments, totalAmount: total };
}

// Export a default for convenience when imported via CommonJS/ESM
export default { processOrder };
