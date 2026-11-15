// checkout.ts – realistic e‑commerce order fulfillment engine (Phase 1 baseline)
// No external dependencies, pure TypeScript, Node built‑in execution.

// Types
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
  id: string; // Order identifier
  items: OrderItem[];
  shippingAddress: Address;
  billingAddress: Address;
  payment: PaymentInfo;
  promos?: PromoCode[];
}

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

// Mock Data (in‑memory)
// Stock per warehouse: { [warehouseId]: { [sku]: quantity } }
const stock: Record<WarehouseId, Record<SKU, number>> = {
  "WH1": { "SKU-01": 5, "SKU-02": 10 },
  "WH2": { "SKU-01": 3, "SKU-03": 7 },
};

function carrierCost(carrier: CarrierId, weightKg: number, domestic: boolean): number {
  // cost in cents
  let base: number;
  if (carrier === "EXPRESS") {
    base = 1500;
  } else if (carrier === "FREIGHT") {
    base = 3000;
  } else {
    base = 800; // other carriers
  }
  const perKg = domestic ? 200 : 400;
  return base + perKg * weightKg;
}

const carriersByCountry: Record<string, CarrierId[]> = {
  "US": ["EXPRESS", "FREIGHT"],
  "CA": ["EXPRESS"],
  "CN": ["FREIGHT"],
  "DE": ["EXPRESS", "FREIGHT"],
};

// Simple fraud rule: block orders over $5000 or with mismatched billing/shipping
const FRAUD_LIMIT = 500_00; // $500.00 in cents (example)

function validateAddress(addr: Address): boolean {
  // Very naive validation – just check required fields are non‑empty
  return Boolean(
    addr.line1 &&
    addr.city &&
    addr.state &&
    addr.zip &&
    addr.country
  );
}

function screenFraud(order: Order, totalBeforeDiscount: number): boolean {
  if (totalBeforeDiscount > FRAUD_LIMIT) return false;
  const same = JSON.stringify(order.shippingAddress) === JSON.stringify(order.billingAddress);
  return same;
}

function applyPromos(order: Order, subtotal: number): {
  discounted: number;
  applied: PromoCode[];
} {
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
  const allocations: Allocation[] = [];
  const needed = new Map<SKU, number>();
  for (const it of order.items) needed.set(it.sku, (needed.get(it.sku) ?? 0) + it.quantity);

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
      whStock[sku] = available - allocate;
    }
  }

  const stillNeeded = Array.from(needed.values()).some(q => q > 0);
  return stillNeeded ? null : allocations;
}

function selectCarrier(address: Address, weightKg: number): CarrierId | null {
  const country = address.country.toUpperCase();
  const carriers = carriersByCountry[country] ?? [];
  if (carriers.length === 0) return null;

  const domestic = country === "US";
  if (domestic && carriers.includes("EXPRESS")) return "EXPRESS";

  let best: CarrierId | null = null;
  let bestCost = Infinity;
  for (const c of carriers) {
    const cost = carrierCost(c, weightKg, domestic);
    if (cost < bestCost) {
      bestCost = cost;
      best = c;
    }
  }
  return best;
}

export function processOrder(order: Order): ProcessResult {
  if (!validateAddress(order.shippingAddress) || !validateAddress(order.billingAddress)) {
    return { orderId: order.id, success: false, error: "Invalid address", totalAmount: 0 };
  }

  const subtotal = sumValues(order.items, it => it.price * it.quantity);

  if (!screenFraud(order, subtotal)) {
    return { orderId: order.id, success: false, error: "Fraud detected", totalAmount: subtotal };
  }

  const { discounted: afterPromo, applied: appliedPromos } = applyPromos(order, subtotal);

  const allocations = allocateInventory(order);
  if (!allocations) {
    return { orderId: order.id, success: false, error: "Insufficient stock", totalAmount: afterPromo };
  }

  const shipmentsMap: Record<WarehouseId, Allocation[]> = {};
  for (const al of allocations) {
    if (!shipmentsMap[al.warehouse]) shipmentsMap[al.warehouse] = [];
    shipmentsMap[al.warehouse].push(al);
  }

  const shipments: Shipment[] = [];
  for (const [wh, allocs] of Object.entries(shipmentsMap)) {
    const weightKg = allocs.reduce((w, a) => w + a.quantity * 0.5, 0);
    const carrier = selectCarrier(order.shippingAddress, weightKg);
    if (!carrier) {
      return {
        orderId: order.id,
        success: false,
        error: `No carrier for ${order.shippingAddress.country}`,
        totalAmount: afterPromo,
      };
    }
    const cost = carrierCost(
      carrier,
      weightKg,
      order.shippingAddress.country.toUpperCase() === "US"
    );
    shipments.push({ carrier, allocations: allocs, cost });
  }

  const total = afterPromo + shipments.reduce((s, sh) => s + sh.cost, 0);
  return { orderId: order.id, success: true, shipments, totalAmount: total };
}

export default { processOrder };
function sumValues<T>(arr: T[], fn: (item: T) => number): number {
  return arr.reduce((s, i) => s + fn(i), 0);
}
