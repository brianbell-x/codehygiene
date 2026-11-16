function buildShipments(shipmentsMap: Record<string, Allocation[]>, order: Order): Shipment[] {
  const shipments: Shipment[] = [];
  for (const [warehouse, allocs] of Object.entries(shipmentsMap)) {
    // approximate weight: 0.5kg per item unit
    const weightKg = allocs.reduce((sum, alloc) => sum + alloc.quantity * 0.5, 0);
    const carrier = selectCarrier(order.shippingAddress, weightKg);
    shipments.push({ warehouse, carrier, weightKg });
  }
  return shipments;
}
