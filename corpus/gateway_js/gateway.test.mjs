/*
  gateway.test.mjs – Node –test suite for the Phase 1 gateway implementation.
  It imports the implementation based on the GATEWAY_IMPL env variable
  ("before" or "after") and runs a series of end‑to‑end scenarios covering:
  1️⃣ successful public GET
  2️⃣ successful private GET with proper tenant tier
  3️⃣ forbidden private GET from a free tier
  4️⃣ successful admin POST with valid payload
  5️⃣ bad payload on admin POST
  6️⃣ rate‑limit exceeded (repeat request quickly)
  7️⃣ circuit‑breaker open after repeated permission failures
  8️⃣ unknown route → 404

  The tests use the built‑in Node.js test runner (node --test) and assert
  that the gatewayHandler returns the expected HTTP‑like objects.
*/

import { strict as assert } from "node:assert";
import { describe, it, beforeEach } from "node:test";

// Dynamically import the target implementation.
const impl = process.env.GATEWAY_IMPL || "before";
const { gatewayHandler, __internal } = await import(`./${impl}/gateway.mjs`);

// Helper to invoke the handler with a minimal request.
function makeReq({ method = "GET", url = "/", headers = {}, body = undefined }) {
  return { method, url, headers, body };
}

describe("Phase 1 gateway", () => {
  beforeEach(() => {
    // Reset in‑memory globals between tests.
    __internal.rateBuckets.clear();
    __internal.circuitState.clear();
  });

  it("public GET succeeds", async () => {
    const resp = await gatewayHandler(makeReq({ url: "/public" }));
    assert.equal(resp.status, 200);
    assert.deepEqual(resp.body, { message: "public ok" });
  });

  it("private GET succeeds for pro tier", async () => {
    const resp = await gatewayHandler(
      makeReq({ url: "/private", headers: { "x-tenant-id": "t1", "x-tenant-tier": "pro" } })
    );
    assert.equal(resp.status, 200);
    assert.deepEqual(resp.body, { message: "private ok" });
  });

  it("private GET forbidden for free tier", async () => {
    const resp = await gatewayHandler(
      makeReq({ url: "/private", headers: { "x-tenant-id": "t2", "x-tenant-tier": "free" } })
    );
    assert.equal(resp.status, 403);
    assert.equal(resp.body.error, "insufficient permission");
  });

  it("admin POST succeeds with valid payload", async () => {
    const resp = await gatewayHandler(
      makeReq({
        method: "POST",
        url: "/admin",
        headers: { "x-tenant-id": "t3", "x-tenant-tier": "enterprise" },
        body: { command: "restart" },
      })
    );
    assert.equal(resp.status, 200);
    assert.equal(resp.body.result, "executed restart");
  });

  it("admin POST rejects bad payload", async () => {
    const resp = await gatewayHandler(
      makeReq({
        method: "POST",
        url: "/admin",
        headers: { "x-tenant-id": "t3", "x-tenant-tier": "enterprise" },
        body: { cmd: "oops" },
      })
    );
    assert.equal(resp.status, 400);
    assert.equal(resp.body.error, "invalid payload");
  });

  it("rate limiting blocks excess requests", async () => {
    // free tier = 5 per minute; fire 6 rapid requests.
    const headers = { "x-tenant-id": "t4", "x-tenant-tier": "free" };
    for (let i = 0; i < 5; i++) {
      const r = await gatewayHandler(makeReq({ url: "/public", headers }));
      assert.equal(r.status, 200);
    }
    const r6 = await gatewayHandler(makeReq({ url: "/public", headers }));
    assert.equal(r6.status, 429);
    assert.equal(r6.body.error, "rate limit exceeded");
  });

  it("circuit breaker opens after repeated permission failures", async () => {
    const headers = { "x-tenant-id": "t5", "x-tenant-tier": "free" };
    // Three successive forbidden attempts trigger circuit.
    for (let i = 0; i < 3; i++) {
      await gatewayHandler(makeReq({ url: "/private", headers }));
    }
    // Fourth attempt should hit circuit open.
    const resp = await gatewayHandler(makeReq({ url: "/private", headers }));
    assert.equal(resp.status, 503);
    assert.equal(resp.body.error, "service unavailable (circuit open)");
  });

  it("unknown route returns 404", async () => {
    const resp = await gatewayHandler(makeReq({ url: "/unknown" }));
    assert.equal(resp.status, 404);
    assert.equal(resp.body.error, "not found");
  });
});
