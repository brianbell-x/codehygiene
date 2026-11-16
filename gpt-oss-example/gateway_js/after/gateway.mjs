/*
  gateway.mjs – Refactored multi-tenant API gateway middleware (Phase 2)
  Features (unchanged from Phase 1) with code‑hygiene improvements:
  - API‑key / Bearer token parsing
  - Tenant‑tier rate limiting
  - Route permission checks
  - Header normalization (no internal headers)
  - Payload validation
  - Circuit‑breaker with fallback response
  - Structured JSON logging
*/

// ---- Configuration -------------------------------------------------------
const TENANT_TIERS = {
  free: { limit: 5, intervalMs: 60_000 }, // 5 requests per minute
  pro: { limit: 20, intervalMs: 60_000 },
  enterprise: { limit: 100, intervalMs: 60_000 },
};

// Routes definition – each route declares required permission and a simple validator.
const ROUTES = {
  "/public": {
    method: "GET",
    permission: null,
    validator: null,
    handler: async (req) => ({ status: 200, body: { message: "public ok" } }),
  },
  "/private": {
    method: "GET",
    permission: "read",
    validator: null,
    handler: async (req) => ({ status: 200, body: { message: "private ok" } }),
  },
  "/admin": {
    method: "POST",
    permission: "admin",
    validator: (payload) => payload && typeof payload.command === "string",
    handler: async (req) => ({ status: 200, body: { result: `executed ${req.body.command}` } }),
  },
};

// In‑memory state (rate‑limit buckets, circuit‑breaker counters)
const rateBuckets = new Map(); // key: tenantId, value: { tokens, resetAt }
const circuitState = new Map(); // key: route, value: { failures, openUntil }

const CIRCUIT_BREAKER = { failuresThreshold: 3, coolDownMs: 30_000 };

// ---- Helpers -----------------------------------------------------------
function parseAuth(headers) {
  const apiKey = headers["x-api-key"] || null;
  const auth = headers["authorization"] || "";
  const bearer = auth.startsWith("Bearer ") ? auth.slice(7).trim() : null;
  return { apiKey, bearer };
}

function getTenantInfo(req) {
  const tenantId = req.headers["x-tenant-id"] || "anonymous";
  const tier = req.headers["x-tenant-tier"] || "free"; // free|pro|enterprise
  return { tenantId, tier };
}

function checkRateLimit(tenantId, tier) {
  const config = TENANT_TIERS[tier] || TENANT_TIERS.free;
  const now = Date.now();
  let bucket = rateBuckets.get(tenantId);
  if (!bucket || now > bucket.resetAt) {
    bucket = { tokens: config.limit, resetAt: now + config.intervalMs };
    rateBuckets.set(tenantId, bucket);
  }
  if (bucket.tokens <= 0) return false;
  bucket.tokens -= 1;
  return true;
}

function checkPermission(tenantInfo, required) {
  // Guard against unknown tenant tiers and missing permissions.
  if (!required) return true;
  const tierPermissions = {
    free: [],
    pro: ["read"],
    enterprise: ["read", "admin"],
  };
  const perms = tierPermissions[tenantInfo.tier];
  return Array.isArray(perms) && perms.includes(required);
}

function normalizeHeaders(headers) {
  const out = {};
  for (const [k, v] of Object.entries(headers)) {
    const lower = k.toLowerCase();
    if (lower.startsWith("x-internal-")) continue;
    out[lower] = v;
  }
  return out;
}

function logStructured(entry) {
  console.log(JSON.stringify(entry));
}

// Helper to build and log error responses in a compact multi‑line form
function respondError(req, start, status, message, code) {
  const resp = { status, body: { error: message } };
  logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: code });
  return resp;
}

function circuitAllowed(route) {
  const state = circuitState.get(route) || { failures: 0, openUntil: 0 };
  const now = Date.now();
  return state.openUntil <= now;
}

function recordFailure(route) {
  const now = Date.now();
  const state = circuitState.get(route) || { failures: 0, openUntil: 0 };
  state.failures += 1;
  if (state.failures >= CIRCUIT_BREAKER.failuresThreshold) {
    state.openUntil = now + CIRCUIT_BREAKER.coolDownMs;
    state.failures = 0;
  }
  circuitState.set(route, state);
}

function resetCircuit(route) {
  circuitState.set(route, { failures: 0, openUntil: 0 });
}

// ---- Main handler -------------------------------------------------------
export async function gatewayHandler(rawReq) {
  const start = Date.now();
  const { method = "GET", url = "/", headers = {}, body } = rawReq;
  const req = {
    method: method.toUpperCase(),
    url,
    headers: normalizeHeaders(headers),
    body,
  };

  const { apiKey, bearer } = parseAuth(req.headers);
  const tenantInfo = getTenantInfo(req);
  req.auth = { apiKey, bearer };
  req.tenant = tenantInfo;

  if (!checkRateLimit(tenantInfo.tenantId, tenantInfo.tier)) {
    return respondError(req, start, 429, "rate limit exceeded", "rate_limit");
  }

  const routeDef = Object.entries(ROUTES).find(
    ([path, cfg]) => path === req.url && cfg.method === req.method
  );
  if (!routeDef) {
    const resp = { status: 404, body: { error: "not found" } };
    logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: "no_route" });
    return resp;
  }
  const [path, cfg] = routeDef;

  if (!circuitAllowed(path)) {
    const resp = { status: 503, body: { error: "service unavailable (circuit open)" } };
    logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: "circuit_open" });
    return resp;
  }

  if (!checkPermission(tenantInfo, cfg.permission)) {
    const resp = { status: 403, body: { error: "insufficient permission" } };
    logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: "perm_denied" });
    recordFailure(path);
    return resp;
  }

  if (cfg.validator) {
    const valid = cfg.validator(req.body);
    if (!valid) {
      const resp = respondError(req, start, 400, "invalid payload", "payload_invalid");
      recordFailure(path);
      return resp;
    }
  }

  try {
    const result = await cfg.handler(req);
    resetCircuit(path);
    logStructured({ request: req, response: result, durationMs: Date.now() - start });
    return result;
  } catch (e) {
    recordFailure(path);
    const resp = { status: 502, body: { error: "upstream failure" } };
    logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: e.message });
    return resp;
  }
}

export const __internal = { TENANT_TIERS, ROUTES, rateBuckets, circuitState };