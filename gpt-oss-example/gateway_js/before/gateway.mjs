/*
  gateway.mjs – Minimal multi-tenant API gateway middleware (Phase 1)
  Features implemented:
  • API‑key and Bearer‑token parsing
  • Tenant‑tier rate‑limiting (token‑bucket per tier)
  • Route‑level permission checks
  • Header normalization / stripping of internal headers
  • Payload validation (simple JSON schema)
  • Circuit‑breaker with fallback response
  • Structured request / error logging (JSON to console)

  The implementation is deliberately self‑contained – no external npm packages.
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
  // In a real system tenant would be derived from API‑key / token.
  // For this demo we honour a `x-tenant-id` header; if missing default to "free".
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
  // Simple mock: tenant tier maps to a set of permissions.
  const tierPermissions = {
    free: [],
    pro: ["read"],
    enterprise: ["read", "admin"],
  };
  const perms = tierPermissions[tenantInfo.tier] || [];
  return required ? perms.includes(required) : true;
}

function normalizeHeaders(headers) {
  const out = {};
  for (const [k, v] of Object.entries(headers)) {
    const lower = k.toLowerCase();
    if (lower.startsWith("x-internal-")) continue; // strip internal use only
    out[lower] = v;
  }
  return out;
}

function logStructured(entry) {
  // In production this would be piped to a logger; here a JSON line suffices.
  console.log(JSON.stringify(entry));
}

function circuitAllowed(route) {
  const state = circuitState.get(route) || { failures: 0, openUntil: 0 };
  const now = Date.now();
  if (state.openUntil > now) return false; // circuit open
  return true;
}

function recordFailure(route) {
  const now = Date.now();
  const state = circuitState.get(route) || { failures: 0, openUntil: 0 };
  state.failures += 1;
  if (state.failures >= CIRCUIT_BREAKER.failuresThreshold) {
    state.openUntil = now + CIRCUIT_BREAKER.coolDownMs;
    state.failures = 0; // reset after opening
  }
  circuitState.set(route, state);
}

function resetCircuit(route) {
  circuitState.set(route, { failures: 0, openUntil: 0 });
}

// ---- Main handler -------------------------------------------------------
export async function gatewayHandler(rawReq) {
  const start = Date.now();
  // Normalise request shape (allow plain objects in tests)
  const req = {
    method: rawReq.method?.toUpperCase() || "GET",
    url: rawReq.url || "/",
    headers: normalizeHeaders(rawReq.headers || {}),
    body: rawReq.body,
  };

  const { apiKey, bearer } = parseAuth(req.headers);
  const tenantInfo = getTenantInfo(req);

  // Basic auth extraction – for demo we just attach to request.
  req.auth = { apiKey, bearer };
  req.tenant = tenantInfo;

  // Rate limiting
  if (!checkRateLimit(tenantInfo.tenantId, tenantInfo.tier)) {
    const resp = { status: 429, body: { error: "rate limit exceeded" } };
    logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: "rate_limit" });
    return resp;
  }

  // Route lookup
  const routeDef = Object.entries(ROUTES).find(
    ([path, cfg]) => path === req.url && cfg.method === req.method
  );
  if (!routeDef) {
    const resp = { status: 404, body: { error: "not found" } };
    logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: "no_route" });
    return resp;
  }
  const [path, cfg] = routeDef;

  // Circuit‑breaker check
  if (!circuitAllowed(path)) {
    const resp = { status: 503, body: { error: "service unavailable (circuit open)" } };
    logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: "circuit_open" });
    return resp;
  }

  // Permission check
  if (!checkPermission(tenantInfo, cfg.permission)) {
    const resp = { status: 403, body: { error: "insufficient permission" } };
    logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: "perm_denied" });
    recordFailure(path);
    return resp;
  }

  // Payload validation (if a validator exists)
  if (cfg.validator) {
    const valid = cfg.validator(req.body);
    if (!valid) {
      const resp = { status: 400, body: { error: "invalid payload" } };
      logStructured({ request: req, response: resp, durationMs: Date.now() - start, error: "payload_invalid" });
      recordFailure(path);
      return resp;
    }
  }

  // Actual handler – wrap to catch unexpected errors.
  try {
    const result = await cfg.handler(req);
    // Reset circuit on success
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

// Export route map for tests (optional)
export const __internal = { TENANT_TIERS, ROUTES, rateBuckets, circuitState };
