import {
  ARCHETYPE_NAMES,
  SCORE_NAMES,
  SQUIGGLE_MESSAGES,
  collectSquiggles,
  scoreCode,
} from "./engine.js";

export const EXAMPLE_REPOS = [
  { label: "codehygiene", slug: "brianbell-x/codehygiene" },
  { label: "gitingest", slug: "cyclotruc/gitingest" },
  { label: "micrograd", slug: "karpathy/micrograd" },
  { label: "clsx", slug: "lukeed/clsx" },
  { label: "itsdangerous", slug: "pallets/itsdangerous" },
];

const CODE_EXTENSIONS = new Set([
  ".py",
  ".pyi",
  ".js",
  ".mjs",
  ".cjs",
  ".ts",
  ".tsx",
  ".jsx",
  ".c",
  ".h",
  ".cc",
  ".cpp",
  ".cxx",
  ".hpp",
  ".rs",
  ".go",
  ".zig",
  ".java",
  ".cs",
  ".lua",
  ".rb",
  ".php",
  ".swift",
  ".kt",
  ".scala",
  ".sh",
]);

const SKIP_DIRS = new Set([
  ".git",
  ".hg",
  ".svn",
  "__pycache__",
  "node_modules",
  "target",
  "build",
  "dist",
  "venv",
  ".venv",
  ".tox",
  ".mypy_cache",
  ".pytest_cache",
]);

const SECONDARY_DIRS = new Set([
  "test",
  "tests",
  "__tests__",
  "spec",
  "example",
  "examples",
  "benchmark",
  "benchmarks",
  "fixtures",
]);

const TEST_ONLY_DIRS = new Set([
  "test",
  "tests",
  "__tests__",
  "spec",
  "fixtures",
]);

const REPO_HOSTS = new Set([
  "github.com",
  "www.github.com",
  "gitingest.com",
  "www.gitingest.com",
]);

const MINIFIED_SUFFIXES = [".min.js", ".min.mjs", ".min.cjs", ".bundle.js"];
const TEST_INFIXES = [".test.", ".spec.", "_test.", "_spec."];
const SAFE_SEGMENT = /^[A-Za-z0-9_.-]+$/;
const MAX_FILE_BYTES = 131072;
const MAX_SCORED_FILES = 200;
const FETCH_CONCURRENCY = 12;

let sharedWorkerPoolPromise = null;

function makeRepoError(code, message) {
  const err = new Error(message);
  err.repoCode = code;
  return err;
}

function stripRepoGitSuffix(repoName) {
  return repoName.endsWith(".git") ? repoName.slice(0, -4) : repoName;
}

function stripSchemePrefix(text) {
  const lower = text.toLowerCase();
  if (lower.startsWith("https://")) {
    return text.slice(8);
  }
  if (lower.startsWith("http://")) {
    return text.slice(7);
  }
  return text;
}

function stripQueryAndSlashes(text) {
  let cutIdx = text.length;
  for (const sep of ["?", "#"]) {
    const pos = text.indexOf(sep);
    if (pos >= 0 && pos < cutIdx) {
      cutIdx = pos;
    }
  }
  let sliced = text.slice(0, cutIdx);
  while (sliced.endsWith("/")) {
    sliced = sliced.slice(0, -1);
  }
  while (sliced.startsWith("/")) {
    sliced = sliced.slice(1);
  }
  return sliced;
}

function buildRepoDescriptor(owner, rawRepo, ref, subpath, blobPath) {
  const repo = stripRepoGitSuffix(rawRepo);
  if (!SAFE_SEGMENT.test(owner) || !SAFE_SEGMENT.test(repo)) {
    return null;
  }

  let normalizedPath = `${owner}/${repo}`;
  if (blobPath && ref) {
    normalizedPath = `${owner}/${repo}/blob/${ref}/${blobPath}`;
  } else if (subpath && ref) {
    normalizedPath = `${owner}/${repo}/tree/${ref}/${subpath}`;
  } else if (ref) {
    normalizedPath = `${owner}/${repo}@${ref}`;
  }

  return {
    owner,
    repo,
    ref,
    subpath,
    blobPath,
    slug: `${owner}/${repo}`,
    normalizedPath,
  };
}

function parseUrlPathSegments(segments) {
  if (segments.length < 2) {
    return null;
  }
  const [owner, repo, kind, refPart, ...restParts] = segments;
  if (kind === "tree" && refPart) {
    return buildRepoDescriptor(owner, repo, refPart, restParts.join("/"), "");
  }
  if (kind === "blob" && refPart && restParts.length > 0) {
    return buildRepoDescriptor(owner, repo, refPart, "", restParts.join("/"));
  }
  return buildRepoDescriptor(owner, repo, "", "", "");
}

export function isFullRepoUrl(rawText) {
  const trimmed = rawText.trim().toLowerCase();
  return (
    trimmed.startsWith("https://github.com/") ||
    trimmed.startsWith("http://github.com/") ||
    trimmed.startsWith("github.com/") ||
    trimmed.startsWith("https://gitingest.com/")
  );
}

export function parseRepoSpec(rawInput) {
  const trimmed = rawInput.trim();
  if (!trimmed || trimmed.includes("\n") || trimmed.includes(" ")) {
    return null;
  }

  let pathText = stripSchemePrefix(trimmed);
  const firstSlash = pathText.indexOf("/");
  if (firstSlash <= 0) {
    return null;
  }

  const leadingHost = pathText.slice(0, firstSlash).toLowerCase();
  if (REPO_HOSTS.has(leadingHost)) {
    pathText = pathText.slice(firstSlash + 1);
  }

  const cleaned = stripQueryAndSlashes(pathText);
  const parts = cleaned.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts.length === 2 && parts[1].includes("@")) {
    const [owner, repoWithRef] = parts;
    const atIdx = repoWithRef.indexOf("@");
    if (atIdx > 0) {
      return buildRepoDescriptor(
        owner,
        repoWithRef.slice(0, atIdx),
        repoWithRef.slice(atIdx + 1),
        "",
        "",
      );
    }
  }

  return parseUrlPathSegments(parts);
}

export function formatRepoFileOption(relPath) {
  const slashIdx = relPath.lastIndexOf("/");
  if (slashIdx < 0) {
    return relPath;
  }
  const baseName = relPath.slice(slashIdx + 1);
  const dirName = relPath.slice(0, slashIdx);
  return `${baseName} — ${dirName}`;
}

export function splitRepoPath(relPath) {
  const slashIdx = relPath.lastIndexOf("/");
  if (slashIdx < 0) {
    return { baseName: relPath, dirPrefix: "" };
  }
  return {
    baseName: relPath.slice(slashIdx + 1),
    dirPrefix: `${relPath.slice(0, slashIdx)}/`,
  };
}

export function groupSquiggleSpans(squiggles) {
  const spans = [];
  for (const squiggle of squiggles) {
    const prev = spans[spans.length - 1];
    if (prev && prev.kind === squiggle.kind && squiggle.line === prev.endLine + 1) {
      prev.endLine = squiggle.line;
      prev.probability = Math.max(prev.probability, squiggle.probability);
      continue;
    }
    spans.push({
      startLine: squiggle.line,
      endLine: squiggle.line,
      kind: squiggle.kind,
      message: squiggle.message,
      probability: squiggle.probability,
    });
  }
  return spans;
}

function fileExtension(filePath) {
  const baseName = filePath.slice(filePath.lastIndexOf("/") + 1);
  const dotIdx = baseName.lastIndexOf(".");
  return dotIdx <= 0 ? "" : baseName.slice(dotIdx).toLowerCase();
}

function hasSkippedDirectory(segments) {
  for (let idx = 0; idx < segments.length - 1; idx += 1) {
    const dirName = segments[idx];
    if (
      dirName.startsWith(".") ||
      dirName.endsWith(".egg-info") ||
      SKIP_DIRS.has(dirName)
    ) {
      return true;
    }
  }
  return false;
}

export function isTestFilePath(relPath) {
  const lower = relPath.toLowerCase();
  const parts = lower.split("/");
  const baseName = parts[parts.length - 1];
  if (parts.slice(0, -1).some((dir) => TEST_ONLY_DIRS.has(dir))) {
    return true;
  }
  if (baseName.startsWith("test_") || baseName.startsWith("tests_")) {
    return true;
  }
  return TEST_INFIXES.some((infix) => baseName.includes(infix));
}

function wildcardMatch(text, pattern) {
  let tIdx = 0;
  let pIdx = 0;
  let starIdx = -1;
  let matchIdx = 0;

  while (tIdx < text.length) {
    if (pIdx < pattern.length && (pattern[pIdx] === "?" || pattern[pIdx] === text[tIdx])) {
      tIdx += 1;
      pIdx += 1;
    } else if (pIdx < pattern.length && pattern[pIdx] === "*") {
      starIdx = pIdx;
      matchIdx = tIdx;
      pIdx += 1;
    } else if (starIdx >= 0) {
      pIdx = starIdx + 1;
      matchIdx += 1;
      tIdx = matchIdx;
    } else {
      return false;
    }
  }

  while (pIdx < pattern.length && pattern[pIdx] === "*") {
    pIdx += 1;
  }
  return pIdx === pattern.length;
}

function normalizeGlobToken(rawToken) {
  let token = rawToken.trim().toLowerCase().replaceAll("\\", "/");
  while (token.startsWith("./")) {
    token = token.slice(2);
  }
  while (token.startsWith("/")) {
    token = token.slice(1);
  }
  if (!token) {
    return null;
  }
  const dirOnly = token.endsWith("/");
  const core = dirOnly ? token.slice(0, -1) : token;
  return core ? { core, dirOnly, hasSlash: core.includes("/") } : null;
}

function matchesGlobSpec(lowerPath, spec) {
  const segments = lowerPath.split("/");
  if (spec.dirOnly) {
    if (!spec.hasSlash) {
      return segments.slice(0, -1).some((seg) => wildcardMatch(seg, spec.core));
    }
    return (
      wildcardMatch(lowerPath, `${spec.core}/*`) ||
      wildcardMatch(lowerPath, `*/${spec.core}/*`)
    );
  }
  if (!spec.hasSlash) {
    return segments.some((seg) => wildcardMatch(seg, spec.core));
  }
  return (
    wildcardMatch(lowerPath, spec.core) ||
    wildcardMatch(lowerPath, `${spec.core}/*`)
  );
}

export function compileFilterMatchers(patternText) {
  if (!patternText || !patternText.trim()) {
    return [];
  }
  return patternText
    .split(",")
    .map(normalizeGlobToken)
    .filter(Boolean);
}

export function matchesRepoFilter(relPath, filterOpts = {}) {
  const { mode = "exclude", patternText = "", excludeTests = false } = filterOpts;
  if (excludeTests && isTestFilePath(relPath)) {
    return false;
  }
  const specs = compileFilterMatchers(patternText);
  if (specs.length === 0) {
    return true;
  }
  const lowerPath = relPath.toLowerCase();
  const hit = specs.some((spec) => matchesGlobSpec(lowerPath, spec));
  return mode === "include" ? hit : !hit;
}

export function isScorableRepoPath(relPath, byteSize, subpath) {
  const lowerPath = relPath.toLowerCase();
  for (const suffix of MINIFIED_SUFFIXES) {
    if (lowerPath.endsWith(suffix)) {
      return false;
    }
  }
  if (byteSize !== null && (byteSize <= 0 || byteSize > MAX_FILE_BYTES)) {
    return false;
  }
  if (subpath && relPath !== subpath && !relPath.startsWith(`${subpath}/`)) {
    return false;
  }

  const segments = relPath.split("/").filter(Boolean);
  if (segments.length === 0 || hasSkippedDirectory(segments)) {
    return false;
  }
  return CODE_EXTENSIONS.has(fileExtension(relPath));
}

function pathPriorityRank(relPath, focusPath) {
  if (focusPath && relPath === focusPath) {
    return -10;
  }
  const parts = relPath.toLowerCase().split("/");
  const inSecondary = parts.slice(0, -1).some((dir) => SECONDARY_DIRS.has(dir));
  return (inSecondary ? 100 : 0) + parts.length;
}

function selectTargetEntries(entries, spec, filterOpts) {
  const scorable = entries.filter((entry) =>
    isScorableRepoPath(entry.path, entry.size, spec.subpath),
  );
  const filtered = filterOpts
    ? scorable.filter((entry) => matchesRepoFilter(entry.path, filterOpts))
    : scorable;

  filtered.sort((left, right) => {
    const rankDiff =
      pathPriorityRank(left.path, spec.blobPath) -
      pathPriorityRank(right.path, spec.blobPath);
    return rankDiff !== 0 ? rankDiff : left.path.localeCompare(right.path);
  });

  return {
    selected: filtered.slice(0, MAX_SCORED_FILES),
    totalEligible: filtered.length,
    unfilteredCount: scorable.length,
  };
}

async function safeFetch(url, fetchOpts) {
  try {
    return await fetch(url, fetchOpts);
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw err;
    }
    throw makeRepoError("offline", "Network request failed; check connection");
  }
}

async function fetchJsdelivrTree(spec, effectiveRef, signal) {
  const endpoint =
    `https://data.jsdelivr.com/v1/package/gh/${spec.owner}/${spec.repo}` +
    `@${encodeURIComponent(effectiveRef)}/flat`;
  const resp = await safeFetch(endpoint, { signal });
  if (!resp.ok) {
    return { entries: null, status: resp.status };
  }
  const payload = await resp.json();
  if (!Array.isArray(payload.files)) {
    return { entries: null, status: 502 };
  }

  const entries = payload.files.map((item) => ({
    path: item.name.replace(/^\//, ""),
    size: typeof item.size === "number" ? item.size : null,
  }));
  return { entries, status: 200 };
}

async function fetchGithubApiTree(spec, targetRef, signal) {
  const treeUrl =
    `https://api.github.com/repos/${spec.owner}/${spec.repo}/git/trees/` +
    `${encodeURIComponent(targetRef)}?recursive=1`;
  const treeResp = await safeFetch(treeUrl, { signal });
  if (treeResp.status === 404) {
    throw makeRepoError(
      "not_found",
      `Repository or ref ${spec.owner}/${spec.repo} not found (public repos only)`,
    );
  }
  if (treeResp.status === 403 || treeResp.status === 429) {
    return { entries: null, fallbackReason: "rate_limited" };
  }
  if (!treeResp.ok) {
    return { entries: null, fallbackReason: "http_error" };
  }

  const body = await treeResp.json();
  if (body.truncated || !Array.isArray(body.tree)) {
    return { entries: null, fallbackReason: "truncated" };
  }
  const entries = body.tree
    .filter((node) => node.type === "blob" && typeof node.path === "string")
    .map((node) => ({
      path: node.path,
      size: typeof node.size === "number" ? node.size : null,
    }));
  return { entries, fallbackReason: null };
}

async function discoverRepoTree(spec, signal) {
  const effectiveRef = spec.ref || "HEAD";
  const apiResult = await fetchGithubApiTree(spec, effectiveRef, signal);
  if (apiResult.entries) {
    return { entries: apiResult.entries, resolvedRef: effectiveRef };
  }

  const jsdResult = await fetchJsdelivrTree(spec, effectiveRef, signal);
  if (jsdResult.entries) {
    return { entries: jsdResult.entries, resolvedRef: effectiveRef };
  }

  if (jsdResult.status === 404) {
    throw makeRepoError(
      "not_found",
      `Repository ${spec.owner}/${spec.repo} not found (public repos only)`,
    );
  }
  if (apiResult.fallbackReason === "rate_limited") {
    throw makeRepoError("rate_limited", "GitHub API rate limit reached; try again shortly");
  }
  throw makeRepoError("offline", `Failed to read repository tree (${jsdResult.status})`);
}

function encodeRepoPath(relPath) {
  return relPath.split("/").map(encodeURIComponent).join("/");
}

async function fetchFileText(spec, resolvedRef, relPath, signal) {
  const encodedPath = encodeRepoPath(relPath);
  const encodedRef = encodeURIComponent(resolvedRef);
  const rawUrl =
    `https://raw.githubusercontent.com/${spec.owner}/${spec.repo}/` +
    `${encodedRef}/${encodedPath}`;

  const primary = await safeFetch(rawUrl, { signal });
  if (primary.ok) {
    return primary.text();
  }

  const cdnUrl =
    `https://cdn.jsdelivr.net/gh/${spec.owner}/${spec.repo}` +
    `@${encodedRef}/${encodedPath}`;
  const fallback = await safeFetch(cdnUrl, { signal });
  return fallback.ok ? fallback.text() : null;
}

function yieldToMainThread() {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function spawnScoreWorker(rawBuffer) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });
    const onMessage = (event) => {
      if (event.data && event.data.type === "ready") {
        worker.removeEventListener("message", onMessage);
        resolve(worker);
      }
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", (err) => reject(err), { once: true });
    const copy = rawBuffer.slice(0);
    worker.postMessage({ type: "init", buffer: copy }, [copy]);
  });
}

function getWorkerPool(rawBuffer) {
  if (typeof Worker === "undefined" || !rawBuffer) {
    return Promise.resolve(null);
  }
  if (!sharedWorkerPoolPromise) {
    const hwCores =
      typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;
    const poolSize = Math.min(6, Math.max(2, hwCores - 1));
    sharedWorkerPoolPromise = Promise.all(
      Array.from({ length: poolSize }, () => spawnScoreWorker(rawBuffer)),
    ).catch(() => null);
  }
  return sharedWorkerPoolPromise;
}

function runWorkerJob(worker, jobId, code, threshold) {
  return new Promise((resolve, reject) => {
    const onMessage = (event) => {
      if (event.data && event.data.type === "scored" && event.data.jobId === jobId) {
        worker.removeEventListener("message", onMessage);
        resolve(event.data.result);
      }
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", (err) => reject(err), { once: true });
    worker.postMessage({ type: "score", jobId, code, threshold });
  });
}

async function scoreFilesInParallel(workItems, weights, threshold, signal, onProgress) {
  const pool = await getWorkerPool(weights.rawBuffer);
  if (!pool || pool.length === 0 || workItems.length <= 1) {
    return scoreFilesInline(workItems, weights, threshold, signal, onProgress);
  }

  const results = new Array(workItems.length);
  let cursor = 0;
  let completed = 0;
  let nextJobId = 1;

  async function slotLoop(worker) {
    while (cursor < workItems.length) {
      if (signal && signal.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }
      const itemIdx = cursor;
      cursor += 1;
      const jobId = nextJobId;
      nextJobId += 1;
      const scored = await runWorkerJob(
        worker,
        jobId,
        workItems[itemIdx].source,
        threshold,
      );
      results[itemIdx] = buildScoredFileRecord(workItems[itemIdx], scored);
      completed += 1;
      if (onProgress) {
        onProgress({ phase: "score", completed, total: workItems.length });
      }
    }
  }

  const activeSlots = pool.slice(0, Math.min(pool.length, workItems.length));
  await Promise.all(activeSlots.map((worker) => slotLoop(worker)));
  return results;
}

async function scoreFilesInline(workItems, weights, threshold, signal, onProgress) {
  const results = [];
  let sliceStartMs = performance.now();

  for (let idx = 0; idx < workItems.length; idx += 1) {
    if (signal && signal.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }
    const scored = scoreCode(workItems[idx].source, weights, threshold);
    results.push(buildScoredFileRecord(workItems[idx], scored));
    if (onProgress) {
      onProgress({ phase: "score", completed: results.length, total: workItems.length });
    }
    if (performance.now() - sliceStartMs > 18) {
      await yieldToMainThread();
      sliceStartMs = performance.now();
    }
  }
  return results;
}

function buildScoredFileRecord(workItem, result) {
  return attachFileSpans({
    path: workItem.path,
    originalCode: workItem.source,
    currentCode: workItem.source,
    isEdited: false,
    initialScore: result.score,
    result,
  });
}

export function attachFileSpans(fileItem, nextThreshold = null) {
  if (nextThreshold !== null) {
    fileItem.result.squiggles = collectSquiggles(fileItem.result, nextThreshold);
  }
  const spans = groupSquiggleSpans(fileItem.result.squiggles);
  const kindMaxProb = new Map();
  for (const span of spans) {
    const prev = kindMaxProb.get(span.kind) ?? -1;
    if (span.probability > prev) {
      kindMaxProb.set(span.kind, span.probability);
    }
  }
  const uniqueKinds = [...kindMaxProb.entries()]
    .sort((left, right) => right[1] - left[1])
    .map((pair) => pair[0]);

  fileItem.spans = spans;
  fileItem.uniqueKinds = uniqueKinds;
  return fileItem;
}

export function sortRepoFiles(files) {
  return [...files].sort((left, right) => {
    const leftFlagged = left.spans.length > 0 ? 1 : 0;
    const rightFlagged = right.spans.length > 0 ? 1 : 0;
    if (leftFlagged !== rightFlagged) {
      return rightFlagged - leftFlagged;
    }
    if (Math.abs(left.result.score - right.result.score) > 1e-4) {
      return left.result.score - right.result.score;
    }
    return left.path.localeCompare(right.path);
  });
}

function buildKindBreakdown(files) {
  const byKind = new Map();
  for (let fileIdx = 0; fileIdx < files.length; fileIdx += 1) {
    const item = files[fileIdx];
    const seenInFile = new Set();

    for (const span of item.spans) {
      let entry = byKind.get(span.kind);
      if (!entry) {
        entry = {
          kind: span.kind,
          issueCount: 0,
          fileCount: 0,
          maxProb: 0,
          worstFileIdx: fileIdx,
          worstFileScore: item.result.score,
          worstSpan: span,
        };
        byKind.set(span.kind, entry);
      }

      entry.issueCount += 1;
      seenInFile.add(span.kind);
      if (span.probability > entry.maxProb) {
        entry.maxProb = span.probability;
      }
      if (item.result.score < entry.worstFileScore) {
        entry.worstFileScore = item.result.score;
        entry.worstFileIdx = fileIdx;
        entry.worstSpan = span;
      }
    }

    for (const kind of seenInFile) {
      byKind.get(kind).fileCount += 1;
    }
  }

  return [...byKind.values()].sort((left, right) => {
    if (right.issueCount !== left.issueCount) {
      return right.issueCount - left.issueCount;
    }
    return right.maxProb - left.maxProb;
  });
}

export function summarizeRepoFiles(files) {
  if (files.length === 0) {
    return null;
  }

  let weightSum = 0;
  let weightedScore = 0;
  let weightedRl = 0;
  let totalIssues = 0;
  let flaggedFiles = 0;
  let totalLines = 0;
  const dimSums = Object.fromEntries(SCORE_NAMES.map((name) => [name, 0]));
  const archWeights = Object.fromEntries(ARCHETYPE_NAMES.map((name) => [name, 0]));

  for (const item of files) {
    const res = item.result;
    const lineWeight = Math.max(4, Math.min(300, res.numLines));
    weightSum += lineWeight;
    weightedScore += res.score * lineWeight;
    weightedRl += res.rlHygiene * lineWeight;
    totalLines += res.numLines;
    totalIssues += item.spans.length;
    if (item.spans.length > 0) {
      flaggedFiles += 1;
    }
    for (const dim of SCORE_NAMES) {
      dimSums[dim] += res.scores[dim] * lineWeight;
    }
    if (archWeights[res.archetype] !== undefined) {
      archWeights[res.archetype] += lineWeight;
    }
  }

  const scores = {};
  for (const dim of SCORE_NAMES) {
    scores[dim] = dimSums[dim] / weightSum;
  }

  let topArchetype = ARCHETYPE_NAMES[0];
  for (const name of ARCHETYPE_NAMES) {
    if (archWeights[name] > archWeights[topArchetype]) {
      topArchetype = name;
    }
  }

  return {
    score: weightedScore / weightSum,
    rlHygiene: weightedRl / weightSum,
    scores,
    archetype: topArchetype,
    numLines: totalLines,
    totalIssues,
    flaggedFiles,
    fileCount: files.length,
    hasEdits: files.some((item) => item.isEdited),
    kindBreakdown: buildKindBreakdown(files),
  };
}

function refreshSession(repoSession, nextFiles = null, shouldSort = true) {
  const active = nextFiles ?? repoSession.files;
  repoSession.files = shouldSort ? sortRepoFiles(active) : active;
  repoSession.summary = summarizeRepoFiles(repoSession.files);
  return repoSession.summary;
}

export function applyRepoFilters(repoSession, filterOpts) {
  const pool = repoSession.allScoredFiles ?? repoSession.files;
  const matched = pool.filter((entry) => matchesRepoFilter(entry.path, filterOpts));
  return refreshSession(repoSession, matched, true);
}

export function rescoreRepoThreshold(repoSession, threshold) {
  const pool = repoSession.allScoredFiles ?? repoSession.files;
  pool.forEach((entry) => attachFileSpans(entry, threshold));
  refreshSession(repoSession, null, true);
}

export function updateRepoFileCode(repoSession, fileIdx, sourceCode, weights, threshold) {
  const activeEntry = repoSession.files[fileIdx];
  const nextResult = scoreCode(sourceCode, weights, threshold);
  Object.assign(activeEntry, {
    currentCode: sourceCode,
    isEdited: sourceCode !== activeEntry.originalCode,
    result: nextResult,
  });
  attachFileSpans(activeEntry);
  repoSession.summary = summarizeRepoFiles(repoSession.files);
  return nextResult;
}

export function buildRepoFixPrompt(repoSession) {
  const { spec, summary, files } = repoSession;
  if (!summary) {
    return `# Code Hygiene Report: ${spec.slug}\nNo matching files in current filter.`;
  }

  const dimsLine = SCORE_NAMES.map(
    (name) => `${name}=${summary.scores[name].toFixed(2)}`,
  ).join(" ");
  const scoreStr = summary.score.toFixed(2);
  const rlStr = summary.rlHygiene.toFixed(2);
  const scoreLine = `Score: ${scoreStr} | RL hygiene: ${rlStr} | Style: ${summary.archetype}`;
  const countPart = `Analyzed: ${summary.fileCount} files (${summary.numLines} lines)`;
  const issuePart = `Issues: ${summary.totalIssues} across ${summary.flaggedFiles} files`;
  const lines = [
    `# Code Hygiene Report: ${spec.normalizedPath}`,
    scoreLine,
    `${countPart} | ${issuePart}`,
    `Dimensions: ${dimsLine}`,
    "",
  ];

  const flagged = files.filter((item) => item.spans.length > 0);
  if (flagged.length === 0) {
    lines.push("Zero squiggles above threshold.");
    return lines.join("\n");
  }

  const note =
    "Prefer small, focused edits per file while keeping tests green; on large " +
    "codebases, clean up files incrementally as they get edited " +
    "(`npx skills add brianbell-x/codehygiene`).";
  lines.push(note, "", "## Files Requiring Refactoring");
  for (const item of flagged) {
    lines.push(
      `\n### ${item.path} (score ${item.result.score.toFixed(2)}, ${item.spans.length} issues)`,
    );
    for (const span of item.spans) {
      const range =
        span.startLine === span.endLine
          ? `Ln ${span.startLine}`
          : `Ln ${span.startLine}-${span.endLine}`;
      const guidance = SQUIGGLE_MESSAGES[span.kind] || span.message;
      lines.push(
        `- ${range} [${span.kind} ${span.probability.toFixed(2)}]: ${guidance}`,
      );
    }
  }
  return lines.join("\n");
}

export async function ingestGithubRepo(spec, weights, threshold, options = {}) {
  const { signal, onProgress, filterOpts } = options;
  const startTime = performance.now();
  if (onProgress) {
    onProgress({ phase: "tree", completed: 0, total: 0 });
  }

  const treeData = await discoverRepoTree(spec, signal);
  const { selected, totalEligible, unfilteredCount } = selectTargetEntries(
    treeData.entries,
    spec,
    filterOpts,
  );

  if (selected.length === 0) {
    if (unfilteredCount > 0) {
      throw makeRepoError(
        "no_files",
        `No files in ${spec.owner}/${spec.repo} matched the current filter`,
      );
    }
    const scopeHint = spec.subpath ? ` under ${spec.subpath}` : "";
    throw makeRepoError(
      "no_files",
      `No supported code files found in ${spec.owner}/${spec.repo}${scopeHint}`,
    );
  }

  const rawContents = new Array(selected.length).fill(null);
  let fetchCursor = 0;
  let fetchedCount = 0;

  async function workerLoop() {
    while (fetchCursor < selected.length) {
      const idx = fetchCursor;
      fetchCursor += 1;
      const text = await fetchFileText(
        spec,
        treeData.resolvedRef,
        selected[idx].path,
        signal,
      );
      if (text && text.trim().length > 0 && !text.includes("\0")) {
        rawContents[idx] = text;
      }
      fetchedCount += 1;
      if (onProgress) {
        onProgress({
          phase: "fetch",
          completed: fetchedCount,
          total: selected.length,
        });
      }
    }
  }

  const workerCount = Math.min(FETCH_CONCURRENCY, selected.length);
  await Promise.all(Array.from({ length: workerCount }, () => workerLoop()));

  const workItems = [];
  for (let idx = 0; idx < selected.length; idx += 1) {
    if (rawContents[idx]) {
      workItems.push({ path: selected[idx].path, source: rawContents[idx] });
    }
  }

  if (workItems.length === 0) {
    throw makeRepoError(
      "no_files",
      `No readable source files found in ${spec.owner}/${spec.repo}`,
    );
  }

  const scoredFiles = await scoreFilesInParallel(
    workItems,
    weights,
    threshold,
    signal,
    onProgress,
  );
  const ordered = sortRepoFiles(scoredFiles);
  const summary = summarizeRepoFiles(ordered);

  return {
    spec,
    resolvedRef: treeData.resolvedRef,
    allScoredFiles: ordered,
    files: ordered,
    totalEligible,
    initialScore: summary.score,
    elapsedMs: Math.round(performance.now() - startTime),
    summary,
  };
}
