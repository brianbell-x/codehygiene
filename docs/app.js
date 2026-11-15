import {
  SCORE_NAMES,
  SQUIGGLE_KINDS,
  collectSquiggles,
  computeRlReward,
  loadWeights,
  scoreCode,
} from "./engine.js";
import {
  EXAMPLE_REPOS,
  applyRepoFilters,
  buildRepoFixPrompt,
  formatRepoFileOption,
  groupSquiggleSpans,
  ingestGithubRepo,
  isFullRepoUrl,
  parseRepoSpec,
  rescoreRepoThreshold,
  splitRepoPath,
  updateRepoFileCode,
} from "./repo.js";
import { buildFlatDiff } from "./stage.js";

const SCORE_LABELS = {
  flatness: "Flatness",
  naming: "Naming",
  simplicity: "Simplicity",
  comments: "Comments",
  error_handling: "Error handling",
  layout: "Layout",
};

const ARCHETYPE_LABELS = {
  minimal: "Minimal",
  clean: "Clean",
  enterprise: "Enterprise",
  dense: "Dense",
  slop: "Slop",
};

const KIND_LABELS = {
  deep_nesting: "Deep nesting",
  poor_naming: "Poor naming",
  noisy_comment: "Noisy comment",
  swallowed_error: "Swallowed error",
  dense_line: "Dense line",
  special_casing: "Special casing",
  duplicate_block: "Duplicate block",
  cramped_layout: "Cramped layout",
};

const KIND_GUIDANCE = {
  deep_nesting: "Flatten nested branches with guard clauses",
  poor_naming: "Replace *Factory ceremony, 2-char golf, or magic index",
  noisy_comment: "Restates code or leaves TODO / lint noise",
  swallowed_error: "Ignored exception or silent fallback value",
  dense_line: "Split long line, chained calls, or nested ternary",
  special_casing: "Replace stacked hardcoded branches with a table",
  duplicate_block: "Extract shared helper for repeated statements",
  cramped_layout: "Break unbroken statement wall or align indent",
};

const SHORT_HEADERS = {
  deep_nesting: "Nest",
  poor_naming: "Name",
  noisy_comment: "Comm",
  swallowed_error: "Err",
  dense_line: "Dens",
  special_casing: "Spec",
  duplicate_block: "Clon",
  cramped_layout: "Layo",
};

const state = {
  weights: null,
  presets: [],
  sourceKind: "home",
  activePresetIdx: 0,
  mode: "before",
  isEdited: false,
  editedCode: "",
  threshold: 0.55,
  testsPassing: true,
  xrayMode: "mask",
  beforeResult: null,
  afterResult: null,
  activeResult: null,
  diagSpans: [],
  rovingDiagIdx: -1,
  highlightStart: -1,
  highlightEnd: -1,
  latencySamples: [],
  repoSession: null,
  repoFileIdx: 0,
  rovingRepoIdx: -1,
  hoverRepoKind: "",
  repoLoading: false,
  repoAbort: null,
};

function byId(elemId) {
  const found = document.getElementById(elemId);
  if (!found) {
    throw new Error(`Missing DOM element: ${elemId}`);
  }
  return found;
}

function escapeHtml(rawText) {
  return rawText
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function archiveKbText() {
  if (!state.weights) {
    return "13.0";
  }
  return (state.weights.stats.archiveBytes / 1024).toFixed(1);
}

function recordWarmLatency(elapsedMs) {
  state.latencySamples.push(elapsedMs);
  if (state.latencySamples.length > 7) {
    state.latencySamples.shift();
  }
  const sorted = [...state.latencySamples].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function currentFilterOpts() {
  return {
    mode: byId("filter-mode").value,
    patternText: byId("filter-pattern").value,
    excludeTests: byId("exclude-tests-check").checked,
  };
}

function syncTextareaHeight() {
  const input = byId("code-input");
  const mirror = byId("code-mirror");
  const linesCount = input.value.split("\n").length;
  const lineHeight = window.matchMedia("(pointer: coarse)").matches ? 24 : 22;
  const minPx = Math.max(280, linesCount * lineHeight + 12);
  input.style.height = `${minPx}px`;
  mirror.style.minHeight = `${minPx}px`;
}

function maskTokenClass(maskCode) {
  if (maskCode === 4) {
    return "tok-comm";
  }
  if (maskCode === 3) {
    return "tok-str";
  }
  return "";
}

function renderSegmentedLine(chars, rowMasks, squiggle) {
  if (chars.length === 0) {
    return " ";
  }
  const sqStart = squiggle ? squiggle.col_start - 1 : -1;
  const sqEnd = squiggle ? squiggle.col_end : -1;
  const parts = [];
  let cursor = 0;

  while (cursor < chars.length) {
    const maskCode = rowMasks && cursor < rowMasks.length ? rowMasks[cursor] : 0;
    const tokCls = maskTokenClass(maskCode);
    const inSq = cursor >= sqStart && cursor < sqEnd;
    let endIdx = cursor + 1;

    while (endIdx < chars.length) {
      const nextMask = rowMasks && endIdx < rowMasks.length ? rowMasks[endIdx] : 0;
      const nextInSq = endIdx >= sqStart && endIdx < sqEnd;
      if (maskTokenClass(nextMask) !== tokCls || nextInSq !== inSq) {
        break;
      }
      endIdx += 1;
    }

    const textSlice = escapeHtml(chars.slice(cursor, endIdx).join(""));
    const classes = [];
    if (tokCls) {
      classes.push(tokCls);
    }
    if (inSq) {
      classes.push("squiggle-span");
    }
    const html =
      classes.length > 0
        ? `<span class="${classes.join(" ")}">${textSlice}</span>`
        : textSlice;
    parts.push(html);
    cursor = endIdx;
  }
  return parts.join("");
}

function renderEditorOverlay(rawCode, result) {
  const visualLines = rawCode.split("\n");
  const squiggleByLine = new Map();
  for (const squiggle of result.squiggles) {
    squiggleByLine.set(squiggle.line, squiggle);
  }

  const gutterParts = [];
  const mirrorParts = [];

  for (let rowIdx = 0; rowIdx < visualLines.length; rowIdx += 1) {
    const lineNum = rowIdx + 1;
    const squiggle = squiggleByLine.get(lineNum) ?? null;
    const gutterCls = squiggle ? "gutter-line has-issue" : "gutter-line";
    gutterParts.push(`<div class="${gutterCls}">${lineNum}</div>`);

    const chars = Array.from(visualLines[rowIdx]);
    const hasMaskRow = rowIdx < result.numLines;
    const maskStart = rowIdx * result.width;
    const rowMasks = hasMaskRow
      ? result.cellMasks.subarray(maskStart, maskStart + result.width)
      : null;
    const lineBody = renderSegmentedLine(chars, rowMasks, squiggle);
    const inFocus =
      lineNum >= state.highlightStart && lineNum <= state.highlightEnd;
    const lineCls = inFocus ? "mirror-line active-focus" : "mirror-line";
    mirrorParts.push(`<div class="${lineCls}">${lineBody}</div>`);
  }

  byId("editor-gutter").innerHTML = gutterParts.join("");
  byId("code-mirror").innerHTML = mirrorParts.join("");
}

function formatLineSpan(startLine, endLine) {
  return startLine === endLine ? `Ln ${startLine}` : `Ln ${startLine}–${endLine}`;
}

function buildDiagRowHtml(cls, attrs, kindLabel, metaHtml, guidance) {
  return [
    `<button type="button" class="${cls}" ${attrs}>`,
    `<span class="diag-top">`,
    `<span class="diag-kind-wrap"><span class="diag-dot"></span>`,
    `<span class="diag-kind">${kindLabel}</span></span>`,
    `<span class="diag-meta">${metaHtml}</span></span>`,
    `<span class="diag-hint-copy">${escapeHtml(guidance)}</span>`,
    `</button>`,
  ].join("");
}

function scoreToPipLevel(val) {
  if (val < 0.35) {
    return 1;
  }
  if (val < 0.6) {
    return 2;
  }
  if (val < 0.82) {
    return 3;
  }
  return 4;
}

function buildDimensionsHtml(scores) {
  return SCORE_NAMES.map((name) => {
    const val = scores[name];
    const filled = scoreToPipLevel(val);
    const activeCls = filled <= 2 ? "pip pip-bad" : "pip pip-on";
    const pips = [1, 2, 3, 4]
      .map((idx) => `<span class="${idx <= filled ? activeCls : "pip"}"></span>`)
      .join("");
    const label = SCORE_LABELS[name] ?? name;
    return [
      `<div class="dim-cell" title="${label}: ${val.toFixed(2)} (${filled}/4)">`,
      `<span class="dim-name">${label}</span>`,
      `<span class="pips" aria-label="${filled} of 4">${pips}</span>`,
      `</div>`,
    ].join("");
  }).join("");
}

function renderFileDiagnostics(result) {
  const spans = groupSquiggleSpans(result.squiggles);
  state.diagSpans = spans;
  const count = spans.length;
  byId("issue-stepper").hidden = state.sourceKind !== "repo" || count === 0;

  if (count === 0) {
    state.rovingDiagIdx = -1;
    const lineUnit = result.numLines === 1 ? "line" : "lines";
    byId("diag-count-text").textContent = "Line diagnostics";
    byId("diag-list").innerHTML =
      `<div class="diag-empty">No issues across ${result.numLines} ${lineUnit}</div>`;
    return;
  }

  const noun = count === 1 ? "issue" : "issues";
  const resolvedSuffix =
    state.sourceKind === "example" &&
    state.mode === "diff" &&
    state.afterResult &&
    state.afterResult.squiggles.length === 0
      ? " resolved"
      : "";
  byId("diag-count-text").textContent = `${count} ${noun}${resolvedSuffix}`;

  if (state.rovingDiagIdx >= count) {
    state.rovingDiagIdx = count - 1;
  }

  const rowsHtml = spans.map((span, idx) => {
    const kindLabel = KIND_LABELS[span.kind] ?? span.kind;
    const guidance = KIND_GUIDANCE[span.kind] ?? span.message;
    const lineRef = formatLineSpan(span.startLine, span.endLine);
    const meta = `<span>${span.probability.toFixed(2)}</span><span>${lineRef}</span>`;
    const cls = idx === state.rovingDiagIdx ? "diag-row is-roving" : "diag-row";
    const attrs = `data-start-line="${span.startLine}" data-end-line="${span.endLine}"`;
    return buildDiagRowHtml(cls, attrs, kindLabel, meta, guidance);
  });
  byId("diag-list").innerHTML = rowsHtml.join("");
}

function renderRepoKindList(summary) {
  if (!summary || summary.totalIssues === 0) {
    const count = summary ? summary.fileCount : 0;
    const unit = count === 1 ? "file" : "files";
    byId("repo-diag-heading").textContent = "Issues by kind";
    byId("repo-kind-list").innerHTML =
      `<div class="diag-empty">Zero squiggles across ${count} ${unit}</div>`;
    return;
  }

  const issueNoun = summary.totalIssues === 1 ? "issue" : "issues";
  const fileNoun = summary.flaggedFiles === 1 ? "file" : "files";
  byId("repo-diag-heading").textContent =
    `${summary.totalIssues} ${issueNoun} in ${summary.flaggedFiles} ${fileNoun}`;

  const rowsHtml = summary.kindBreakdown.map((entry) => {
    const kindLabel = KIND_LABELS[entry.kind] ?? entry.kind;
    const guidance = KIND_GUIDANCE[entry.kind] ?? "";
    const unit = entry.fileCount === 1 ? "file" : "files";
    const meta = `<span>${entry.issueCount} in ${entry.fileCount} ${unit}</span>`;
    const attrs =
      `data-repo-kind="${entry.kind}" data-worst-file="${entry.worstFileIdx}" ` +
      `data-start-line="${entry.worstSpan.startLine}" data-end-line="${entry.worstSpan.endLine}"`;
    return buildDiagRowHtml("diag-row", attrs, kindLabel, meta, guidance);
  });
  byId("repo-kind-list").innerHTML = rowsHtml.join("");
}

function renderMaskXray(result) {
  const maxLines = Math.min(result.numLines, 44);
  const symbols = [".", "·", "c", "s", "#"];
  const classes = ["mc-pad", "mc-ind", "mc-code", "mc-str", "mc-comm"];
  const rows = [];

  for (let rowIdx = 0; rowIdx < maxLines; rowIdx += 1) {
    const base = rowIdx * result.width;
    const cols = Math.min(result.width, 72);
    const cells = [];
    for (let colIdx = 0; colIdx < cols; colIdx += 1) {
      const code = result.cellMasks[base + colIdx];
      cells.push(`<span class="${classes[code]}">${symbols[code]}</span>`);
    }
    const ln = String(rowIdx + 1).padStart(2, "0");
    rows.push(
      `<div class="mask-row"><span class="mask-ln">${ln}</span><span>${cells.join("")}</span></div>`,
    );
  }

  return [
    `<p class="xray-note">Block 0 classifies indent (·), code (c), string (s), and comment (#).</p>`,
    `<div class="mask-grid-view">${rows.join("")}</div>`,
  ].join("");
}

function renderProbsXray(result) {
  const maxLines = Math.min(result.numLines, 40);
  const numKinds = SQUIGGLE_KINDS.length;
  const heads = SQUIGGLE_KINDS.map((kind) => `<th>${SHORT_HEADERS[kind]}</th>`).join("");
  const bodyRows = [];

  for (let rowIdx = 0; rowIdx < maxLines; rowIdx += 1) {
    const base = rowIdx * numKinds;
    const cells = [];
    for (let kindIdx = 0; kindIdx < numKinds; kindIdx += 1) {
      const prob = result.lineProbs[base + kindIdx];
      const cls = prob >= state.threshold ? "prob-hot" : "";
      cells.push(`<td class="${cls}">${prob.toFixed(2)}</td>`);
    }
    bodyRows.push(`<tr><td>Ln ${rowIdx + 1}</td>${cells.join("")}</tr>`);
  }

  return [
    `<p class="xray-note">Block 3 per-line probabilities across 8 squiggle heads.</p>`,
    `<div class="mask-grid-view">`,
    `<table class="prob-table"><thead><tr><th>Line</th>${heads}</tr></thead>`,
    `<tbody>${bodyRows.join("")}</tbody></table></div>`,
  ].join("");
}

function syncXrayPanel() {
  const details = byId("xray-details");
  if (!state.activeResult) {
    return;
  }
  const reward = computeRlReward(state.activeResult, state.testsPassing, 0.25, 0.5);
  byId("rl-reward-val").textContent = reward.toFixed(3);
  if (!details.open) {
    return;
  }
  byId("xray-body").innerHTML =
    state.xrayMode === "mask"
      ? renderMaskXray(state.activeResult)
      : renderProbsXray(state.activeResult);
}

function formatDiffParts(parts, wordCls) {
  return parts
    .map((part) => {
      const safe = escapeHtml(part.text);
      return part.changed ? `<span class="${wordCls}">${safe}</span>` : safe;
    })
    .join("");
}

function renderFlatDiffView() {
  const preset = state.presets[state.activePresetIdx];
  if (!preset) {
    return;
  }
  const sourceCode = state.isEdited ? state.editedCode : preset.before;
  const diffRows = buildFlatDiff(sourceCode, preset.after);
  const container = byId("diff-body");

  if (diffRows.length === 0) {
    container.innerHTML = `<div class="diff-empty">No differences vs. After</div>`;
    return;
  }

  const html = diffRows.map((row) => {
    if (row.type === "gap") {
      return `<div class="diff-gap-row">⋯ ${row.hiddenCount} unchanged lines</div>`;
    }
    const oldNum = row.oldLine === null ? "" : String(row.oldLine);
    const newNum = row.newLine === null ? "" : String(row.newLine);

    if (row.type === "del") {
      const body = row.parts
        ? formatDiffParts(row.parts, "diff-word-del")
        : escapeHtml(row.text);
      return [
        `<div class="diff-row diff-row-del">`,
        `<span class="diff-num">${oldNum}</span><span class="diff-num"></span>`,
        `<span class="diff-sigil diff-sigil-del">−</span>`,
        `<span class="diff-code">${body}</span></div>`,
      ].join("");
    }

    if (row.type === "add") {
      const body = row.parts
        ? formatDiffParts(row.parts, "diff-word-add")
        : escapeHtml(row.text);
      return [
        `<div class="diff-row diff-row-add">`,
        `<span class="diff-num"></span><span class="diff-num">${newNum}</span>`,
        `<span class="diff-sigil diff-sigil-add">+</span>`,
        `<span class="diff-code">${body}</span></div>`,
      ].join("");
    }

    return [
      `<div class="diff-row">`,
      `<span class="diff-num">${oldNum}</span><span class="diff-num">${newNum}</span>`,
      `<span class="diff-sigil"> </span>`,
      `<span class="diff-code">${escapeHtml(row.text)}</span></div>`,
    ].join("");
  });

  container.innerHTML = html.join("");
}

function buildRepoFileRowHtml(item, idx) {
  const { baseName, dirPrefix } = splitRepoPath(item.path);
  const issueCount = item.spans.length;
  const classes = ["repo-row"];
  if (issueCount > 0) {
    classes.push("has-issue");
  }
  if (idx === state.repoFileIdx) {
    classes.push("is-active");
  }
  if (idx === state.rovingRepoIdx) {
    classes.push("is-roving");
  }
  if (state.hoverRepoKind && item.uniqueKinds.includes(state.hoverRepoKind)) {
    classes.push("is-kind-match");
  }

  const kindsText = item.uniqueKinds
    .slice(0, 2)
    .map((kind) => KIND_LABELS[kind] ?? kind)
    .join(" · ");
  const issueNoun = issueCount === 1 ? "issue" : "issues";
  const issueLabel = issueCount > 0 ? `${issueCount} ${issueNoun}` : "";
  const dirSpan = dirPrefix
    ? `<span class="repo-file-dir">${escapeHtml(dirPrefix)}</span>`
    : "";
  const editedSpan = item.isEdited
    ? `<span class="repo-file-edited">edited</span>`
    : "";
  const kindsSpan = kindsText
    ? `<span class="repo-row-kinds">${escapeHtml(kindsText)}</span>`
    : "";

  return [
    `<button type="button" class="${classes.join(" ")}" data-repo-file-idx="${idx}">`,
    `<span class="repo-row-gutter">${item.result.score.toFixed(2)}</span>`,
    `<span class="repo-row-path">`,
    `<span class="repo-file-base">${escapeHtml(baseName)}</span>`,
    `${dirSpan}${editedSpan}</span>`,
    `<span class="repo-row-right">${kindsSpan}`,
    `<span class="repo-row-issues">${issueLabel}</span></span>`,
    `</button>`,
  ].join("");
}

function renderRepoSummaryAndFiles() {
  const session = state.repoSession;
  if (!session) {
    return;
  }

  const extLink = byId("repo-ext-link");
  extLink.href = `https://github.com/${session.spec.slug}`;
  extLink.textContent = `${session.spec.slug} ↗`;

  const summary = session.summary;
  const heroScoreEl = byId("repo-hero-score");

  if (!summary) {
    heroScoreEl.textContent = "—";
    heroScoreEl.classList.remove("is-low");
    byId("repo-meta-summary").textContent = "0 matching files";
    byId("repo-subscores-body").innerHTML = "";
    renderRepoKindList(null);
    byId("repo-files-meta").textContent = "0 files";
    byId("repo-file-list").innerHTML =
      `<div class="repo-empty-filter">No files match the current filter.</div>`;
    return;
  }

  const scoreChanged =
    summary.hasEdits && Math.abs(summary.score - session.initialScore) >= 0.005;
  if (scoreChanged) {
    heroScoreEl.innerHTML =
      `${session.initialScore.toFixed(2)}<span class="score-arrow">→</span>` +
      `${summary.score.toFixed(2)}`;
  } else {
    heroScoreEl.textContent = summary.score.toFixed(2);
  }
  heroScoreEl.classList.toggle("is-low", summary.score < 0.5);

  const archLabel = ARCHETYPE_LABELS[summary.archetype] ?? summary.archetype;
  const fileUnit = summary.fileCount === 1 ? "file" : "files";
  byId("repo-meta-summary").textContent =
    `${archLabel} · ${summary.fileCount} ${fileUnit} · ${summary.numLines.toLocaleString()} lines`;
  byId("repo-subscores-body").innerHTML = buildDimensionsHtml(summary.scores);
  renderRepoKindList(summary);

  const cleanCount = summary.fileCount - summary.flaggedFiles;
  byId("repo-files-meta").textContent =
    `${summary.flaggedFiles} flagged · ${cleanCount} clean`;

  const flaggedIndices = [];
  const cleanIndices = [];
  for (let idx = 0; idx < session.files.length; idx += 1) {
    if (session.files[idx].spans.length > 0) {
      flaggedIndices.push(idx);
    } else {
      cleanIndices.push(idx);
    }
  }

  const htmlParts = [];
  for (const idx of flaggedIndices) {
    htmlParts.push(buildRepoFileRowHtml(session.files[idx], idx));
  }
  if (flaggedIndices.length > 0 && cleanIndices.length > 0) {
    const cleanNoun = cleanIndices.length === 1 ? "file" : "files";
    htmlParts.push(
      `<div class="repo-gap-row">⋯ ${cleanIndices.length} ${cleanNoun} with no issues</div>`,
    );
  }
  for (const idx of cleanIndices) {
    htmlParts.push(buildRepoFileRowHtml(session.files[idx], idx));
  }
  byId("repo-file-list").innerHTML = htmlParts.join("");
}

function syncExampleRepoChipState() {
  const activeSlug =
    state.sourceKind === "repo" && state.repoSession
      ? state.repoSession.spec.slug.toLowerCase()
      : "";
  for (const chip of document.querySelectorAll(".example-chip")) {
    const slug = (chip.getAttribute("data-repo-slug") || "").toLowerCase();
    chip.classList.toggle("is-active", Boolean(activeSlug) && slug === activeSlug);
  }
}

function syncSelectDropdown() {
  const select = byId("example-select");
  if (state.sourceKind === "repo" && state.repoSession) {
    select.innerHTML = state.repoSession.files
      .map((item, idx) => {
        const label = escapeHtml(formatRepoFileOption(item.path));
        return `<option value="${idx}">${label}</option>`;
      })
      .join("");
    select.value = String(state.repoFileIdx);
    return;
  }

  select.innerHTML = state.presets
    .map((preset, idx) => `<option value="${idx}">${escapeHtml(preset.filename)}</option>`)
    .join("");
  select.value = String(state.activePresetIdx);
}

function syncPageLayout() {
  const isHome = state.sourceKind === "home" && !state.repoLoading;
  const isRepo = state.sourceKind === "repo" && Boolean(state.repoSession);
  const hasRepoFiles = isRepo && state.repoSession.files.length > 0;
  const showWorkbench =
    state.sourceKind === "example" || (isRepo && hasRepoFiles);

  byId("hero-section").hidden = !isHome;
  byId("helper-line").hidden = isRepo;
  byId("toggle-snippet-btn").textContent =
    state.sourceKind === "example"
      ? "Hide snippet workbench"
      : "Paste a snippet or compare Before → After examples";
  byId("repo-results-card").hidden = !isRepo;
  byId("workbench-card").hidden = !showWorkbench;
  byId("example-tabs").hidden = state.sourceKind !== "example";
  byId("snippet-score-block").hidden = state.sourceKind !== "example";

  const showDiff = state.sourceKind === "example" && state.mode === "diff";
  byId("editor-view").hidden = showDiff;
  byId("diff-view").hidden = !showDiff;

  syncExampleRepoChipState();
}

function syncFileScoreBadge(result) {
  const badge = byId("active-file-pill");
  badge.textContent = result.score.toFixed(2);
  badge.classList.toggle("is-low", result.score < 0.5);
}

function loadCodeIntoEditor(sourceCode, result) {
  const input = byId("code-input");
  if (input.value !== sourceCode) {
    input.value = sourceCode;
  }
  syncTextareaHeight();
  renderEditorOverlay(sourceCode, result);
  syncFileScoreBadge(result);
  renderFileDiagnostics(result);
  syncXrayPanel();
}

function syncSnippetInspector() {
  if (!state.beforeResult || !state.afterResult) {
    return;
  }
  const heroEl = byId("hero-score");
  const archEl = byId("archetype-summary");

  if (state.mode === "diff") {
    const bScore = state.beforeResult.score.toFixed(2);
    const aScore = state.afterResult.score.toFixed(2);
    heroEl.innerHTML = `${bScore}<span class="score-arrow">→</span>${aScore}`;
    heroEl.classList.remove("is-low");
    const bArch = ARCHETYPE_LABELS[state.beforeResult.archetype];
    const aArch = ARCHETYPE_LABELS[state.afterResult.archetype];
    archEl.textContent = `${bArch} → ${aArch}`;
    byId("subscores-body").innerHTML = buildDimensionsHtml(state.afterResult.scores);
    syncFileScoreBadge(state.afterResult);
    renderFileDiagnostics(state.beforeResult);
    return;
  }

  const active = state.mode === "after" ? state.afterResult : state.beforeResult;
  heroEl.textContent = active.score.toFixed(2);
  heroEl.classList.toggle("is-low", active.score < 0.5);
  archEl.textContent = ARCHETYPE_LABELS[active.archetype] ?? active.archetype;
  byId("subscores-body").innerHTML = buildDimensionsHtml(active.scores);

  byId("before-label").textContent = state.isEdited ? "Edited" : "Before";
  byId("before-score-pill").textContent = state.beforeResult.score.toFixed(2);
  byId("after-score-pill").textContent = state.afterResult.score.toFixed(2);
  byId("reset-btn").hidden = !state.isEdited;

  for (const modeName of ["before", "after", "diff"]) {
    byId(`tab-${modeName}`).setAttribute(
      "aria-selected",
      String(state.mode === modeName),
    );
  }
}

function applyExampleState() {
  const preset = state.presets[state.activePresetIdx];
  if (!preset || !state.weights) {
    return;
  }

  const beforeSrc = state.isEdited ? state.editedCode : preset.before;
  state.beforeResult = scoreCode(beforeSrc, state.weights, state.threshold);
  state.afterResult = scoreCode(preset.after, state.weights, state.threshold);
  state.activeResult =
    state.mode === "after" ? state.afterResult : state.beforeResult;

  const medianMs = recordWarmLatency(state.activeResult.elapsedMs);
  byId("model-meta-text").textContent =
    `${archiveKbText()} KiB · 1.58-bit · ${medianMs.toFixed(1)} ms`;

  syncSelectDropdown();
  syncPageLayout();
  syncSnippetInspector();

  if (state.mode === "diff") {
    renderFlatDiffView();
    syncXrayPanel();
  } else {
    const codeToShow = state.mode === "after" ? preset.after : beforeSrc;
    loadCodeIntoEditor(codeToShow, state.activeResult);
  }
}

function loadActiveRepoFileIntoWorkbench() {
  const session = state.repoSession;
  if (!session || session.files.length === 0) {
    syncPageLayout();
    return;
  }
  if (state.repoFileIdx >= session.files.length) {
    state.repoFileIdx = 0;
  }

  const fileItem = session.files[state.repoFileIdx];
  state.activeResult = fileItem.result;
  byId("reset-btn").hidden = !fileItem.isEdited;
  syncSelectDropdown();
  syncPageLayout();
  loadCodeIntoEditor(fileItem.currentCode, fileItem.result);
}

function syncUrlState(pushHistoryEntry) {
  const url = new URL(window.location.href);
  if (state.sourceKind === "repo" && state.repoSession) {
    url.searchParams.set("repo", state.repoSession.spec.normalizedPath);
    const activeFile = state.repoSession.files[state.repoFileIdx];
    if (activeFile && state.repoFileIdx > 0) {
      url.searchParams.set("file", activeFile.path);
    } else {
      url.searchParams.delete("file");
    }
    const scoreStr = state.repoSession.summary
      ? state.repoSession.summary.score.toFixed(2)
      : "—";
    document.title = `${state.repoSession.spec.slug} ${scoreStr} · codehygiene`;
  } else {
    url.searchParams.delete("repo");
    url.searchParams.delete("file");
    document.title = "codehygiene — Score any codebase";
  }

  if (pushHistoryEntry) {
    window.history.pushState(null, "", url);
  } else {
    window.history.replaceState(null, "", url);
  }
}

function setRepoError(message) {
  const errEl = byId("repo-error");
  const card = document.querySelector(".ingest-card");
  if (!message) {
    errEl.hidden = true;
    errEl.textContent = "";
    card.classList.remove("has-error");
    return;
  }
  errEl.hidden = false;
  errEl.textContent = message;
  card.classList.add("has-error");
}

function setLoadingUi(isLoading, buttonLabel, progressRatio) {
  state.repoLoading = isLoading;
  const submitBtn = byId("repo-submit-btn");
  const progressWrap = byId("topbar-progress");
  const progressFill = byId("topbar-progress-fill");

  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? buttonLabel : "Score";
  progressWrap.hidden = !isLoading;
  if (isLoading) {
    const pct = Math.max(8, Math.min(98, Math.round(progressRatio * 100)));
    progressFill.style.width = `${pct}%`;
  }
  byId("repo-clear-btn").hidden = byId("repo-input").value.trim().length === 0;
}

function scrollEditorToLine(startLine, endLine) {
  state.highlightStart = startLine;
  state.highlightEnd = endLine;
  if (state.activeResult) {
    renderEditorOverlay(byId("code-input").value, state.activeResult);
  }
  const well = byId("editor-view");
  const lineHeight = window.matchMedia("(pointer: coarse)").matches ? 24 : 22;
  const targetTop = Math.max(0, (startLine - 4) * lineHeight);
  well.scrollTo({ top: targetTop, behavior: "smooth" });
}

function selectRepoFile(fileIdx, focusSpan = null, scrollWorkbench = false) {
  const session = state.repoSession;
  if (!session || fileIdx < 0 || fileIdx >= session.files.length) {
    return;
  }
  state.repoFileIdx = fileIdx;
  state.rovingDiagIdx = -1;
  state.highlightStart = focusSpan ? focusSpan.startLine : -1;
  state.highlightEnd = focusSpan ? focusSpan.endLine : -1;

  renderRepoSummaryAndFiles();
  loadActiveRepoFileIntoWorkbench();
  syncUrlState(false);

  if (focusSpan) {
    scrollEditorToLine(focusSpan.startLine, focusSpan.endLine);
  }
  if (scrollWorkbench) {
    byId("workbench-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function submitRepoLoad(rawInput, requestedFilePath = "", pushHistory = true) {
  const spec = parseRepoSpec(rawInput);
  if (!spec) {
    setRepoError("Enter a valid GitHub repository URL or owner/repo (for example: karpathy/micrograd).");
    return;
  }

  if (state.repoAbort) {
    state.repoAbort.abort();
  }
  const abortCtrl = new AbortController();
  state.repoAbort = abortCtrl;

  setRepoError("");
  byId("repo-input").value = `https://github.com/${spec.normalizedPath}`;
  setLoadingUi(true, "Tree…", 0.12);

  try {
    const session = await ingestGithubRepo(spec, state.weights, state.threshold, {
      signal: abortCtrl.signal,
      filterOpts: currentFilterOpts(),
      onProgress(prog) {
        if (prog.phase === "tree") {
          setLoadingUi(true, "Tree…", 0.14);
        } else if (prog.phase === "fetch" && prog.total > 0) {
          const ratio = 0.15 + 0.4 * (prog.completed / prog.total);
          setLoadingUi(true, `Fetch ${prog.completed}/${prog.total}`, ratio);
        } else if (prog.phase === "score" && prog.total > 0) {
          const ratio = 0.55 + 0.43 * (prog.completed / prog.total);
          setLoadingUi(true, `Score ${prog.completed}/${prog.total}`, ratio);
        }
      },
    });

    if (abortCtrl.signal.aborted) {
      return;
    }

    document.documentElement.removeAttribute("data-boot-repo");
    state.repoSession = session;
    state.sourceKind = "repo";
    state.hoverRepoKind = "";
    state.highlightStart = -1;
    state.highlightEnd = -1;

    const targetPath = requestedFilePath || spec.blobPath;
    const matchIdx = targetPath
      ? session.files.findIndex((item) => item.path === targetPath)
      : 0;
    state.repoFileIdx = matchIdx >= 0 ? matchIdx : 0;

    setLoadingUi(false, "Score", 1);
    const count = session.files.length;
    const scoreStr = session.summary.score.toFixed(2);
    byId("model-meta-text").textContent =
      `${archiveKbText()} KiB · 1.58-bit · ${count} files in ${session.elapsedMs} ms`;
    byId("repo-live-region").textContent =
      `Scored ${session.spec.slug}: ${scoreStr} across ${count} files.`;

    renderRepoSummaryAndFiles();
    loadActiveRepoFileIntoWorkbench();
    syncUrlState(pushHistory);
  } catch (err) {
    document.documentElement.removeAttribute("data-boot-repo");
    if (err && err.name === "AbortError") {
      return;
    }
    setLoadingUi(false, "Score", 0);
    setRepoError(err.message || "Failed to score repository.");
    syncPageLayout();
  }
}

function clearRepoSession() {
  if (state.repoAbort) {
    state.repoAbort.abort();
    state.repoAbort = null;
  }
  state.repoSession = null;
  state.repoLoading = false;
  state.sourceKind = "home";
  state.highlightStart = -1;
  state.highlightEnd = -1;
  setRepoError("");
  byId("repo-input").value = "";
  setLoadingUi(false, "Score", 0);
  syncPageLayout();
  syncUrlState(true);
  byId("repo-input").focus();
}

function handleEditorInput() {
  const code = byId("code-input").value;
  state.highlightStart = -1;
  state.highlightEnd = -1;

  if (state.sourceKind === "repo" && state.repoSession) {
    const updated = updateRepoFileCode(
      state.repoSession,
      state.repoFileIdx,
      code,
      state.weights,
      state.threshold,
    );
    state.activeResult = updated;
    byId("reset-btn").hidden = !state.repoSession.files[state.repoFileIdx].isEdited;
    syncTextareaHeight();
    renderEditorOverlay(code, updated);
    syncFileScoreBadge(updated);
    renderFileDiagnostics(updated);
    renderRepoSummaryAndFiles();
    syncXrayPanel();
    return;
  }

  state.mode = "before";
  state.isEdited = true;
  state.editedCode = code;
  applyExampleState();
}

function handleResetClick() {
  state.highlightStart = -1;
  state.highlightEnd = -1;

  if (state.sourceKind === "repo" && state.repoSession) {
    const item = state.repoSession.files[state.repoFileIdx];
    updateRepoFileCode(
      state.repoSession,
      state.repoFileIdx,
      item.originalCode,
      state.weights,
      state.threshold,
    );
    renderRepoSummaryAndFiles();
    loadActiveRepoFileIntoWorkbench();
    return;
  }

  state.isEdited = false;
  state.editedCode = "";
  applyExampleState();
}

function stepRovingDiagnostic(delta) {
  if (state.diagSpans.length === 0) {
    return;
  }
  const total = state.diagSpans.length;
  const next =
    state.rovingDiagIdx < 0
      ? 0
      : (state.rovingDiagIdx + delta + total) % total;
  state.rovingDiagIdx = next;
  const span = state.diagSpans[next];
  if (state.activeResult) {
    renderFileDiagnostics(state.activeResult);
  }
  scrollEditorToLine(span.startLine, span.endLine);
}

async function copyTextWithFeedback(buttonEl, textToCopy) {
  try {
    await navigator.clipboard.writeText(textToCopy);
    buttonEl.classList.add("is-copied");
    window.setTimeout(() => {
      buttonEl.classList.remove("is-copied");
    }, 1200);
  } catch (_) {
    // Clipboard access unavailable
  }
}

function refreshActiveRepoFilters() {
  if (state.sourceKind !== "repo" || !state.repoSession) {
    return;
  }
  const prevPath = state.repoSession.files[state.repoFileIdx]?.path || "";
  applyRepoFilters(state.repoSession, currentFilterOpts());
  const nextIdx = prevPath
    ? state.repoSession.files.findIndex((item) => item.path === prevPath)
    : 0;
  state.repoFileIdx = nextIdx >= 0 ? nextIdx : 0;
  renderRepoSummaryAndFiles();
  loadActiveRepoFileIntoWorkbench();
}

function renderExampleRepoChips() {
  const html = EXAMPLE_REPOS.map((repo) => {
    const attrs = `type="button" class="example-chip" data-repo-slug="${repo.slug}"`;
    return `<button ${attrs}>${escapeHtml(repo.label)}</button>`;
  }).join("");
  byId("example-repo-chips").innerHTML = html;
}

function bindRepoFormEvents() {
  const repoInput = byId("repo-input");

  byId("repo-form").addEventListener("submit", (event) => {
    event.preventDefault();
    submitRepoLoad(repoInput.value);
  });

  repoInput.addEventListener("input", () => {
    setRepoError("");
    byId("repo-clear-btn").hidden = repoInput.value.trim().length === 0;
  });

  repoInput.addEventListener("paste", (event) => {
    const pasted = event.clipboardData?.getData("text") || "";
    if (isFullRepoUrl(pasted)) {
      window.setTimeout(() => {
        submitRepoLoad(repoInput.value);
      }, 0);
    }
  });

  byId("repo-clear-btn").addEventListener("click", () => {
    clearRepoSession();
  });

  byId("example-repo-chips").addEventListener("click", (event) => {
    const chip = event.target.closest("[data-repo-slug]");
    if (!chip) {
      return;
    }
    const slug = chip.getAttribute("data-repo-slug");
    if (slug) {
      submitRepoLoad(slug);
    }
  });

  let filterTimer = 0;
  byId("filter-mode").addEventListener("change", refreshActiveRepoFilters);
  byId("exclude-tests-check").addEventListener("change", refreshActiveRepoFilters);
  byId("filter-pattern").addEventListener("input", () => {
    window.clearTimeout(filterTimer);
    filterTimer = window.setTimeout(refreshActiveRepoFilters, 120);
  });
}

function bindWorkbenchEvents() {
  byId("toggle-snippet-btn").addEventListener("click", () => {
    if (state.sourceKind === "example") {
      state.sourceKind = "home";
      syncPageLayout();
      return;
    }
    state.sourceKind = "example";
    state.mode = "before";
    applyExampleState();
  });

  byId("example-select").addEventListener("change", (event) => {
    const idx = Number.parseInt(event.target.value, 10);
    if (state.sourceKind === "repo") {
      selectRepoFile(idx);
      return;
    }
    state.activePresetIdx = idx;
    state.mode = "before";
    state.isEdited = false;
    state.editedCode = "";
    state.highlightStart = -1;
    state.highlightEnd = -1;
    applyExampleState();
  });

  byId("example-tabs").addEventListener("click", (event) => {
    const btn = event.target.closest("[data-mode]");
    if (!btn) {
      return;
    }
    state.mode = btn.getAttribute("data-mode");
    state.highlightStart = -1;
    state.highlightEnd = -1;
    applyExampleState();
  });

  byId("code-input").addEventListener("input", handleEditorInput);
  byId("reset-btn").addEventListener("click", handleResetClick);
  byId("prev-issue-btn").addEventListener("click", () => stepRovingDiagnostic(-1));
  byId("next-issue-btn").addEventListener("click", () => stepRovingDiagnostic(1));

  byId("diag-list").addEventListener("click", (event) => {
    const row = event.target.closest("[data-start-line]");
    if (!row) {
      return;
    }
    const startLine = Number.parseInt(row.getAttribute("data-start-line"), 10);
    const endLine = Number.parseInt(row.getAttribute("data-end-line"), 10);
    if (state.sourceKind === "example" && state.mode === "diff") {
      state.mode = "before";
      applyExampleState();
    }
    scrollEditorToLine(startLine, endLine);
  });
}

function bindRepoResultsEvents() {
  byId("repo-file-list").addEventListener("click", (event) => {
    const row = event.target.closest("[data-repo-file-idx]");
    if (!row) {
      return;
    }
    const idx = Number.parseInt(row.getAttribute("data-repo-file-idx"), 10);
    selectRepoFile(idx, null, true);
  });

  const kindList = byId("repo-kind-list");
  kindList.addEventListener("mouseover", (event) => {
    const row = event.target.closest("[data-repo-kind]");
    const nextKind = row ? row.getAttribute("data-repo-kind") : "";
    if (nextKind !== state.hoverRepoKind) {
      state.hoverRepoKind = nextKind;
      renderRepoSummaryAndFiles();
    }
  });

  kindList.addEventListener("mouseleave", () => {
    if (state.hoverRepoKind) {
      state.hoverRepoKind = "";
      renderRepoSummaryAndFiles();
    }
  });

  kindList.addEventListener("click", (event) => {
    const row = event.target.closest("[data-worst-file]");
    if (!row) {
      return;
    }
    const fileIdx = Number.parseInt(row.getAttribute("data-worst-file"), 10);
    const startLine = Number.parseInt(row.getAttribute("data-start-line"), 10);
    const endLine = Number.parseInt(row.getAttribute("data-end-line"), 10);
    selectRepoFile(fileIdx, { startLine, endLine }, true);
  });

  byId("copy-prompt-btn").addEventListener("click", (event) => {
    if (!state.repoSession) {
      return;
    }
    const promptText = buildRepoFixPrompt(state.repoSession);
    copyTextWithFeedback(event.currentTarget, promptText);
  });

  byId("repo-share-btn").addEventListener("click", (event) => {
    syncUrlState(false);
    copyTextWithFeedback(event.currentTarget, window.location.href);
  });
}

function bindGlobalControls() {
  byId("theme-toggle-btn").addEventListener("click", () => {
    const root = document.documentElement;
    const nextTheme = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", nextTheme);
    try {
      localStorage.setItem("ch-theme", nextTheme);
    } catch (_) {
      // Storage unavailable
    }
  });

  byId("threshold-range").addEventListener("input", (event) => {
    state.threshold = Number.parseFloat(event.target.value);
    byId("threshold-val").textContent = state.threshold.toFixed(2);
    if (state.sourceKind === "repo" && state.repoSession) {
      rescoreRepoThreshold(state.repoSession, state.threshold);
      renderRepoSummaryAndFiles();
      loadActiveRepoFileIntoWorkbench();
    } else if (state.sourceKind === "example") {
      applyExampleState();
    }
  });

  byId("tests-checkbox").addEventListener("change", (event) => {
    state.testsPassing = event.target.checked;
    syncXrayPanel();
  });

  byId("xray-details").addEventListener("toggle", () => {
    syncXrayPanel();
  });

  document.querySelector(".xray-tabs").addEventListener("click", (event) => {
    const btn = event.target.closest("[data-xray]");
    if (!btn) {
      return;
    }
    state.xrayMode = btn.getAttribute("data-xray");
    byId("tab-mask").setAttribute("aria-selected", String(state.xrayMode === "mask"));
    byId("tab-probs").setAttribute("aria-selected", String(state.xrayMode === "probs"));
    syncXrayPanel();
  });

  byId("copy-install-btn").addEventListener("click", (event) => {
    copyTextWithFeedback(
      event.currentTarget,
      "npx skills add brianbell-x/codehygiene -g",
    );
  });

  document.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      return;
    }
    if (event.key === "j" || event.key === "J") {
      event.preventDefault();
      stepRovingDiagnostic(1);
    } else if (event.key === "k" || event.key === "K") {
      event.preventDefault();
      stepRovingDiagnostic(-1);
    }
  });

  window.addEventListener("popstate", () => {
    const params = new URLSearchParams(window.location.search);
    const repoParam = params.get("repo");
    const fileParam = params.get("file") || "";
    if (repoParam) {
      submitRepoLoad(repoParam, fileParam, false);
    } else {
      clearRepoSession();
    }
  });
}

function extractPresetList(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (payload && Array.isArray(payload.examples)) {
    return payload.examples;
  }
  return [];
}

async function initApp() {
  renderExampleRepoChips();
  bindRepoFormEvents();
  bindWorkbenchEvents();
  bindRepoResultsEvents();
  bindGlobalControls();

  const [weights, presetsResp] = await Promise.all([
    loadWeights("./weights.npz"),
    fetch("./presets.json").then((resp) => resp.json()),
  ]);

  state.weights = weights;
  state.presets = extractPresetList(presetsResp);
  byId("model-meta-text").textContent =
    `${archiveKbText()} KiB · 1.58-bit ternary`;

  const params = new URLSearchParams(window.location.search);
  const initialRepo = params.get("repo");
  const initialFile = params.get("file") ?? "";

  if (initialRepo) {
    await submitRepoLoad(initialRepo, initialFile, false);
  } else {
    syncPageLayout();
  }
}

initApp();
