const MAX_COLS = 128;
const MAX_REL_DIST = 32;
const SKETCH_DIM = 40;
const NUM_CHANNELS_HC = 52;

const HC_ACT = 0;
const HC_COMM = 1;
const HC_CODE = 2;
const HC_IND = 5;
const HC_STR = 48;

const POW3 = [1, 3, 9, 27, 81];
const SCORE_LEVELS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];

export const SCORE_NAMES = [
  "flatness",
  "naming",
  "simplicity",
  "comments",
  "error_handling",
  "layout",
];

export const ARCHETYPE_NAMES = [
  "minimal",
  "clean",
  "enterprise",
  "dense",
  "slop",
];

const SQUIGGLE_SPEC = [
  ["deep_nesting", "Deep nesting; flatten with guard clauses"],
  ["poor_naming", "Over-abstracted (*Factory), cryptic short name, or magic index"],
  ["noisy_comment", "Comment restates code or leaves TODO/lint noise"],
  ["swallowed_error", "Ignored error or silent fallback value"],
  ["dense_line", "Long line, chained calls, or nested ternary"],
  ["special_casing", "Stacked hardcoded special-case branches"],
  ["duplicate_block", "Duplicated logic block"],
  ["cramped_layout", "Unbroken statement wall or inconsistent indent"],
];

export const SQUIGGLE_KINDS = SQUIGGLE_SPEC.map((entry) => entry[0]);
export const SQUIGGLE_MESSAGES = Object.fromEntries(SQUIGGLE_SPEC);

function findZipEndOffset(bytes, view) {
  const minOffset = Math.max(0, bytes.length - 65557);
  for (let pos = bytes.length - 22; pos >= minOffset; pos -= 1) {
    if (view.getUint32(pos, true) === 0x06054b50) {
      return pos;
    }
  }
  throw new Error("Invalid weights.npz archive: missing central directory");
}

async function inflateRawBytes(compressed) {
  const stream = new Blob([compressed])
    .stream()
    .pipeThrough(new DecompressionStream("deflate-raw"));
  const buffer = await new Response(stream).arrayBuffer();
  return new Uint8Array(buffer);
}

function parseNpyArray(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = bytes[6];
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const dataOffset = (major === 1 ? 10 : 12) + headerLen;
  const headerText = new TextDecoder("latin1").decode(bytes.subarray(8, dataOffset));
  const descrMatch = headerText.match(/'descr':\s*'([^']+)'/);
  const descr = descrMatch ? descrMatch[1] : "";
  const body = bytes.subarray(dataOffset);

  if (descr.endsWith("u1")) {
    return new Uint8Array(body);
  }
  if (descr === "<f4") {
    const aligned = body.slice();
    return new Float32Array(aligned.buffer);
  }
  if (descr.startsWith("<U")) {
    const aligned = body.slice();
    const codePoints = new Uint32Array(aligned.buffer);
    const chars = new Array(codePoints.length);
    for (let idx = 0; idx < codePoints.length; idx += 1) {
      chars[idx] = String.fromCodePoint(codePoints[idx]);
    }
    return chars.join("");
  }
  throw new Error(`Unsupported .npy descriptor: ${descr}`);
}

export async function unpackNpzBuffer(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const view = new DataView(arrayBuffer);
  const eocdPos = findZipEndOffset(bytes, view);
  const entryCount = view.getUint16(eocdPos + 10, true);
  let dirPos = view.getUint32(eocdPos + 16, true);
  const entries = {};
  const decoder = new TextDecoder("utf-8");

  for (let idx = 0; idx < entryCount; idx += 1) {
    const method = view.getUint16(dirPos + 10, true);
    const compSize = view.getUint32(dirPos + 20, true);
    const nameLen = view.getUint16(dirPos + 28, true);
    const extraLen = view.getUint16(dirPos + 30, true);
    const commentLen = view.getUint16(dirPos + 32, true);
    const localPos = view.getUint32(dirPos + 42, true);
    const entryName = decoder.decode(bytes.subarray(dirPos + 46, dirPos + 46 + nameLen));

    const localNameLen = view.getUint16(localPos + 26, true);
    const localExtraLen = view.getUint16(localPos + 28, true);
    const dataStart = localPos + 30 + localNameLen + localExtraLen;
    const rawSlice = bytes.subarray(dataStart, dataStart + compSize);
    const unzipped = method === 8 ? await inflateRawBytes(rawSlice) : rawSlice;
    const stem = entryName.replace(/\.npy$/, "");
    entries[stem] = parseNpyArray(unzipped);
    dirPos += 46 + nameLen + extraLen + commentLen;
  }

  return unpackTernaryArchive(entries.trits, entries.vecs, entries.meta);
}

function buildSparse2D(dense, rows, cols) {
  const rowOffsets = new Int32Array(rows + 1);
  const activeList = [];
  let nonZeroCount = 0;
  for (let idx = 0; idx < dense.length; idx += 1) {
    if (dense[idx] !== 0) {
      nonZeroCount += 1;
    }
  }

  const colIndices = new Int32Array(nonZeroCount);
  const values = new Float32Array(nonZeroCount);
  let cursor = 0;
  for (let row = 0; row < rows; row += 1) {
    rowOffsets[row] = cursor;
    const rowBase = row * cols;
    for (let col = 0; col < cols; col += 1) {
      const val = dense[rowBase + col];
      if (val !== 0) {
        colIndices[cursor] = col;
        values[cursor] = val;
        cursor += 1;
      }
    }
    if (cursor > rowOffsets[row]) {
      activeList.push(row);
    }
  }
  rowOffsets[rows] = cursor;
  const activeRows = Int32Array.from(activeList);
  return { data: dense, rows, cols, rowOffsets, colIndices, values, activeRows, nonZeroCount };
}

function buildSparseConv(dense, kSize, inChannels, outChannels) {
  const taps = [];
  for (let tap = 0; tap < kSize; tap += 1) {
    const tapBase = tap * inChannels * outChannels;
    const slice = dense.subarray(tapBase, tapBase + inChannels * outChannels);
    taps.push(buildSparse2D(slice, inChannels, outChannels));
  }
  return { data: dense, kSize, inChannels, outChannels, taps };
}

export function unpackTernaryArchive(tritBuf, vecBuf, metaStr) {
  const [tritPart, vecPart] = metaStr.split("|");
  const vecs = {};
  let vecOffset = 0;

  for (const entry of vecPart.split(" ; ")) {
    const [name, dimsText, countText] = entry.split(":");
    const count = Number.parseInt(countText, 10);
    const shape = dimsText.split(",").map((dim) => Number.parseInt(dim, 10));
    vecs[name] = {
      data: vecBuf.subarray(vecOffset, vecOffset + count),
      shape,
    };
    vecOffset += count;
  }

  const flatTrits = new Int8Array(tritBuf.length * 5);
  for (let byteIdx = 0; byteIdx < tritBuf.length; byteIdx += 1) {
    const byteVal = tritBuf[byteIdx];
    const dstBase = byteIdx * 5;
    for (let slot = 0; slot < 5; slot += 1) {
      const digit = Math.floor(byteVal / POW3[slot]) % 3;
      flatTrits[dstBase + slot] = ((digit + 1) % 3) - 1;
    }
  }

  const params = {};
  let byteOffset = 0;
  let totalTrits = 0;
  let nonZeroTrits = 0;

  for (const entry of tritPart.split(" ; ")) {
    const [name, dimsText, packedText] = entry.split(":");
    const packedLen = Number.parseInt(packedText, 10);
    const shape = dimsText.split(",").map((dim) => Number.parseInt(dim, 10));
    const numElems = shape.reduce((acc, dim) => acc * dim, 1);
    const tritStart = byteOffset * 5;
    byteOffset += packedLen;

    let scaleKey = name.endsWith("_w") ? `${name.slice(0, -2)}_s` : `${name}_s`;
    if (name === "col_sum_proj") {
      scaleKey = "col_sum_s";
    }
    const scaleVec = vecs[scaleKey] ? vecs[scaleKey].data : null;
    const lastDim = shape[shape.length - 1];
    const dense = new Float32Array(numElems);

    for (let idx = 0; idx < numElems; idx += 1) {
      const trit = flatTrits[tritStart + idx];
      if (trit !== 0) {
        nonZeroTrits += 1;
        dense[idx] = scaleVec ? trit * scaleVec[idx % lastDim] : trit;
      }
    }
    totalTrits += numElems;

    if (shape.length === 3) {
      params[name] = buildSparseConv(dense, shape[0], shape[1], shape[2]);
    } else {
      params[name] = buildSparse2D(dense, shape[0], shape[1]);
    }
  }

  for (const [name, vecObj] of Object.entries(vecs)) {
    if (!name.endsWith("_s")) {
      params[name] = vecObj;
    }
  }

  params.stats = {
    packedBytes: tritBuf.byteLength + vecBuf.byteLength,
    totalTrits,
    nonZeroTrits,
    zeroRatio: 1 - nonZeroTrits / totalTrits,
    scalarCount: vecBuf.length,
  };
  return params;
}

let cachedWeightsPromise = null;

export function loadWeights(url = "./weights.npz") {
  if (!cachedWeightsPromise) {
    cachedWeightsPromise = fetch(url).then(async (resp) => {
      if (!resp.ok) {
        throw new Error(`Failed to load weights (${resp.status})`);
      }
      const buffer = await resp.arrayBuffer();
      const unpacked = await unpackNpzBuffer(buffer);
      unpacked.rawBuffer = buffer;
      unpacked.stats.archiveBytes = buffer.byteLength;
      return unpacked;
    });
  }
  return cachedWeightsPromise;
}

function expandTabsTwo(text) {
  if (!text.includes("\t")) {
    return text;
  }
  const chunks = [];
  let colPos = 0;
  for (const ch of text) {
    if (ch === "\t") {
      const pad = 2 - (colPos % 2);
      chunks.push(" ".repeat(pad));
      colPos += pad;
    } else if (ch === "\n" || ch === "\r") {
      chunks.push(ch);
      colPos = 0;
    } else {
      chunks.push(ch);
      colPos += 1;
    }
  }
  return chunks.join("");
}

function splitPythonLines(text) {
  if (!text) {
    return [""];
  }
  const trimmed = text.replace(/(?:\r\n|\r|\n)$/, "");
  return trimmed.split(/\r\n|\r|\n/);
}

export function encodeByteGrid(code, maxLines = 2048) {
  const expanded = expandTabsTwo(code);
  const lines = splitPythonLines(expanded).slice(0, maxLines);
  const numLines = lines.length;
  const grid = new Uint8Array(numLines * MAX_COLS);

  for (let row = 0; row < numLines; row += 1) {
    const lineText = lines[row];
    if (!lineText) {
      continue;
    }
    const rowBase = row * MAX_COLS;
    let col = 0;
    for (const ch of lineText) {
      if (col >= MAX_COLS) {
        break;
      }
      const codePoint = ch.codePointAt(0);
      const ascii = codePoint > 127 ? 127 : codePoint & 0x7f;
      grid[rowBase + col] = ascii === 0 ? 32 : ascii;
      col += 1;
    }
  }
  return { grid, lines, numLines, width: MAX_COLS };
}

function buildActiveSlices(arrays, dims, activeRows) {
  const slices = [];
  let inOffset = 0;
  let actCursor = 0;
  const numActive = activeRows.length;

  for (let part = 0; part < arrays.length; part += 1) {
    const nextOffset = inOffset + dims[part];
    const actStart = actCursor;
    while (actCursor < numActive && activeRows[actCursor] < nextOffset) {
      actCursor += 1;
    }
    if (actCursor > actStart) {
      slices.push({
        src: arrays[part],
        dim: dims[part],
        inOffset,
        actStart,
        actEnd: actCursor,
      });
    }
    inOffset = nextOffset;
  }
  return slices;
}

function accumulateActiveSlice(slice, row, weight, out, outBase) {
  const { src, dim, inOffset, actStart, actEnd } = slice;
  const { activeRows, rowOffsets, colIndices, values } = weight;
  const srcBase = row * dim - inOffset;

  for (let actIdx = actStart; actIdx < actEnd; actIdx += 1) {
    const inIdx = activeRows[actIdx];
    const inVal = src[srcBase + inIdx];
    if (inVal === 0) {
      continue;
    }
    const end = rowOffsets[inIdx + 1];
    for (let ptr = rowOffsets[inIdx]; ptr < end; ptr += 1) {
      out[outBase + colIndices[ptr]] += inVal * values[ptr];
    }
  }
}

function linearSparseParts(arrays, dims, numRows, weight, biasVec, applyRelu) {
  const outDim = weight.cols;
  const out = new Float32Array(numRows * outDim);
  const slices = buildActiveSlices(arrays, dims, weight.activeRows);

  for (let row = 0; row < numRows; row += 1) {
    const outBase = row * outDim;
    if (biasVec) {
      out.set(biasVec, outBase);
    }
    for (let idx = 0; idx < slices.length; idx += 1) {
      accumulateActiveSlice(slices[idx], row, weight, out, outBase);
    }
    if (applyRelu) {
      for (let col = 0; col < outDim; col += 1) {
        if (out[outBase + col] < 0) {
          out[outBase + col] = 0;
        }
      }
    }
  }
  return out;
}

function linearSparse(input, numRows, weight, biasVec, applyRelu) {
  return linearSparseParts([input], [weight.rows], numRows, weight, biasVec, applyRelu);
}

function applyConvTap(cells, out, lineCellBase, numCols, inChannels, outChannels, tapMat, shift) {
  const { rowOffsets, colIndices, values, activeRows } = tapMat;
  const colStart = Math.max(0, -shift);
  const colEnd = Math.min(numCols, numCols - shift);

  for (let col = colStart; col < colEnd; col += 1) {
    const srcBase = (lineCellBase + col + shift) * inChannels;
    const dstBase = (lineCellBase + col) * outChannels;
    accumulateSparseRow(cells, srcBase, activeRows, out, dstBase, rowOffsets, colIndices, values);
  }
}

function accumulateSparseRow(input, srcBase, activeRows, out, dstBase, rowOffsets, colIndices, values) {
  const numActive = activeRows.length;
  for (let actIdx = 0; actIdx < numActive; actIdx += 1) {
    const inIdx = activeRows[actIdx];
    const inVal = input[srcBase + inIdx];
    if (inVal === 0) {
      continue;
    }
    const end = rowOffsets[inIdx + 1];
    for (let ptr = rowOffsets[inIdx]; ptr < end; ptr += 1) {
      out[dstBase + colIndices[ptr]] += inVal * values[ptr];
    }
  }
}

function conv1dGrid(cells, numLines, numCols, convWeight, biasVec, applyRelu, colLimits) {
  const { kSize, inChannels, outChannels, taps } = convWeight;
  const pad = (kSize - 1) >> 1;
  const totalCells = numLines * numCols;
  const out = new Float32Array(totalCells * outChannels);

  for (let row = 0; row < numLines; row += 1) {
    const rowCols = colLimits ? colLimits[row] : numCols;
    const lineCellBase = row * numCols;
    if (biasVec) {
      for (let col = 0; col < rowCols; col += 1) {
        const base = (lineCellBase + col) * outChannels;
        for (let chan = 0; chan < outChannels; chan += 1) {
          out[base + chan] = biasVec[chan];
        }
      }
    }
    for (let tap = 0; tap < kSize; tap += 1) {
      if (taps[tap].nonZeroCount > 0 && rowCols > 0) {
        applyConvTap(cells, out, lineCellBase, rowCols, inChannels, outChannels, taps[tap], tap - pad);
      }
    }
    if (applyRelu) {
      const startIdx = lineCellBase * outChannels;
      const endIdx = (lineCellBase + rowCols) * outChannels;
      for (let idx = startIdx; idx < endIdx; idx += 1) {
        if (out[idx] < 0) {
          out[idx] = 0;
        }
      }
    }
  }
  return out;
}

function causalLinePrefix(causalMat, projData, numLines, width, projDim) {
  const out = new Float32Array(numLines * width * projDim);
  const includeDiag = causalMat.data[0] === 1;
  const running = new Float32Array(projDim);

  for (let line = 0; line < numLines; line += 1) {
    running.fill(0);
    const lineBase = line * width * projDim;
    for (let col = 0; col < width; col += 1) {
      const cellBase = lineBase + col * projDim;
      if (includeDiag) {
        for (let chan = 0; chan < projDim; chan += 1) {
          running[chan] += projData[cellBase + chan];
          out[cellBase + chan] = running[chan];
        }
      } else {
        for (let chan = 0; chan < projDim; chan += 1) {
          out[cellBase + chan] = running[chan];
          running[chan] += projData[cellBase + chan];
        }
      }
    }
  }
  return out;
}

function concatLastAxis(arrays, dims, numRows) {
  let totalDim = 0;
  for (let idx = 0; idx < dims.length; idx += 1) {
    totalDim += dims[idx];
  }
  const merged = new Float32Array(numRows * totalDim);
  let colStart = 0;
  for (let part = 0; part < arrays.length; part += 1) {
    const src = arrays[part];
    const partDim = dims[part];
    for (let row = 0; row < numRows; row += 1) {
      const dstBase = row * totalDim + colStart;
      const srcBase = row * partDim;
      for (let chan = 0; chan < partDim; chan += 1) {
        merged[dstBase + chan] = src[srcBase + chan];
      }
    }
    colStart += partDim;
  }
  return merged;
}

function compactCells(cells, totalCells, inChannels) {
  let nonZero = 0;
  for (let idx = 0; idx < cells.length; idx += 1) {
    if (cells[idx] !== 0) {
      nonZero += 1;
    }
  }

  const offsets = new Int32Array(totalCells + 1);
  const chans = new Int32Array(nonZero);
  const vals = new Float32Array(nonZero);
  let cursor = 0;

  for (let cell = 0; cell < totalCells; cell += 1) {
    offsets[cell] = cursor;
    const base = cell * inChannels;
    for (let chan = 0; chan < inChannels; chan += 1) {
      const val = cells[base + chan];
      if (val !== 0) {
        chans[cursor] = chan;
        vals[cursor] = val;
        cursor += 1;
      }
    }
  }
  offsets[totalCells] = cursor;
  return { offsets, chans, vals };
}

function accumulateCompactCell(sparseCells, cellIdx, tapMat, cellBuf) {
  const { offsets, chans, vals } = sparseCells;
  const { rowOffsets, colIndices, values } = tapMat;
  const cellEnd = offsets[cellIdx + 1];

  for (let idx = offsets[cellIdx]; idx < cellEnd; idx += 1) {
    const chan = chans[idx];
    const inVal = vals[idx];
    const rowEnd = rowOffsets[chan + 1];
    for (let ptr = rowOffsets[chan]; ptr < rowEnd; ptr += 1) {
      cellBuf[colIndices[ptr]] += inVal * values[ptr];
    }
  }
}

function sumConvColumn(sparseCells, lineCellBase, col, numCols, convWeight, biasVec, cellBuf) {
  const { kSize, taps } = convWeight;
  const pad = (kSize - 1) >> 1;
  cellBuf.set(biasVec);

  for (let tap = 0; tap < kSize; tap += 1) {
    const srcCol = col + tap - pad;
    if (srcCol >= 0 && srcCol < numCols && taps[tap].nonZeroCount > 0) {
      accumulateCompactCell(sparseCells, lineCellBase + srcCol, taps[tap], cellBuf);
    }
  }
}

function conv1dSumColumns(sparseCells, numLines, numCols, convWeight, biasVec, colLimits) {
  const { outChannels } = convWeight;
  const sums = new Float32Array(numLines * outChannels);
  const cellBuf = new Float32Array(outChannels);

  for (let row = 0; row < numLines; row += 1) {
    const rowCols = colLimits ? colLimits[row] : numCols;
    const lineCellBase = row * numCols;
    const sumBase = row * outChannels;

    for (let col = 0; col < rowCols; col += 1) {
      sumConvColumn(sparseCells, lineCellBase, col, numCols, convWeight, biasVec, cellBuf);
      for (let chan = 0; chan < outChannels; chan += 1) {
        if (cellBuf[chan] > 0) {
          sums[sumBase + chan] += cellBuf[chan];
        }
      }
    }
  }
  return sums;
}

function sumLineCells(sparseCells, numLines, width, numChannels) {
  const { offsets, chans, vals } = sparseCells;
  const sums = new Float32Array(numLines * numChannels);
  for (let line = 0; line < numLines; line += 1) {
    const dstBase = line * numChannels;
    const startIdx = offsets[line * width];
    const endIdx = offsets[(line + 1) * width];
    for (let idx = startIdx; idx < endIdx; idx += 1) {
      sums[dstBase + chans[idx]] += vals[idx];
    }
  }
  return sums;
}

function normalizeSketchSlices(rawSketch, numLines) {
  const totalSketch = SKETCH_DIM * 2;
  const out = new Float32Array(numLines * totalSketch);
  for (let line = 0; line < numLines; line += 1) {
    const base = line * totalSketch;
    for (let half = 0; half < 2; half += 1) {
      const sliceStart = base + half * SKETCH_DIM;
      let normSq = 2.5;
      for (let idx = 0; idx < SKETCH_DIM; idx += 1) {
        const clamped = Math.min(1.0, Math.max(0.0, rawSketch[sliceStart + idx]));
        out[sliceStart + idx] = clamped;
        normSq += clamped * clamped;
      }
      const invNorm = 1.0 / Math.sqrt(normSq);
      for (let idx = 0; idx < SKETCH_DIM; idx += 1) {
        out[sliceStart + idx] *= invNorm;
      }
    }
  }
  return out;
}

function attendHeadQuery(qLine, numLines, headCfg, qProj, kProj, vProj, merged, logits, validKeys) {
  const { qkTotal, valTotal, headDim, valDim, invScale } = headCfg;
  const { qkHeadOff, valHeadOff, relBase, bRel, sinkLogit, hasQuery } = headCfg;
  const qBase = qLine * qkTotal + qkHeadOff;
  const kStart = bRel[relBase] <= -1e3 ? Math.max(0, qLine - MAX_REL_DIST + 1) : 0;
  const farBias = bRel[relBase + 2 * MAX_REL_DIST];
  const kEnd = farBias <= -1e3 ? Math.min(numLines, qLine + MAX_REL_DIST) : numLines;
  let maxLogit = sinkLogit;
  let validCount = 0;

  for (let kLine = kStart; kLine < kEnd; kLine += 1) {
    const relBin = Math.max(-MAX_REL_DIST, Math.min(MAX_REL_DIST, kLine - qLine)) + MAX_REL_DIST;
    const relBias = bRel[relBase + relBin];
    if (relBias <= -1e3) {
      continue;
    }
    let dot = 0.0;
    if (hasQuery) {
      const kBase = kLine * qkTotal + qkHeadOff;
      for (let dim = 0; dim < headDim; dim += 1) {
        dot += qProj[qBase + dim] * kProj[kBase + dim];
      }
    }
    const scoreVal = dot * invScale + relBias;
    validKeys[validCount] = kLine;
    logits[validCount] = scoreVal;
    validCount += 1;
    if (scoreVal > maxLogit) {
      maxLogit = scoreVal;
    }
  }

  let expSum = Math.exp(sinkLogit - maxLogit);
  for (let idx = 0; idx < validCount; idx += 1) {
    const weightVal = Math.exp(logits[idx] - maxLogit);
    logits[idx] = weightVal;
    expSum += weightVal;
  }

  const invSum = 1.0 / expSum;
  const dstBase = qLine * valTotal + valHeadOff;
  for (let idx = 0; idx < validCount; idx += 1) {
    const attnWeight = logits[idx] * invSum;
    const vBase = validKeys[idx] * valTotal + valHeadOff;
    for (let dim = 0; dim < valDim; dim += 1) {
      merged[dstBase + dim] += attnWeight * vProj[vBase + dim];
    }
  }
}

function sinkAttention(state, numLines, wq, wk, wv, wo, bRel, bSink) {
  const numHeads = bSink.length;
  const qkTotal = wq.cols;
  const valTotal = wv.cols;
  const headDim = qkTotal / numHeads;
  const valDim = valTotal / numHeads;
  const invScale = 1.0 / Math.sqrt(headDim);
  const relSpan = 2 * MAX_REL_DIST + 1;
  const hasQuery = wq.nonZeroCount > 0 && wk.nonZeroCount > 0;

  const qProj = hasQuery ? linearSparse(state, numLines, wq, null, false) : null;
  const kProj = hasQuery ? linearSparse(state, numLines, wk, null, false) : null;
  const vProj = linearSparse(state, numLines, wv, null, false);
  const merged = new Float32Array(numLines * valTotal);
  const logits = new Float32Array(numLines);
  const validKeys = new Int32Array(numLines);

  for (let head = 0; head < numHeads; head += 1) {
    const headCfg = {
      qkTotal,
      valTotal,
      headDim,
      valDim,
      invScale,
      qkHeadOff: head * headDim,
      valHeadOff: head * valDim,
      relBase: head * relSpan,
      bRel,
      sinkLogit: bSink[head],
      hasQuery,
    };
    for (let qLine = 0; qLine < numLines; qLine += 1) {
      attendHeadQuery(qLine, numLines, headCfg, qProj, kProj, vProj, merged, logits, validKeys);
    }
  }

  return linearSparse(merged, numLines, wo, null, false);
}

function sigmoidScalar(val) {
  const clipped = Math.max(-30.0, Math.min(30.0, val));
  return 1.0 / (1.0 + Math.exp(-clipped));
}

function softmaxSlice(arr, start, count) {
  let maxVal = -Infinity;
  for (let idx = 0; idx < count; idx += 1) {
    if (arr[start + idx] > maxVal) {
      maxVal = arr[start + idx];
    }
  }
  const probs = new Float32Array(count);
  let sumExp = 0.0;
  for (let idx = 0; idx < count; idx += 1) {
    const expVal = Math.exp(arr[start + idx] - maxVal);
    probs[idx] = expVal;
    sumExp += expVal;
  }
  for (let idx = 0; idx < count; idx += 1) {
    probs[idx] /= sumExp;
  }
  return probs;
}

function classifyCellMask(cells, cellOff) {
  if (cells[cellOff + HC_COMM] > 0.5) {
    return 4;
  }
  if (cells[cellOff + HC_STR] > 0.5) {
    return 3;
  }
  if (cells[cellOff + HC_IND] > 0.5) {
    return 1;
  }
  if (cells[cellOff + HC_CODE] > 0.5 && cells[cellOff + HC_ACT] > 0.5) {
    return 2;
  }
  return 5;
}

function extractCellMasks(cells, lines, numLines, width) {
  const masks = new Uint8Array(numLines * width);
  for (let row = 0; row < numLines; row += 1) {
    const lineLen = Math.min(width, Array.from(lines[row]).length);
    const rowMaskBase = row * width;
    const rowCellBase = row * width * NUM_CHANNELS_HC;
    for (let col = 0; col < lineLen; col += 1) {
      masks[rowMaskBase + col] = classifyCellMask(cells, rowCellBase + col * NUM_CHANNELS_HC);
    }
  }
  return masks;
}

export function forwardModel(params, grid, lines, numLines, width = MAX_COLS) {
  const totalCells = numLines * width;
  const embDim = params.tok_emb.cols;
  const emb = new Float32Array(totalCells * embDim);
  const tokData = params.tok_emb.data;
  const colData = params.col_emb.data;

  for (let row = 0; row < numLines; row += 1) {
    const gridRow = row * width;
    for (let col = 0; col < width; col += 1) {
      const dstOff = (gridRow + col) * embDim;
      const tokOff = grid[gridRow + col] * embDim;
      const colOff = col * embDim;
      for (let chan = 0; chan < embDim; chan += 1) {
        emb[dstOff + chan] = tokData[tokOff + chan] + colData[colOff + chan];
      }
    }
  }

  const lex = conv1dGrid(emb, numLines, width, params.pre_conv3_w, params.pre_conv3_b.data, true);
  const lexDim = params.pre_conv3_w.outChannels;
  const prefDim = params.pref1_proj.cols;
  const pref1In = linearSparseParts(
    [emb, lex],
    [embDim, lexDim],
    totalCells,
    params.pref1_proj,
    null,
    false,
  );
  const prefix1 = causalLinePrefix(params.W_causal1, pref1In, numLines, width, prefDim);

  const gates = linearSparseParts(
    [emb, lex, prefix1],
    [embDim, lexDim, prefDim],
    totalCells,
    params.gate1_w,
    params.gate1_b.data,
    true,
  );
  const pref2In = linearSparse(gates, totalCells, params.pref2_proj, null, false);
  const prefix2 = causalLinePrefix(params.W_causal2, pref2In, numLines, width, params.pref2_proj.cols);

  const fuseDims = [embDim, lexDim, params.gate1_w.cols, params.pref2_proj.cols];
  const cells = linearSparseParts(
    [emb, lex, gates, prefix2],
    fuseDims,
    totalCells,
    params.col_fuse_w,
    params.col_fuse_b.data,
    true,
  );
  const cellMasks = extractCellMasks(cells, lines, numLines, width);
  const sparseCells = compactCells(cells, totalCells, NUM_CHANNELS_HC);

  const colLimits = new Int32Array(numLines);
  for (let row = 0; row < numLines; row += 1) {
    const len = lines[row].length;
    colLimits[row] = len === 0 ? 0 : Math.min(width, len + 13);
  }

  const conv3Sum = conv1dSumColumns(
    sparseCells, numLines, width, params.conv3_w, params.conv3_b.data, null,
  );
  const conv7Sum = conv1dSumColumns(
    sparseCells, numLines, width, params.conv7_w, params.conv7_b.data, colLimits,
  );
  const conv13Sum = conv1dSumColumns(
    sparseCells, numLines, width, params.conv13_w, params.conv13_b.data, colLimits,
  );

  const cellSums = sumLineCells(sparseCells, numLines, width, NUM_CHANNELS_HC);
  const colSums = linearSparse(cellSums, numLines, params.col_sum_proj, null, false);
  const rawSketch = conv1dSumColumns(
    sparseCells, numLines, width, params.sketch_w, params.sketch_b.data, colLimits,
  );

  const lineFeatDims = [
    params.col_sum_proj.cols,
    params.conv3_w.outChannels,
    params.conv7_w.outChannels,
    params.conv13_w.outChannels,
  ];
  const lineHidden = linearSparseParts(
    [colSums, conv3Sum, conv7Sum, conv13Sum],
    lineFeatDims,
    numLines,
    params.lmlp1_w,
    params.lmlp1_b.data,
    true,
  );
  const lineStats = linearSparse(lineHidden, numLines, params.lmlp2_w, params.lmlp2_b.data, false);
  const normSketches = normalizeSketchSlices(rawSketch, numLines);
  let state = concatLastAxis([lineStats, normSketches], [params.lmlp2_w.cols, SKETCH_DIM * 2], numLines);

  for (const layer of ["1", "2"]) {
    const attnDelta = sinkAttention(
      state,
      numLines,
      params[`attn${layer}_wq`],
      params[`attn${layer}_wk`],
      params[`attn${layer}_wv`],
      params[`attn${layer}_wo`],
      params[`attn${layer}_brel`].data,
      params[`attn${layer}_bsink`].data,
    );
    for (let idx = 0; idx < state.length; idx += 1) {
      state[idx] += attnDelta[idx];
    }

    const ctxHidden = linearSparse(
      state,
      numLines,
      params[`cmlp${layer}_w`],
      params[`cmlp${layer}_b`].data,
      true,
    );
    const ctxDelta = linearSparse(
      ctxHidden,
      numLines,
      params[`cmlp${layer}_2_w`],
      params[`cmlp${layer}_2_b`].data,
      false,
    );
    for (let idx = 0; idx < state.length; idx += 1) {
      state[idx] += ctxDelta[idx];
    }
  }

  const sqLogits = linearSparse(state, numLines, params.sq_w, params.sq_b.data, false);
  const lineProbs = new Float32Array(sqLogits.length);
  for (let idx = 0; idx < sqLogits.length; idx += 1) {
    lineProbs[idx] = sigmoidScalar(sqLogits[idx]);
  }

  const gateLogits = linearSparse(state, numLines, params.pool_gate_w, params.pool_gate_b.data, false);
  const activeGate = new Float32Array(numLines);
  let gateSum = 1e-5;
  for (let row = 0; row < numLines; row += 1) {
    const gateVal = sigmoidScalar(gateLogits[row]);
    activeGate[row] = gateVal;
    gateSum += gateVal;
  }

  const stateDim = params.sq_w.rows;
  const docVec = new Float32Array(stateDim * 2);
  for (let row = 0; row < numLines; row += 1) {
    const gateVal = activeGate[row];
    const rowBase = row * stateDim;
    for (let chan = 0; chan < stateDim; chan += 1) {
      const val = state[rowBase + chan];
      docVec[chan] += val * gateVal;
      const posWeighted = val > 0 ? val * gateVal : 0.0;
      if (posWeighted > docVec[stateDim + chan]) {
        docVec[stateDim + chan] = posWeighted;
      }
    }
  }
  for (let chan = 0; chan < stateDim; chan += 1) {
    docVec[chan] /= gateSum;
  }

  const docHidden = linearSparse(docVec, 1, params.doc1_w, params.doc1_b.data, true);
  const logits = linearSparse(docHidden, 1, params.doc2_w, params.doc2_b.data, false);
  const overallScore = sigmoidScalar(logits[0]);
  const rlHygiene = sigmoidScalar(logits[1]);

  const subscores = {};
  for (let idx = 0; idx < SCORE_NAMES.length; idx += 1) {
    const levelProbs = softmaxSlice(logits, 2 + idx * 4, 4);
    let expected = 0.0;
    for (let lvl = 0; lvl < 4; lvl += 1) {
      expected += levelProbs[lvl] * SCORE_LEVELS[lvl];
    }
    subscores[SCORE_NAMES[idx]] = expected;
  }

  const styleStart = 2 + SCORE_NAMES.length * 4;
  const styleArr = softmaxSlice(logits, styleStart, ARCHETYPE_NAMES.length);
  const archetypeProbs = {};
  let bestArchIdx = 0;
  for (let idx = 0; idx < ARCHETYPE_NAMES.length; idx += 1) {
    archetypeProbs[ARCHETYPE_NAMES[idx]] = styleArr[idx];
    if (styleArr[idx] > styleArr[bestArchIdx]) {
      bestArchIdx = idx;
    }
  }

  return {
    score: overallScore,
    rlHygiene,
    scores: subscores,
    archetype: ARCHETYPE_NAMES[bestArchIdx],
    archetypeProbs,
    lineProbs,
    activeGate,
    cellMasks,
    lines,
    numLines,
    width,
  };
}

export function collectSquiggles(forwardOut, threshold = 0.55) {
  const { lineProbs, lines } = forwardOut;
  const numKinds = SQUIGGLE_KINDS.length;
  const squiggles = [];

  for (let row = 0; row < lines.length; row += 1) {
    const rawLine = lines[row];
    const probBase = row * numKinds;
    let bestKind = 0;
    for (let kindIdx = 1; kindIdx < numKinds; kindIdx += 1) {
      if (lineProbs[probBase + kindIdx] > lineProbs[probBase + bestKind]) {
        bestKind = kindIdx;
      }
    }
    const probability = lineProbs[probBase + bestKind];
    if (probability < threshold) {
      continue;
    }
    const stripped = rawLine.replace(/^ +/, "");
    const colStart = stripped ? rawLine.length - stripped.length + 1 : 1;
    const kind = SQUIGGLE_KINDS[bestKind];
    squiggles.push({
      line: row + 1,
      col_start: colStart,
      col_end: Math.max(colStart, rawLine.length),
      probability,
      kind,
      message: SQUIGGLE_MESSAGES[kind],
    });
  }
  return squiggles;
}

export function scoreCode(code, weights, threshold = 0.55) {
  const startMs = performance.now();
  const { grid, lines, numLines, width } = encodeByteGrid(code);
  const raw = forwardModel(weights, grid, lines, numLines, width);
  const squiggles = collectSquiggles(raw, threshold);
  const elapsedMs = performance.now() - startMs;
  return { ...raw, squiggles, elapsedMs };
}

export function computeRlReward(result, testsPassed = true, weight = 0.25, hygieneBlend = 0.5) {
  const gate = Number(testsPassed);
  if (gate <= 0) {
    return 0.0;
  }
  const blended = (1.0 - hygieneBlend) * result.score + hygieneBlend * result.rlHygiene;
  return gate * (1.0 + weight * blended);
}

export function formatJsonResult(result) {
  const round4 = (val) => Math.round(val * 10000) / 10000;
  const scores = {};
  for (const [key, val] of Object.entries(result.scores)) {
    scores[key] = round4(val);
  }
  const archetypeProbs = {};
  for (const [key, val] of Object.entries(result.archetypeProbs)) {
    archetypeProbs[key] = round4(val);
  }
  return {
    score: round4(result.score),
    rl_hygiene: round4(result.rlHygiene),
    scores,
    archetype: result.archetype,
    archetype_probs: archetypeProbs,
    squiggles: result.squiggles.map((sq) => ({
      ...sq,
      probability: round4(sq.probability),
    })),
  };
}
