function buildLcsTable(leftItems, rightItems) {
  const leftLen = leftItems.length;
  const rightLen = rightItems.length;
  const width = rightLen + 1;
  const table = new Int32Array((leftLen + 1) * width);

  for (let leftIdx = leftLen - 1; leftIdx >= 0; leftIdx -= 1) {
    for (let rightIdx = rightLen - 1; rightIdx >= 0; rightIdx -= 1) {
      const cellIdx = leftIdx * width + rightIdx;
      if (leftItems[leftIdx] === rightItems[rightIdx]) {
        table[cellIdx] = table[(leftIdx + 1) * width + (rightIdx + 1)] + 1;
      } else {
        const downVal = table[(leftIdx + 1) * width + rightIdx];
        const rightVal = table[leftIdx * width + (rightIdx + 1)];
        table[cellIdx] = downVal >= rightVal ? downVal : rightVal;
      }
    }
  }
  return table;
}

function backtrackLcs(leftItems, rightItems) {
  const table = buildLcsTable(leftItems, rightItems);
  const width = rightItems.length + 1;
  const steps = [];
  let leftIdx = 0;
  let rightIdx = 0;

  while (leftIdx < leftItems.length && rightIdx < rightItems.length) {
    if (leftItems[leftIdx] === rightItems[rightIdx]) {
      steps.push({ type: "ctx", leftIdx, rightIdx });
      leftIdx += 1;
      rightIdx += 1;
    } else if (
      table[(leftIdx + 1) * width + rightIdx] >= table[leftIdx * width + (rightIdx + 1)]
    ) {
      steps.push({ type: "del", leftIdx, rightIdx: -1 });
      leftIdx += 1;
    } else {
      steps.push({ type: "add", leftIdx: -1, rightIdx });
      rightIdx += 1;
    }
  }

  while (leftIdx < leftItems.length) {
    steps.push({ type: "del", leftIdx, rightIdx: -1 });
    leftIdx += 1;
  }
  while (rightIdx < rightItems.length) {
    steps.push({ type: "add", leftIdx: -1, rightIdx });
    rightIdx += 1;
  }
  return steps;
}

function tokenizeCodeWords(lineText) {
  const tokens = lineText.match(/\w+|\s+|[^\w\s]+/g);
  if (!tokens) {
    return [];
  }
  return tokens;
}

export function diffWordTokens(deletedLine, addedLine) {
  const delTokens = tokenizeCodeWords(deletedLine);
  const addTokens = tokenizeCodeWords(addedLine);
  if (delTokens.length * addTokens.length > 3600) {
    return {
      delParts: [{ text: deletedLine, changed: false }],
      addParts: [{ text: addedLine, changed: false }],
    };
  }

  const delParts = [];
  const addParts = [];
  for (const step of backtrackLcs(delTokens, addTokens)) {
    if (step.type === "ctx") {
      delParts.push({ text: delTokens[step.leftIdx], changed: false });
      addParts.push({ text: addTokens[step.rightIdx], changed: false });
    } else if (step.type === "del") {
      const text = delTokens[step.leftIdx];
      delParts.push({ text, changed: text.trim().length > 0 });
    } else {
      const text = addTokens[step.rightIdx];
      addParts.push({ text, changed: text.trim().length > 0 });
    }
  }
  return { delParts, addParts };
}

function computeLineOps(oldLines, newLines) {
  if (oldLines.length * newLines.length > 90000) {
    const fallback = oldLines.map((text, idx) => ({
      type: "del",
      text,
      oldLine: idx + 1,
      newLine: null,
    }));
    for (let idx = 0; idx < newLines.length; idx += 1) {
      fallback.push({ type: "add", text: newLines[idx], oldLine: null, newLine: idx + 1 });
    }
    return fallback;
  }

  return backtrackLcs(oldLines, newLines).map((step) => {
    if (step.type === "ctx") {
      return {
        type: "ctx",
        text: oldLines[step.leftIdx],
        oldLine: step.leftIdx + 1,
        newLine: step.rightIdx + 1,
      };
    }
    if (step.type === "del") {
      return {
        type: "del",
        text: oldLines[step.leftIdx],
        oldLine: step.leftIdx + 1,
        newLine: null,
      };
    }
    return {
      type: "add",
      text: newLines[step.rightIdx],
      oldLine: null,
      newLine: step.rightIdx + 1,
    };
  });
}

function collapseContextRuns(ops) {
  const rows = [];
  let cursor = 0;

  while (cursor < ops.length) {
    if (ops[cursor].type !== "ctx") {
      rows.push(ops[cursor]);
      cursor += 1;
      continue;
    }

    let endIdx = cursor;
    while (endIdx + 1 < ops.length && ops[endIdx + 1].type === "ctx") {
      endIdx += 1;
    }

    const runLen = endIdx - cursor + 1;
    const isEdge = cursor === 0 || endIdx === ops.length - 1;
    const maxVisible = isEdge ? 3 : 6;

    if (runLen <= maxVisible + 2) {
      for (let idx = cursor; idx <= endIdx; idx += 1) {
        rows.push(ops[idx]);
      }
    } else if (cursor === 0) {
      rows.push({ type: "gap", hiddenCount: runLen - 3 });
      for (let idx = endIdx - 2; idx <= endIdx; idx += 1) {
        rows.push(ops[idx]);
      }
    } else if (endIdx === ops.length - 1) {
      for (let idx = cursor; idx < cursor + 3; idx += 1) {
        rows.push(ops[idx]);
      }
      rows.push({ type: "gap", hiddenCount: runLen - 3 });
    } else {
      for (let idx = cursor; idx < cursor + 3; idx += 1) {
        rows.push(ops[idx]);
      }
      rows.push({ type: "gap", hiddenCount: runLen - 6 });
      for (let idx = endIdx - 2; idx <= endIdx; idx += 1) {
        rows.push(ops[idx]);
      }
    }

    cursor = endIdx + 1;
  }

  return rows;
}

export function buildFlatDiff(sourceCode, targetCode) {
  if (sourceCode === targetCode) {
    return [];
  }

  const oldLines = sourceCode.split("\n");
  const newLines = targetCode.split("\n");
  const rawOps = collapseContextRuns(computeLineOps(oldLines, newLines));
  const enriched = [];
  let cursor = 0;

  while (cursor < rawOps.length) {
    if (rawOps[cursor].type !== "del") {
      enriched.push(rawOps[cursor]);
      cursor += 1;
      continue;
    }

    const delRun = [];
    while (cursor < rawOps.length && rawOps[cursor].type === "del") {
      delRun.push(rawOps[cursor]);
      cursor += 1;
    }

    const addRun = [];
    while (cursor < rawOps.length && rawOps[cursor].type === "add") {
      addRun.push(rawOps[cursor]);
      cursor += 1;
    }

    if (delRun.length === addRun.length) {
      const pairedAdds = [];
      for (let idx = 0; idx < delRun.length; idx += 1) {
        const { delParts, addParts } = diffWordTokens(delRun[idx].text, addRun[idx].text);
        enriched.push({ ...delRun[idx], parts: delParts });
        pairedAdds.push({ ...addRun[idx], parts: addParts });
      }
      enriched.push(...pairedAdds);
    } else {
      enriched.push(...delRun, ...addRun);
    }
  }

  return enriched;
}
