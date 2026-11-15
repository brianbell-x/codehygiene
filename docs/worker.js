import { scoreCode, unpackNpzBuffer } from "./engine.js";

let workerWeights = null;

globalThis.addEventListener("message", async (event) => {
  const payload = event.data;
  if (!payload) {
    return;
  }

  if (payload.type === "init") {
    workerWeights = await unpackNpzBuffer(payload.buffer);
    globalThis.postMessage({ type: "ready" });
    return;
  }

  if (payload.type === "score" && workerWeights) {
    const result = scoreCode(payload.code, workerWeights, payload.threshold);
    const transfers = [
      result.lineProbs.buffer,
      result.activeGate.buffer,
      result.cellMasks.buffer,
    ];
    globalThis.postMessage({ type: "scored", jobId: payload.jobId, result }, transfers);
  }
});
