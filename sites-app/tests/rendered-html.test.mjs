import assert from "node:assert/strict";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${path}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }), { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } }, { waitUntil() {}, passThroughOnException() {} });
}

test("오늘의 경기 화면을 서버 렌더링한다", async () => {
  const response = await render("/");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<html lang="ko">/);
  assert.match(html, /MLB2/);
  assert.match(html, /오늘의 MLB 경기/);
  assert.match(html, /모델 검증/);
  assert.match(html, /방법론/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|베팅|ROI|수익/);
});

test("검증과 방법론 화면을 렌더링한다", async () => {
  const validation = await (await render("/model-validation")).text();
  const methodology = await (await render("/methodology")).text();
  assert.match(validation, /2025 봉인 홀드아웃/);
  assert.match(validation, /0\.6819/);
  assert.match(validation, /현재 예측 제공/);
  assert.doesNotMatch(validation, /활성화하지 않습니다/);
  assert.match(methodology, /Fail-closed/);
  assert.match(methodology, /선발투수 이름은 표시용/);
  assert.match(methodology, /미래 성능 모니터링/);
});
