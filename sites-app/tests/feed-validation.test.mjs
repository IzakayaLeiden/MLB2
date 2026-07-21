import assert from "node:assert/strict";
import test from "node:test";
import { demoFeed } from "../lib/demo-feed.mjs";
import { validatePredictionFeed } from "../lib/feed-validation.mjs";

const now = new Date("2026-07-21T12:00:00Z");
const date = "2026-07-21";

test("정상 당일 피드를 허용한다", () => {
  const result = validatePredictionFeed(demoFeed(date, now), { now, expectedDate: date });
  assert.equal(result.available, true);
  assert.equal(result.state, "available");
});

test("오래된 피드는 확률을 차단한다", () => {
  const feed = demoFeed(date, new Date("2026-07-19T00:00:00Z"));
  const result = validatePredictionFeed(feed, { now, expectedDate: date });
  assert.equal(result.available, false);
  assert.equal(result.code, "stale_feed");
});

test("빈 정상 피드는 empty 상태로 구분한다", () => {
  const feed = demoFeed(date, now);
  feed.predictions = [];
  const result = validatePredictionFeed(feed, { now, expectedDate: date });
  assert.equal(result.available, true);
  assert.equal(result.state, "empty");
});

test("부분·오류·날짜 불일치 피드는 차단한다", () => {
  const partial = demoFeed(date, now);
  delete partial.predictions[0].home_team;
  assert.equal(validatePredictionFeed(partial, { now, expectedDate: date }).code, "partial_feed");
  const failed = demoFeed(date, now);
  failed.quality.status = "failed";
  assert.equal(validatePredictionFeed(failed, { now, expectedDate: date }).code, "quality_failed");
  assert.equal(validatePredictionFeed(demoFeed("2026-07-20", now), { now, expectedDate: date }).code, "date_mismatch");
});
