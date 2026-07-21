import assert from "node:assert/strict";
import test from "node:test";
import { predictionDisplay } from "../lib/prediction-display.mjs";

test("홈 승리확률이 50%를 넘으면 홈팀을 우세팀으로 표시한다", () => {
  const result = predictionDisplay(0.62);
  assert.equal(result.favoredSide, "home");
  assert.equal(result.favoredProbability, 0.62);
  assert.equal(result.awayWinProbability, 0.38);
});

test("홈 승리확률이 50%보다 낮으면 원정팀 확률로 변환한다", () => {
  const result = predictionDisplay(0.432);
  assert.equal(result.favoredSide, "away");
  assert.equal(result.favoredProbability, 0.5680000000000001);
  assert.equal(result.homeWinProbability + result.awayWinProbability, 1);
});

test("정확히 50%이면 우열 없음으로 표시한다", () => {
  const result = predictionDisplay(0.5);
  assert.equal(result.favoredSide, "even");
  assert.equal(result.favoredProbability, 0.5);
});

test("확률 범위를 벗어난 값은 거부한다", () => {
  assert.throws(() => predictionDisplay(1.01), RangeError);
});
