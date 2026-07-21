const MAX_AGE_MS = 26 * 60 * 60 * 1000;

export function easternDate(value = new Date()) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(value);
}

function failure(reason, code = "invalid_feed") {
  return { available: false, state: "unavailable", code, reason, feed: null };
}

export function validatePredictionFeed(payload, { now = new Date(), expectedDate = easternDate(now) } = {}) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return failure("피드 형식이 올바르지 않습니다.");
  if (payload.schema_version !== "prediction-feed-v1") return failure("지원하지 않는 피드 버전입니다.", "schema_mismatch");
  if (payload.target_date_et !== expectedDate) return failure("당일 예측 피드가 아닙니다.", "date_mismatch");
  const created = new Date(payload.created_at_utc);
  if (Number.isNaN(created.getTime())) return failure("생성 시각이 올바르지 않습니다.", "invalid_timestamp");
  const age = now.getTime() - created.getTime();
  if (age > MAX_AGE_MS || age < -10 * 60 * 1000) return failure("예측 데이터가 26시간 이상 오래됐습니다.", "stale_feed");
  if (payload.quality?.status !== "passed") return failure("데이터 품질 게이트를 통과하지 못했습니다.", "quality_failed");
  if (!Array.isArray(payload.predictions)) return failure("경기 목록이 없습니다.", "partial_feed");
  for (const prediction of payload.predictions) {
    const probability = prediction?.home_win_probability;
    if (!prediction || typeof prediction.game_id !== "number" || typeof probability !== "number" || probability < 0 || probability > 1 || !prediction.home_team?.name || !prediction.away_team?.name || "home_win" in prediction || "result" in prediction) {
      return failure("일부 경기 데이터가 불완전합니다.", "partial_feed");
    }
  }
  return {
    available: true,
    state: payload.predictions.length ? "available" : "empty",
    code: null,
    reason: payload.predictions.length ? null : "오늘 예정된 정규시즌 경기가 없습니다.",
    feed: payload,
  };
}
