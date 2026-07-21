"use client";

import { useEffect, useMemo, useState } from "react";

type Team = { id: number; name: string; abbreviation?: string; probable_pitcher?: string | null };
type Diagnostics = { elo_home_win_probability: number; home_elo: number; away_elo: number; home_recent_win_pct: number; away_recent_win_pct: number; home_rest_days: number; away_rest_days: number; data_through_date: string | null };
type Prediction = { game_id: number; game_start_utc: string; away_team: Team; home_team: Team; venue: { name?: string | null }; home_win_probability: number; diagnostics: Diagnostics };
type Feed = { created_at_utc: string; data_through_date: string; target_date_et: string; model_version: string; predictions: Prediction[] };
type FutureValidation = { state: "passed" | "monitoring" | "unknown"; historical_passed: boolean; elapsed_days: number; required_days: number; graded_games: number; required_games: number; coverage: number | null };
type FeedResponse = { available: boolean; state: "available" | "empty" | "unavailable"; reason?: string | null; feed?: Feed | null; validation?: FutureValidation };

const koreanTeams: Record<string, string> = {
  "Chicago White Sox": "시카고 화이트삭스", "Detroit Tigers": "디트로이트 타이거스", "Houston Astros": "휴스턴 애스트로스", "Minnesota Twins": "미네소타 트윈스", "San Diego Padres": "샌디에이고 파드리스", "Chicago Cubs": "시카고 컵스",
};

function etToday() {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());
}

function shiftDate(value: string, days: number) {
  const date = new Date(`${value}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

function dateLabel(value: string, weekday = false) {
  return new Intl.DateTimeFormat("ko-KR", { timeZone: "UTC", year: "numeric", month: "long", day: "numeric", ...(weekday ? { weekday: "short" as const } : {}) }).format(new Date(`${value}T12:00:00Z`));
}

function timeLabel(value: string) {
  return `${new Intl.DateTimeFormat("en-GB", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(value))} ET`;
}

function teamLabel(team: Team) { return koreanTeams[team.name] || team.name; }
function monogram(team: Team) {
  if (team.name.includes("White Sox")) return "S";
  if (team.name.includes("Padres")) return "P";
  return team.name.trim().slice(0, 1).toUpperCase();
}
function homeProbabilityLabel(team: Team) {
  return ({ "Detroit Tigers": "디트로이트", "Minnesota Twins": "미네소타", "Chicago Cubs": "시카고 컵스" } as Record<string, string>)[team.name] || teamLabel(team);
}

function GameRow({ prediction }: { prediction: Prediction }) {
  const [open, setOpen] = useState(false);
  const percent = prediction.home_win_probability * 100;
  return (
    <article className="game-entry">
      <div className="game-row">
        <div className="team away-team"><span className="monogram" aria-hidden="true">{monogram(prediction.away_team)}</span><div><h3>{teamLabel(prediction.away_team)}</h3><p><span>선발</span> {prediction.away_team.probable_pitcher || "미정"}</p></div></div>
        <div className="game-meta"><strong>{timeLabel(prediction.game_start_utc)}</strong><span>{prediction.venue.name || "구장 미정"}</span></div>
        <div className="team home-team"><span className="monogram" aria-hidden="true">{monogram(prediction.home_team)}</span><div><h3>{teamLabel(prediction.home_team)}</h3><p><span>선발</span> {prediction.home_team.probable_pitcher || "미정"}</p></div></div>
        <div className="probability" aria-label={`${teamLabel(prediction.home_team)} 홈 승리 확률 ${percent.toFixed(1)}퍼센트`}>
          <span>{homeProbabilityLabel(prediction.home_team)} 홈 승리</span><strong>{percent.toFixed(1)}%</strong>
          <div className="probability-track" aria-hidden="true"><i style={{ width: `${percent}%` }} /></div>
        </div>
        <button className="expand-button" type="button" aria-expanded={open} aria-controls={`details-${prediction.game_id}`} onClick={() => setOpen((value) => !value)}>{open ? "⌃" : "⌄"}<span className="sr-only">경기 근거 {open ? "접기" : "펼치기"}</span></button>
      </div>
      {open && <div className="game-details" id={`details-${prediction.game_id}`}>
        <dl><div><dt>Elo</dt><dd>{Math.round(prediction.diagnostics.away_elo)} – {Math.round(prediction.diagnostics.home_elo)}</dd></div><div><dt>최근 승률</dt><dd>{Math.round(prediction.diagnostics.away_recent_win_pct * 100)}% – {Math.round(prediction.diagnostics.home_recent_win_pct * 100)}%</dd></div><div><dt>휴식일</dt><dd>{prediction.diagnostics.away_rest_days}일 – {prediction.diagnostics.home_rest_days}일</dd></div><div><dt>데이터 기준일</dt><dd>{prediction.diagnostics.data_through_date || "없음"}</dd></div></dl>
      </div>}
    </article>
  );
}

export function TodayDashboard() {
  const [targetDate, setTargetDate] = useState(etToday);
  const [result, setResult] = useState<FeedResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true;
    fetch(`/api/feed?date=${targetDate}`, { cache: "no-store" })
      .then((response) => response.json() as Promise<FeedResponse>)
      .then((payload) => { if (active) setResult(payload); })
      .catch(() => { if (active) setResult({ available: false, state: "unavailable", reason: "예측 피드에 연결할 수 없습니다." }); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [targetDate, revision]);
  useEffect(() => {
    const refresh = () => { setLoading(true); setRevision((value) => value + 1); };
    window.addEventListener("mlb2-refresh", refresh);
    return () => window.removeEventListener("mlb2-refresh", refresh);
  }, []);
  const feed = result?.feed || null;
  const validation = result?.validation;
  const updated = useMemo(() => feed ? new Intl.DateTimeFormat("en-GB", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(feed.created_at_utc)) : "--:--", [feed]);

  return (
    <div className="dashboard-page">
      <section className="dashboard-heading">
        <div><h1>오늘의 MLB 경기</h1><p>{dateLabel(targetDate)}</p></div>
        <div className="date-controls" aria-label="경기 날짜 선택"><button type="button" onClick={() => { setLoading(true); setTargetDate((date) => shiftDate(date, -1)); }} aria-label="이전 날짜">‹</button><span>▣&nbsp; {dateLabel(targetDate, true)}</span><button type="button" onClick={() => { setLoading(true); setTargetDate((date) => shiftDate(date, 1)); }} aria-label="다음 날짜">›</button></div>
      </section>
      <div className="data-standard">▱&nbsp; 데이터 기준 {updated} ET · {feed?.model_version || "model-v1"} · 홈 승리 확률</div>
      {!loading && result?.available && <div className={`validation-progress ${validation?.state === "passed" ? "is-passed" : ""}`} role="status">
        <b>{validation?.historical_passed === false ? "역사 검증 상태 확인 중" : "과거 데이터 검증 완료"}</b>
        <span>{validation?.state === "passed" ? "미래 검증 완료" : validation?.state === "unknown" ? "미래 검증 상태 확인 중" : `미래 검증 진행 중 · ${validation?.elapsed_days ?? 0}/${validation?.required_days ?? 30}일 · ${validation?.graded_games ?? 0}/${validation?.required_games ?? 300}경기`}</span>
      </div>}
      <section className="games-list" aria-live="polite" aria-busy={loading}>
        {loading && <div className="state-panel"><strong>경기 정보를 확인하고 있습니다.</strong></div>}
        {!loading && result?.available && result.state === "empty" && <div className="state-panel"><strong>오늘 예정된 정규시즌 경기가 없습니다.</strong></div>}
        {!loading && !result?.available && <div className="state-panel unavailable"><strong>예측 사용 불가</strong><p>{result?.reason || "검증된 당일 피드가 없습니다."}</p><span>품질 조건이 회복될 때까지 확률을 표시하지 않습니다.</span></div>}
        {!loading && result?.available && feed?.predictions.map((prediction) => <GameRow key={prediction.game_id} prediction={prediction} />)}
      </section>
      <section className="explanation-band">
        <div className="explanation-copy"><h2>이 확률은 어떻게 만들어지나요?</h2><div className="explanation-items"><p><b>경기 전 데이터만</b><span>Elo, 시즌 및 최근 성적, 득실점, 휴식일만 사용합니다.</span></p><p><b>날짜 단위 검증</b><span>같은 날 결과가 다른 경기 피처에 들어가지 않습니다.</span></p><p><b>실전 성능 추적</b><span>현재 예측을 제공하면서 30일·300경기 미래 성능을 별도로 검증합니다.</span></p></div></div>
        <div className="validation-summary"><h2>모델 검증 요약 <small>(2025 홀드아웃)</small></h2><div><span>Log Loss<br /><b>0.6819</b></span><span>Brier Score<br /><b>0.2445</b></span><span>워크포워드<br /><b>2022–24</b></span></div></div>
      </section>
      <footer className="disclaimer">ⓘ <span>확률은 정보 제공용이며 결과를 보장하지 않습니다. 정규시즌 승패만 다룹니다.</span></footer>
    </div>
  );
}
