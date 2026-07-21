import { NextRequest, NextResponse } from "next/server";
import { demoFeed } from "../../../lib/demo-feed.mjs";
import { easternDate, validatePredictionFeed } from "../../../lib/feed-validation.mjs";

export const dynamic = "force-dynamic";

type GateStatus = {
  passed?: boolean;
  checks?: Record<string, boolean>;
  metrics?: { elapsed_days?: number; graded_games?: number; coverage?: number };
};

function validationSummary(status: GateStatus | null) {
  return {
    state: status?.passed === true ? "passed" : status ? "monitoring" : "unknown",
    historical_passed: status?.checks?.historical_holdout_passed === true,
    elapsed_days: status?.metrics?.elapsed_days ?? 0,
    required_days: 30,
    graded_games: status?.metrics?.graded_games ?? 0,
    required_games: 300,
    coverage: status?.metrics?.coverage ?? null,
  };
}

export async function GET(request: NextRequest) {
  const now = new Date();
  const targetDate = request.nextUrl.searchParams.get("date") || easternDate(now);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(targetDate)) {
    return NextResponse.json({ available: false, state: "unavailable", reason: "날짜 형식이 올바르지 않습니다." }, { status: 400 });
  }
  try {
    let payload: unknown;
    let gateStatus: GateStatus | null = null;
    if (process.env.SITE_DEMO_FEED === "1") {
      payload = demoFeed(targetDate, now);
      gateStatus = { passed: false, checks: { historical_holdout_passed: true }, metrics: { elapsed_days: 0, graded_games: 0, coverage: 1 } };
    } else {
      const base = process.env.PAGES_FEED_URL;
      if (!base) throw new Error("공개 피드 주소가 아직 설정되지 않았습니다.");
      const source = new URL(targetDate === easternDate(now) ? "latest.json" : `archive/${targetDate}.json`, base.endsWith("/") ? base : `${base}/`);
      const response = await fetch(source, { cache: "no-store", headers: { accept: "application/json" } });
      if (!response.ok) throw new Error(`피드 응답 오류: ${response.status}`);
      payload = await response.json();
      try {
        const statusResponse = await fetch(new URL("status.json", base.endsWith("/") ? base : `${base}/`), { cache: "no-store", headers: { accept: "application/json" } });
        if (statusResponse.ok) gateStatus = await statusResponse.json() as GateStatus;
      } catch {
        gateStatus = null;
      }
    }
    return NextResponse.json({ ...validatePredictionFeed(payload, { now, expectedDate: targetDate }), validation: validationSummary(gateStatus) }, { headers: { "cache-control": "no-store" } });
  } catch (error) {
    return NextResponse.json({ available: false, state: "unavailable", code: "fetch_failed", reason: error instanceof Error ? error.message : "피드를 가져오지 못했습니다.", feed: null }, { headers: { "cache-control": "no-store" } });
  }
}
