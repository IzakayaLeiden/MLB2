import { NextRequest, NextResponse } from "next/server";
import { demoFeed } from "../../../lib/demo-feed.mjs";
import { easternDate, validatePredictionFeed } from "../../../lib/feed-validation.mjs";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const now = new Date();
  const targetDate = request.nextUrl.searchParams.get("date") || easternDate(now);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(targetDate)) {
    return NextResponse.json({ available: false, state: "unavailable", reason: "날짜 형식이 올바르지 않습니다." }, { status: 400 });
  }
  try {
    let payload: unknown;
    if (process.env.SITE_DEMO_FEED === "1") {
      payload = demoFeed(targetDate, now);
    } else {
      const base = process.env.PAGES_FEED_URL;
      if (!base) throw new Error("공개 피드 주소가 아직 설정되지 않았습니다.");
      const source = new URL(targetDate === easternDate(now) ? "latest.json" : `archive/${targetDate}.json`, base.endsWith("/") ? base : `${base}/`);
      const response = await fetch(source, { cache: "no-store", headers: { accept: "application/json" } });
      if (!response.ok) throw new Error(`피드 응답 오류: ${response.status}`);
      payload = await response.json();
    }
    return NextResponse.json(validatePredictionFeed(payload, { now, expectedDate: targetDate }), { headers: { "cache-control": "no-store" } });
  } catch (error) {
    return NextResponse.json({ available: false, state: "unavailable", code: "fetch_failed", reason: error instanceof Error ? error.message : "피드를 가져오지 못했습니다.", feed: null }, { headers: { "cache-control": "no-store" } });
  }
}
