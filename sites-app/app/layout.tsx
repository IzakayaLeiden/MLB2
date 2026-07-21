import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";
import { SiteHeader } from "./site-header";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") || requestHeaders.get("host") || "localhost";
  const protocol = requestHeaders.get("x-forwarded-proto") || (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const title = "MLB2 | 검증형 MLB 승리 확률";
  const description = "날짜 누수를 차단하고 미래 검증 상태를 공개하는 MLB 정규시즌 승리 확률 정보 서비스입니다.";
  return {
    metadataBase: new URL(origin), title, description,
    openGraph: { title, description, type: "website", images: [{ url: "/og.png", width: 1200, height: 630, alt: "MLB2 검증형 MLB 승리 확률" }] },
    twitter: { card: "summary_large_image", title, description, images: ["/og.png"] },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>
        <SiteHeader />
        <main>{children}</main>
      </body>
    </html>
  );
}
