"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const navigation = [["/", "오늘의 경기"], ["/model-validation", "모델 검증"], ["/methodology", "방법론"]] as const;

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <header className="site-header">
      <Link className="wordmark" href="/" aria-label="MLB2 홈">MLB<span>2</span></Link>
      <nav className={open ? "primary-nav is-open" : "primary-nav"} aria-label="주요 메뉴">
        {navigation.map(([href, label]) => (
          <Link key={href} href={href} className={pathname === href ? "active" : ""} onClick={() => setOpen(false)}>{label}</Link>
        ))}
      </nav>
      <div className="header-tools">
        <Link className="help-link" href="/methodology">ⓘ <span>도움말</span></Link>
        <button type="button" className="refresh-button" onClick={() => window.dispatchEvent(new Event("mlb2-refresh"))} aria-label="데이터 업데이트">↻ <span>업데이트</span></button>
        <time className="header-time">ET</time>
        <button type="button" className="menu-button" aria-expanded={open} aria-label="메뉴 열기" onClick={() => setOpen((value) => !value)}>☰</button>
      </div>
    </header>
  );
}
