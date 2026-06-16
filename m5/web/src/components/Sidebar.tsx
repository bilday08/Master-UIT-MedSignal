"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, BarChart3 } from "lucide-react";

import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Chẩn đoán", icon: Activity, desc: "Dự đoán cho 1 bệnh nhân" },
  { href: "/results", label: "Hiệu năng", icon: BarChart3, desc: "So sánh các mô hình" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-60 shrink-0 border-r bg-card md:flex md:flex-col">
      <Link
        href="/"
        className="flex items-center gap-3 border-b px-5 py-4 transition-colors hover:bg-accent"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/logo.jpg"
          alt="MedSignal logo"
          className="size-10 shrink-0 rounded-md object-cover"
        />
        <div className="min-w-0">
          <p className="text-base font-bold tracking-tight">MedSignal</p>
          <p className="text-xs leading-tight text-muted-foreground">
            Chẩn đoán xơ vữa động mạch cảnh
          </p>
        </div>
      </Link>
      <nav className="flex flex-col gap-1 p-3">
        {NAV.map(({ href, label, icon: Icon, desc }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-start gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
                active
                  ? "bg-primary text-primary-foreground"
                  : "text-foreground hover:bg-accent",
              )}
            >
              <Icon className="mt-0.5 size-4 shrink-0" />
              <span className="flex flex-col">
                <span className="font-medium">{label}</span>
                <span
                  className={cn(
                    "text-[11px]",
                    active ? "text-primary-foreground/70" : "text-muted-foreground",
                  )}
                >
                  {desc}
                </span>
              </span>
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto p-4 text-[10px] leading-relaxed text-muted-foreground">
        Công cụ hỗ trợ, không thay thế chẩn đoán của bác sĩ.
      </div>
    </aside>
  );
}
