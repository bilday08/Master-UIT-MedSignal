"use client";

import * as React from "react";

import { PredictionForm } from "@/components/PredictionForm";
import { ResultCard } from "@/components/ResultCard";
import { ResultSkeleton } from "@/components/ResultSkeleton";
import type { PredictResult } from "@/lib/api";

export default function HomePage() {
  const [result, setResult] = React.useState<PredictResult | null>(null);
  const [loading, setLoading] = React.useState(false);

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight">
          Chẩn đoán xơ vữa động mạch cảnh
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Kết hợp chỉ số lipid và ảnh siêu âm để dự đoán mảng xơ vữa, độ hồi âm, và điểm nguy cơ.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-5">
        <div className="md:col-span-3">
          <PredictionForm onResult={setResult} onLoadingChange={setLoading} />
        </div>
        <div className="md:col-span-2">
          {loading ? (
            <ResultSkeleton />
          ) : result ? (
            <ResultCard result={result} />
          ) : (
            <div className="flex h-full items-center justify-center rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
              Nhập chỉ số + ảnh rồi bấm <span className="mx-1 font-medium">Dự đoán</span>
              để xem kết quả.
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
