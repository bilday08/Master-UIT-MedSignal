"use client";

import * as React from "react";

import { PredictionForm } from "@/components/PredictionForm";
import { ResultCard } from "@/components/ResultCard";
import { ResultSkeleton } from "@/components/ResultSkeleton";
import type { PredictResult, ShapContribution } from "@/lib/api";

export default function HomePage() {
  const [result, setResult] = React.useState<PredictResult | null>(null);
  const [gradcamUrl, setGradcamUrl] = React.useState<string | null>(null);
  const [shap, setShap] = React.useState<ShapContribution[] | null>(null);
  const [loading, setLoading] = React.useState(false);

  return (
    <main className="mx-auto max-w-7xl px-6 py-6">
      <div className="grid gap-6 md:grid-cols-5">
        <div className="md:col-span-3">
          <PredictionForm
            onResult={setResult}
            onLoadingChange={setLoading}
            onGradcam={setGradcamUrl}
            onShap={setShap}
          />
        </div>
        <div className="md:col-span-2">
          {loading ? (
            <ResultSkeleton />
          ) : result ? (
            <ResultCard result={result} gradcamUrl={gradcamUrl} shap={shap} />
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
