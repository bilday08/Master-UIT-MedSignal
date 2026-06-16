import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { PredictResult, ShapContribution } from "@/lib/api";

function pct(x: number) {
  return `${(x * 100).toFixed(1)}%`;
}

const ECHO_VI: Record<string, string> = {
  Low: "Thấp",
  Intermediate: "Trung gian",
  High: "Cao",
};

const FEATURE_VI: Record<string, string> = {
  Age: "Tuổi",
  "Lp(a)_mg_dL": "Lp(a)",
  ApoB_mg_dL: "ApoB",
  LDL_C_mg_dL: "LDL-C",
  Triglyceride_mg_dL: "Triglyceride",
  Total_Cholesterol_mg_dL: "Total-C",
  Non_HDL_mg_dL: "Non-HDL",
  IMT_mm: "IMT",
  Sex: "Giới tính",
};

function ShapSection({ shap }: { shap: ShapContribution[] }) {
  const top = shap.slice(0, 5);
  const maxAbs = Math.max(...top.map((s) => Math.abs(s.value)), 1e-6);
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-muted-foreground">
        Chỉ số ảnh hưởng tới dự đoán
      </p>
      <div className="space-y-1.5">
        {top.map((s) => {
          const up = s.value > 0;
          const w = (Math.abs(s.value) / maxAbs) * 100;
          return (
            <div key={s.feature} className="flex items-center gap-2 text-xs">
              <span className="w-24 shrink-0 font-medium">
                {FEATURE_VI[s.feature] ?? s.feature}
              </span>
              <div className="flex h-2 flex-1 items-center">
                <div
                  className={up ? "h-full rounded-full bg-destructive" : "h-full rounded-full bg-emerald-600"}
                  style={{ width: `${w}%` }}
                />
              </div>
              <span className={up ? "text-destructive" : "text-emerald-600"}>
                {up ? "↑" : "↓"}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-[10px] text-muted-foreground">
        <span className="text-destructive">↑ đỏ</span> đẩy về phía có mảng xơ vữa,{" "}
        <span className="text-emerald-600">↓ xanh</span> đẩy về âm tính.
      </p>
    </div>
  );
}

export function ResultCard({
  result,
  gradcamUrl,
  shap,
}: {
  result: PredictResult;
  gradcamUrl?: string | null;
  shap?: ShapContribution[] | null;
}) {
  const positive = result.plaque_label === 1;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">Kết quả dự đoán</CardTitle>
        <CardDescription>
          Ngưỡng quyết định plaque: {result.threshold}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Plaque */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Mảng xơ vữa</span>
            <Badge variant={positive ? "destructive" : "success"}>
              {positive ? "Có nguy cơ" : "Âm tính"}
            </Badge>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
            <div
              className={positive ? "h-full bg-destructive" : "h-full bg-emerald-600"}
              style={{ width: pct(result.plaque_prob) }}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Xác suất: <span className="font-semibold">{pct(result.plaque_prob)}</span>
          </p>
        </div>

        {/* Echogenicity + Risk */}
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border p-3">
            <p className="text-xs text-muted-foreground">Độ hồi âm</p>
            <p className="text-lg font-semibold">
              {ECHO_VI[result.echo_class] ?? result.echo_class}
            </p>
            <p className="mt-1 text-[10px] leading-tight text-muted-foreground">
              {result.echo_note}
            </p>
          </div>
          <div className="rounded-lg border p-3">
            <p className="text-xs text-muted-foreground">Điểm nguy cơ</p>
            <p className="text-lg font-semibold">{result.risk_score.toFixed(3)}</p>
            <p className="mt-1 text-[10px] leading-tight text-muted-foreground">
              Thang điểm nguy cơ liên tục.
            </p>
          </div>
        </div>

        {shap && shap.length > 0 && <ShapSection shap={shap} />}

        {gradcamUrl && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-muted-foreground">
              Vùng ảnh model chú ý (Grad-CAM)
            </p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={gradcamUrl}
              alt="Grad-CAM heatmap"
              className="w-full max-w-64 rounded-lg border"
            />
            <p className="text-[10px] text-muted-foreground">
              Màu nóng là vùng ảnh ảnh hưởng nhiều nhất tới dự đoán mảng xơ vữa.
            </p>
          </div>
        )}

        <p className="rounded-md bg-amber-50 p-2 text-[11px] text-amber-800 dark:bg-amber-950 dark:text-amber-200">
          Kết quả mang tính tham khảo, cần bác sĩ chuyên khoa diễn giải trên bối cảnh lâm sàng.
        </p>
      </CardContent>
    </Card>
  );
}
