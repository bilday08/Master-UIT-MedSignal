import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { PredictResult } from "@/lib/api";

function pct(x: number) {
  return `${(x * 100).toFixed(1)}%`;
}

const ECHO_VI: Record<string, string> = {
  Low: "Thấp",
  Intermediate: "Trung gian",
  High: "Cao",
};

export function ResultCard({ result }: { result: PredictResult }) {
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

        <p className="rounded-md bg-amber-50 p-2 text-[11px] text-amber-800 dark:bg-amber-950 dark:text-amber-200">
          Kết quả mang tính tham khảo, cần bác sĩ chuyên khoa diễn giải trên bối cảnh lâm sàng.
        </p>
      </CardContent>
    </Card>
  );
}
