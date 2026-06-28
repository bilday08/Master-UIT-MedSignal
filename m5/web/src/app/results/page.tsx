"use client";

import * as React from "react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  getAblation,
  getDiscordance,
  type AblationTable,
  type DiscordanceData,
} from "@/lib/api";

function fmt(x: number | null | undefined, d = 3) {
  return x == null ? "—" : x.toFixed(d);
}

function best(rows: Record<string, number | string>[], key: string, mode: "max" | "min") {
  const nums = rows.map((r) => r[key] as number).filter((v) => typeof v === "number");
  if (!nums.length) return null;
  return mode === "max" ? Math.max(...nums) : Math.min(...nums);
}

export default function ResultsPage() {
  const [data, setData] = React.useState<AblationTable | null>(null);
  const [disc, setDisc] = React.useState<DiscordanceData | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    getAblation().then(setData).catch((e) => setError(String(e)));
    getDiscordance().then(setDisc).catch(() => {});
  }, []);

  return (
    <main className="mx-auto max-w-7xl px-6 py-6">
      {error && (
        <p className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          Không tải được số liệu: {error}. Backend đang chạy ở cổng 8000 chứ?
        </p>
      )}
      {!data && !error && (
        <p className="text-sm text-muted-foreground">Đang tải số liệu…</p>
      )}

      {data && (
        <div className="space-y-6">
          {/* Plaque */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Phát hiện plaque</CardTitle>
              <CardDescription>
                Do tỷ lệ ca dương thấp, PR-AUC phản ánh chất lượng tốt hơn AUC-ROC. Ô đậm là tốt nhất mỗi cột.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tbl
                rows={data.plaque}
                cols={[
                  { key: "auc_roc", label: "AUC-ROC", mode: "max" },
                  { key: "pr_auc", label: "PR-AUC", mode: "max" },
                  { key: "sensitivity", label: "Sens", mode: "max" },
                  { key: "specificity", label: "Spec", mode: "max" },
                  { key: "f1", label: "F1", mode: "max" },
                ]}
              />
            </CardContent>
          </Card>

          {/* Echo + Risk */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Phân loại echogenicity</CardTitle>
                <CardDescription>Macro-F1 trên 3 mức (Low/Inter/High).</CardDescription>
              </CardHeader>
              <CardContent>
                <Tbl rows={data.echo} cols={[{ key: "macro_f1", label: "Macro-F1", mode: "max" }]} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Điểm nguy cơ</CardTitle>
                <CardDescription>R² càng cao càng tốt, MAE càng thấp càng tốt.</CardDescription>
              </CardHeader>
              <CardContent>
                <Tbl
                  rows={data.risk}
                  cols={[
                    { key: "mae", label: "MAE", mode: "min" },
                    { key: "r2", label: "R²", mode: "max" },
                  ]}
                />
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {disc && <Discordance disc={disc} />}
    </main>
  );
}

function Discordance({ disc }: { disc: DiscordanceData }) {
  const bestSens = Math.max(
    ...disc.comparison.map((c) => c.sensitivity ?? 0),
  );
  return (
    <div className="mt-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            Nhóm discordance: LDL thấp nhưng Lp(a) cao
          </CardTitle>
          <CardDescription>
            {disc.n_total} ca (LDL-C &lt; 130 và Lp(a) ≥ 50 mg/dL), trong đó {disc.n_positive} ca
            có mảng xơ vữa. Đây là nhóm mà chỉ nhìn LDL-C dễ bỏ sót.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-4 font-medium">Mô hình</th>
                  <th className="px-3 py-2 text-right font-medium">Độ nhạy</th>
                  <th className="px-3 py-2 text-right font-medium">Độ đặc hiệu</th>
                  <th className="px-3 py-2 text-right font-medium">F1</th>
                </tr>
              </thead>
              <tbody>
                {disc.comparison.map((c) => {
                  const isMM = c.model === "Multimodal";
                  const isBestSens = c.sensitivity === bestSens;
                  return (
                    <tr key={c.model} className="border-b last:border-0">
                      <td className={isMM ? "py-2 pr-4 font-semibold" : "py-2 pr-4"}>
                        {c.model}
                      </td>
                      <td
                        className={
                          "px-3 py-2 text-right tabular-nums " +
                          (isBestSens ? "font-bold text-emerald-600" : "")
                        }
                      >
                        {fmt(c.sensitivity, 3)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {fmt(c.specificity, 3)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">{fmt(c.f1, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="rounded-md bg-amber-50 p-2 text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-200">
            ⚠️ {disc.warning} Dùng dự đoán out-of-fold (không rò rỉ).
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Hiệu năng theo dải Lp(a)</CardTitle>
          <CardDescription>
            Trên toàn bộ 300 ca, không phụ thuộc cỡ mẫu nhỏ của nhóm discordance.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th className="py-2 pr-4 font-medium">Dải Lp(a)</th>
                <th className="px-3 py-2 text-right font-medium">n (dương)</th>
                <th className="px-3 py-2 text-right font-medium">AUC-ROC</th>
                <th className="px-3 py-2 text-right font-medium">PR-AUC</th>
                <th className="px-3 py-2 text-right font-medium">Độ nhạy</th>
              </tr>
            </thead>
            <tbody>
              {disc.lpa_stratified.map((t) => (
                <tr key={t.tier} className="border-b last:border-0">
                  <td className="py-2 pr-4">{t.tier}</td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {t.n} ({t.n_pos})
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmt(t.auc_roc, 3)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmt(t.pr_auc, 3)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmt(t.sensitivity, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}

function Tbl({
  rows,
  cols,
}: {
  rows: Record<string, number | string>[];
  cols: { key: string; label: string; mode: "max" | "min" }[];
}) {
  const bests = Object.fromEntries(cols.map((c) => [c.key, best(rows, c.key, c.mode)]));
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="py-2 pr-4 font-medium">Model</th>
            {cols.map((c) => (
              <th key={c.key} className="px-3 py-2 text-right font-medium">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isFusion = String(r.model).includes("Multimodal");
            return (
              <tr key={String(r.model)} className="border-b last:border-0">
                <td className={isFusion ? "py-2 pr-4 font-semibold" : "py-2 pr-4"}>
                  {String(r.model)}
                </td>
                {cols.map((c) => {
                  const v = r[c.key] as number;
                  const isBest = bests[c.key] != null && v === bests[c.key];
                  return (
                    <td
                      key={c.key}
                      className={
                        "px-3 py-2 text-right tabular-nums " +
                        (isBest ? "font-bold text-emerald-600" : "")
                      }
                    >
                      {fmt(v)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
