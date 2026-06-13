import { Microscope } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function ExplainPage() {
  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight">Diễn giải kết quả</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Trực quan hoá cơ sở dẫn tới dự đoán: vùng ảnh và chỉ số ảnh hưởng nhiều nhất.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Grad-CAM</CardTitle>
            <CardDescription>Vùng ảnh siêu âm model chú ý khi dự đoán plaque.</CardDescription>
          </CardHeader>
          <CardContent className="flex h-40 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
            <span className="flex items-center gap-2">
              <Microscope className="size-4" /> Sắp có: heatmap trên ảnh IMT/CCA
            </span>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">SHAP</CardTitle>
            <CardDescription>Chỉ số lipid nào đẩy dự đoán lên hay xuống.</CardDescription>
          </CardHeader>
          <CardContent className="flex h-40 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
            <span className="flex items-center gap-2">
              <Microscope className="size-4" /> Sắp có: summary plot + waterfall theo ca
            </span>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
