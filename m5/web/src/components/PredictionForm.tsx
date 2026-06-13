"use client";

import * as React from "react";
import {
  Check,
  ChevronsUpDown,
  HelpCircle,
  Loader2,
  Upload,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  fetchImageFile,
  getCase,
  getCases,
  gradcam,
  NUMERIC_FEATURES,
  predict,
  shapLocal,
  type CaseSummary,
  type PredictResult,
  type ShapContribution,
  type TabularInput,
} from "@/lib/api";

const LIPID = [
  "Lp(a)_mg_dL",
  "ApoB_mg_dL",
  "LDL_C_mg_dL",
  "Triglyceride_mg_dL",
  "Total_Cholesterol_mg_dL",
  "Non_HDL_mg_dL",
] as const;
const CLINICAL = ["Age", "IMT_mm"] as const;

const LABELS: Record<string, string> = {
  Age: "Tuổi",
  "Lp(a)_mg_dL": "Lp(a)",
  ApoB_mg_dL: "ApoB",
  LDL_C_mg_dL: "LDL-C",
  Triglyceride_mg_dL: "Triglyceride",
  Total_Cholesterol_mg_dL: "Total-C",
  Non_HDL_mg_dL: "Non-HDL",
  IMT_mm: "IMT",
};
const UNITS: Record<string, string> = {
  Age: "năm",
  IMT_mm: "mm",
  "Lp(a)_mg_dL": "mg/dL",
  ApoB_mg_dL: "mg/dL",
  LDL_C_mg_dL: "mg/dL",
  Triglyceride_mg_dL: "mg/dL",
  Total_Cholesterol_mg_dL: "mg/dL",
  Non_HDL_mg_dL: "mg/dL",
};
// Giai thich tung chi so (hien khi re vao dau ?).
const HELP: Record<string, string> = {
  Age: "Tuổi bệnh nhân. Nguy cơ xơ vữa tăng theo tuổi.",
  "Lp(a)_mg_dL":
    "Lipoprotein(a): yếu tố nguy cơ xơ vữa do di truyền, độc lập với LDL. Cao nghĩa là nguy cơ cao dù LDL bình thường.",
  ApoB_mg_dL:
    "Apolipoprotein B: đếm số hạt lipoprotein gây xơ vữa. Phản ánh nguy cơ tốt hơn LDL-C đơn thuần.",
  LDL_C_mg_dL:
    "Cholesterol LDL (loại 'xấu'): mục tiêu điều trị chính của rối loạn lipid máu.",
  Triglyceride_mg_dL:
    "Mỡ trung tính trong máu. Cao thường đi kèm hội chứng chuyển hoá.",
  Total_Cholesterol_mg_dL: "Cholesterol toàn phần = HDL + LDL + VLDL.",
  Non_HDL_mg_dL:
    "Cholesterol không HDL (= TC trừ HDL): gồm toàn bộ hạt gây xơ vữa.",
  IMT_mm:
    "Độ dày lớp nội-trung mạc động mạch cảnh (mm): chỉ dấu xơ vữa sớm trên siêu âm.",
  Sex: "Giới tính sinh học, ảnh hưởng tới nguy cơ tim mạch.",
};

const DEFAULTS: Record<string, string> = {
  Age: "62",
  "Lp(a)_mg_dL": "80",
  ApoB_mg_dL: "110",
  LDL_C_mg_dL: "120",
  Triglyceride_mg_dL: "150",
  Total_Cholesterol_mg_dL: "210",
  Non_HDL_mg_dL: "160",
  IMT_mm: "0.95",
};

function HelpTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          tabIndex={-1}
          aria-label="Giải thích"
          className="cursor-help text-muted-foreground transition-colors hover:text-foreground"
        >
          <HelpCircle className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent>{text}</TooltipContent>
    </Tooltip>
  );
}

function NumField({
  name,
  value,
  onChange,
}: {
  name: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={name} className="flex items-center justify-between gap-1">
        <span className="flex items-center gap-1">
          {LABELS[name]}
          <HelpTip text={HELP[name]} />
        </span>
        <span className="text-[10px] font-normal text-muted-foreground">
          {UNITS[name]}
        </span>
      </Label>
      <Input
        id={name}
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required
      />
    </div>
  );
}

// Tao object URL TRONG effect (khong dung useMemo): tranh bi revoke som boi
// Strict Mode double-invoke lam anh vo. URL duoc tao lai sau moi mount.
function Thumb({ file, label }: { file: File; label: string }) {
  const [url, setUrl] = React.useState<string>("");
  React.useEffect(() => {
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return (
    <div className="flex flex-col items-center gap-1">
      {url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={url}
          alt={label}
          className="size-14 rounded-md border object-cover"
        />
      ) : (
        <div className="size-14 rounded-md border bg-muted" />
      )}
      <span className="max-w-14 truncate text-[9px] text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

function Dropzone({
  multiple,
  files,
  onFiles,
  hint,
}: {
  multiple?: boolean;
  files: File[];
  onFiles: (f: File[]) => void;
  hint: string;
}) {
  const [drag, setDrag] = React.useState(false);
  const ref = React.useRef<HTMLInputElement>(null);

  function accept(list: FileList | null) {
    if (!list) return;
    const imgs = Array.from(list).filter((f) => f.type.startsWith("image/"));
    onFiles(multiple ? imgs.slice(0, 4) : imgs.slice(0, 1));
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        setDrag(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        accept(e.dataTransfer.files);
      }}
      onClick={() => ref.current?.click()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed p-3 text-center text-xs transition-colors",
        drag ? "border-primary bg-accent" : "hover:bg-accent/50",
      )}
    >
      <input
        ref={ref}
        type="file"
        accept="image/*"
        multiple={multiple}
        className="hidden"
        onChange={(e) => accept(e.target.files)}
      />
      <Upload className="size-4 text-muted-foreground" />
      <span className="text-muted-foreground">
        {files.length ? `${files.length} ảnh đã chọn (bấm để đổi)` : hint}
      </span>
      {files.length > 0 && (
        <div
          className="mt-1 flex flex-wrap justify-center gap-2"
          onClick={(e) => e.stopPropagation()}
        >
          {files.map((f, i) => (
            <Thumb key={`${f.name}-${i}`} file={f} label={f.name} />
          ))}
        </div>
      )}
    </div>
  );
}

const selectCls =
  "flex h-9 w-full cursor-pointer rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";

function caseLabel(c: CaseSummary) {
  return `${c.patient_id} (${c.plaque ? "có plaque" : "âm tính"}, ${c.n_cca} CCA)`;
}

// Combobox go text + chon (shadcn Popover + Command). Loc 300 ca theo ma benh nhan.
function CaseCombobox({
  cases,
  picked,
  onPick,
}: {
  cases: CaseSummary[];
  picked: string;
  onPick: (id: string) => void;
}) {
  const [open, setOpen] = React.useState(false);
  const current = cases.find((c) => c.patient_id === picked);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={!cases.length}
          className="w-full justify-between font-normal"
        >
          <span className="truncate">
            {current ? caseLabel(current) : "Chọn hoặc gõ mã ca"}
          </span>
          <ChevronsUpDown className="size-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0">
        <Command>
          <CommandInput placeholder="Gõ mã ca, vd P012" />
          <CommandList>
            <CommandEmpty>Không tìm thấy ca.</CommandEmpty>
            <CommandGroup>
              {cases.map((c) => (
                <CommandItem
                  key={c.patient_id}
                  value={`${c.patient_id} ${c.plaque ? "plaque" : "am tinh"}`}
                  onSelect={() => {
                    onPick(c.patient_id);
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "size-4",
                      picked === c.patient_id ? "opacity-100" : "opacity-0",
                    )}
                  />
                  <span className="font-medium">{c.patient_id}</span>
                  <span className="ml-auto text-xs text-muted-foreground">
                    {c.plaque ? "có plaque" : "âm tính"}, {c.n_cca} CCA
                  </span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

export function PredictionForm({
  onResult,
  onLoadingChange,
  onGradcam,
  onShap,
}: {
  onResult: (r: PredictResult) => void;
  onLoadingChange?: (loading: boolean) => void;
  onGradcam?: (url: string | null) => void;
  onShap?: (shap: ShapContribution[] | null) => void;
}) {
  const [values, setValues] = React.useState<Record<string, string>>(DEFAULTS);
  const [sex, setSex] = React.useState<"Male" | "Female">("Male");
  const [imt, setImt] = React.useState<File[]>([]);
  const [cca, setCca] = React.useState<File[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const [cases, setCases] = React.useState<CaseSummary[]>([]);
  const [picked, setPicked] = React.useState<string>("");
  const [gt, setGt] = React.useState<string | null>(null);
  const [loadingCase, setLoadingCase] = React.useState(false);

  React.useEffect(() => {
    // Khong tu chon ca nao: ban dau de trong, combobox hien placeholder.
    getCases()
      .then(setCases)
      .catch(() => {});
  }, []);

  function setField(k: string, v: string) {
    setValues((s) => ({ ...s, [k]: v }));
  }

  async function loadCase(pid: string) {
    if (!pid) return;
    setPicked(pid);
    setLoadingCase(true);
    setError(null);
    try {
      const c = await getCase(pid);
      const next: Record<string, string> = {};
      for (const f of NUMERIC_FEATURES) next[f] = String(c.tabular[f]);
      setValues(next);
      setSex(c.tabular.Sex);
      setImt([await fetchImageFile(c.imt_image)]);
      setCca(await Promise.all(c.cca_images.map(fetchImageFile)));
      setGt(
        `Chẩn đoán thực tế ${c.patient_id}: plaque=${c.ground_truth.plaque}, echo=${c.ground_truth.echo}, risk=${c.ground_truth.risk.toFixed(2)}`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingCase(false);
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!imt.length) {
      setError("Cần ảnh IMT. Kéo thả/chọn ảnh hoặc bấm Tải ca.");
      return;
    }
    const tabular: TabularInput = { Sex: sex };
    for (const f of NUMERIC_FEATURES) tabular[f] = Number(values[f]);
    setLoading(true);
    onLoadingChange?.(true);
    onGradcam?.(null);
    onShap?.(null);
    try {
      const r = await predict(tabular, imt[0], cca);
      // Grad-CAM + SHAP local (best-effort, khong chan ket qua chinh).
      const [cam, shap] = await Promise.all([
        gradcam(tabular, imt[0], cca).catch(() => null),
        shapLocal(tabular).catch(() => null),
      ]);
      onResult(r);
      onGradcam?.(cam);
      onShap?.(shap);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
      onLoadingChange?.(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Nhập chỉ số bệnh nhân</CardTitle>
        <CardDescription>
          Chọn một ca có sẵn để tự điền chỉ số và ảnh, hoặc nhập tay và kéo thả ảnh.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Chon ca co san (combobox go text) -> chon la nap luon */}
        <div className="space-y-1.5 rounded-lg border bg-muted/40 p-3">
          <Label className="flex items-center gap-2">
            Ca có sẵn ({cases.length})
            {loadingCase && <Loader2 className="size-3.5 animate-spin text-muted-foreground" />}
          </Label>
          <CaseCombobox cases={cases} picked={picked} onPick={loadCase} />
          <p className="text-[11px] text-muted-foreground">
            Chọn một ca để tự điền chỉ số và ảnh.
          </p>
        </div>

        <form onSubmit={onSubmit} className="space-y-4">
          <section className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Bộ lipid
            </h3>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {LIPID.map((f) => (
                <NumField key={f} name={f} value={values[f] ?? ""} onChange={(v) => setField(f, v)} />
              ))}
            </div>
          </section>

          <section className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Lâm sàng
            </h3>
            <div className="grid grid-cols-3 gap-3">
              {CLINICAL.map((f) => (
                <NumField key={f} name={f} value={values[f] ?? ""} onChange={(v) => setField(f, v)} />
              ))}
              <div className="space-y-1.5">
                <Label htmlFor="Sex" className="flex items-center gap-1">
                  Giới tính
                  <HelpTip text={HELP.Sex} />
                </Label>
                <Select value={sex} onValueChange={(v) => setSex(v as "Male" | "Female")}>
                  <SelectTrigger id="Sex">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Male">Nam</SelectItem>
                    <SelectItem value="Female">Nữ</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </section>

          <section className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Ảnh siêu âm (kéo thả hoặc bấm để chọn)
            </h3>
            {/* IMT chi 1 anh nen cho hep (4/12); CCA can rong (8/12) de 4 anh tren 1 hang */}
            <div className="grid grid-cols-12 gap-3">
              <div className="col-span-4 space-y-1.5">
                <Label className="flex items-center gap-1">
                  Ảnh IMT <span className="text-destructive">*</span>
                  <HelpTip text="1 ảnh dọc động mạch cảnh để đo IMT và phát hiện mảng xơ vữa." />
                </Label>
                <Dropzone files={imt} onFiles={setImt} hint="Kéo thả ảnh IMT" />
              </div>
              <div className="col-span-8 space-y-1.5">
                <Label className="flex items-center gap-1">
                  Ảnh CCA (0 đến 4)
                  <HelpTip text="Ảnh cắt ngang động mạch cảnh chung, dùng để phân loại độ hồi âm của mảng xơ vữa." />
                </Label>
                <Dropzone multiple files={cca} onFiles={setCca} hint="Kéo thả tối đa 4 ảnh CCA" />
              </div>
            </div>
          </section>

          {gt && <p className="text-[11px] text-muted-foreground">{gt} (để đối chiếu).</p>}

          {error && (
            <p className="rounded-md bg-destructive/10 p-2 text-sm text-destructive">
              {error}
            </p>
          )}

          <Button type="submit" disabled={loading} className="w-full">
            {loading && <Loader2 className="size-4 animate-spin" />}
            {loading ? "Đang dự đoán" : "Dự đoán"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
