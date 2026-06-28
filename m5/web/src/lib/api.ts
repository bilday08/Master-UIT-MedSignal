// Typed client cho FastAPI backend (m5_serving).
// Base URL doc tu NEXT_PUBLIC_API_URL, mac dinh localhost:8000.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// 8 numeric + Sex — khop feature_columns cua backend.
export const NUMERIC_FEATURES = [
  "Age",
  "Lp(a)_mg_dL",
  "ApoB_mg_dL",
  "LDL_C_mg_dL",
  "Triglyceride_mg_dL",
  "Total_Cholesterol_mg_dL",
  "Non_HDL_mg_dL",
  "IMT_mm",
] as const;

export type NumericFeature = (typeof NUMERIC_FEATURES)[number];

export interface TabularInput extends Record<string, number | string> {
  Sex: "Male" | "Female";
}

export interface PredictResult {
  plaque_prob: number;
  plaque_label: 0 | 1;
  threshold: number;
  echo_class: "Low" | "Intermediate" | "High";
  echo_note: string;
  risk_score: number;
}

export interface HealthResult {
  status: string;
  model_ready: boolean;
  meta: Record<string, unknown>;
}

export interface SampleCase {
  patient_id: string;
  tabular: TabularInput;
  imt_image: string;
  cca_images: string[];
  ground_truth: { plaque: number; echo: string; risk: number };
}

export interface AblationTable {
  plaque: Array<{
    model: string;
    auc_roc: number;
    pr_auc: number;
    sensitivity: number;
    specificity: number;
    f1: number;
  }>;
  echo: Array<{ model: string; macro_f1: number }>;
  risk: Array<{ model: string; mae: number; r2: number }>;
}

export async function getHealth(): Promise<HealthResult> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/health ${res.status}`);
  return res.json();
}

export interface CaseSummary {
  patient_id: string;
  plaque: number;
  n_cca: number;
}

export async function getSamples(): Promise<SampleCase[]> {
  const res = await fetch(`${API_URL}/samples`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/samples ${res.status}`);
  return (await res.json()).cases;
}

// Danh sach nhe toan bo 300 ca cho dropdown.
export async function getCases(): Promise<CaseSummary[]> {
  const res = await fetch(`${API_URL}/cases`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/cases ${res.status}`);
  return (await res.json()).cases;
}

// Chi tiet 1 ca (tabular + ten anh) theo patient_id.
export async function getCase(pid: string): Promise<SampleCase> {
  const res = await fetch(`${API_URL}/case/${encodeURIComponent(pid)}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`/case/${pid} ${res.status}`);
  return res.json();
}

export async function getAblation(): Promise<AblationTable> {
  const res = await fetch(`${API_URL}/ablation`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/ablation ${res.status}`);
  return res.json();
}

export interface DiscordanceData {
  n_total: number;
  n_positive: number;
  warning: string;
  method: string;
  comparison: Array<{
    model: string;
    sensitivity: number | null;
    specificity: number | null;
    f1: number | null;
  }>;
  lpa_stratified: Array<{
    tier: string;
    n: number;
    n_pos: number;
    auc_roc: number;
    pr_auc: number;
    sensitivity: number;
    specificity: number;
  }>;
  cases: Array<{
    patient_id: string;
    ldl: number;
    lpa: number;
    plaque_true: number;
    plaque_pred: number;
    plaque_prob: number;
  }>;
}

export async function getDiscordance(): Promise<DiscordanceData> {
  const res = await fetch(`${API_URL}/discordance`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/discordance ${res.status}`);
  return res.json();
}

// Grad-CAM: heatmap PNG tren anh IMT theo task plaque. Tra ve object URL.
export async function gradcam(
  tabular: TabularInput,
  imtImage: File,
  ccaImages: File[] = [],
): Promise<string> {
  const form = new FormData();
  form.append("tabular", JSON.stringify(tabular));
  form.append("imt_image", imtImage);
  for (const f of ccaImages) form.append("cca_images", f);
  const res = await fetch(`${API_URL}/gradcam`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`/gradcam ${res.status}`);
  return URL.createObjectURL(await res.blob());
}

export interface ShapFeature {
  feature: string;
  importance: number;
}

export async function getShapGlobal(): Promise<ShapFeature[]> {
  const res = await fetch(`${API_URL}/shap/global`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/shap/global ${res.status}`);
  return (await res.json()).features;
}

// SHAP cho 1 ca: value>0 day du doan ve "co plaque", value<0 day ve "am tinh".
export interface ShapContribution {
  feature: string;
  value: number;
}

export async function shapLocal(
  tabular: TabularInput,
): Promise<ShapContribution[]> {
  const form = new FormData();
  form.append("tabular", JSON.stringify(tabular));
  const res = await fetch(`${API_URL}/shap/local`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`/shap/local ${res.status}`);
  return (await res.json()).features;
}

export function imageUrl(name: string): string {
  return `${API_URL}/image/${encodeURIComponent(name)}`;
}

// Tai 1 anh tu backend ve thanh File (de submit lai qua /predict).
export async function fetchImageFile(name: string): Promise<File> {
  const res = await fetch(imageUrl(name));
  if (!res.ok) throw new Error(`/image/${name} ${res.status}`);
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type || "image/png" });
}

export async function predict(
  tabular: TabularInput,
  imtImage: File,
  ccaImages: File[] = [],
): Promise<PredictResult> {
  const form = new FormData();
  form.append("tabular", JSON.stringify(tabular));
  form.append("imt_image", imtImage);
  for (const f of ccaImages) form.append("cca_images", f);

  const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `/predict ${res.status}`);
  }
  return res.json();
}
