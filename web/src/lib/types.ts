export type LayerId =
  | "aorc" | "nearest" | "bilinear" | "xgboost"
  | "cnn" | "cnn_spectral" | "swin" | "diffusion" | "diffusion_member"
  /** Colorado and POWER ship one downscaled product rather than four. */
  | "ml";

export interface DayMetrics {
  rmse: number; mae: number; bias: number; r: number | null;
  max: number; mean: number; wet: number;
  pod: number | null; far: number | null; csi: number | null;
}

export interface Day {
  date: string;
  obsMean: number; obsMax: number; obsWet: number; obsHeavy: number;
  m: Partial<Record<Exclude<LayerId, "aorc">, DayMetrics>>;
}

export interface Meta {
  vmax: number; heavy: number; floor: number;
  width: number; height: number;
  encoding: string; bbox: [number, number, number, number];
  layers: LayerId[]; nDays: number; period: [string, string];
  /** Refinement factor: 12 for IMERG (144 fine cells per coarse), 60 for POWER (3,600). */
  factor?: number;
}

export interface Field {
  /** mm/day, row-major, length = width*height. NaN where no data. */
  data: Float32Array;
  width: number; height: number;
}

export const LAYER_META: Record<LayerId, {
  label: string; short: string; kind: "truth" | "input" | "baseline" | "learned" | "generative";
  note: string; color: string;
}> = {
  aorc:             { label: "AORC",                  short: "AORC",        kind: "truth",      color: "#111a22", note: "The rainfall record everything here is scored against — radar and rain gauges, on a 1 km grid." },
  nearest:          { label: "IMERG 10 km",           short: "IMERG",       kind: "input",      color: "#8296a6", note: "What the satellite delivers: one number spread flat over a 12 × 12 block of kilometres." },
  bilinear:         { label: "Bilinear",              short: "Bilinear",    kind: "baseline",   color: "#7b4f9d", note: "Blur the satellite estimate up to 1 km. The bar every model has to clear." },
  xgboost:          { label: "XGBoost 2-stage",       short: "XGBoost",     kind: "learned",    color: "#2d6ca8", note: "Thousands of small yes/no rules learned from past storms. Closest of the lot — and the blurriest." },
  cnn:              { label: "CNN (U-Net)",           short: "CNN",         kind: "learned",    color: "#1f9d85", note: "1.7 million settings, learned from the maps themselves." },
  cnn_spectral:     { label: "CNN + texture penalty", short: "CNN+tex",     kind: "learned",    color: "#b5468c", note: "The same CNN, trained to match the fine-scale detail of real rain as well as its value. Four to eight times the texture, for almost no change in error." },
  swin:             { label: "Swin transformer",      short: "Swin",        kind: "learned",    color: "#c98a04", note: "7.9 million settings. Best at finding where the heavy rain lands." },
  diffusion:        { label: "Diffusion, 6-member mean", short: "Diff mean", kind: "generative", color: "#c4503c", note: "Six attempts averaged together. Averaging smooths away the detail each attempt invented." },
  diffusion_member: { label: "Diffusion, one member", short: "Diff member", kind: "generative", color: "#e0714f", note: "A single attempt. The only one whose rainfall looks as patchy as the real thing." },
  ml:               { label: "XGBoost 2-stage",       short: "XGBoost",     kind: "learned",    color: "#2d6ca8", note: "Thousands of small yes/no rules learned from past storms, trained for this area." },
};

export const ORDER: LayerId[] = [
  "aorc", "nearest", "bilinear", "ml", "xgboost", "cnn", "cnn_spectral", "swin", "diffusion", "diffusion_member",
];
