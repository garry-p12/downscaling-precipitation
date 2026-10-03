export type LayerId =
  | "aorc" | "nearest" | "bilinear" | "xgboost"
  | "cnn" | "swin" | "diffusion" | "diffusion_member"
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
  aorc:             { label: "AORC",                short: "Truth",      kind: "truth",      color: "#111a22", note: "1 km gridded analysis — the reference every score is measured against." },
  nearest:          { label: "IMERG 10 km",          short: "IMERG 10 km", kind: "input",     color: "#8296a6", note: "The satellite retrieval as delivered — one flat value per 12×12 block of 1 km cells." },
  bilinear:         { label: "Bilinear",            short: "Bilinear",   kind: "baseline",   color: "#7b4f9d", note: "Smooth interpolation. The bar every model must clear." },
  xgboost:          { label: "XGBoost 2-stage",     short: "XGBoost",    kind: "learned",    color: "#2d6ca8", note: "Best RMSE of any product — and the smoothest, at 2 % of observed texture." },
  cnn:              { label: "CNN (U-Net)",         short: "CNN",        kind: "learned",    color: "#1f9d85", note: "1.65 M parameters over the 1 km grid." },
  swin:             { label: "Swin transformer",    short: "Swin",       kind: "learned",    color: "#c98a04", note: "7.9 M parameters. Best heavy-event detection and the only KGE above bilinear." },
  diffusion:        { label: "Diffusion (ens. mean)", short: "Diff mean", kind: "generative", color: "#c4503c", note: "Mean of 6 diffusion members. Averaging re-smooths what sampling created." },
  diffusion_member: { label: "Diffusion (1 member)", short: "Diff member", kind: "generative", color: "#e0714f", note: "A single draw. 84 % of observed fine-scale variance — the only realistic texture here." },
  ml:               { label: "XGBoost 2-stage",       short: "XGBoost",    kind: "learned",    color: "#2d6ca8", note: "The shipped 2-stage gradient-boosted product for this domain." },
};

export const ORDER: LayerId[] = [
  "aorc", "nearest", "bilinear", "ml", "xgboost", "cnn", "swin", "diffusion", "diffusion_member",
];
