/** The three studies the playground can show. All share a 420 x 360 fine grid. */
export interface DomainDef {
  id: string;
  label: string;
  sub: string;
  input: string;
  dir: string;
  /** RMSE reduction against bilinear, on this domain's scored box. */
  gain: number;
  rmse: number;
  bilinear: number;
  note: string;
  accent: string;
}

export const DOMAINS: DomainDef[] = [
  {
    id: "austin", label: "Austin, Texas", sub: "flat · convective",
    input: "IMERG 10 km", dir: "/data",
    gain: 6.8, rmse: 4.583, bilinear: 4.919,
    note: "No cell exceeds 5° slope. Rain falls where a storm happens to fire, and nothing in a 10 km average predicts that.",
    accent: "#8296a6",
  },
  {
    id: "colorado", label: "Colorado Front Range", sub: "orographic · 25.5 % steep",
    input: "IMERG 10 km", dir: "/data-colorado",
    gain: 15.0, rmse: 2.825, bilinear: 3.324,
    note: "Elevation 1,221–4,245 m. Terrain decides where precipitation lands, so the mapping becomes learnable — skill more than doubles.",
    accent: "#2d6ca8",
  },
  {
    id: "power", label: "NASA POWER", sub: "50 km reanalysis input",
    input: "POWER 0.5°", dir: "/data-power",
    gain: 9.9, rmse: 5.679, bilinear: 6.303,
    note: "Same Austin box, but the input is 0.5° MERRA-2: 3,600 fine cells per coarse cell instead of 144. A larger relative gain off a far worse starting point.",
    accent: "#cf5f2e",
  },
];

/** Seasonal decomposition of the Colorado domain — the study's central result. */
export const LADDER = [
  { label: "Colorado, cool season", sub: "Oct–Apr · orographic", gain: 20.8, accent: "#2d6ca8" },
  { label: "Colorado, all year", sub: "mixed", gain: 15.0, accent: "#7d8f5f" },
  { label: "Colorado, warm season", sub: "May–Sep · convective", gain: 8.0, accent: "#c98a04" },
  { label: "Austin, all year", sub: "convective, flat", gain: 6.8, accent: "#8296a6" },
];

export const SEEDS = [
  { family: "XGBoost", n: 4, values: [4.578, 4.583, 4.585, 4.583], accent: "#2d6ca8" },
  { family: "Swin 7.9 M", n: 3, values: [4.614, 4.646, 4.653], accent: "#cf5f2e" },
  { family: "CNN 1.65 M", n: 3, values: [4.663, 4.669, 4.7005], accent: "#c98a04" },
];

export const TRANSFER = [
  { train: "Austin", score: "Austin", rmse: 4.583, gain: 6.8 },
  { train: "Texas + west", score: "Austin", rmse: 4.609, gain: -0.6 },
  { train: "Colorado", score: "Colorado", rmse: 2.825, gain: 15.0 },
  { train: "Colorado", score: "Austin", rmse: 5.593, gain: -13.7 },
];
