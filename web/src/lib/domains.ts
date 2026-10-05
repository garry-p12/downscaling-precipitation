/**
 * Where the field tiles are served from. Empty means "this site", i.e. the
 * ~70 MB of PNGs under `public/`. Set `NEXT_PUBLIC_DATA_BASE` to an absolute
 * URL to keep them out of the deploy and serve them from a bucket or CDN
 * instead; the tiles are plain static files, so any host will do.
 */
export const DATA_BASE = (process.env.NEXT_PUBLIC_DATA_BASE ?? "").replace(/\/+$/, "");

/** Resolve a tile path against that base. */
export const dataUrl = (path: string) => `${DATA_BASE}${path}`;

/** The three places the viewer can show. All share a 420 x 360 fine grid. */
export interface DomainDef {
  id: string;
  label: string;
  sub: string;
  input: string;
  /** The coarse cell size, as the headline says it out loud. */
  coarse: string;
  /** What the coarse data is, for a sentence: "what the satellite delivers". */
  source: string;
  dir: string;
  /** How much closer than bilinear interpolation, as a percentage, over this area. */
  gain: number;
  rmse: number;
  bilinear: number;
  note: string;
  accent: string;
}

export const DOMAINS: DomainDef[] = [
  {
    id: "austin", label: "Austin, Texas", sub: "flat · summer thunderstorms",
    input: "Satellite, 10 km", coarse: "10 km", source: "satellite", dir: "/data",
    gain: 6.8, rmse: 4.583, bilinear: 4.919,
    note: "Nowhere here is steeper than a gentle hill. Rain falls where a storm happens to fire, and a 10 km average has no way of knowing where that is.",
    accent: "#8296a6",
  },
  {
    id: "colorado", label: "Colorado Front Range", sub: "mountains · a quarter of it steep",
    input: "Satellite, 10 km", coarse: "10 km", source: "satellite", dir: "/data-colorado",
    gain: 15.0, rmse: 2.825, bilinear: 3.324,
    note: "From 1,200 m to 4,200 m. The mountains decide where rain lands, so there is a real pattern to learn — and the models get more than twice as much out of it.",
    accent: "#2d6ca8",
  },
  {
    id: "power", label: "NASA POWER", sub: "a much blurrier starting point",
    input: "Weather model, 50 km", coarse: "50 km", source: "weather model", dir: "/data-power",
    gain: 9.9, rmse: 5.679, bilinear: 6.303,
    note: "The same patch of Texas, but starting from 50 km squares instead of 10 km — 25 times blurrier. More to gain, from a far worse start.",
    accent: "#cf5f2e",
  },
];

/** Colorado split by season — the clearest result in the study. */
export const LADDER = [
  { label: "Colorado, Oct–Apr", sub: "storms pushed up over the mountains", gain: 20.8, accent: "#2d6ca8" },
  { label: "Colorado, all year", sub: "a mix of both", gain: 15.0, accent: "#7d8f5f" },
  { label: "Colorado, May–Sep", sub: "summer thunderstorms", gain: 8.0, accent: "#c98a04" },
  { label: "Austin, all year", sub: "thunderstorms, flat ground", gain: 6.8, accent: "#8296a6" },
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
