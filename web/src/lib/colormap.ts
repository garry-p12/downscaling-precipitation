/** Precipitation ramps and a diverging ramp for differences. */
type Stop = [number, [number, number, number]];

export type RampId = "warm" | "nws";

/**
 * The reference design's single-hue sequential ramp, cream through terracotta.
 * One hue means the eye reads intensity rather than category, which is the
 * right default here: these are all the same quantity at the same scale.
 */
const WARM: Stop[] = [
  [0.1,  [253, 244, 232]],
  [1,    [250, 232, 206]],
  [2.5,  [246, 217, 174]],
  [5,    [242, 200, 147]],
  [10,   [235, 174, 110]],
  [20,   [226, 146,  82]],
  [30,   [215, 120,  60]],
  [40,   [200,  95,  44]],
  [60,   [180,  69,  28]],
  [80,   [158,  52,  24]],
  [120,  [128,  38,  22]],
  [160,  [ 96,  28,  20]],
  [200,  [ 64,  20,  16]],
];

/** The NWS multi-hue ramp, kept for anyone reading these as weather maps. */
const NWS: Stop[] = [
  [0.1,  [238, 243, 248]],
  [1,    [185, 220, 240]],
  [2.5,  [124, 195, 232]],
  [5,    [ 74, 159, 216]],
  [10,   [ 59, 127, 196]],
  [20,   [ 76, 176,  95]],
  [30,   [134, 201,  79]],
  [40,   [232, 212,  60]],
  [60,   [240, 160,  42]],
  [80,   [226,  98,  43]],
  [120,  [201,  47,  47]],
  [160,  [155,  42, 107]],
  [200,  [107,  45, 143]],
];

const RAMPS: Record<RampId, Stop[]> = { warm: WARM, nws: NWS };

export const RAMP_LABEL: Record<RampId, string> = { warm: "Simple", nws: "Weather map" };

/** Cool blue on one side, the ramp's terracotta on the other. */
const DIVERGE: Stop[] = [
  [-1,    [ 29,  78, 138]],
  [-0.5,  [ 88, 142, 196]],
  [-0.15, [178, 206, 232]],
  [0,     [246, 248, 249]],
  [0.15,  [248, 222, 195]],
  [0.5,   [224, 146,  86]],
  [1,     [164,  56,  26]],
];

function interp(stops: Stop[], v: number): [number, number, number] {
  if (v <= stops[0][0]) return stops[0][1];
  const last = stops[stops.length - 1];
  if (v >= last[0]) return last[1];
  for (let i = 1; i < stops.length; i++) {
    const [b, cb] = stops[i];
    if (v <= b) {
      const [a, ca] = stops[i - 1];
      const t = (v - a) / (b - a);
      return [
        Math.round(ca[0] + (cb[0] - ca[0]) * t),
        Math.round(ca[1] + (cb[1] - ca[1]) * t),
        Math.round(ca[2] + (cb[2] - ca[2]) * t),
      ];
    }
  }
  return last[1];
}

/** 256-entry LUT over a sqrt-stretched 0..vmax range. */
export function precipLUT(vmax: number, ramp: RampId = "warm"): Uint8ClampedArray {
  const stops = RAMPS[ramp];
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const mm = vmax * (i / 255) ** 2;
    const o = i * 4;
    if (mm < 0.1) { lut[o] = 0; lut[o + 1] = 0; lut[o + 2] = 0; lut[o + 3] = 0; continue; }
    const [r, g, b] = interp(stops, mm);
    lut[o] = r; lut[o + 1] = g; lut[o + 2] = b; lut[o + 3] = 255;
  }
  return lut;
}

export function divergeColor(norm: number): [number, number, number] {
  return interp(DIVERGE, Math.max(-1, Math.min(1, norm)));
}

export function precipColor(mm: number, ramp: RampId = "warm"): [number, number, number] {
  return interp(RAMPS[ramp], mm);
}

export const css = ([r, g, b]: [number, number, number]) => `rgb(${r},${g},${b})`;

/**
 * The reference legend is a row of discrete blocks rather than a gradient, so
 * a reader can match a map cell to a band by eye. Bands are cut on the same
 * sqrt stretch the canvas uses, so block k is genuinely the colour of that
 * slice of the display range.
 */
export function bands(vmax: number, ramp: RampId, n = 7) {
  return Array.from({ length: n }, (_, i) => {
    const lo = vmax * (i / n) ** 2;
    const hi = vmax * ((i + 1) / n) ** 2;
    return { lo, hi, color: css(precipColor(vmax * ((i + 0.5) / n) ** 2, ramp)) };
  });
}

export function diffBands(scale: number, n = 8) {
  return Array.from({ length: n }, (_, i) => {
    const t = (i + 0.5) / n * 2 - 1;
    return { lo: scale * ((i / n) * 2 - 1), hi: scale * (((i + 1) / n) * 2 - 1), color: css(divergeColor(t)) };
  });
}
