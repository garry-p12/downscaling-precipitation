/**
 * One place for the words the portal puts in front of a reader. The scores
 * underneath are ordinary RMSE, bias, POD and domain maximum; the names here
 * are what those mean to someone who has not met them before, and the `help`
 * line is what a tooltip says when the plain name is not enough.
 */
export type MetricId = "miss" | "wetdry" | "heavy" | "peak";

export const METRIC: Record<MetricId, { label: string; axis: string; help: string }> = {
  miss: {
    label: "How far off",
    axis: "how far off, mm of rain",
    help: "How far the map is from what fell at the typical square kilometre, in millimetres of rain. Big misses count for more than small ones. Statisticians call it RMSE.",
  },
  wetdry: {
    label: "Too wet or dry",
    axis: "too wet (+) or too dry (−), mm",
    help: "Average over- or under-estimate across the whole area. Near zero can still hide big errors that cancel out — one half of the map too wet, the other too dry.",
  },
  heavy: {
    label: "Heavy rain caught",
    axis: "share of heavy-rain spots found",
    help: "Of the places that got more than 30 mm of rain, the share this version also put above 30 mm.",
  },
  peak: {
    label: "Heaviest spot",
    axis: "heaviest square kilometre, mm",
    help: "The wettest single square kilometre on the map that day, next to the wettest one that actually fell.",
  },
};

/** Plain-language rounding: readers do not need three decimals of rain. */
export const mm = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1));

/** Scores here cluster within a millimetre, so one decimal has to survive. */
export const score = (v: number) => v.toFixed(1);

export function ordinal(n: number) {
  const r10 = n % 10, r100 = n % 100;
  if (r10 === 1 && r100 !== 11) return `${n}st`;
  if (r10 === 2 && r100 !== 12) return `${n}nd`;
  if (r10 === 3 && r100 !== 13) return `${n}rd`;
  return `${n}th`;
}
