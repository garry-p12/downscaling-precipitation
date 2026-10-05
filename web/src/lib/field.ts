import type { Field, LayerId, Meta } from "./types";
import { dataUrl } from "./domains";

const cache = new Map<string, Field>();
const inflight = new Map<string, Promise<Field>>();

const key = (layer: LayerId, date: string, dir: string) => `${dir}/${layer}/${date}`;

/**
 * Tiles carry the value in the luminance channel on a sqrt ramp and the
 * validity mask in alpha, so a decoded tile is real mm/day — differences and
 * point readouts are computed here, not baked into the image.
 */
export async function loadField(layer: LayerId, date: string, meta: Meta,
                                dir = "/data"): Promise<Field> {
  const k = key(layer, date, dir);
  const hit = cache.get(k);
  if (hit) return hit;
  const pending = inflight.get(k);
  if (pending) return pending;

  const p = (async () => {
    const res = await fetch(dataUrl(`${dir}/fields/${layer}/${date}.png`));
    if (!res.ok) throw new Error(`tile ${k}: ${res.status}`);
    const bmp = await createImageBitmap(await res.blob());
    const { width, height } = bmp;
    const cv = new OffscreenCanvas(width, height);
    const ctx = cv.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    const px = ctx.getImageData(0, 0, width, height).data;
    const data = new Float32Array(width * height);
    const vmax = meta.vmax;
    for (let i = 0, j = 0; i < data.length; i++, j += 4) {
      if (px[j + 3] === 0) { data[i] = NaN; continue; }
      const t = px[j] / 255;
      data[i] = vmax * t * t;
    }
    const f: Field = { data, width, height };
    cache.set(k, f);
    inflight.delete(k);
    if (cache.size > 420) {
      const first = cache.keys().next().value;
      if (first) cache.delete(first);
    }
    return f;
  })();
  inflight.set(k, p);
  return p;
}

export function prefetch(layers: LayerId[], dates: string[], meta: Meta, dir = "/data") {
  for (const d of dates) for (const l of layers) loadField(l, d, meta, dir).catch(() => {});
}

export function sampleAt(f: Field | undefined, x: number, y: number): number {
  if (!f) return NaN;
  if (x < 0 || y < 0 || x >= f.width || y >= f.height) return NaN;
  return f.data[y * f.width + x];
}

/** Zoom windows, in fractional grid coordinates. */
export const WINDOWS: Record<string, { label: string; box: [number, number, number, number] | null; hint: string }> = {
  full:    { label: "Everything",    box: null,                     hint: "300 by 390 km, every square kilometre of it" },
  austin:  { label: "Austin",        box: [0.28, 0.36, 0.52, 0.60], hint: "roughly 72 by 94 km around the city" },
  hill:    { label: "Hill Country",  box: [0.04, 0.30, 0.34, 0.60], hint: "the only hills anywhere in this view" },
  east:    { label: "Eastern plains", box: [0.62, 0.30, 0.96, 0.62], hint: "flat, wetter, thunderstorm country" },
  zoom:    { label: "Close in",      box: [0.34, 0.40, 0.57, 0.63], hint: "tight enough to pick out single kilometres" },
};
