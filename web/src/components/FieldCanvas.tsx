"use client";
import { useEffect, useMemo, useRef } from "react";
import type { Field, Meta } from "@/lib/types";
import { precipLUT, divergeColor, type RampId } from "@/lib/colormap";

export interface View { x0: number; y0: number; x1: number; y1: number }

interface Props {
  field?: Field; ref_?: Field;
  meta: Meta; view: View; mode: "value" | "diff";
  diffScale?: number; ramp?: RampId;
  cursor?: { x: number; y: number } | null;
  onHover?: (p: { x: number; y: number } | null) => void;
  className?: string;
}

export default function FieldCanvas({
  field, ref_, meta, view, mode, diffScale = 20, ramp = "warm", cursor, onHover, className,
}: Props) {
  const cv = useRef<HTMLCanvasElement>(null);
  const lut = useMemo(() => precipLUT(meta.vmax, ramp), [meta.vmax, ramp]);

  useEffect(() => {
    const c = cv.current; if (!c || !field) return;
    const W = meta.width, H = meta.height;
    const x0 = Math.floor(view.x0 * W), x1 = Math.ceil(view.x1 * W);
    const y0 = Math.floor(view.y0 * H), y1 = Math.ceil(view.y1 * H);
    const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
    c.width = w; c.height = h;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(w, h);
    const px = img.data;
    const L = lut;
    for (let yy = 0; yy < h; yy++) {
      const sy = y0 + yy;
      for (let xx = 0; xx < w; xx++) {
        const sx = x0 + xx;
        const o = (yy * w + xx) * 4;
        const v = field.data[sy * W + sx];
        if (!isFinite(v)) { px[o + 3] = 0; continue; }
        if (mode === "diff") {
          const r = ref_?.data[sy * W + sx];
          if (r == null || !isFinite(r)) { px[o + 3] = 0; continue; }
          const [cr, cg, cb] = divergeColor((v - r) / diffScale);
          px[o] = cr; px[o + 1] = cg; px[o + 2] = cb; px[o + 3] = 255;
        } else {
          const i = Math.max(0, Math.min(255, Math.round(Math.sqrt(Math.max(0, v) / meta.vmax) * 255)));
          const q = i * 4;
          px[o] = L[q]; px[o + 1] = L[q + 1]; px[o + 2] = L[q + 2]; px[o + 3] = L[q + 3];
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [field, ref_, meta, view, mode, diffScale, lut]);

  const toGrid = (e: React.MouseEvent) => {
    const c = cv.current; if (!c) return null;
    const r = c.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const W = meta.width, H = meta.height;
    const x0 = Math.floor(view.x0 * W), y0 = Math.floor(view.y0 * H);
    return { x: Math.floor(x0 + fx * c.width), y: Math.floor(y0 + fy * c.height) };
  };

  const W = meta.width, H = meta.height;
  const x0 = Math.floor(view.x0 * W), y0 = Math.floor(view.y0 * H);
  const vw = Math.ceil(view.x1 * W) - x0, vh = Math.ceil(view.y1 * H) - y0;
  const cx = cursor ? ((cursor.x - x0 + 0.5) / vw) * 100 : 0;
  const cy = cursor ? ((cursor.y - y0 + 0.5) / vh) * 100 : 0;
  const inView = cursor && cursor.x >= x0 && cursor.x < x0 + vw && cursor.y >= y0 && cursor.y < y0 + vh;

  return (
    <div className={`relative ${className ?? ""}`}>
      <canvas
        ref={cv}
        className="w-full h-auto block [image-rendering:pixelated] cursor-crosshair"
        onMouseMove={(e) => onHover?.(toGrid(e))}
        onMouseLeave={() => onHover?.(null)}
      />
      {inView && (
        <>
          <div className="pointer-events-none absolute inset-y-0 w-px opacity-45"
            style={{ left: `${cx}%`, background: "repeating-linear-gradient(to bottom,var(--cross) 0 3px,transparent 3px 6px)" }} />
          <div className="pointer-events-none absolute inset-x-0 h-px opacity-45"
            style={{ top: `${cy}%`, background: "repeating-linear-gradient(to right,var(--cross) 0 3px,transparent 3px 6px)" }} />
        </>
      )}
      {!field && (
        <div className="absolute inset-0 grid place-items-center mono text-[11px] text-[var(--ink3)]">loading…</div>
      )}
    </div>
  );
}
