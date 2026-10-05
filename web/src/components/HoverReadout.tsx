"use client";
import type { Field, LayerId, Meta } from "@/lib/types";
import { LAYER_META } from "@/lib/types";
import { sampleAt } from "@/lib/field";
import { precipColor, css, type RampId } from "@/lib/colormap";
import Info from "./Info";

interface Props {
  cursor: { x: number; y: number } | null;
  layers: LayerId[]; fields: Partial<Record<LayerId, Field>>; meta: Meta; ramp: RampId;
}

export default function HoverReadout({ cursor, layers, fields, meta, ramp }: Props) {
  const truth = cursor ? sampleAt(fields.aorc, cursor.x, cursor.y) : NaN;
  const [lon0, lat0, lon1, lat1] = meta.bbox;
  const lon = cursor ? lon0 + ((cursor.x + 0.5) / meta.width) * (lon1 - lon0) : 0;
  const lat = cursor ? lat1 - ((cursor.y + 0.5) / meta.height) * (lat1 - lat0) : 0;

  return (
    <div className="card p-4">
      <p className="hint">One spot</p>
      <h3 className="text-[16px] font-bold leading-tight mt-0.5 flex items-center gap-2">
        One square kilometre
        <Info text="How much rain every version put at the square kilometre under your pointer, and how far each one is from what actually fell." />
      </h3>
      <p className="mono text-[10.5px] text-[var(--ink3)] mt-1">
        {cursor
          ? `${lat.toFixed(3)} N  ${Math.abs(lon).toFixed(3)} W · cell ${cursor.x}, ${cursor.y}`
          : "\u2014"}
      </p>

      {!cursor ? (
        <p className="hint mt-3">Hover a map to read every version at one square kilometre.</p>
      ) : (
        <div className="mt-3 space-y-[3px]">
          {layers.map((l) => {
            const v = sampleAt(fields[l], cursor.x, cursor.y);
            const d = l === "aorc" ? null : v - truth;
            const ok = isFinite(v) && v >= 0.1;
            return (
              <div key={l} className="flex items-center gap-2 text-[11.5px]">
                <span className="w-3 h-3 rounded-[3px] shrink-0 border border-black/10"
                  style={{ background: ok ? css(precipColor(v, ramp)) : "var(--nodata)" }} />
                <span className="flex-1 truncate"
                  style={{ color: l === "aorc" ? "var(--ink)" : "var(--ink2)", fontWeight: l === "aorc" ? 700 : 400 }}>
                  {LAYER_META[l].short}
                </span>
                <span className="mono w-12 text-right text-[var(--ink)]">{isFinite(v) ? v.toFixed(1) : "—"}</span>
                <span className="mono w-14 text-right"
                  style={{ color: d == null ? "var(--ink3)" : Math.abs(d) < 1 ? "var(--ink3)" : d > 0 ? "var(--accent)" : "var(--cool)" }}>
                  {d == null ? "what fell" : `${d > 0 ? "+" : ""}${d.toFixed(1)}`}
                </span>
              </div>
            );
          })}
        </div>
      )}
      <p className="hint mt-3">mm · right column is the error</p>
    </div>
  );
}
