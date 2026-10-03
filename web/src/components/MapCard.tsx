"use client";
import FieldCanvas, { type View } from "./FieldCanvas";
import { PrecipLegend, DiffLegend } from "./Legend";
import type { Field, LayerId, Meta } from "@/lib/types";
import { LAYER_META } from "@/lib/types";
import type { RampId } from "@/lib/colormap";

interface Props {
  layer: LayerId; meta: Meta; view: View;
  field?: Field; ref_?: Field;
  mode: "value" | "diff"; diffScale?: number; ramp: RampId;
  value?: number | null; unit?: string; valueLabel?: string;
  sub?: string; selected?: boolean; tag?: string;
  cursor: { x: number; y: number } | null;
  onHover: (p: { x: number; y: number } | null) => void;
}

export default function MapCard({
  layer, meta, view, field, ref_, mode, diffScale = 20, ramp,
  value, unit = "RMSE mm day⁻¹", valueLabel, sub, selected, tag, cursor, onHover,
}: Props) {
  const L = LAYER_META[layer];
  return (
    <figure className={`card overflow-hidden ${selected ? "card-sel" : ""}`}>
      {/* fixed header height keeps the maps top-aligned when a title wraps */}
      <figcaption className="flex items-start gap-3 px-3.5 pt-3 pb-3 h-[116px] overflow-hidden">
        <span className="flex-1 min-w-0">
          {tag && (
            <span className="eyebrow block !text-[10px] mb-[3px]"
              style={{ color: selected ? "var(--accent)" : "var(--ink3)" }}>{tag}</span>
          )}
          <span className="text-[14.5px] font-semibold leading-tight line-clamp-2">{valueLabel ?? L.label}</span>
          <span className="hint mt-1 line-clamp-2">{sub ?? L.note}</span>
        </span>
        {value != null && (
          <span className="text-right shrink-0">
            <span className="mono block text-[26px] leading-none tracking-tight"
              style={{ color: selected ? "var(--accent)" : "var(--ink)" }}>
              {value.toFixed(2)}
            </span>
            <span className="lbl block mt-1.5">{unit}</span>
          </span>
        )}
      </figcaption>
      <div className="px-3.5">
        <div className="mapframe">
          <FieldCanvas field={field} ref_={ref_} meta={meta} view={view} mode={mode}
            diffScale={diffScale} ramp={ramp} cursor={cursor} onHover={onHover} />
        </div>
      </div>
      <div className="mt-3">
        {mode === "diff" ? <DiffLegend scale={diffScale} /> : <PrecipLegend vmax={meta.vmax} ramp={ramp} />}
      </div>
    </figure>
  );
}
