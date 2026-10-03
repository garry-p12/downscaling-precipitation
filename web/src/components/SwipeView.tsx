"use client";
import { useRef, useState } from "react";
import FieldCanvas, { type View } from "./FieldCanvas";
import { PrecipLegend } from "./Legend";
import type { Field, LayerId, Meta } from "@/lib/types";
import { LAYER_META } from "@/lib/types";
import type { RampId } from "@/lib/colormap";

interface Props {
  meta: Meta; view: View; left: LayerId; right: LayerId; ramp: RampId;
  fields: Partial<Record<LayerId, Field>>;
  cursor: { x: number; y: number } | null;
  onHover: (p: { x: number; y: number } | null) => void;
}

export default function SwipeView({ meta, view, left, right, ramp, fields, cursor, onHover }: Props) {
  const [split, setSplit] = useState(50);
  const box = useRef<HTMLDivElement>(null);

  const drag = (e: React.PointerEvent) => {
    e.preventDefault();
    const move = (ev: PointerEvent) => {
      const r = box.current?.getBoundingClientRect(); if (!r) return;
      setSplit(Math.min(98, Math.max(2, ((ev.clientX - r.left) / r.width) * 100)));
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  };

  return (
    <figure className="card overflow-hidden">
      <figcaption className="px-3.5 pt-3.5 pb-3">
        <span className="block text-[14.5px] font-semibold leading-tight">
          {LAYER_META[left].label} <span className="text-[var(--ink3)] font-normal">against</span> {LAYER_META[right].label}
        </span>
        <span className="hint block mt-1">
          Drag the divider. Same day, same colour scale, same grid — the only difference is the product.
        </span>
      </figcaption>
      <div className="px-3.5">
        <div className="mapframe">
          <div ref={box} className="relative select-none">
            <FieldCanvas field={fields[right]} meta={meta} view={view} mode="value" ramp={ramp}
              cursor={cursor} onHover={onHover} />
            <div className="absolute inset-0 overflow-hidden" style={{ width: `${split}%` }}>
              <div style={{ width: `${(100 / split) * 100}%` }}>
                <FieldCanvas field={fields[left]} meta={meta} view={view} mode="value" ramp={ramp}
                  cursor={cursor} onHover={onHover} />
              </div>
            </div>
            <div className="absolute inset-y-0 w-0.5 cursor-ew-resize"
              style={{ left: `${split}%`, background: "#fff", boxShadow: "0 0 0 1px rgba(20,26,33,.35)" }}
              onPointerDown={drag}>
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 grid place-items-center text-[13px] rounded-[var(--r-ctl)] border border-[var(--line)]"
                style={{ background: "#fff", color: "var(--ink)", boxShadow: "var(--shadow-pop)" }}>↔</div>
            </div>
            <span className="absolute top-2.5 left-2.5 text-[11px] font-medium px-2.5 py-1 rounded-[5px]"
              style={{ background: "var(--dark)", color: "var(--dark-ink)" }}>{LAYER_META[left].short}</span>
            <span className="absolute top-2.5 right-2.5 text-[11px] font-medium px-2.5 py-1 rounded-[5px]"
              style={{ background: "var(--dark)", color: "var(--dark-ink)" }}>{LAYER_META[right].short}</span>
          </div>
        </div>
      </div>
      <div className="mt-3"><PrecipLegend vmax={meta.vmax} ramp={ramp} /></div>
    </figure>
  );
}
