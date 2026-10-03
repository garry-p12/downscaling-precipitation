"use client";
import { useMemo } from "react";
import type { Day, LayerId } from "@/lib/types";
import { LAYER_META } from "@/lib/types";

type Metric = "rmse" | "bias" | "pod" | "max";

interface Props {
  days: Day[]; index: number; layers: LayerId[]; metric: Metric;
  onIndex: (i: number) => void;
}

export default function SeasonChart({ days, index, layers, metric, onIndex }: Props) {
  const W = 1000, H = 330, ml = 54, mr = 120, mt = 26, mb = 34;
  const models = layers.filter((l) => l !== "aorc") as Exclude<LayerId, "aorc">[];

  const series = useMemo(() => models.map((l) => ({
    id: l,
    pts: days.map((d, i) => {
      const m = d.m[l];
      const v = !m ? null : metric === "max" ? m.max : metric === "pod" ? m.pod : m[metric];
      return { i, v: v == null || !isFinite(v) ? null : v };
    }),
  })), [days, models, metric]);

  const vals = series.flatMap((s) => s.pts.map((p) => p.v)).filter((v): v is number => v != null);
  if (metric === "max") vals.push(...days.map((d) => d.obsMax));
  const rawLo = Math.min(...vals, metric === "bias" ? 0 : Infinity);
  const hi = Math.max(...vals);
  const pad = (hi - rawLo) * 0.08 || 1;
  const lo = metric === "bias" ? rawLo : Math.max(0, rawLo);
  const Y = (v: number) => mt + ((hi + pad - v) / (hi - lo + 2 * pad)) * (H - mt - mb);
  const X = (i: number) => ml + (i / Math.max(1, days.length - 1)) * (W - ml - mr);
  const ticks = Array.from({ length: 5 }, (_, i) => lo - (metric === "bias" ? pad : 0) + ((hi + pad - lo + (metric === "bias" ? pad : 0)) * i) / 4);
  const label = { rmse: "RMSE, mm day⁻¹", bias: "bias, mm day⁻¹",
                  pod: "POD above 30 mm", max: "domain peak, mm day⁻¹" }[metric];

  return (
    <div className="card p-4">
      <div className="flex items-baseline gap-3 mb-1">
        <h3 className="text-[14px] font-bold">Every day in the record</h3>
        <span className="lbl">{label}</span>
        <span className="lbl ml-auto">click to jump to a day</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block cursor-pointer"
        onClick={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const f = ((e.clientX - r.left) / r.width) * W;
          onIndex(Math.min(days.length - 1, Math.max(0, Math.round(((f - ml) / (W - ml - mr)) * (days.length - 1)))));
        }}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={ml} y1={Y(t)} x2={W - mr} y2={Y(t)} stroke="var(--line)" />
            <text x={ml - 8} y={Y(t) + 3.5} textAnchor="end" fill="var(--ink3)"
              style={{ fontSize: 10, fontFamily: "var(--font-mono)" }}>{t.toFixed(metric === "pod" ? 2 : 0)}</text>
          </g>
        ))}
        {metric === "bias" && <line x1={ml} y1={Y(0)} x2={W - mr} y2={Y(0)} stroke="var(--ink3)" strokeWidth={1} />}
        {metric === "max" && (
          <path d={days.map((d, i) => `${i ? "L" : "M"}${X(i)} ${Y(d.obsMax)}`).join(" ")}
            fill="none" stroke="var(--ink)" strokeWidth={2} opacity={0.8} />
        )}
        {series.map((s) => (
          <path key={s.id}
            d={s.pts.filter((p) => p.v != null).map((p, k) => `${k ? "L" : "M"}${X(p.i)} ${Y(p.v!)}`).join(" ")}
            fill="none" stroke={LAYER_META[s.id].color} strokeWidth={1.4} opacity={0.92}
            strokeDasharray={s.id === "diffusion_member" ? "4 3" : undefined} />
        ))}
        <line x1={X(index)} y1={mt - 8} x2={X(index)} y2={H - mb} stroke="var(--dark)" strokeWidth={1.5} />
        <circle cx={X(index)} cy={mt - 10} r={4} fill="var(--dark)" />
        {series.map((s, k) => (
          <text key={s.id} x={W - mr + 10} y={mt + 11 + k * 15} fill={LAYER_META[s.id].color}
            style={{ fontSize: 10.5, fontFamily: "var(--font-mono)" }}>{LAYER_META[s.id].short}</text>
        ))}
        {metric === "max" && (
          <text x={W - mr + 10} y={mt + 11 + series.length * 15} fill="var(--ink)"
            style={{ fontSize: 10.5, fontFamily: "var(--font-mono)", fontWeight: 600 }}>AORC</text>
        )}
        <text x={ml} y={H - 8} fill="var(--ink3)" style={{ fontSize: 10, fontFamily: "var(--font-mono)" }}>{days[0].date}</text>
        <text x={W - mr} y={H - 8} textAnchor="end" fill="var(--ink3)"
          style={{ fontSize: 10, fontFamily: "var(--font-mono)" }}>{days[days.length - 1].date}</text>
      </svg>
    </div>
  );
}
