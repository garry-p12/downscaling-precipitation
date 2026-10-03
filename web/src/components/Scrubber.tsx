"use client";
import { useMemo } from "react";
import type { Day, LayerId } from "@/lib/types";
import { LAYER_META } from "@/lib/types";
import Info from "./Info";

interface Props {
  days: Day[]; index: number; playing: boolean; speed: number; focus: LayerId;
  onIndex: (i: number) => void; onPlay: () => void; onSpeed: (s: number) => void;
}

export default function Scrubber({ days, index, playing, speed, focus, onIndex, onPlay, onSpeed }: Props) {
  const max = useMemo(() => Math.max(...days.map((d) => d.obsMean)), [days]);
  const cur = days[index];
  const frac = index / Math.max(1, days.length - 1);

  const m = focus === "aorc" ? null : cur.m[focus];
  const caption = !m
    ? `observed peak ${cur.obsMax.toFixed(1)} mm in one square kilometre`
    : `${LAYER_META[focus].short} was ${Math.abs(m.bias).toFixed(2)} mm ${m.bias < 0 ? "too dry" : "too wet"} on average · peak ${m.max.toFixed(0)} vs ${cur.obsMax.toFixed(0)} observed`;

  const path = useMemo(() => {
    const W = 1000, H = 34;
    return days.map((d, i) => {
      const x = (i / Math.max(1, days.length - 1)) * W;
      const y = H - Math.sqrt(d.obsMean / max) * (H - 3) - 1.5;
      return `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(" ");
  }, [days, max]);

  const area = `${path} L1000 34 L0 34 Z`;

  const scrub = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const apply = (clientX: number) => {
      const r = el.getBoundingClientRect();
      const f = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      onIndex(Math.round(f * (days.length - 1)));
    };
    apply(e.clientX);
    const move = (ev: PointerEvent) => apply(ev.clientX);
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  };

  return (
    <div className="card p-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
        <button onClick={onPlay} className="btn-dark">
          <span className="w-2.5 text-center not-italic">{playing ? "❙❙" : "▶"}</span>
          {playing ? "Pause" : "Play storms"}
        </button>
        <div className="mono text-[19px] tracking-tight">{cur.date}</div>
        <p className="text-[12.5px] text-[var(--ink2)] flex-1 min-w-[16rem]">
          {caption.split("·").map((part, i) => (
            <span key={i}>
              {i > 0 && <span className="text-[var(--ink3)]"> · </span>}
              {part.trim()}
            </span>
          ))}
        </p>
        <div className="flex items-center gap-2">
          <span className="eyebrow">speed</span>
          <div className="seg">
            {[[1400, "0.7×"], [900, "1×"], [450, "2×"], [200, "4×"]].map(([v, l]) => (
              <button key={v} aria-pressed={speed === v} onClick={() => onSpeed(v as number)}>{l}</button>
            ))}
          </div>
        </div>
      </div>

      <p className="eyebrow mt-4 flex items-center gap-2">
        rainfall, each day of the record
        <Info text="Height is the observed domain-mean rainfall for that day. The record is the 130 wettest test days plus every 11th day for seasonal spread." />
      </p>

      <div className="relative mt-2 select-none cursor-pointer" onPointerDown={scrub}>
        <svg viewBox="0 0 1000 34" preserveAspectRatio="none" className="w-full h-[34px] block">
          <path d={area} fill="var(--ramp-2)" />
          <path d={path} fill="none" stroke="var(--ramp-5)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
          <line x1={frac * 1000} y1={0} x2={frac * 1000} y2={34} stroke="var(--dark)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="relative h-[3px] rounded-full mt-1.5" style={{ background: "var(--track)" }}>
          <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${frac * 100}%`, background: "var(--dark)" }} />
          <div className="absolute -top-[5px] w-[13px] h-[13px] rounded-full border-2 border-white"
            style={{ left: `calc(${frac * 100}% - 6.5px)`, background: "var(--dark)", boxShadow: "var(--shadow-pop)" }} />
        </div>
      </div>

      <div className="flex justify-between mono text-[10.5px] text-[var(--ink3)] mt-2">
        <span>{days[0].date}</span>
        <span>day {index + 1} of {days.length} · {cur.obsHeavy.toLocaleString()} cells over 30 mm</span>
        <span>{days[days.length - 1].date}</span>
      </div>
    </div>
  );
}
