"use client";
import { bands, diffBands, type RampId } from "@/lib/colormap";

const fmt = (v: number) => (Math.abs(v) >= 10 || Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1));

/**
 * Discrete blocks rather than a gradient, as in the reference: a reader can
 * carry a block's colour back to the map and land on a band, not a guess.
 * Tick labels are positioned on the band edges they actually mark.
 */
function Strip({ title, blocks, lead, tail, ticks }: {
  title: string;
  blocks: { hi: number; color: string }[];
  lead: string; tail: string; ticks: number[];
}) {
  const n = blocks.length;
  return (
    <div className="px-3.5 pb-3.5" title={title}>
      <div className="flex gap-px">
        {blocks.map((b, i) => (
          <i key={i} className="flex-1 h-[11px] block first:rounded-l-[2px] last:rounded-r-[2px]"
            style={{ background: b.color }} />
        ))}
      </div>
      <div className="relative h-[13px] mt-1">
        <span className="absolute left-0 top-0 mono text-[9.5px] text-[var(--ink3)]">{lead}</span>
        {ticks.filter((i) => i < n - 1).map((i) => (
          <span key={i} className="absolute top-0 mono text-[9.5px] text-[var(--ink3)] -translate-x-1/2"
            style={{ left: `${((i + 1) / n) * 100}%` }}>
            {fmt(blocks[i].hi)}
          </span>
        ))}
        <span className="absolute right-0 top-0 mono text-[9.5px] text-[var(--ink3)]">{tail}</span>
      </div>
    </div>
  );
}

export function PrecipLegend({ vmax, ramp }: { vmax: number; ramp: RampId }) {
  const b = bands(vmax, ramp, 7);
  return <Strip title="Rain that day, mm" blocks={b} lead="0" tail={`${vmax}+`} ticks={[1, 3]} />;
}

export function DiffLegend({ scale }: { scale: number }) {
  const b = diffBands(scale, 8);
  return (
    <Strip title="Too dry ← → too wet, mm" blocks={b}
      lead={`−${scale}`} tail={`+${scale}`} ticks={[1, 3, 5]} />
  );
}
