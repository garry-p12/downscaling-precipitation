"use client";
import { LADDER, SEEDS, TRANSFER, DOMAINS } from "@/lib/domains";
import Info from "./Info";

/** Position on the shared sequential ramp, so a longer bar is also a darker one. */
function shade(gain: number) {
  const t = Math.max(0, Math.min(1, gain / 22));
  return ["var(--ramp-2)", "var(--ramp-3)", "var(--ramp-4)", "var(--ramp-5)", "var(--ramp-6)"][
    Math.min(4, Math.floor(t * 5))];
}

function Bar({ v, max, accent }: { v: number; max: number; accent: string }) {
  const neg = v < 0;
  const zero = (0 - Math.min(0, -2)) / (max - Math.min(0, -2)) * 100;
  const w = (Math.abs(v) / (max - Math.min(0, -2))) * 100;
  return (
    <span className="relative block h-[18px] rounded-[3px]" style={{ background: "var(--track)" }}>
      <span className="absolute inset-y-0 rounded-[3px]"
        style={{ background: accent, left: neg ? `${zero - w}%` : `${zero}%`, width: `${w}%` }} />
    </span>
  );
}

export default function Findings() {
  const max = 22;
  return (
    <div className="space-y-5">
      <div className="card p-5">
        <h3 className="text-[19px] font-bold mb-1">Skill tracks orographic forcing</h3>
        <p className="text-[13px] text-[var(--ink2)] mb-4 max-w-[62ch]">
          RMSE reduction against bilinear interpolation. The bottom two rows are the result:
          warm-season Colorado and flat Austin score within 1.2 points of each other, despite
          25.5 percentage points of difference in steep terrain. Terrain only buys skill when
          the season engages it.
        </p>
        <div className="space-y-2.5">
          {LADDER.map((r) => (
            <div key={r.label} className="grid grid-cols-[190px_1fr_58px] gap-3 items-center">
              <span>
                <span className="block text-[13px] font-semibold leading-tight">{r.label}</span>
                <span className="block text-[11px] text-[var(--ink3)]">{r.sub}</span>
              </span>
              <Bar v={r.gain} max={max} accent={shade(r.gain)} />
              <span className="mono text-[13px] text-right" style={{ color: r.gain >= 12 ? "var(--ramp-6)" : "var(--ink2)" }}>
                {r.gain.toFixed(1)} %
              </span>
            </div>
          ))}
        </div>
        <p className="lbl mt-4">dashed target in the paper is 20 % · only the cool season reaches it</p>
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        <div className="card p-5">
          <h3 className="text-[14px] font-bold flex items-center gap-2 mb-1">
            Seed repeats <Info text="Ten training runs. Each family's range across random seeds, on the Austin test set." />
          </h3>
          <p className="text-[12.5px] text-[var(--ink2)] mb-4">
            Swin beats the CNN in all nine pairwise comparisons and the ranges do not
            overlap — but a single run put them 0.010 apart, four times smaller than the
            true difference.
          </p>
          {SEEDS.map((f) => {
            const lo = Math.min(...f.values), hi = Math.max(...f.values);
            const L = 4.56, R = 4.72;
            return (
              <div key={f.family} className="mb-3.5">
                <div className="flex justify-between mono text-[11px] mb-1">
                  <span className="text-[var(--ink)]">{f.family} · {f.n} seeds</span>
                  <span style={{ color: f.accent }}>{(hi - lo).toFixed(3)} spread</span>
                </div>
                <div className="relative h-[20px] rounded-[3px]" style={{ background: "var(--track)" }}>
                  <span className="absolute inset-y-[5px] rounded-full opacity-40"
                    style={{ background: f.accent, left: `${(lo - L) / (R - L) * 100}%`,
                             width: `${(hi - lo) / (R - L) * 100}%` }} />
                  {f.values.map((v, i) => (
                    <span key={i} className="absolute top-1/2 w-[7px] h-[7px] rounded-full -translate-y-1/2 -translate-x-1/2"
                      style={{ background: f.accent, left: `${(v - L) / (R - L) * 100}%` }} />
                  ))}
                </div>
              </div>
            );
          })}
          <p className="lbl">test RMSE mm day⁻¹ · 4.56 → 4.72</p>
        </div>

        <div className="card p-5">
          <h3 className="text-[14px] font-bold flex items-center gap-2 mb-1">
            Cross-domain transfer <Info text="Training on one domain and scoring on another, to see what the model actually learned." />
          </h3>
          <p className="text-[12.5px] text-[var(--ink2)] mb-4">
            A Colorado-trained model is 13.7 % <em>worse</em> than bilinear on Austin. It
            swings bias from −0.08 to +0.35, because what it learned was that its own
            retrieval runs dry — not how orography shapes rain.
          </p>
          <div className="space-y-1.5">
            {TRANSFER.map((r, i) => (
              <div key={i} className="flex items-center gap-2 mono text-[11.5px] px-2.5 py-2 rounded-[8px]"
                style={{ background: r.gain < 0 ? "var(--accent-soft)" : "var(--card)",
                         border: `1px solid ${r.gain < 0 ? "var(--accent-line)" : "var(--line)"}` }}>
                <span className="flex-1 text-[var(--ink2)]">
                  {r.train} <span className="text-[var(--ink3)]">→</span> {r.score}
                </span>
                <span className="tabular-nums text-[var(--ink)] w-12 text-right">{r.rmse.toFixed(3)}</span>
                <span className="tabular-nums w-16 text-right font-semibold"
                  style={{ color: r.gain < 0 ? "var(--accent)" : "var(--good)" }}>
                  {r.gain > 0 ? "+" : ""}{r.gain.toFixed(1)} %
                </span>
              </div>
            ))}
          </div>
          <p className="lbl mt-3">trained → scored · RMSE · vs bilinear</p>
        </div>
      </div>

      <div className="card p-5">
        <h3 className="text-[14px] font-bold mb-1">Three studies, one viewer</h3>
        <p className="text-[12.5px] text-[var(--ink2)] mb-4 max-w-[64ch]">
          All three share a 420 × 360 one-kilometre grid and the same 2019–2020 test days,
          so the maps are directly comparable. Switch between them above.
        </p>
        <div className="grid sm:grid-cols-3 gap-4">
          {DOMAINS.map((d) => (
            <div key={d.id} className="rounded-[10px] border border-[var(--line)] p-3.5">
              <span className="block text-[13.5px] font-semibold">{d.label}</span>
              <span className="lbl">{d.input} · {d.sub}</span>
              <span className="mono block text-[26px] mt-2 leading-none" style={{ color: d.gain >= 12 ? "var(--ramp-6)" : "var(--ramp-5)" }}>
                {d.gain.toFixed(1)} %
              </span>
              <span className="lbl">vs bilinear · RMSE {d.rmse.toFixed(3)}</span>
              <p className="text-[12px] text-[var(--ink2)] mt-2.5 leading-snug">{d.note}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
