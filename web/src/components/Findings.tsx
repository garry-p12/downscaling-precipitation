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
        <h3 className="text-[19px] font-bold mb-1">Mountains help. Flat ground doesn&rsquo;t.</h3>
        <p className="text-[13px] text-[var(--ink2)] mb-4 max-w-[62ch]">
          How much closer each model gets than plain bilinear smoothing. The bottom two bars are the point:
          a Colorado summer and flat Austin land within a point of each other, even though a quarter
          of Colorado is steep ground. Terrain only helps when the weather is using it — storms
          shoved up a mountainside land in a place you can predict, and summer thunderstorms don&rsquo;t.
        </p>
        <div className="space-y-2.5">
          {LADDER.map((r) => (
            <div key={r.label} className="grid grid-cols-[190px_1fr_58px] gap-3 items-center">
              <span>
                <span className="block text-[13px] font-semibold leading-tight">{r.label}</span>
                <span className="block text-[11px] text-[var(--ink3)]">{r.sub}</span>
              </span>
              <Bar v={r.gain} max={max} accent={shade(r.gain)} />
              <span className="mono text-[13px] text-right"
                style={{ color: r.gain >= 12 ? "var(--ramp-6)" : "var(--ink2)" }}>
                {r.gain.toFixed(1)} %
              </span>
            </div>
          ))}
        </div>
        <p className="hint mt-4">closer than bilinear · the study set out to reach 20 %, and only the mountain winter does</p>
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        <div className="card p-5">
          <h3 className="text-[14px] font-bold flex items-center gap-2 mb-1">
            Train it twice, get two answers
            <Info text="Ten training runs in all. Each row is the range of scores one recipe produced when it was re-run with a different random start." />
          </h3>
          <p className="text-[12.5px] text-[var(--ink2)] mb-4">
            Training has randomness in it, so the same recipe lands on a slightly different answer
            each time. Swin beats the CNN in all nine head-to-head runs — but one pair of runs
            finished 0.010 apart, four times closer than the real gap between them.
            A single run proves nothing.
          </p>
          {SEEDS.map((f) => {
            const lo = Math.min(...f.values), hi = Math.max(...f.values);
            const L = 4.56, R = 4.72;
            return (
              <div key={f.family} className="mb-3.5">
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-[var(--ink)]">{f.family} · {f.n} runs</span>
                  <span className="mono" style={{ color: f.accent }}>{(hi - lo).toFixed(3)} apart</span>
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
          <p className="hint">how far off, in mm · the scale runs 4.56 to 4.72</p>
        </div>

        <div className="card p-5">
          <h3 className="text-[14px] font-bold flex items-center gap-2 mb-1">
            A model doesn&rsquo;t travel
            <Info text="Training in one place and using it in another, to see what the model actually picked up." />
          </h3>
          <p className="text-[12.5px] text-[var(--ink2)] mb-4">
            Train in Colorado, use it over Texas, and it does <em>worse</em> than bilinear by
            13.7 %. It flips from slightly too dry to clearly too wet — because what it really learned
            was that its own satellite feed runs dry over Colorado, not how mountains shape rain.
          </p>
          <div className="space-y-1.5">
            {TRANSFER.map((r, i) => (
              <div key={i} className="flex items-center gap-2 text-[12px] px-2.5 py-2 rounded-[8px]"
                style={{ background: r.gain < 0 ? "var(--accent-soft)" : "var(--card)",
                         border: `1px solid ${r.gain < 0 ? "var(--accent-line)" : "var(--line)"}` }}>
                <span className="flex-1 text-[var(--ink2)]">
                  {r.train} <span className="text-[var(--ink3)]">→</span> {r.score}
                </span>
                <span className="mono text-[var(--ink)] w-12 text-right">{r.rmse.toFixed(2)}</span>
                <span className="mono w-16 text-right font-semibold"
                  style={{ color: r.gain < 0 ? "var(--accent)" : "var(--good)" }}>
                  {r.gain > 0 ? "+" : ""}{r.gain.toFixed(1)} %
                </span>
              </div>
            ))}
          </div>
          <p className="hint mt-3">trained → used · how far off, in mm · and how that compares with bilinear</p>
        </div>
      </div>

      <div className="card p-5">
        <h3 className="text-[14px] font-bold mb-1">Three places, one viewer</h3>
        <p className="text-[12.5px] text-[var(--ink2)] mb-4 max-w-[64ch]">
          All three cover the same size of grid and the same days in 2019 and 2020, so the maps line
          up against each other. Switch between them on the left.
        </p>
        <div className="grid sm:grid-cols-3 gap-4">
          {DOMAINS.map((d) => (
            <div key={d.id} className="rounded-[10px] border border-[var(--line)] p-3.5">
              <span className="block text-[13.5px] font-semibold">{d.label}</span>
              <span className="hint block mt-0.5">{d.input} · {d.sub}</span>
              <span className="mono block text-[26px] mt-2 leading-none"
                style={{ color: d.gain >= 12 ? "var(--ramp-6)" : "var(--ramp-5)" }}>
                {d.gain.toFixed(1)} %
              </span>
              <span className="hint block">closer than bilinear · off by {d.rmse.toFixed(1)} mm</span>
              <p className="text-[12px] text-[var(--ink2)] mt-2.5 leading-snug">{d.note}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
