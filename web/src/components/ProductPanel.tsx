"use client";
import type { Day, LayerId } from "@/lib/types";
import { LAYER_META, ORDER } from "@/lib/types";
import Info from "./Info";

const RAMP = ["var(--ramp-2)", "var(--ramp-3)", "var(--ramp-4)", "var(--ramp-5)", "var(--ramp-6)", "var(--ramp-6)"];

interface Props {
  day: Day; value: LayerId; onChange: (l: LayerId) => void;
  metric: "rmse" | "bias";
  /** Layers shown elsewhere on the page already (the coarse input has its own panel). */
  exclude?: LayerId[];
  /** Layers this domain ships; Austin has four models, Colorado and POWER one. */
  only?: string[];
  cells: number;
  onSwipe: () => void; onDiff: () => void;
}

export default function ProductPanel({
  day, value, onChange, metric, exclude = [], only, cells, onSwipe, onDiff,
}: Props) {
  const rows = (ORDER.filter((l) => l !== "aorc" && !exclude.includes(l)
    && (!only || only.includes(l))) as Exclude<LayerId, "aorc">[])
    .filter((l) => day.m[l]);

  const score = (l: Exclude<LayerId, "aorc">) => {
    const m = day.m[l]!;
    return metric === "rmse" ? m.rmse : Math.abs(m.bias);
  };
  const ranked = [...rows].sort((a, b) => score(a) - score(b));
  const worst = ranked.length ? score(ranked[ranked.length - 1]) : 1;

  const focus = value as Exclude<LayerId, "aorc">;
  const rank = ranked.indexOf(focus);
  const m = day.m[focus];
  const base = day.m.bilinear;
  const vs = m && base && base.rmse > 0 ? ((base.rmse - m.rmse) / base.rmse) * 100 : null;

  const badge =
    rank === 0 ? { cls: "badge-good", text: `Best today · ranked 1 of ${ranked.length}` }
    : rank === ranked.length - 1 ? { cls: "", text: `Weakest today · ranked ${rank + 1} of ${ranked.length}` }
    : { cls: "badge-flat", text: `Ranked ${rank + 1} of ${ranked.length}` };

  const stats: [string, string][] = m ? [
    ["Mean bias", `${m.bias > 0 ? "+" : ""}${m.bias.toFixed(2)} mm`],
    ["Domain peak", `${m.max.toFixed(0)} vs ${day.obsMax.toFixed(0)} obs`],
    ["Detection > 30 mm", m.pod == null ? "—" : `${(m.pod * 100).toFixed(0)} %`],
    ["Correlation", m.r == null ? "—" : m.r.toFixed(2)],
  ] : [];

  return (
    <div className="card p-4">
      <p className="hint">Selected product</p>
      <h2 className="text-[23px] font-bold leading-tight mt-0.5">{LAYER_META[focus].label}</h2>
      <p className="mt-2.5"><span className={`badge ${badge.cls}`}>{badge.text}</span></p>

      <p className="text-[14px] leading-[1.5] mt-3">
        {m == null ? (
          <>Not scored on this day.</>
        ) : metric === "bias" ? (
          <>Averaged over the domain it ran <b>{Math.abs(m.bias).toFixed(2)} mm</b> too {m.bias < 0 ? "dry" : "wet"} on
            this day. A near-zero average can still hide large errors that cancel.</>
        ) : (
          <>On this day it was off by <b>{m.rmse.toFixed(2)} mm</b> at the typical square kilometre, over{" "}
            {cells.toLocaleString()} cells{vs == null ? "." : <>— <b>{Math.abs(vs).toFixed(1)} %</b> {vs >= 0 ? "better" : "worse"} than bilinear.</>}</>
        )}
      </p>

      {stats.length > 0 && (
        <div className="grid grid-cols-2 gap-2 mt-3.5">
          {stats.map(([k, v]) => (
            <div key={k} className="border border-[var(--line)] rounded-[var(--r-ctl)] px-2.5 py-2">
              <span className="hint block leading-tight">{k}</span>
              <span className="mono block text-[13.5px] mt-1">{v}</span>
            </div>
          ))}
        </div>
      )}

      <h3 className="text-[13px] font-semibold mt-4 flex items-center gap-2">
        How the products compare
        <Info text={metric === "rmse"
          ? "Scored on this day alone. The headline numbers in the study average every test day."
          : "Mean signed error over the domain for this day."} />
      </h3>
      <div className="mt-2 space-y-[5px]">
        {ranked.map((l, i) => {
          const v = score(l);
          const on = l === focus;
          const signed = metric === "bias" ? day.m[l]!.bias : v;
          return (
            <button key={l} onClick={() => onChange(l)}
              className={`w-full grid grid-cols-[86px_1fr_54px] items-center gap-2 text-left rounded-[5px] px-1 py-[3px] ${on ? "bar-row-top" : ""}`}
              style={{ background: on ? "var(--line2)" : "transparent" }}>
              <span className="text-[11.5px] truncate" style={{ fontWeight: on ? 700 : 400 }}>
                {LAYER_META[l].short}
              </span>
              <span className="bar">
                <i style={{ left: 0, width: `${Math.max(3, (v / worst) * 100)}%`, background: RAMP[Math.min(RAMP.length - 1, i)] }} />
              </span>
              <span className="mono text-[11.5px] text-right" style={{ fontWeight: on ? 700 : 400 }}>
                {metric === "bias" && signed > 0 ? "+" : ""}{signed.toFixed(2)}
              </span>
            </button>
          );
        })}
      </div>
      <p className="hint mt-2">
        {metric === "rmse" ? "RMSE" : "bias"} mm day⁻¹ on {day.date} · {metric === "rmse" ? "lower is better" : "zero is unbiased"}
      </p>

      <button className="btn-primary mt-4" onClick={onSwipe}>Swipe it against the truth</button>
      <p className="text-center mt-2.5">
        <button className="link" onClick={onDiff}>See where this product goes wrong</button>
      </p>
    </div>
  );
}
