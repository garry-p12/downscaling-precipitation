"use client";
import type { Day, LayerId } from "@/lib/types";
import { LAYER_META, ORDER } from "@/lib/types";
import { METRIC, mm, score as fmt, ordinal } from "@/lib/metrics";
import Info from "./Info";

const RAMP = ["var(--ramp-2)", "var(--ramp-3)", "var(--ramp-4)", "var(--ramp-5)", "var(--ramp-6)", "var(--ramp-6)"];

interface Props {
  day: Day; value: LayerId; onChange: (l: LayerId) => void;
  metric: "miss" | "wetdry";
  /** Versions shown elsewhere on the page already (the satellite has its own panel). */
  exclude?: LayerId[];
  /** Versions this place ships; Austin has four models, the others one. */
  only?: string[];
  onSwipe: () => void; onDiff: () => void;
}

export default function ProductPanel({
  day, value, onChange, metric, exclude = [], only, onSwipe, onDiff,
}: Props) {
  const rows = (ORDER.filter((l) => l !== "aorc" && !exclude.includes(l)
    && (!only || only.includes(l))) as Exclude<LayerId, "aorc">[])
    .filter((l) => day.m[l]);

  const score = (l: Exclude<LayerId, "aorc">) => {
    const m = day.m[l]!;
    return metric === "miss" ? m.rmse : Math.abs(m.bias);
  };
  const ranked = [...rows].sort((a, b) => score(a) - score(b));
  const worst = ranked.length ? score(ranked[ranked.length - 1]) : 1;

  const focus = value as Exclude<LayerId, "aorc">;
  const rank = ranked.indexOf(focus);
  const n = ranked.length;
  const m = day.m[focus];
  const base = day.m.bilinear;
  const vs = m && base && base.rmse > 0 ? ((base.rmse - m.rmse) / base.rmse) * 100 : null;

  const word = metric === "miss" ? "closest" : "most even";
  const worstWord = metric === "miss" ? "furthest off" : "most lopsided";
  const badge =
    rank === 0 ? { cls: "badge-good", text: `Today's ${word}, of ${n}` }
    : rank === n - 1 ? { cls: "", text: `Today's ${worstWord}, of ${n}` }
    : { cls: "badge-flat", text: `${ordinal(rank + 1)} ${word} of ${n}` };

  const stats: { k: string; v: string; note?: string; help: string }[] = m ? [
    { k: METRIC.wetdry.label, v: `${mm(Math.abs(m.bias))} mm too ${m.bias < 0 ? "dry" : "wet"}`,
      help: METRIC.wetdry.help },
    { k: METRIC.peak.label, v: `${m.max.toFixed(0)} mm`, note: `what fell: ${day.obsMax.toFixed(0)} mm`,
      help: METRIC.peak.help },
    { k: METRIC.heavy.label, v: m.pod == null ? "—" : `${(m.pod * 100).toFixed(0)} in 100`,
      help: METRIC.heavy.help },
    { k: "Pattern match", v: m.r == null ? "—" : m.r.toFixed(2),
      help: "How closely the shape of the rain matches what fell, from 0 (no resemblance) to 1 (identical)." },
  ] : [];

  return (
    <div className="card p-4">
      <p className="hint">Showing</p>
      <h2 className="text-[23px] font-bold leading-tight mt-0.5 flex items-center gap-2">
        {LAYER_META[focus].label} <Info text={LAYER_META[focus].note} />
      </h2>
      <p className="mt-2.5"><span className={`badge ${badge.cls}`}>{badge.text}</span></p>

      <p className="text-[14px] leading-[1.5] mt-3">
        {m == null ? (
          <>Not scored on this day.</>
        ) : metric === "wetdry" ? (
          <>Across the whole area it ran <b>{mm(Math.abs(m.bias))} mm too {m.bias < 0 ? "dry" : "wet"}</b> on
            this day. An average close to zero can still hide big errors that cancel out.</>
        ) : (
          <>On this day it was off by about <b>{mm(m.rmse)} mm of rain</b> at the typical spot
            {vs == null ? "." : <> — {Math.abs(vs).toFixed(0)} % {vs >= 0 ? "closer" : "further off"} than bilinear.</>}</>
        )}
      </p>

      {stats.length > 0 && (
        <div className="grid grid-cols-2 gap-2 mt-3.5">
          {stats.map((st) => (
            <div key={st.k} className="border border-[var(--line)] rounded-[var(--r-ctl)] px-2.5 py-2">
              <span className="hint flex items-start gap-1.5 leading-tight min-h-[28px]">
                {st.k} <Info text={st.help} />
              </span>
              <span className="mono block text-[13px]">{st.v}</span>
              {st.note && <span className="hint block mt-0.5">{st.note}</span>}
            </div>
          ))}
        </div>
      )}

      <h3 className="text-[13px] font-semibold mt-4 flex items-center gap-2">
        How the versions compare
        <Info text={metric === "miss"
          ? `${METRIC.miss.help} This is one day only — the headline numbers average every day in the record.`
          : METRIC.wetdry.help} />
      </h3>
      <div className="mt-2 space-y-[5px]">
        {ranked.map((l, i) => {
          const v = score(l);
          const on = l === focus;
          const signed = metric === "wetdry" ? day.m[l]!.bias : v;
          return (
            <button key={l} onClick={() => onChange(l)}
              className="w-full grid grid-cols-[86px_1fr_54px] items-center gap-2 text-left rounded-[5px] px-1 py-[3px]"
              style={{ background: on ? "var(--line2)" : "transparent" }}>
              <span className="text-[11.5px] truncate" style={{ fontWeight: on ? 700 : 400 }}>
                {LAYER_META[l].short}
              </span>
              <span className="bar">
                <i style={{ left: 0, width: `${Math.max(3, (v / worst) * 100)}%`, background: RAMP[Math.min(RAMP.length - 1, i)] }} />
              </span>
              <span className="mono text-[11.5px] text-right" style={{ fontWeight: on ? 700 : 400 }}>
                {metric === "wetdry" && signed > 0 ? "+" : metric === "wetdry" ? "−" : ""}
                {fmt(Math.abs(signed))}
              </span>
            </button>
          );
        })}
      </div>
      <p className="hint mt-2">mm, {day.date} · shorter is better</p>

      <button className="btn-primary mt-4" onClick={onSwipe}>Compare it with what fell</button>
      <p className="text-center mt-2.5">
        <button className="link" onClick={onDiff}>See where it gets it wrong</button>
      </p>
    </div>
  );
}
