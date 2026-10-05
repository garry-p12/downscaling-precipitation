"use client";
import { DOMAINS, type DomainDef } from "@/lib/domains";
import { WINDOWS } from "@/lib/field";
import { LAYER_META, ORDER, type LayerId, type Meta } from "@/lib/types";
import { RAMP_LABEL, type RampId } from "@/lib/colormap";
import Info from "./Info";

interface Props {
  domain: DomainDef; onDomain: (d: DomainDef) => void;
  win: string; onWin: (k: string) => void;
  ramp: RampId; onRamp: (r: RampId) => void;
  tab: string;
  diffScale: number; onDiffScale: (n: number) => void;
  meta: Meta; sel: LayerId[]; onToggle: (l: LayerId) => void;
  nDays: number;
}

export default function Rail({
  domain, onDomain, win, onWin, ramp, onRamp, tab, diffScale, onDiffScale, meta, sel, onToggle, nDays,
}: Props) {
  const picks = tab === "all" || tab === "diff" || tab === "season";
  const years = [meta.period[0].slice(0, 4), meta.period[1].slice(0, 4)];

  return (
    <aside className="border-b lg:border-b-0 lg:border-r border-[var(--line)] px-4 lg:px-5 py-5
                      [&>*]:max-w-[560px] lg:[&>*]:max-w-none
                      lg:sticky lg:top-[var(--head)] lg:max-h-[calc(100vh-var(--head))] lg:overflow-y-auto">
      <div>
        <span className="field flex items-center gap-2">
          Place <Info text={domain.note} />
        </span>
        <div className="space-y-1.5">
          {DOMAINS.map((d) => (
            <button key={d.id} className="ctl w-full !text-left !py-2.5" aria-pressed={d.id === domain.id}
              onClick={() => onDomain(d)}>
              <span className="block text-[13.5px] leading-tight">{d.label}</span>
              <span className="block text-[11px] mt-[3px] opacity-70">{d.sub}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5">
        <span className="field flex items-center gap-2">
          Zoom <Info text={WINDOWS[win].hint} />
        </span>
        <select className="select" value={win} onChange={(e) => onWin(e.target.value)}>
          {Object.entries(WINDOWS).map(([k, w]) => <option key={k} value={k}>{w.label}</option>)}
        </select>
      </div>

      <div className="mt-5">
        <span className="field">Colours</span>
        <div className="seg w-full">
          {(["warm", "nws"] as RampId[]).map((r) => (
            <button key={r} className="flex-1" aria-pressed={ramp === r} onClick={() => onRamp(r)}>
              {RAMP_LABEL[r]}
            </button>
          ))}
        </div>
      </div>

      {tab === "diff" && (
        <div className="mt-5">
          <span className="field flex items-center gap-2">
            Colour range
            <Info text="How much rain it takes to reach full colour. Turn it down to bring out faint patterns in the blurrier versions." />
          </span>
          <div className="flex items-center gap-3">
            <input type="range" min={5} max={60} step={5} value={diffScale} className="flex-1"
              onChange={(e) => onDiffScale(+e.target.value)} />
            <span className="mono text-[12.5px] w-16 text-right">{diffScale} mm</span>
          </div>
        </div>
      )}

      {picks && (
        <div className="mt-5">
          <span className="field">Which versions to show</span>
          <div className="grid grid-cols-2 gap-1.5">
            {ORDER.filter((l) => meta.layers.includes(l)).map((l) => {
              const on = sel.includes(l);
              return (
                <button key={l} className="ctl !px-2 !py-2 !text-[12px] flex items-center gap-1.5 justify-start"
                  aria-pressed={on} onClick={() => onToggle(l)}>
                  <i className="w-2 h-2 rounded-full shrink-0"
                    style={{ background: LAYER_META[l].color,
                             boxShadow: on ? "0 0 0 1px rgba(255,255,255,.65)" : "0 0 0 1px rgba(20,26,33,.12)" }} />
                  <span className="truncate">{LAYER_META[l].short}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="statbox mt-6 flex items-baseline gap-2">
        <span className="mono text-[17px] leading-none">{nDays}</span>
        <span className="hint">
          rainy days, {years[0]}&ndash;{years[1]}
          <Info text={`The ${nDays} wettest days of the test period plus a spread through the seasons. Every square kilometre is scored, every day.`} />
        </span>
      </div>
    </aside>
  );
}
