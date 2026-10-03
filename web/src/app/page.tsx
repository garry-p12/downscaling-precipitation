"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { type View } from "@/components/FieldCanvas";
import AppBar from "@/components/AppBar";
import Rail from "@/components/Rail";
import MapCard from "@/components/MapCard";
import ProductPanel from "@/components/ProductPanel";
import Scrubber from "@/components/Scrubber";
import SwipeView from "@/components/SwipeView";
import SeasonChart from "@/components/SeasonChart";
import HoverReadout from "@/components/HoverReadout";
import Info from "@/components/Info";
import Findings from "@/components/Findings";
import { DOMAINS } from "@/lib/domains";
import { loadField, prefetch, WINDOWS } from "@/lib/field";
import type { RampId } from "@/lib/colormap";
import { LAYER_META, type Day, type Field, type LayerId, type Meta } from "@/lib/types";

type Tab = "day" | "all" | "diff" | "season" | "findings";
type SeasonMetric = "rmse" | "bias" | "pod" | "max";

const TABS = [
  { id: "day",      label: "Day by day" },
  { id: "all",      label: "All products" },
  { id: "diff",     label: "Where it goes wrong" },
  { id: "season",   label: "Whole record" },
  { id: "findings", label: "What we found" },
];

const HEADLINE: Record<Tab, { h: string; info: string }> = {
  day:    { h: "Ten kilometres to one, day by day", info: "Left is the IMERG retrieval as delivered: 1,050 coarse cells, each one flat across the 12×12 block of kilometres beneath it. Centre is what actually fell. Right is the model's reconstruction. All three share one colour scale." },
  all:    { h: "Every product, one storm", info: "All eight layers on the selected day. Differences between panels are the models and nothing else." },
  diff:   { h: "Where each product goes wrong", info: "Prediction minus AORC. Blue is too dry, terracotta is too wet. Narrow the range to see structure in the smooth products." },
  season: { h: "Skill across the whole record", info: "Per-day scores for every product. The lines sit almost on top of each other, which is the finding rather than a plotting error." },
  findings: { h: "What the three studies show", info: "Headline results across all domains, inputs and seed repeats. Every number here comes from a scored pipeline run." },
};

export default function Page() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [days, setDays] = useState<Day[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [domain, setDomain] = useState(DOMAINS[0]);
  const [tab, setTab] = useState<Tab>("day");
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(900);
  const [win, setWin] = useState<string>("full");
  const [ramp, setRamp] = useState<RampId>("warm");
  const [focus, setFocus] = useState<Exclude<LayerId, "aorc">>("xgboost");
  const [mode, setMode] = useState<"split" | "swipe">("split");
  const [swipeLeft, setSwipeLeft] = useState<LayerId>("nearest");
  const [sel, setSel] = useState<LayerId[]>(["aorc", "bilinear", "xgboost", "swin", "diffusion_member"]);
  const [diffScale, setDiffScale] = useState(20);
  const [seasonMetric, setSeasonMetric] = useState<SeasonMetric>("rmse");
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [fields, setFields] = useState<Partial<Record<LayerId, Field>>>({});

  // Track which domain the loaded data belongs to, rather than clearing state
  // synchronously in the effect: a stale meta/days pair must never render
  // against a new domain's tiles.
  const [loadedFor, setLoadedFor] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    Promise.all([fetch(`${domain.dir}/meta.json`).then((r) => r.json()),
                 fetch(`${domain.dir}/days.json`).then((r) => r.json())])
      .then(([m, d]: [Meta, Day[]]) => {
        if (!live) return;
        setMeta(m); setDays(d); setFields({}); setErr(null); setLoadedFor(domain.id);
        // open on the biggest storm; a dry day renders as a blank panel
        let best = 0;
        for (let i = 1; i < d.length; i++) if (d[i].obsMean > d[best].obsMean) best = i;
        setIndex(best);
        // each domain ships its own layer set: Austin has four models, the others one.
        // Focus must land on a MODEL -- "nearest" and "bilinear" are baselines and
        // already have their own panels.
        const avail = m.layers.filter((l) => l !== "aorc");
        const model = (["xgboost", "ml", "cnn", "swin"] as const).find((l) => avail.includes(l))
          ?? avail[avail.length - 1];
        setFocus(model as Exclude<LayerId, "aorc">);
        setSel(["aorc", ...avail.slice(0, 4)] as LayerId[]);
      })
      .catch((e) => live && setErr(String(e)));
    return () => { live = false; };
  }, [domain]);

  const date = days?.[index]?.date;
  const need = useMemo<LayerId[]>(() => {
    const s = new Set<LayerId>(["aorc"]);
    if (tab === "all" || tab === "diff" || tab === "season") sel.forEach((l) => s.add(l));
    else { s.add(focus); s.add("nearest"); }
    return [...s];
  }, [tab, sel, focus]);

  useEffect(() => {
    if (!meta || !date || loadedFor !== domain.id) return;
    let live = true;
    Promise.all(need.map((l) => loadField(l, date, meta, domain.dir).then((f) => [l, f] as const).catch(() => null)))
      .then((pairs) => {
        if (!live) return;
        const next: Partial<Record<LayerId, Field>> = {};
        for (const p of pairs) if (p) next[p[0]] = p[1];
        setFields(next);
      });
    return () => { live = false; };
  }, [meta, date, need, domain, loadedFor]);

  useEffect(() => {
    if (!meta || !days || loadedFor !== domain.id) return;
    prefetch(need, [1, 2, 3].map((k) => days[index + k]?.date).filter(Boolean) as string[], meta, domain.dir);
  }, [meta, days, index, need, domain, loadedFor]);

  useEffect(() => {
    if (!playing || !days) return;
    const t = setInterval(() => setIndex((i) => (i + 1) % days.length), speed);
    return () => clearInterval(t);
  }, [playing, speed, days]);

  const onKey = useCallback((e: KeyboardEvent) => {
    if (!days) return;
    const el = e.target as HTMLElement | null;
    if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return;
    if (e.key === "ArrowRight") setIndex((i) => Math.min(days.length - 1, i + 1));
    if (e.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
    if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
  }, [days]);
  useEffect(() => { window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey); }, [onKey]);

  if (err) return <main className="p-10 mono text-sm">Could not load tiles: {err}</main>;
  if (!meta || !days || loadedFor !== domain.id) return (
    <main className="min-h-screen grid place-items-center mono text-[12px] text-[var(--ink3)]">loading fields…</main>
  );

  const box = WINDOWS[win].box;
  const view: View = box ? { x0: box[0], y0: box[1], x1: box[2], y1: box[3] } : { x0: 0, y0: 0, x1: 1, y1: 1 };
  const day = days[index];
  const fm = day.m[focus];
  const cells = meta.width * meta.height;
  const wide = tab === "findings";
  const toggle = (l: LayerId) =>
    setSel((s) => (s.includes(l) ? (s.length > 1 ? s.filter((x) => x !== l) : s) : [...s, l]));

  const level = domain.gain >= 15 ? "chip-high" : domain.gain >= 8 ? "chip-elev" : "chip-low";

  return (
    <div className="min-h-screen flex flex-col">
      <AppBar tabs={TABS} value={tab} onChange={(t) => setTab(t as Tab)}
        days={days} index={index} onIndex={(i) => { setIndex(i); setPlaying(false); }}
        onAbout={() => setTab("findings")} />

      <div className={`mx-auto w-full max-w-[1760px] flex-1 grid grid-cols-1 ${
        wide ? "" : "lg:grid-cols-[308px_minmax(0,1fr)] xl:grid-cols-[308px_minmax(0,1fr)_344px]"}`}>

        {!wide && (
          <Rail domain={domain} onDomain={setDomain} win={win} onWin={setWin}
            ramp={ramp} onRamp={setRamp} tab={tab}
            diffScale={diffScale} onDiffScale={setDiffScale}
            meta={meta} sel={sel} onToggle={toggle} nDays={meta.nDays} />
        )}

        <main className="min-w-0 px-4 lg:px-6 py-5">
          {/* the reference's alert band, carrying this domain's headline result */}
          <div className="band">
            <span className={`band-chip ${level}`}>
              {domain.gain.toFixed(1)} % better than bilinear
            </span>
            <div className="min-w-[240px] flex-1">
              <p className="text-[13.5px] font-semibold leading-snug">
                {domain.label} — the shipped product cuts RMSE to {domain.rmse.toFixed(3)} mm day⁻¹,
                from {domain.bilinear.toFixed(3)} for plain interpolation
              </p>
              <p className="hint mt-0.5">
                Averaged over the full {meta.period[0]} – {meta.period[1]} test period, not the day shown below
              </p>
            </div>
            <button className="link shrink-0" onClick={() => setTab("findings")}>How this was measured</button>
          </div>

          <div className="flex items-start gap-4 flex-wrap mt-5">
            <div className="min-w-0">
              <h1 className="text-[21px] font-bold leading-tight tracking-[-0.012em] flex items-center gap-2 flex-wrap">
                {HEADLINE[tab].h} <Info text={HEADLINE[tab].info} />
              </h1>
              <p className="hint mt-1">
                {domain.input} → 1 km · {domain.label} · day {index + 1} of {days.length} · {day.date}
              </p>
            </div>

            <div className="ml-auto flex flex-wrap items-center gap-2">
              {tab === "day" && (
                <div className="seg">
                  <button aria-pressed={mode === "split"} onClick={() => setMode("split")}>Input · truth · output</button>
                  <button aria-pressed={mode === "swipe"} onClick={() => setMode("swipe")}>Swipe</button>
                </div>
              )}
              {tab === "day" && mode === "swipe" && (
                <div className="seg">
                  <button aria-pressed={swipeLeft === "nearest"} onClick={() => setSwipeLeft("nearest")}>vs the 10 km input</button>
                  <button aria-pressed={swipeLeft === "aorc"} onClick={() => setSwipeLeft("aorc")}>vs the truth</button>
                </div>
              )}
              {tab === "season" && (
                <div className="seg">
                  {(["rmse", "bias", "pod", "max"] as SeasonMetric[]).map((m) => (
                    <button key={m} aria-pressed={seasonMetric === m} onClick={() => setSeasonMetric(m)}>
                      {({ rmse: "RMSE", bias: "Bias", pod: "POD", max: "Peak" } as const)[m]}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="space-y-4 mt-4">
            {tab === "day" && mode === "split" && (
              <div className="grid md:grid-cols-2 2xl:grid-cols-3 gap-4">
                <MapCard layer="nearest" meta={meta} view={view} field={fields.nearest} mode="value" ramp={ramp}
                  valueLabel={`Input — ${domain.input}`} tag="input"
                  sub={`each value flat across ${(meta.factor ?? 12) ** 2} square kilometres`}
                  value={day.m.nearest?.rmse ?? null} unit="RMSE before downscaling"
                  cursor={cursor} onHover={setCursor} />
                <MapCard layer="aorc" meta={meta} view={view} field={fields.aorc} mode="value" ramp={ramp}
                  valueLabel="Observed — AORC 1 km" tag="truth"
                  sub={`${cells.toLocaleString()} cells. What actually fell.`}
                  value={day.obsMax} unit="peak mm day⁻¹"
                  cursor={cursor} onHover={setCursor} />
                <MapCard layer={focus} meta={meta} view={view} field={fields[focus]} mode="value" ramp={ramp}
                  valueLabel={`Downscaled — ${LAYER_META[focus].short}`} tag="selected"
                  value={fm?.rmse ?? null} unit="RMSE after downscaling" selected
                  cursor={cursor} onHover={setCursor} />
              </div>
            )}

            {tab === "day" && mode === "swipe" && (
              <SwipeView meta={meta} view={view} left={swipeLeft} right={focus} ramp={ramp}
                fields={fields} cursor={cursor} onHover={setCursor} />
            )}

            {tab === "all" && (
              <div className="grid md:grid-cols-2 2xl:grid-cols-3 gap-4">
                {sel.map((l) => (
                  <MapCard key={l} layer={l} meta={meta} view={view} field={fields[l]} mode="value" ramp={ramp}
                    value={l === "aorc" ? day.obsMax : day.m[l]?.rmse ?? null}
                    unit={l === "aorc" ? "peak mm day⁻¹" : "RMSE mm day⁻¹"}
                    selected={l === focus}
                    cursor={cursor} onHover={setCursor} />
                ))}
              </div>
            )}

            {tab === "diff" && (
              <div className="grid md:grid-cols-2 2xl:grid-cols-3 gap-4">
                {sel.filter((l) => l !== "aorc").map((l) => (
                  <MapCard key={l} layer={l} meta={meta} view={view} field={fields[l]} ref_={fields.aorc}
                    mode="diff" diffScale={diffScale} ramp={ramp}
                    value={day.m[l]?.bias ?? null} unit="bias mm day⁻¹"
                    selected={l === focus}
                    cursor={cursor} onHover={setCursor} />
                ))}
              </div>
            )}

            {tab === "season" && (
              <SeasonChart days={days} index={index} layers={sel} metric={seasonMetric} onIndex={setIndex} />
            )}

            {tab === "findings" && <Findings />}

            {tab !== "findings" && (
              <Scrubber days={days} index={index} playing={playing} speed={speed} focus={focus}
                onIndex={(i) => { setIndex(i); setPlaying(false); }}
                onPlay={() => setPlaying((p) => !p)} onSpeed={setSpeed} />
            )}
          </div>
        </main>

        {!wide && (
          <aside className="border-t xl:border-t-0 xl:border-l border-[var(--line)] px-4 lg:px-5 py-5 space-y-4
                            lg:col-span-2 xl:col-span-1
                            xl:sticky xl:top-[var(--head)] xl:max-h-[calc(100vh-var(--head))] xl:overflow-y-auto">
            <ProductPanel day={day} value={focus} onChange={(l) => setFocus(l as Exclude<LayerId, "aorc">)}
              only={meta.layers} exclude={tab === "day" ? ["nearest"] : []}
              metric={tab === "diff" ? "bias" : "rmse"} cells={cells}
              onSwipe={() => { setTab("day"); setMode("swipe"); setSwipeLeft("aorc"); }}
              onDiff={() => setTab("diff")} />
            <HoverReadout cursor={cursor} layers={need} fields={fields} meta={meta} ramp={ramp} />
          </aside>
        )}
      </div>

      <footer className="border-t border-[var(--line)]">
        <div className="mx-auto max-w-[1760px] px-4 lg:px-6 py-4 flex flex-wrap gap-x-6 gap-y-2 justify-between
                        text-[11.5px] text-[var(--ink3)]">
          <span>
            Data: {domain.input} + AORC 1 km · {meta.nDays} days, {meta.period[0]} – {meta.period[1]} ·{" "}
            Method: <button className="link !text-[11.5px]" onClick={() => setTab("findings")}>the study</button>
          </span>
          <span>Research product — not an operational forecast</span>
        </div>
      </footer>
    </div>
  );
}
