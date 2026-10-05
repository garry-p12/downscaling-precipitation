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
import { DOMAINS, dataUrl } from "@/lib/domains";
import { loadField, prefetch, WINDOWS } from "@/lib/field";
import { METRIC } from "@/lib/metrics";
import type { RampId } from "@/lib/colormap";
import { LAYER_META, type Day, type Field, type LayerId, type Meta } from "@/lib/types";

type Tab = "day" | "all" | "diff" | "season" | "findings";
type SeasonMetric = "rmse" | "bias" | "pod" | "max";

const TABS = [
  { id: "day",      label: "Day by day" },
  { id: "all",      label: "All versions" },
  { id: "diff",     label: "Where it's wrong" },
  { id: "season",   label: "Two-year record" },
  { id: "findings", label: "What we learned" },
];

/** The season tab's four scores, in the words the rest of the page uses. */
const SEASON_LABEL: Record<SeasonMetric, string> = {
  rmse: METRIC.miss.label, bias: METRIC.wetdry.label,
  pod: METRIC.heavy.label, max: METRIC.peak.label,
};

const HEADLINE: Record<Tab, { h: string; info: string }> = {
  day: { h: "", info: "" },   // built per place in the component -- see dayHead
  all: {
    h: "Every version of the same storm",
    info: "Every version, on the day you have picked. Anything that differs between the panels is the model and nothing else.",
  },
  diff: {
    h: "Where each one gets it wrong",
    info: "Each version with what actually fell subtracted from it. Blue means too dry, orange means too wet. Turn the colour range down on the left to bring out faint patterns in the blurrier versions.",
  },
  season: {
    h: "How they do across two years",
    info: "A score for every day in the record. The lines sit almost on top of one another — that is the finding, not a drawing error.",
  },
  findings: {
    h: "What we learned",
    info: "The headline results across all three places. Every number came out of a scored run.",
  },
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

  // Track which place the loaded data belongs to, rather than clearing state
  // synchronously in the effect: a stale meta/days pair must never render
  // against a new place's tiles.
  const [loadedFor, setLoadedFor] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    Promise.all([fetch(dataUrl(`${domain.dir}/meta.json`)).then((r) => r.json()),
                 fetch(dataUrl(`${domain.dir}/days.json`)).then((r) => r.json())])
      .then(([m, d]: [Meta, Day[]]) => {
        if (!live) return;
        setMeta(m); setDays(d); setFields({}); setErr(null); setLoadedFor(domain.id);
        // open on the biggest storm; a dry day renders as a blank panel
        let best = 0;
        for (let i = 1; i < d.length; i++) if (d[i].obsMean > d[best].obsMean) best = i;
        setIndex(best);
        // each place ships its own set: Austin has four models, the others one.
        // Focus must land on a MODEL -- the satellite and plain smoothing are
        // the starting point and the baseline, and already have their own panels.
        const avail = m.layers.filter((l) => l !== "aorc");
        const model = (["xgboost", "ml", "cnn", "swin"] as const).find((l) => avail.includes(l))
          ?? avail[avail.length - 1];
        setFocus(model as Exclude<LayerId, "aorc">);
        setSel(["aorc", ...avail.slice(0, 5)] as LayerId[]);
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

  if (err) return <main className="p-10 text-sm">The maps could not be loaded: <span className="mono">{err}</span></main>;
  if (!meta || !days || loadedFor !== domain.id) return (
    <main className="min-h-screen grid place-items-center mono text-[12px] text-[var(--ink3)]">loading maps…</main>
  );

  const box = WINDOWS[win].box;
  const view: View = box ? { x0: box[0], y0: box[1], x1: box[2], y1: box[3] } : { x0: 0, y0: 0, x1: 1, y1: 1 };
  const day = days[index];
  const fm = day.m[focus];
  const years = [meta.period[0].slice(0, 4), meta.period[1].slice(0, 4)];
  const wide = tab === "findings";
  const toggle = (l: LayerId) =>
    setSel((s) => (s.includes(l) ? (s.length > 1 ? s.filter((x) => x !== l) : s) : [...s, l]));

  const level = domain.gain >= 15 ? "chip-high" : domain.gain >= 8 ? "chip-elev" : "chip-low";
  const perCoarse = (meta.factor ?? 12) ** 2;
  const head = tab === "day" ? {
    h: `From a ${domain.coarse} blur to a 1 km map`,
    info: `On the left is what the ${domain.source} delivers: one rainfall number per ${domain.coarse} `
        + `square, held flat across the ${perCoarse.toLocaleString()} square kilometres underneath it. `
        + `In the middle is what actually fell. On the right is a model's attempt at filling in the `
        + `detail. All three use the same colours.`,
  } : HEADLINE[tab];

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
          {/* the headline result for this place, in the reference's alert band */}
          <div className="band">
            <span className={`band-chip ${level}`}>{domain.gain.toFixed(1)} % closer</span>
            <div className="min-w-[240px] flex-1">
              <p className="text-[13.5px] font-semibold leading-snug">
                Best model lands within {domain.rmse.toFixed(1)} mm; plain smoothing is{" "}
                {domain.bilinear.toFixed(1)} mm out.
                <Info text={`Averaged over every day in ${years[0]} and ${years[1]} over ${domain.label}, not the single day shown below.`} />
              </p>
            </div>
            <button className="link shrink-0" onClick={() => setTab("findings")}>How we measured this</button>
          </div>

          <div className="flex items-start gap-4 flex-wrap mt-5">
            <div className="min-w-0">
              <h1 className="text-[21px] font-bold leading-tight tracking-[-0.012em] flex items-center gap-2 flex-wrap">
                {head.h} <Info text={head.info} />
              </h1>
              <p className="hint mt-1">
                {domain.coarse} &rarr; 1 km{!wide && <> &middot; {day.date}</>}
              </p>
            </div>

            <div className="ml-auto flex flex-wrap items-center gap-2">
              {tab === "day" && (
                <div className="seg">
                  <button aria-pressed={mode === "split"} onClick={() => setMode("split")}>Side by side</button>
                  <button aria-pressed={mode === "swipe"} onClick={() => setMode("swipe")}>Slide to compare</button>
                </div>
              )}
              {tab === "day" && mode === "swipe" && (
                <div className="seg">
                  <button aria-pressed={swipeLeft === "nearest"} onClick={() => setSwipeLeft("nearest")}>against the satellite</button>
                  <button aria-pressed={swipeLeft === "aorc"} onClick={() => setSwipeLeft("aorc")}>against what fell</button>
                </div>
              )}
              {tab === "season" && (
                <div className="seg">
                  {(["rmse", "bias", "pod", "max"] as SeasonMetric[]).map((m) => (
                    <button key={m} aria-pressed={seasonMetric === m} onClick={() => setSeasonMetric(m)}>
                      {SEASON_LABEL[m]}
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
                  valueLabel="Where we start" tag={domain.input.toLowerCase()}
                  sub={`One number held flat across ${perCoarse.toLocaleString()} km².`}
                  value={day.m.nearest?.rmse ?? null} unit="how far off, mm"
                  cursor={cursor} onHover={setCursor} />
                <MapCard layer="aorc" meta={meta} view={view} field={fields.aorc} mode="value" ramp={ramp}
                  valueLabel="What actually fell" tag="measured, 1 km"
                  sub="Every square kilometre measured separately."
                  value={day.obsMax} unit="heaviest spot, mm" decimals={0}
                  cursor={cursor} onHover={setCursor} />
                <MapCard layer={focus} meta={meta} view={view} field={fields[focus]} mode="value" ramp={ramp}
                  valueLabel={`Where we end — ${LAYER_META[focus].short}`} tag="model, 1 km"
                  sub="The model's attempt at the detail the input missed."
                  value={fm?.rmse ?? null} unit="how far off, mm" selected
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
                    unit={l === "aorc" ? "heaviest spot, mm" : "how far off, mm"} decimals={l === "aorc" ? 0 : 1}
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
                    value={day.m[l]?.bias ?? null} unit="too wet or dry, mm"
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
              <Scrubber days={days} index={index} playing={playing} speed={speed}
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
              metric={tab === "diff" ? "wetdry" : "miss"}
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
            {meta.nDays} days, {years[0]}&ndash;{years[1]} ·{" "}
            <button className="link !text-[11.5px]" onClick={() => setTab("findings")}>how it works</button>
          </span>
          <span>Research project — not a forecast</span>
        </div>
      </footer>
    </div>
  );
}
