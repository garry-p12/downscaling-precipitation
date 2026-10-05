"use client";
import { useState } from "react";
import type { Day } from "@/lib/types";

export interface TabDef { id: string; label: string }

interface Props {
  tabs: TabDef[]; value: string; onChange: (id: string) => void;
  days: Day[] | null; index: number; onIndex: (i: number) => void;
}

export default function AppBar({ tabs, value, onChange, days, index, onIndex }: Props) {
  const [typed, setTyped] = useState("");

  /** The record is sparse — 183 of 731 test days — so a typed date snaps to
      the nearest day that actually has tiles rather than failing. */
  const jump = (s: string) => {
    setTyped(s);
    if (!days || !/^\d{4}-\d{2}-\d{2}$/.test(s)) return;
    const t = Date.parse(s);
    if (!isFinite(t)) return;
    let best = 0, bd = Infinity;
    for (let i = 0; i < days.length; i++) {
      const d = Math.abs(Date.parse(days[i].date) - t);
      if (d < bd) { bd = d; best = i; }
    }
    onIndex(best);
  };

  return (
    <header className="sticky top-0 z-40 bg-[var(--page)] border-b border-[var(--line)]">
      {/* one row from lg up; below that the view switcher wraps to its own
          scrollable line rather than forcing the page wider than the screen */}
      <div className="mx-auto max-w-[1760px] px-4 lg:px-6 lg:min-h-[var(--head)]
                      flex flex-wrap items-center gap-x-4 lg:gap-x-6 gap-y-3 py-3">
        <div className="min-w-0">
          <div className="text-[17px] font-bold leading-tight tracking-[-0.01em] truncate">Rain at One Kilometre</div>
          <div className="mono text-[10.5px] text-[var(--ink3)] mt-[3px] truncate">
            Sharpening satellite rain maps · research project
          </div>
        </div>

        <div className="order-last lg:order-none w-full lg:w-auto min-w-0 overflow-x-auto">
          <nav className="seg seg-dark" aria-label="Views">
            {tabs.map((t) => (
              <button key={t.id} aria-pressed={t.id === value} onClick={() => onChange(t.id)}>{t.label}</button>
            ))}
          </nav>
        </div>

        <label className="ml-auto hidden xl:block w-[250px] shrink-0">
          <span className="hint block mb-1">Jump to a date</span>
          <input className="input mono text-[13px]" list="record-dates" placeholder={days?.[index]?.date ?? "YYYY-MM-DD"}
            value={typed} onChange={(e) => jump(e.target.value)} />
          <datalist id="record-dates">
            {days?.map((d) => <option key={d.date} value={d.date} />)}
          </datalist>
        </label>
      </div>
    </header>
  );
}
