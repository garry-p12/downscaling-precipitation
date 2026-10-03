# Rain at One Kilometre — downscaling playground

Interactive comparison of IMERG precipitation downscaled from 10 km to 1 km against
AORC, over Austin, Texas. Four model families, 183 days drawn from the 2019–2020
test period.

## Running

```bash
npm install
npm run dev          # http://localhost:3000
```

The app needs its field tiles in `public/data`. They are not committed (~45 MB);
generate them from the project root:

```bash
python scripts/gen_web_tiles.py --out web/public/data --wettest 130 --stride 11
```

That selects the 130 wettest test days plus every 11th day for seasonal spread,
and writes:

```
public/data/meta.json              grid, bbox, encoding, layer list
public/data/days.json              exact per-day metrics for every product
public/data/fields/<layer>/<date>.png
```

## How the tiles work

Each tile is a 360 × 420 PNG in greyscale+alpha. **It is a data file, not a picture**:

- luminance carries the value on a square-root ramp — `mm = 200 * (L/255)²`
- alpha carries the validity mask (0 = no data)
- values below 0.1 mm/day are floored to zero, which is below retrieval
  significance and roughly halves the file size

The browser decodes a tile back to a `Float32Array` of mm/day
([`src/lib/field.ts`](src/lib/field.ts)), so colour maps, difference fields and
point readouts are all computed client-side from real values. Per-day metrics in
`days.json` are computed in NumPy at full float32 precision — the quantised tiles
are only for display.

## Views

| tab | what it answers |
|---|---|
| **Side by side** | every product on one day, one colour scale |
| **A / B swipe** | drag a divider between any two products |
| **Difference** | prediction − AORC, diverging scale, adjustable range |
| **Across days** | per-day RMSE / bias / POD / peak through the period |

Hovering any map reports the value at that square kilometre for **every visible
product at once**, with its error against AORC.

Keyboard: `←` `→` step days, `space` plays.

## Layers

| id | what it is |
|---|---|
| `aorc` | 1 km reference analysis — the truth every score is against |
| `nearest` | IMERG repeated across each 12 × 12 block |
| `bilinear` | smooth interpolation — the baseline to beat |
| `xgboost` | 2-stage gradient-boosted trees — best RMSE |
| `cnn` | 1.65 M-parameter U-Net |
| `swin` | 7.9 M-parameter shifted-window transformer — best heavy-event detection |
| `diffusion` | mean of 6 diffusion members |
| `diffusion_member` | a single diffusion draw — the only realistic texture |

## Design

The shell follows the Austin Rainfall Risk Mapper wireframes: a white masthead over
a hairline rule with a dark segmented view switcher, a control rail on the left, the
map stage in the middle, and a selected-product panel on the right. Tokens live at the
top of [`src/app/globals.css`](src/app/globals.css) — one warm cream-to-terracotta
sequential ramp is shared by the maps, the comparison bars and the legend blocks,
dark slate carries active controls, and blue is reserved for the single primary action
on a panel. Numbers are monospace; everything else is sans.

The legend is a row of discrete blocks cut on the same sqrt stretch the canvas uses,
so a block is genuinely the colour of that slice of the display range. `Colour scale`
in the rail switches the maps between that warm ramp and the multi-hue NWS ramp.

## Stack

Next.js 16 (App Router), React 19, TypeScript, Tailwind 4. No charting or mapping
library — the canvas renderer and SVG charts are in `src/components`.
