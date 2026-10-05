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
| **Day by day** | the 10 km input, what fell, and one model's reconstruction, side by side or under a draggable divider |
| **All versions** | every product on one day, on one colour scale |
| **Where it's wrong** | prediction − AORC, diverging scale, adjustable range |
| **Two-year record** | per-day scores for every product across the whole period |
| **What we learned** | the headline results: seasons, seed spread, cross-domain transfer |

Hovering any map reports the value at that square kilometre for **every visible
product at once**, with its error against AORC.

Keyboard: `←` `→` step days, `space` plays.

## Layers

The portal is written for a general reader, so the interface never says RMSE,
bias or POD. [`src/lib/metrics.ts`](src/lib/metrics.ts) holds that vocabulary in
one place — the label, the axis caption and the tooltip for each score. Product
names are left alone: AORC, IMERG, Bilinear, XGBoost, CNN, Swin and Diffusion
appear under their own names, and the plain-language explanation of each one
lives in its `note` in `LAYER_META`
([`src/lib/types.ts`](src/lib/types.ts)), which the interface shows beside the
name rather than instead of it.

| id | label | what it is |
|---|---|---|
| `aorc` | AORC | 1 km gridded analysis — the reference every score is measured against |
| `nearest` | IMERG 10 km | IMERG repeated across each 12 × 12 block |
| `bilinear` | Bilinear | smooth interpolation — the baseline to beat |
| `xgboost` | XGBoost 2-stage | 2-stage gradient-boosted trees — best RMSE |
| `cnn` | CNN (U-Net) | 1.65 M-parameter U-Net |
| `cnn_spectral` | CNN + texture penalty | the same U-Net trained with `--spectral-weight 0.01`; 4x the sub-10 km power at the same RMSE |
| `swin` | Swin transformer | 7.9 M-parameter shifted-window transformer — best heavy-event detection |
| `diffusion` | Diffusion, 6-member mean | mean of 6 diffusion members |
| `diffusion_member` | Diffusion, one member | a single diffusion draw — the only realistic texture |

| score | shown as |
|---|---|
| RMSE | how far off, in mm of rain |
| bias | too wet or too dry |
| POD above 30 mm | heavy rain caught |
| domain maximum | heaviest spot |

## Deploying to Netlify

The app is a **static export** — plain HTML, JS and data files, no serverless
functions. [`netlify.toml`](../netlify.toml) at the repository root points
Netlify at this subdirectory (`base = "web"`, `publish = "out"`).

**Connect the repository and it just works.** The field tiles are committed, so
a clean clone has everything the build needs:

```
Build command     npm run build      (from netlify.toml)
Publish directory out
Node version      20
```

`next build` copies `public/` into `out/`, producing a self-contained 73 MB
site — 2,817 tiles plus the app.

That choice is deliberate and has a cost. The tiles are generated output, not
source, and they are ~70 MB that every future clone pays for; regenerating them
adds another 70 MB to history. The alternative, if the repository becomes
unwieldy, is to host them separately and set an environment variable in the
Netlify UI:

```
NEXT_PUBLIC_DATA_BASE = https://your-bucket.example.com
```

Every metadata and tile request is prefixed with it (`src/lib/domains.ts`).
Unset, the app loads tiles from its own origin. On another domain it needs CORS
to allow your site — the browser decodes the PNGs to real values, so they are
fetched as data rather than loaded as images.

To deploy a one-off build without git:

```bash
npm run build && npx netlify deploy --prod --dir=out
```

Caching is set in `netlify.toml`: hashed `_next/static` assets are immutable,
tiles get an hour with background revalidation, and `meta.json` / `days.json`
get five minutes — those list which layers exist, so a stale copy makes the
viewer ask for tiles that are no longer there.

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
