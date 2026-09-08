# EarthPulse 🛰️

### Ask Earth. Understand Change. See the Evidence.

**SIH26227** · Space Technology · AI-powered Geospatial Intelligence Platform

Ask a question in plain English. EarthPulse finds real Sentinel-2 satellite imagery,
compares it across time, filters out false change, and explains what it found —
with measurements, confidence and evidence.

> This is a **working prototype using live satellite data**, not a mockup.
> No API keys, no mock data, no pre-baked results.

---

## Quick start

**Windows:** double-click `run.bat` — **macOS / Linux:** `chmod +x run.sh && ./run.sh`

Then open **http://localhost:8000** and click any example chip.

<details>
<summary>Manual setup</summary>

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd backend                       # important — must run from here
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
</details>

Requires **Python 3.10+** and an internet connection (it streams live satellite data).
GDAL is bundled with the `rasterio` wheel — don't install it separately.

📖 **New to this? See [SETUP.md](SETUP.md)** for VS Code instructions and troubleshooting.

---

## What it does

```
"Show me where urban expansion occurred in Kokapet between 2019 and 2025"
                              ↓
  interpret → locate → search → load → quality → detect → filter → render → explain
                              ↓
  75.2 ha of built-up expansion · 91 clusters · 97% confidence · 8.2 seconds
```

Every stage streams to the browser as it completes, so you watch the agent work.

---

## Try these

| Query | What it demonstrates |
|---|---|
| `Show me where urban expansion occurred in Kokapet between 2019 and 2025` | Urban growth, staggered onset detection |
| `Find new construction near lakes in Osman Sagar from 2018 to 2025` | Proximity constraint + trust-and-safety language |
| `Which water bodies have significantly reduced around Sambhar between 2019 and 2024?` | Compositional parsing, water loss, tile mosaicking |
| `Show vegetation loss around Wayanad over the last 6 years` | Relative time window, forest margin |
| `What changed in Amaravati between 2018 and 2025?` | Open question, greenfield construction |

---

## The interesting part: rejecting false change

Detecting change is easy — threshold an image difference. **Not being fooled is hard.**

A first run on Kokapet confidently reported **681 hectares of new construction**.
610 of those hectares were a **dried lake bed**: water receding to bare soil raises
the built-up index (NDBI) exactly the way fresh concrete does.

EarthPulse now runs five independent false-change filters:

| Filter | Removes |
|---|---|
| Season locking | phenology — April is only ever compared to April |
| SCL masking | cloud, cirrus, shadow, snow |
| Scene-wide median removal | atmosphere, sun angle, illumination drift |
| **Spectral plausibility gate** | **transitions that move the index but aren't the requested phenomenon** |
| Persistence testing | transient conditions that don't hold across epochs |

Result: **681 ha → 75.2 ha**, and the report explicitly states
*"drying water is not construction, so these were excluded."*

Three more silent-failure bugs found and fixed during the build are documented in
[`docs/EarthPulse-TRD.md §9`](docs/EarthPulse-TRD.md) — a mixed-archive radiometric
defect, a morphology operation that destroyed 86% of true detections, and a STAC
pagination bug that silently truncated recent years.

---

## Measured performance

| AOI | Mode | Epochs | Found | Confidence | Time |
|---|---|---|---|---|---|
| Kokapet | urban | 7 | 75.2 ha | 97% | 8.2 s |
| Wayanad | vegetation | 7 | 314.6 ha | 97% | 12.2 s |
| Sambhar | water loss | 6 | 644.8 ha | 97% | 10.5 s |
| Amaravati | urban | 8 | 198.7 ha | 97% | 10.5 s |

Query interpretation: **13/13** on the intent test set. Frontend: **0 console errors**.

---

## Architecture

```
backend/app/
  config.py       GDAL tuning, thresholds, SCL semantics
  nlp.py          natural language → executable QueryPlan
  geocode.py      place name → bounding box (gazetteer + Nominatim)
  catalog.py      STAC search, season locking, tile mosaicking
  raster.py       COG streaming onto one common grid, spectral indices
  changedet.py    change detection, false-change filters, confidence
  explain.py      evidence engine, report writer, safety guard
  similarity.py   interpretable 13-D signatures
  render.py       before/after/change/heatmap/SCL as data URIs
  main.py         FastAPI orchestrator, 9-stage NDJSON stream

frontend/         zero-build SPA (index.html, app.js, styles.css)
docs/             Technical Requirements Document
```

**Data:** Copernicus Sentinel-2 L2A (ESA) via AWS Open Data · Element84 Earth Search STAC · OSM Nominatim. All free, all keyless.

---

## Trust & safety

Enforced in code, not convention. `explain.sanitise()` rewrites accusatory language
in every generated sentence:

> ❌ "Illegal construction detected."
> ✅ "New construction detected. Regulatory verification may be required."

Every result carries a confidence band, the reasons behind it, stated limitations,
and a recommendation to verify.

---

## Documentation

- **[Technical Requirements Document](docs/EarthPulse-TRD.md)** — architecture, algorithms, measured results, defect analysis, roadmap
