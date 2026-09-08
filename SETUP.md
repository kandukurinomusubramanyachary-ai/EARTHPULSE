# Running EarthPulse on your own machine

Short answer: **download the whole `earthpulse` folder, open that folder in VS Code, and run one command.** Don't copy-paste files one by one — the imports depend on the folder structure.

---

## Prerequisites

| Need | Version | Check with |
|---|---|---|
| Python | 3.10 or newer | `python --version` |
| Internet | required | EarthPulse streams live satellite data |

**Windows users:** when installing Python, tick **"Add Python to PATH"** on the first screen of the installer. This is the single most common cause of setup failure.

You do **not** need to install GDAL, QGIS, Anaconda, or Node.js.

---

## Option A — one command (easiest)

Open a terminal in the `earthpulse` folder:

**Windows**
```
run.bat
```
(or just double-click `run.bat` in File Explorer)

**macOS / Linux**
```bash
chmod +x run.sh
./run.sh
```

This creates a virtual environment, installs dependencies on first run (~1–2 min), and starts the server.

Then open **http://localhost:8000**

---

## Option B — VS Code, step by step

### 1. Open the right folder

`File → Open Folder…` → select the **`earthpulse`** folder itself.

Your Explorer sidebar should look like this:

```
earthpulse/          ← this folder must be the one you opened
├── backend/
│   └── app/
├── frontend/
├── docs/
├── requirements.txt
├── run.sh
└── run.bat
```

If you see `app/` at the top level instead of `backend/`, you opened one folder too deep.

### 2. Create a virtual environment

`Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → **Python: Create Environment** → **Venv** → pick your Python 3.10+ interpreter → tick **`requirements.txt`** when it offers to install dependencies.

VS Code will now use this environment automatically.

<details>
<summary>Prefer the terminal? (click)</summary>

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

### 3. Run it

Press **F5** (a launch config is already included), or in the terminal:

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

⚠️ You must be **inside the `backend` folder** when running uvicorn. Running it from `earthpulse/` gives `ModuleNotFoundError: No module named 'app'`.

### 4. Open the app

**http://localhost:8000**

Click any example chip. First analysis takes ~10–15 seconds (it's downloading real Sentinel-2 imagery); later ones are faster once GDAL's cache warms up.

Stop the server with **Ctrl+C**.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'app'`**
You're in the wrong directory. `cd backend` first, then run uvicorn.

**`ModuleNotFoundError: No module named 'fastapi'`**
The virtual environment isn't active. Look for `(.venv)` at the start of your terminal prompt. If it's missing, run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux).
In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one containing `.venv`.

**`'python' is not recognized`** (Windows)
Python isn't on PATH. Reinstall it and tick "Add Python to PATH", or try `py` instead of `python`.

**`Address already in use` / port 8000 busy**
Use a different port: `python -m uvicorn app.main:app --port 8080` and open `http://localhost:8080`.

**Page loads but every analysis fails**
EarthPulse needs outbound internet access to three hosts. On a college or corporate network these are sometimes blocked:
- `earth-search.aws.element84.com` (satellite catalog)
- `sentinel-cogs.s3.us-west-2.amazonaws.com` (imagery)
- `nominatim.openstreetmap.org` (place names)

Test with: `curl https://earth-search.aws.element84.com/v1/` — if that hangs, try a mobile hotspot.

**`No usable Sentinel-2 imagery was found`**
Genuine data gap, not a bug. Widen the year range, raise the "Max cloud" setting, or try a different location.

**Analysis is slow (>30 s)**
Normal on a slow connection — it's streaming real imagery from AWS. Use the **Fast (360 px)** detail setting to speed it up.

---

## Before your demo

1. Start the server a few minutes early.
2. Run one query to warm the cache — the first is always slowest.
3. Best two for a jury:
   - **Urban expansion** (Kokapet) — staggered onset, 91 clusters
   - **Water-body shrinkage** (Sambhar) — the swipe view shows the lake visibly receding

Have a screen recording as backup in case the venue Wi-Fi fails.

---

## Project layout

```
earthpulse/
├── backend/app/
│   ├── main.py         FastAPI orchestrator, 9-stage NDJSON stream
│   ├── nlp.py          natural language → executable QueryPlan
│   ├── geocode.py      place name → bounding box
│   ├── catalog.py      STAC search, season locking, tile mosaicking
│   ├── raster.py       COG streaming, spectral indices
│   ├── changedet.py    change detection + false-change filters
│   ├── explain.py      evidence engine, report writer, safety guard
│   ├── similarity.py   interpretable 13-D signatures
│   ├── render.py       before/after/change imagery
│   └── config.py       thresholds, GDAL tuning
├── frontend/           zero-build SPA — edit and just refresh
├── docs/               Technical Requirements Document
├── requirements.txt
├── run.sh / run.bat
└── SETUP.md            this file
```

**Editing tips**
- Frontend (`frontend/*`): save, refresh the browser. No build step.
- Backend (`backend/app/*`): restart the server, or add `--reload` to the uvicorn command to auto-restart on save.
- Tuning thresholds: `backend/app/config.py` — `Z_THRESHOLD` (sensitivity), `MAX_CLOUD_COVER`, `GRID_SIZE`.
