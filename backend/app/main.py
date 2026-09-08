"""
EarthPulse — FastAPI application and agentic orchestrator (PRD §21, §22).

The `/api/analyze` endpoint streams newline-delimited JSON so the frontend can
show each pipeline stage as it completes — the agent is visibly *reasoning and
acting*, not returning one opaque blob.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import gc
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from . import aoi
from . import catalog, changedet, explain, geocode, nlp, raster, render, similarity

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_POOL = ThreadPoolExecutor(max_workers=8)
# Hard ceiling on epochs per analysis — bounds peak memory regardless of AOI
# size or how many years the user asks for.
MAX_EPOCHS = int(os.getenv("EP_MAX_EPOCHS", "8"))
# Cache the last full analysis so /api/report and /api/similar can reuse it.
_LAST: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="Natural-language Earth question")
    place: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    bbox: list[float] | None = None
    grid: int | None = None
    max_cloud: float | None = None


class InterpretRequest(BaseModel):
    query: str
    place: str | None = None
    start_year: int | None = None
    end_year: int | None = None


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "server_date": _dt.date.today().isoformat(),
        "data_source": "Copernicus Sentinel-2 L2A via AWS Open Data (Element84 Earth Search STAC)",
    }


@app.post("/api/interpret")
async def interpret(req: InterpretRequest) -> dict[str, Any]:
    """Feature 2 — parse only, no imagery. Used for the live query preview."""
    plan = nlp.interpret(
        req.query,
        place_override=req.place,
        start_override=req.start_year,
        end_override=req.end_year,
    )
    return plan.to_dict()


@app.get("/api/examples")
async def examples() -> dict[str, Any]:
    return {"examples": EXAMPLES}


@app.get("/api/report.md")
async def report_md() -> PlainTextResponse:
    md = _LAST.get("markdown")
    if not md:
        return PlainTextResponse("No analysis has been run yet.", status_code=404)
    return PlainTextResponse(md, media_type="text/markdown")


@app.get("/api/similar")
async def similar() -> JSONResponse:
    """Feature 12 — similarity search over the last analysed AOI."""
    vec = _LAST.get("signature")
    if not vec:
        return JSONResponse({"error": "No analysis has been run yet."}, status_code=404)
    return JSONResponse({"matches": similarity.find_similar(vec, exclude_key=_LAST.get("key"), k=4)})


EXAMPLES = [
    {
        "title": "Urban expansion",
        "query": "Show me where urban expansion occurred in Kokapet between 2019 and 2025",
        "icon": "building",
        "note": "West Hyderabad's fastest-growing corridor",
    },
    {
        "title": "Construction near lakes",
        "query": "Find new construction near lakes in Osman Sagar from 2018 to 2025",
        "icon": "droplet",
        "note": "Tests the proximity + trust-and-safety layer",
    },
    {
        "title": "Water-body shrinkage",
        "query": "Which water bodies have significantly reduced around Sambhar between 2019 and 2024?",
        "icon": "waves",
        "note": "India's largest inland salt lake",
    },
    {
        "title": "Vegetation loss",
        "query": "Show vegetation loss around Wayanad over the last 6 years",
        "icon": "tree",
        "note": "Western Ghats forest margin",
    },
    {
        "title": "Planned city growth",
        "query": "What changed in Amaravati between 2018 and 2025?",
        "icon": "sparkles",
        "note": "Greenfield capital construction",
    },
    {
        "title": "Reservoir dynamics",
        "query": "Did water expand or shrink at Nagarjuna Sagar between 2018 and 2024?",
        "icon": "droplet",
        "note": "Major Krishna-basin reservoir",
    },
]


# ---------------------------------------------------------------------------
# Streaming orchestrator
# ---------------------------------------------------------------------------
def _sse(step: str, status: str, message: str, **extra: Any) -> str:
    payload = {"step": step, "status": status, "message": message, "t": round(time.time(), 3)}
    payload.update(extra)
    return json.dumps(payload, default=_json_default) + "\n"


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (v != v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
    except Exception:
        pass
    return str(o)


def _clean_floats(obj: Any) -> Any:
    """Replace NaN/Inf with None so the JSON is strictly valid."""
    import math as _m
    if isinstance(obj, dict):
        return {k: _clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_floats(v) for v in obj]
    if isinstance(obj, float):
        if _m.isnan(obj) or _m.isinf(obj):
            return None
    return obj


async def _run_pipeline(req: AnalyzeRequest) -> AsyncIterator[str]:
    t_start = time.time()
    loop = asyncio.get_running_loop()

    try:
        # ---------------- Step 1: interpret ---------------------------------
        yield _sse("interpret", "running", "Reading your question\u2026")
        plan = nlp.interpret(
            req.query,
            place_override=req.place,
            start_override=req.start_year,
            end_override=req.end_year,
        )
        yield _sse(
            "interpret", "done",
            f"Understood: {plan.mode_label} \u2014 {plan.start_year} to {plan.end_year}",
            plan=plan.to_dict(),
        )
        await asyncio.sleep(0)

        # ---------------- Step 2: locate ------------------------------------
        yield _sse("locate", "running", "Locating the area on Earth\u2026")
        if req.bbox and len(req.bbox) == 4:
            bbox = tuple(float(v) for v in req.bbox)  # type: ignore[assignment]
            place_label = plan.place or "Selected area"
            place_info = {"label": place_label, "bbox": list(bbox), "source": "user-drawn"}
        else:
            if not plan.place:
                yield _sse("locate", "error",
                           "I could not identify a place in that question. Add a location, "
                           "for example \u201cnear Hyderabad\u201d, or pick an example query.")
                return
            place = await geocode.geocode(plan.place)
            if place is None:
                yield _sse("locate", "error", f"Could not find \u201c{plan.place}\u201d on the map. Try a nearby larger town or district.")
                return
            bbox = place.bbox
            place_label = place.label
            place_info = place.to_dict()

            # --- Country-scale reduction -----------------------------------
            # geocode() clamps any AOI to MAX_AOI_DEG around its CENTROID. For
            # a country that silently relocates the analysis: Nepal's centroid
            # is in the Annapurna massif at ~2200-4100 m, so a flood query was
            # answered from Himalayan snow/glacier instead of the Terai
            # floodplain where Nepal actually floods.
            #
            # Reduce to a representative study window that is inside the real
            # national boundary, biased to low-lying terrain for water queries.
            # Never fatal: any failure leaves the clamped bbox untouched.
            try:
                _study = await aoi.resolve_study_area(
                    plan.place, bbox, place.lon, place.lat,
                    prefer="water" if plan.index == "MNDWI" else "any",
                )
                if _study.reduced:
                    bbox = _study.bbox
                    place_info = dict(place_info)
                    place_info["bbox"] = [round(v, 6) for v in bbox]
                    place_info["study_area"] = _study.to_dict()
                    _study_notes = list(_study.notes)
                else:
                    _study_notes = []
            except Exception as exc:  # noqa: BLE001
                _study_notes = []
                _study = None
                print(f"[aoi] study-area reduction failed: {exc}")
                # If the AOI really is country-scale and we could not reduce
                # it to something processable, say so plainly instead of
                # crashing or silently analysing the wrong terrain.
                #
                # NOTE: `bbox` here is already clamped to MAX_AOI_DEG, so it
                # never looks country-scale. Test the TRUE extent of the named
                # place instead — that is what we failed to reduce.
                try:
                    _true = None
                    try:
                        _b = await aoi.fetch_boundary(plan.place)
                        _true = (_b or {}).get("bbox")
                    except Exception:
                        _true = None
                    if _true is not None and aoi.is_country_scale(_true):
                        yield _sse(
                            "locate", "error",
                            "Country-scale analysis is too large. Please select a "
                            "province, district, municipality, or smaller study area.",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                        return
                except Exception:
                    pass

        # --- Large-area protection -----------------------------------------
        # Country-scale queries ("Nepal", "Bangladesh") resolve to AOIs two
        # orders of magnitude larger than a tahsil. The grid stays bounded, but
        # coarse grids over huge areas produce pixels so large that small water
        # bodies fall below the minimum mapping unit. Cap the requested detail
        # for very large AOIs so the run stays inside memory AND stays
        # scientifically meaningful, and tell the user what was adjusted.
        _span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        _requested_grid = req.grid or settings.GRID_SIZE
        _grid_px = _requested_grid
        _aoi_notes: list[str] = list(_study_notes)
        if _span >= 0.40:
            _grid_px = min(_requested_grid, 420)
            _aoi_notes.append(
                "This is a very large study area, so it is analysed at a coarser ground "
                "sampling. Zoom into a district or river reach for finer detail."
            )
        try:
            grid = raster.build_grid(bbox, _grid_px)
        except Exception as exc:  # noqa: BLE001
            yield _sse("locate", "error",
                       f"Could not build a valid analysis grid for that area: {exc}")
            return
        km_w = (bbox[2] - bbox[0]) * 111.32
        km_h = (bbox[3] - bbox[1]) * 110.57
        yield _sse(
            "locate", "done",
            f"{place_label} \u2014 about {km_w:.0f} \u00d7 {km_h:.0f} km",
            place=place_info, grid=grid.to_dict(),
            area_km2=round(km_w * km_h, 1),
            notes=_aoi_notes,
        )

        # ---------------- Step 3: search catalog ------------------------------
        yield _sse("search", "running", "Searching the Sentinel-2 archive\u2026")
        try:
            features = await catalog.search_scenes(
                bbox, plan.start_year, plan.end_year,
                max_cloud=req.max_cloud or settings.MAX_CLOUD_COVER,
            )
            if not features:
                features = await catalog.search_scenes(
                    bbox, plan.start_year, plan.end_year, max_cloud=45)
        except Exception as exc:  # noqa: BLE001
            yield _sse("search", "error",
                       "Could not reach the Sentinel-2 catalog. Check your internet "
                       f"connection and try again. ({type(exc).__name__})",
                       detail=str(exc)[:400])
            return
        if not features:
            yield _sse("search", "error",
                       f"No suitable Sentinel-2 imagery found for the selected "
                       f"{place_label} study area and period "
                       f"({plan.start_year}\u2013{plan.end_year}). "
                       f"Try a wider date range or a higher cloud limit.")
            return

        try:
            sel = catalog.select_epochs(features, plan.start_year, plan.end_year)
        except Exception as exc:  # noqa: BLE001
            yield _sse("search", "error",
                       f"Could not select comparable dates from the archive: {exc}",
                       detail=traceback.format_exc(limit=3))
            return
        if len(sel.scenes) < 2:
            yield _sse("search", "error",
                       f"No suitable Sentinel-2 imagery found for the selected {place_label} "
                       f"study area and period: only {len(sel.scenes)} cloud-free date was "
                       f"available and at least two are needed to compare. "
                       f"Try a wider date range or a higher cloud limit.")
            return

        # Bound the number of epochs so a long window over a big AOI cannot
        # allocate an unbounded number of rasters.
        if len(sel.scenes) > MAX_EPOCHS:
            keep = [sel.scenes[0]] + sel.scenes[1:-1][: MAX_EPOCHS - 2] + [sel.scenes[-1]]
            sel.notes.append(
                f"{len(sel.scenes)} usable dates were found; the {MAX_EPOCHS} most "
                f"representative were kept (endpoints always retained) to bound processing."
            )
            sel.scenes = keep

        yield _sse(
            "search", "done",
            f"{len(sel.scenes)} clear epochs selected from {sel.total_candidates} candidate scenes",
            scenes=[s.to_dict() for s in sel.scenes],
            notes=sel.notes, rejected=sel.rejected,
            total_candidates=sel.total_candidates,
        )

        # ---------------- Step 4: load imagery --------------------------------
        yield _sse("load", "running", f"Streaming {len(sel.scenes)} satellite images onto a common grid\u2026")

        def _load_all():
            # Load sequentially and tolerate individual epoch failures, so one
            # bad scene cannot abort an otherwise valid multi-temporal stack.
            out = []
            errs = []
            for s in sel.scenes:
                try:
                    out.append(raster.load_epoch(s, grid))
                except Exception as exc:  # noqa: BLE001
                    errs.append(f"{s.date}: {type(exc).__name__}: {exc}")
            return out, errs

        epochs, load_errors = await loop.run_in_executor(_POOL, _load_all)

        if len(epochs) < 2:
            yield _sse("load", "error",
                       f"No suitable Sentinel-2 imagery found for the selected {place_label} "
                       f"study area and period: only {len(epochs)} date could be read. "
                       f"Try a wider date range or a higher cloud limit.",
                       detail="; ".join(load_errors)[:400])
            return

        usable = [e for e in epochs if e.valid_fraction >= 0.25]
        dropped = [e for e in epochs if e.valid_fraction < 0.25]
        if len(usable) < 2:
            usable = sorted(epochs, key=lambda e: -e.valid_fraction)[:2]
            dropped = [e for e in epochs if e not in usable]

        yield _sse(
            "load", "done",
            f"{len(usable)} images loaded and co-registered on one {grid.width}\u00d7{grid.height} grid",
            epochs=[{"date": e.date, "year": e.year,
                     "valid_fraction": round(e.valid_fraction, 3),
                     "cloud_cover": round(e.cloud_cover, 2)} for e in usable],
            dropped=[{"date": e.date, "reason": f"only {e.valid_fraction * 100:.0f}% usable pixels"}
                     for e in dropped],
        )

        # ---------------- Step 5: quality + change ----------------------------
        yield _sse("quality", "running", "Checking clouds, shadows and image alignment\u2026")
        await asyncio.sleep(0)

        def _detect():
            return changedet.detect(
                usable, grid,
                index=plan.index, direction=plan.direction, mode=plan.mode,
            )

        try:
            result = await loop.run_in_executor(_POOL, _detect)
        except ValueError as exc:
            # Raised when the epochs have no cloud-free overlap at all.
            yield _sse("quality", "error",
                       f"The selected dates for {place_label} do not have enough cloud-free "
                       f"overlap to compare ({exc}). Try a wider date range or a higher "
                       f"cloud limit.")
            return
        except Exception as exc:  # noqa: BLE001
            yield _sse("detect", "error",
                       f"Change analysis failed for this area: {type(exc).__name__}: {exc}",
                       detail=traceback.format_exc(limit=3))
            return

        yield _sse(
            "quality", "done",
            f"Alignment {result.quality.get('registration_shift_px', 0):.1f} px \u00b7 "
            f"{result.quality.get('analysis_valid_fraction', 0) * 100:.0f}% of the area usable",
            quality=_clean_floats(result.quality),
        )

        yield _sse("detect", "running", f"Comparing {plan.index} across {len(usable)} dates\u2026")
        await asyncio.sleep(0)
        yield _sse(
            "detect", "done",
            (f"{len(result.clusters)} change cluster(s) \u00b7 {result.stats['changed_area_ha']:,.1f} ha"
             if result.clusters else "No significant change passed the filters"),
            stats=_clean_floats(result.stats),
        )

        yield _sse("filter", "running", "Removing false changes from cloud, shadow and season\u2026")
        await asyncio.sleep(0)
        rej = result.quality.get("noise_rejection_rate")
        yield _sse(
            "filter", "done",
            (f"{rej * 100:.0f}% of raw candidates rejected as noise"
             if rej else "False-change filter applied"),
            evidence=result.evidence, caveats=result.caveats,
        )

        # ---------------- Step 6: render --------------------------------------
        yield _sse("render", "running", "Rendering before / after / change views\u2026")

        def _render():
            first, last = usable[0], usable[-1]
            return {
                "before": render.render_rgb(first),
                "after": render.render_rgb(last),
                "overlay": render.render_change_overlay(last, result.change_mask, direction=plan.direction),
                "mask": render.render_mask(result.change_mask, direction=plan.direction),
                "heatmap": render.render_heatmap(
                    result.magnitude, first.valid & last.valid, direction=plan.direction),
                "scl_before": render.render_scl(first),
                "scl_after": render.render_scl(last),
                "thumbs": [{"date": e.date, "year": e.year, "src": render.render_thumb(e)} for e in usable],
            }

        try:
            images = await loop.run_in_executor(_POOL, _render)
        except Exception as exc:  # noqa: BLE001
            yield _sse("render", "error",
                       f"Could not render the imagery views: {type(exc).__name__}: {exc}",
                       detail=traceback.format_exc(limit=3))
            return
        yield _sse("render", "done", "Views ready", images=images)

        # ---------------- Step 7: explain -------------------------------------
        yield _sse("explain", "running", "Writing the evidence report\u2026")
        narrative = explain.build_narrative(plan, result, place_label)
        md = explain.markdown_report(
            plan, result, narrative, place_label, [s.to_dict() for s in sel.scenes]
        )

        # Similarity signature (Feature 12).
        def _sig():
            return similarity.build_signature(usable, result.change_mask, result.clusters, result.stats)

        vec = await loop.run_in_executor(_POOL, _sig)
        key = f"{place_label}|{plan.start_year}-{plan.end_year}|{plan.mode}"
        similarity.remember(similarity.Signature(
            key=key, label=place_label,
            lon=(bbox[0] + bbox[2]) / 2, lat=(bbox[1] + bbox[3]) / 2,
            period=f"{plan.start_year}\u2013{plan.end_year}", mode=plan.mode,
            vector=vec, headline=narrative["headline"],
            area_ha=result.stats["changed_area_ha"],
        ))
        matches = similarity.find_similar(vec, exclude_key=key, k=4)

        _LAST.clear()
        _LAST.update({"markdown": md, "signature": vec, "key": key})

        elapsed = time.time() - t_start
        yield _sse(
            "explain", "done", "Analysis complete",
            narrative=_clean_floats(narrative),
            clusters=[c.to_dict() for c in result.clusters[:20]],
            timeline=[t.to_dict() for t in result.timeline],
            similar=matches,
            confidence=round(result.confidence, 3),
            confidence_band=result.confidence_band,
            elapsed_s=round(elapsed, 2),
        )
        yield _sse("complete", "done", f"Done in {elapsed:.1f}s", elapsed_s=round(elapsed, 2))

    except Exception as exc:  # noqa: BLE001
        yield _sse(
            "error", "error",
            f"Analysis failed: {exc}",
            detail=traceback.format_exc(limit=3),
        )
    finally:
        # Release the large NumPy arrays before the next request.
        #
        # A single analysis holds ~8 epochs x 6 float32 bands on the analysis
        # grid, plus the rendered PNG buffers. Python frees these eventually,
        # but the generator frame can keep them alive long enough for two
        # back-to-back requests to overlap and trip the container OOM killer
        # (observed: exit 137). Dropping the references and forcing a
        # collection here keeps steady-state RSS flat across many analyses.
        for _name in ("epochs", "usable", "dropped", "result", "images", "features"):
            if _name in locals():
                try:
                    del locals()[_name]
                except Exception:
                    pass
        epochs = usable = dropped = result = images = features = None  # noqa: F841
        gc.collect()


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest) -> StreamingResponse:
    return StreamingResponse(
        _run_pipeline(req),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api/* wins)
# ---------------------------------------------------------------------------
import os as _os

_FRONTEND = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "frontend")
if _os.path.isdir(_FRONTEND):
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
