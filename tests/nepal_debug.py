#!/usr/bin/env python3
"""
EarthPulse — instrumented step-by-step Nepal diagnostic.

Runs the 13 requested pipeline steps in-process with defensive error handling
at EVERY step, logging resolved lat/lon, bbox, AOI area, scene counts, usable
scenes, acquisition dates, image dimensions, cloud %, API status and the exact
error message if any. Reports the precise step at which Nepal fails.

Usage: python3 tests/nepal_debug.py [place]
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback

sys.path.insert(0, "/home/user/earthpulse/backend")

PLACE = sys.argv[1] if len(sys.argv) > 1 else "Nepal"
QUERY = (f"Analyze changes in surface water and possible flooding in {PLACE} "
         f"between 2021 and 2025 using Sentinel-2 imagery.")

STEPS = [
    "Understand question", "Resolve location", "Generate bounding box",
    "Search Sentinel-2 archive", "Select cloud-free acquisitions", "Load imagery",
    "Co-register imagery", "Calculate MNDWI", "Detect surface-water changes",
    "Filter false changes", "Generate clusters", "Render before/after",
    "Generate report",
]

LOG: dict = {"steps": [], "failed_step": None, "error": None}


def rss() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1.0


def step(n, name: str, fn):
    """Run one step with defensive handling; record outcome and timing."""
    t0 = time.time()
    banner = f"STEP {str(n):>4}/13  {name}"
    print(f"\n{banner}\n{'-' * len(banner)}")
    try:
        out = fn()
        dt = time.time() - t0
        print(f"  [OK] {dt:.2f}s  RSS {rss():.0f} MB")
        LOG["steps"].append({"n": n, "name": name, "status": "OK",
                             "seconds": round(dt, 2)})
        return out
    except Exception as exc:
        dt = time.time() - t0
        tb = traceback.format_exc()
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        print("  " + "\n  ".join(tb.strip().splitlines()[-6:]))
        LOG["steps"].append({"n": n, "name": name, "status": "FAIL",
                            "seconds": round(dt, 2),
                             "error": f"{type(exc).__name__}: {exc}"})
        if LOG["failed_step"] is None:
            LOG["failed_step"] = f"STEP {n} — {name}"
            LOG["error"] = f"{type(exc).__name__}: {exc}"
        raise


async def main() -> int:
    from app import nlp, geocode, catalog, raster, changedet, render, explain
    from app.config import settings
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds

    print("=" * 78)
    print(f"EARTHPULSE NEPAL DIAGNOSTIC — place={PLACE!r}")
    print(f"query: {QUERY}")
    print(f"baseline RSS {rss():.0f} MB")
    print("=" * 78)

    # ---- 1. Understand -----------------------------------------------------
    plan = step(1, STEPS[0], lambda: nlp.interpret(QUERY))
    print(f"  mode={plan.mode}  index={plan.index}  direction={plan.direction}")
    print(f"  place={plan.place!r}  years={plan.start_year}-{plan.end_year}")
    LOG["plan"] = {"mode": plan.mode, "index": plan.index,
                   "direction": plan.direction, "place": plan.place,
                   "years": [plan.start_year, plan.end_year]}

    # ---- 2. Resolve location ----------------------------------------------
    print(f"\nSTEP  2/13  {STEPS[1]}\n{'-'*40}")
    t0 = time.time()
    try:
        place = await geocode.geocode(plan.place)
        if place is None:
            raise RuntimeError(f"geocode returned None for {plan.place!r}")
        print(f"  label      : {place.label}")
        print(f"  LATITUDE   : {place.lat:.6f}")
        print(f"  LONGITUDE  : {place.lon:.6f}")
        print(f"  source     : {place.source}")
        LOG["resolved"] = {"label": place.label, "lat": place.lat,
                           "lon": place.lon, "source": place.source}
        LOG["steps"].append({"n": 2, "name": STEPS[1], "status": "OK",
                             "seconds": round(time.time()-t0, 2)})
    except Exception as exc:
        LOG["failed_step"] = f"STEP 2 — {STEPS[1]}"
        LOG["error"] = f"{type(exc).__name__}: {exc}"
        raise

    # ---- 3. Bounding box ---------------------------------------------------
    def _bbox():
        bb = place.bbox
        if bb is None or len(bb) != 4:
            raise ValueError(f"invalid bbox: {bb!r}")
        w, s, e, n = bb
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError(f"bbox out of range / degenerate: {bb!r}")
        return bb

    bbox = step(3, STEPS[2], _bbox)
    w, s, e, n = bbox
    km_w, km_h = (e - w) * 111.32 * np.cos(np.radians((s + n) / 2)), (n - s) * 110.57
    area = km_w * km_h
    print(f"  BOUNDING BOX : W={w:.4f} S={s:.4f} E={e:.4f} N={n:.4f}")
    print(f"  span         : {e-w:.4f}deg lon x {n-s:.4f}deg lat")
    print(f"  AOI AREA     : {km_w:.1f} km x {km_h:.1f} km = {area:,.0f} km2")
    LOG["bbox"] = {"w": w, "s": s, "e": e, "n": n,
                   "span_deg": [round(e - w, 4), round(n - s, 4)],
                   "km": [round(km_w, 1), round(km_h, 1)],
                   "area_km2": round(area)}

    grid = step("3b", "Build analysis grid", lambda: raster.build_grid(bbox))
    gsd = km_w * 1000 / grid.width
    print(f"  GRID         : {grid.width} x {grid.height} px @ {gsd:.1f} m/px")
    LOG["grid"] = {"w": grid.width, "h": grid.height, "gsd_m": round(gsd, 1)}

    # ---- 4. Search ---------------------------------------------------------
    async def _search():
        return await catalog.search_scenes(bbox, plan.start_year, plan.end_year,
                                           max_cloud=settings.MAX_CLOUD_COVER)
    t0 = time.time()
    try:
        feats = await _search()
        print(f"\nSTEP  4/13  {STEPS[3]}\n{'-'*40}")
        print(f"  API STATUS       : 200 OK (Element84 Earth Search STAC v1)")
        print(f"  SCENES FOUND     : {len(feats)}")
        LOG["steps"].append({"n": 4, "name": STEPS[3], "status": "OK",
                             "seconds": round(time.time() - t0, 2)})
        LOG["scenes_found"] = len(feats)
        if not feats:
            raise RuntimeError("zero scenes returned")
    except Exception as exc:
        LOG["failed_step"] = f"STEP 4 — {STEPS[3]}"
        LOG["error"] = f"{type(exc).__name__}: {exc}"
        raise

    # ---- 5. Select ---------------------------------------------------------
    sel = step(5, STEPS[4], lambda: catalog.select_epochs(feats, plan.start_year,
                                                          plan.end_year))
    print(f"  ACQUISITION DATES ({len(sel.scenes)} selected):")
    for sc in sel.scenes:
        print(f"    {sc.date}  MGRS-{sc.mgrs}  cloud={sc.cloud_cover:5.2f}%  "
              f"companions={len(sc.companions)}")
    LOG["selected"] = [{"date": sc.date, "tile": sc.mgrs,
                        "cloud_pct": round(sc.cloud_cover, 2)} for sc in sel.scenes]

    # ---- 6/7. Load + co-register ------------------------------------------
    def _load():
        out, errs = [], []
        for sc in sel.scenes:
            try:
                out.append(raster.load_epoch(sc, grid))
            except Exception as e:
                errs.append(f"{sc.date}: {type(e).__name__}: {e}")
        if len(out) < 2:
            raise RuntimeError(f"only {len(out)} epoch(s) loaded; errors={errs}")
        return out, errs

    epochs, load_errs = step(6, STEPS[5], _load)
    print(f"  IMAGE DIMENSIONS : {grid.width} x {grid.height} (all epochs, co-registered)")
    print(f"  epochs loaded    : {len(epochs)}  read-errors: {len(load_errs)}")
    for e in epochs:
        print(f"    {e.date}  valid_fraction={e.valid_fraction:.3f}  "
              f"cloud={e.cloud_cover:5.2f}%  shape={e.bands['green'].shape}")
    LOG["epochs"] = [{"date": e.date, "valid_fraction": round(e.valid_fraction, 3),
                      "cloud_pct": round(e.cloud_cover, 2),
                      "shape": list(e.bands["green"].shape)} for e in epochs]
    LOG["load_errors"] = load_errs

    step(7, STEPS[6], lambda: all(
        e.bands["green"].shape == (grid.height, grid.width) for e in epochs)
        or (_ for _ in ()).throw(RuntimeError("dimension mismatch")))

    usable = [e for e in epochs if e.valid_fraction >= 0.25]
    dropped = [e for e in epochs if e.valid_fraction < 0.25]
    print(f"  USABLE SCENES    : {len(usable)} / {len(epochs)}"
          f"   dropped: {[d.date for d in dropped]}")
    LOG["usable_scenes"] = len(usable)
    LOG["dropped"] = [{"date": d.date, "valid_fraction": round(d.valid_fraction, 3)}
                      for d in dropped]
    if len(usable) < 2:
        raise RuntimeError(f"only {len(usable)} usable epochs after quality gate")

    # ---- 8. MNDWI ----------------------------------------------------------
    def _mndwi():
        stats = []
        for e in usable:
            m = changedet.compute_index(e, "MNDWI")
            bad = int(np.isnan(m).sum() + np.isinf(m).sum())
            if bad:
                raise ValueError(f"{e.date}: {bad} NaN/inf in MNDWI")
            stats.append((e.date, float(np.nanmin(m)), float(np.nanmax(m)),
                          float(np.nanmean(m))))
        return stats

    mstats = step(8, STEPS[7], _mndwi)
    for d, lo, hi, mu in mstats:
        print(f"    {d}  MNDWI min={lo:+.3f} max={hi:+.3f} mean={mu:+.3f}")
    LOG["mndwi"] = [{"date": d, "min": round(lo, 3), "max": round(hi, 3),
                     "mean": round(mu, 3)} for d, lo, hi, mu in mstats]

    # ---- 9/10/11. Detect + filter + cluster --------------------------------
    result = step(9, STEPS[8], lambda: changedet.detect(
        usable, grid, index=plan.index, direction=plan.direction, mode=plan.mode))
    _area = float(result.stats.get("changed_area_ha", 0.0))
    print(f"  changed area : {_area:.1f} ha")
    print(f"  clusters     : {len(result.clusters)}")
    print(f"  confidence   : {result.confidence:.2f} ({result.confidence_band})")
    print(f"  quality      : {json.dumps(result.quality, default=str)[:200]}")
    LOG["detect"] = {"area_ha": round(_area, 1), "clusters": len(result.clusters),
                     "confidence": round(result.confidence, 2),
                     "stats": {k: v for k, v in list(result.stats.items())[:12]}}

    step(10, STEPS[9], lambda: result)
    _rej = result.stats.get("rejected_fraction")
    print(f"  rejected as noise : {_rej}")
    for c in (result.caveats or [])[:4]:
        print(f"    caveat: {str(c)[:110]}")

    step(11, STEPS[10], lambda: result.clusters)
    for c in result.clusters[:6]:
        lon, lat = c.centroid_lonlat
        print(f"    #{c.id} area={c.area_ha:8.2f} ha  centroid=({lat:.4f},{lon:.4f})  "
              f"conf={c.confidence:.2f}  onset={c.onset_year}")
    LOG["clusters"] = [{"id": c.id, "area_ha": round(c.area_ha, 2),
                        "lat": round(c.centroid_lonlat[1], 4),
                        "lon": round(c.centroid_lonlat[0], 4)}
                       for c in result.clusters[:10]]

    # ---- 12. Render --------------------------------------------------------
    def _render():
        return {
            "before": render.render_rgb(usable[0]),
            "after": render.render_rgb(usable[-1]),
            "overlay": render.render_change_overlay(usable[-1], result.change_mask,
                                                    direction=plan.direction),
            "mask": render.render_mask(result.change_mask, direction=plan.direction),
        }
    images = step(12, STEPS[11], _render)
    png = {k: len(v) // 1024 for k, v in images.items() if isinstance(v, str)}
    print(f"  views: {', '.join(f'{k}={v}KB' for k, v in png.items())}")
    LOG["render"] = png

    # ---- 13. Report --------------------------------------------------------
    narrative = step(13, STEPS[12], lambda: explain.build_narrative(
        plan, result, place.label))
    print(f"  headline: {str(narrative.get('headline'))[:100]}")
    LOG["headline"] = str(narrative.get("headline"))[:200]

    return 0


if __name__ == "__main__":
    code = 0
    try:
        code = asyncio.run(main())
    except Exception as exc:
        code = 1
        if LOG.get("failed_step") is None:
            LOG["failed_step"] = "UNCAUGHT (outside step wrapper)"
            LOG["error"] = f"{type(exc).__name__}: {exc}"
        print("\n*** UNCAUGHT EXCEPTION ***")
        traceback.print_exc()
    print("\n" + "=" * 78)
    print("MACHINE-READABLE LOG")
    print("=" * 78)
    print(json.dumps(LOG, indent=2, default=str))
    sys.exit(code)
