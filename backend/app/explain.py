"""
EarthPulse — Features 10, 13, 15: evidence engine, report writer and
"So What?" contextual intelligence.

Trust & Safety (PRD §28) is enforced here, in code, not by convention:

  * `_SAFE_REWRITES` rewrites accusatory phrasing into verification language.
  * Every headline claim is emitted together with its supporting evidence.
  * Confidence and its drivers are always stated.
  * Recommendations are always framed as "verify", never as "violation".
"""
from __future__ import annotations

import re
from typing import Any

from .changedet import ChangeResult
from .nlp import QueryPlan


# ---------------------------------------------------------------------------
# Trust & Safety guard (Rule 5)
# ---------------------------------------------------------------------------
_SAFE_REWRITES = [
    (r"\billegal\b", "unpermitted-looking"),
    (r"\bencroachment\b", "possible boundary change"),
    (r"\bviolation\b", "condition requiring verification"),
    (r"\bcrime\b", "activity"),
    (r"\bculprit\b", "responsible party"),
    (r"\bproves\b", "indicates"),
    (r"\bproof\b", "evidence"),
    (r"\bcertainly\b", "likely"),
    (r"\bdefinitely\b", "likely"),
    (r"\bguarantee[sd]?\b", "indicates"),
]


def sanitise(text: str) -> str:
    """Strip unsupported accusations from any generated sentence."""
    out = text
    for pat, repl in _SAFE_REWRITES:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# Domain context for the "So What?" layer
# ---------------------------------------------------------------------------
_CONTEXT: dict[str, dict[str, Any]] = {
    "urban_expansion": {
        "verb": "built-up expansion",
        "why": (
            "New impervious surface reduces rainwater infiltration and increases "
            "surface runoff, which raises local flood risk and can intensify the "
            "urban heat-island effect."
        ),
        "who": "urban planning and municipal infrastructure teams",
        "action": "Cross-check the detected polygons against approved layout plans and building permits.",
    },
    "vegetation_loss": {
        "verb": "vegetation loss",
        "why": (
            "Loss of vegetation cover reduces carbon uptake and soil stability, and "
            "can increase erosion and local surface temperature."
        ),
        "who": "forest and environmental departments",
        "action": "Verify whether the loss corresponds to a permitted activity, a harvest cycle, or an unplanned clearance.",
    },
    "vegetation_gain": {
        "verb": "vegetation gain",
        "why": (
            "Increases in vegetation cover can indicate successful plantation, "
            "regeneration, or a shift in cropping intensity."
        ),
        "who": "afforestation and agricultural programme teams",
        "action": "Confirm whether the gain reflects a plantation programme or a seasonal cropping cycle.",
    },
    "water_loss": {
        "verb": "reduction in surface water",
        "why": (
            "Shrinking water bodies affect drinking-water supply, groundwater "
            "recharge and aquatic habitat, and may signal drought or abstraction."
        ),
        "who": "irrigation, water resources and environmental authorities",
        "action": "Compare against reservoir gauge records and rainfall data for the same period.",
    },
    "water_gain": {
        "verb": "water expansion",
        "why": (
            "Expanding water extent can indicate flooding, reservoir filling or "
            "waterlogging, all of which affect settlements and cropland downstream."
        ),
        "who": "disaster management and irrigation authorities",
        "action": "Overlay the detected extent with settlement and cropland layers to estimate exposure.",
    },
    "burn_scar": {
        "verb": "burned area",
        "why": (
            "Burn scars indicate loss of biomass and expose soil to erosion, with "
            "elevated landslide and runoff risk in the following monsoon."
        ),
        "who": "forest fire management teams",
        "action": "Prioritise the largest polygons for ground assessment and post-fire soil stabilisation.",
    },
}

_PROXIMITY_NOTE = {
    "water": "Because the question referred to water bodies, changes close to lakes, tanks and rivers deserve particular attention; buffer-zone rules often apply near water.",
    "forest": "Because the question referred to forest areas, detected changes should be checked against notified forest boundaries.",
    "urban": "Because the question referred to urban areas, changes should be checked against the current master plan.",
    "farmland": "Because the question referred to agricultural land, seasonal cropping cycles should be excluded before drawing conclusions.",
}


def _fmt_area(ha: float) -> str:
    if ha >= 100:
        return f"{ha / 100:.2f} km² ({ha:,.0f} hectares)"
    if ha >= 1:
        return f"{ha:,.1f} hectares"
    return f"{ha * 10_000:,.0f} m²"


def _spatial_bias(result: ChangeResult) -> str | None:
    """Describe where the change concentrates, in plain compass language."""
    if not result.clusters:
        return None
    lons = [c.centroid_lonlat[0] for c in result.clusters]
    lats = [c.centroid_lonlat[1] for c in result.clusters]
    weights = [c.area_ha for c in result.clusters]
    total = sum(weights) or 1.0
    clon = sum(l * w for l, w in zip(lons, weights)) / total
    clat = sum(l * w for l, w in zip(lats, weights)) / total

    all_lon = sorted(lons)
    all_lat = sorted(lats)
    mid_lon = (all_lon[0] + all_lon[-1]) / 2
    mid_lat = (all_lat[0] + all_lat[-1]) / 2

    ns = "northern" if clat > mid_lat else "southern"
    ew = "eastern" if clon > mid_lon else "western"
    if len(result.clusters) == 1:
        return None
    return f"{ns} and {ew}"


def build_narrative(plan: QueryPlan, result: ChangeResult, place_label: str) -> dict[str, Any]:
    """Produce the full evidence-first explanation payload."""
    ctx = _CONTEXT.get(plan.mode, _CONTEXT["urban_expansion"])
    s = result.stats
    # Use the TRUE cluster count, not the number we describe in detail.
    n = int(result.stats.get("cluster_count", len(result.clusters)))
    area = s["changed_area_ha"]
    conf_pct = round(result.confidence * 100)

    # ---- Headline ---------------------------------------------------------
    if n == 0:
        headline = (
            f"No significant {ctx['verb']} was detected in {place_label} "
            f"between {s['baseline_year']} and {s['latest_year']}."
        )
    else:
        headline = (
            f"{_fmt_area(area)} of {ctx['verb']} detected in {place_label} "
            f"between {s['baseline_year']} and {s['latest_year']}, across "
            f"{n} distinct cluster{'s' if n != 1 else ''}."
        )

    # ---- Summary paragraph -------------------------------------------------
    parts: list[str] = []
    parts.append(
        f"EarthPulse compared {s['epoch_count']} Sentinel-2 acquisitions of {place_label}, "
        f"from {s['baseline_date']} to {s['latest_date']}, all selected from the same season "
        f"so that vegetation cycles and sun angle stay comparable."
    )
    parts.append(
        f"Change was measured with {s['index']} "
        f"({s.get('index_meta', {}).get('formula', '')}), at a ground sampling of about "
        f"{s['pixel_size_m']:.0f} m per pixel."
    )
    if n:
        pct = s["changed_percent"]
        parts.append(
            f"After filtering, {_fmt_area(area)} — {pct:.2f}% of the {_fmt_area(s['analysed_area_ha'])} "
            f"assessed — met every criterion for real change. The largest single cluster covers "
            f"{_fmt_area(s['largest_cluster_ha'])}."
        )
        bias = _spatial_bias(result)
        if bias:
            parts.append(f"Detected change concentrates in the {bias} part of the study area.")
        onsets = [c.onset_year for c in result.clusters if c.onset_year]
        if onsets:
            first = min(onsets)
            if len(set(onsets)) == 1:
                parts.append(f"The change signature first becomes clear in {first}.")
            else:
                parts.append(
                    f"Onset is staggered: the earliest cluster becomes detectable in {first}, "
                    f"the latest in {max(onsets)}, which is consistent with progressive rather "
                    f"than sudden change."
                )
    else:
        parts.append(
            f"Across the {_fmt_area(s['analysed_area_ha'])} assessed, no group of pixels was both "
            f"statistically significant and large enough to report. For this indicator and period, "
            f"the area appears stable."
        )
    summary = sanitise(" ".join(parts))

    # ---- Confidence explanation --------------------------------------------
    conf_reasons: list[str] = []
    q = result.quality
    vf = q.get("analysis_valid_fraction", 0)
    if vf >= 0.85:
        conf_reasons.append(f"{vf * 100:.0f}% of the area was cloud-free on both key dates.")
    else:
        conf_reasons.append(f"Only {vf * 100:.0f}% of the area was usable on both key dates, which lowers confidence.")
    if s["epoch_count"] >= 3:
        conf_reasons.append(f"{s['epoch_count']} time steps allowed persistence testing.")
    else:
        conf_reasons.append("Only two time steps were available, so persistence could not be tested.")
    shift = q.get("registration_shift_px", 0)
    if shift <= 2.0:
        conf_reasons.append(f"Image alignment was verified to within {shift:.1f} pixels.")
    else:
        conf_reasons.append(f"Residual misalignment of {shift:.1f} pixels adds uncertainty.")
    if s["mean_persistence"] > 0:
        conf_reasons.append(
            f"Detected changes persist across {s['mean_persistence'] * 100:.0f}% of intermediate observations on average."
        )

    conf_text = (
        f"Overall confidence is {conf_pct}% ({result.confidence_band}). " + " ".join(conf_reasons)
    )
    if result.confidence_band == "low":
        conf_text += " Treat this result as an indication requiring verification rather than a finding."

    # ---- "So What?" (Feature 15) --------------------------------------------
    if n:
        so_what = (
            f"Why this matters: {ctx['why']} This is typically relevant to {ctx['who']}. "
        )
        if plan.proximity and plan.proximity in _PROXIMITY_NOTE:
            so_what += _PROXIMITY_NOTE[plan.proximity] + " "
        so_what += f"Suggested next step: {ctx['action']}"
    else:
        so_what = (
            f"A stable result is still useful: it provides a documented, dated baseline for "
            f"{place_label} that future monitoring can be compared against. Relevant to {ctx['who']}."
        )
    so_what = sanitise(so_what)

    # ---- Recommendations ----------------------------------------------------
    recs: list[str] = []
    if n:
        recs.append(sanitise(ctx["action"]))
        recs.append(
            "Prioritise ground or high-resolution verification for the largest and "
            "highest-confidence polygons first."
        )
        if result.confidence_band != "high":
            recs.append(
                "Re-run the analysis with a wider date window, or with imagery from an "
                "adjacent season, to strengthen the evidence."
            )
        low = [c for c in result.clusters if c.confidence < 0.7]
        if low:
            recs.append(
                f"{len(low)} cluster(s) scored below 70% confidence and should not be acted on "
                f"without independent confirmation."
            )
    else:
        recs.append("No action required for this indicator; consider re-running with a longer time window.")
    recs.append(
        "This analysis is derived from 10–20 m satellite imagery and is intended to focus "
        "human attention, not to replace field verification or legal survey."
    )

    # ---- Structured report (Feature 13) --------------------------------------
    report = {
        "title": f"{ctx['verb'].title()} Assessment — {place_label}",
        "location": place_label,
        "period": f"{s['baseline_year']}–{s['latest_year']}",
        "observation_dates": f"{s['baseline_date']} to {s['latest_date']}",
        "major_observation": headline,
        "estimated_affected_area": _fmt_area(area) if n else "None above the reporting threshold",
        "affected_area_ha": area,
        "cluster_count": n,
        "concentration": _spatial_bias(result) or ("single cluster" if n == 1 else "not applicable"),
        "confidence_percent": conf_pct,
        "confidence_band": result.confidence_band,
        "indicator": f"{s['index']} — {s.get('index_meta', {}).get('name', '')}",
        "evidence": [sanitise(e) for e in result.evidence],
        "limitations": [sanitise(c) for c in result.caveats],
        "recommendations": recs,
        "data_source": "Copernicus Sentinel-2 L2A surface reflectance (ESA), via AWS Open Data",
    }

    return {
        "headline": sanitise(headline),
        "summary": summary,
        "confidence_explanation": sanitise(conf_text),
        "so_what": so_what,
        "recommendations": recs,
        "report": report,
    }


def markdown_report(plan: QueryPlan, result: ChangeResult, narrative: dict[str, Any],
                    place_label: str, scenes: list[dict[str, Any]]) -> str:
    """Render a downloadable Markdown report."""
    r = narrative["report"]
    s = result.stats
    lines: list[str] = []
    a = lines.append

    a(f"# {r['title']}")
    a("")
    a(f"**Generated by** EarthPulse — evidence-first satellite change intelligence  ")
    a(f"**Question asked** “{plan.raw_query}”  ")
    a(f"**Location** {r['location']}  ")
    a(f"**Period** {r['period']} ({r['observation_dates']})  ")
    a(f"**Indicator** {r['indicator']}  ")
    a(f"**Data source** {r['data_source']}")
    a("")
    a("---")
    a("")
    a("## 1. Headline")
    a("")
    a(f"> {r['major_observation']}")
    a("")
    a(f"**Confidence: {r['confidence_percent']}% ({r['confidence_band'].upper()})**")
    a("")
    a("## 2. Summary")
    a("")
    a(narrative["summary"])
    a("")
    a("## 3. Measurements")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Affected area | {r['estimated_affected_area']} |")
    a(f"| Share of analysed area | {s['changed_percent']:.2f}% |")
    a(f"| Area assessed | {s['analysed_area_ha']:,.0f} ha |")
    a(f"| Change clusters | {s['cluster_count']} |")
    a(f"| Largest cluster | {s['largest_cluster_ha']:,.1f} ha |")
    a(f"| Ground sampling | ~{s['pixel_size_m']:.0f} m/pixel |")
    a(f"| Acquisitions used | {s['epoch_count']} |")
    a(f"| Mean persistence | {s['mean_persistence'] * 100:.0f}% |")
    a("")
    a("## 4. Imagery used")
    a("")
    a("| Date | Scene ID | Cloud cover | Usable pixels |")
    a("|---|---|---|---|")
    for sc, eq in zip(scenes, result.quality.get("epochs", [])):
        a(f"| {sc['date']} | `{sc['id']}` | {sc['cloud_cover']:.1f}% | {eq['valid_fraction'] * 100:.0f}% |")
    a("")
    a("## 5. Evidence")
    a("")
    for e in r["evidence"]:
        a(f"- {e}")
    a("")
    if r["limitations"]:
        a("## 6. Limitations and uncertainty")
        a("")
        for c in r["limitations"]:
            a(f"- {c}")
        a("")
    a("## 7. Why this matters")
    a("")
    a(narrative["so_what"])
    a("")
    a("## 8. Recommendations")
    a("")
    for rec in r["recommendations"]:
        a(f"- {rec}")
    a("")
    if result.clusters:
        a(f"## 9. Detected clusters (largest {min(len(result.clusters),15)} of {s['cluster_count']})")
        a("")
        a("| # | Area (ha) | Centroid | Mean Δ | Persistence | Onset | Confidence |")
        a("|---|---|---|---|---|---|---|")
        for c in result.clusters[:15]:
            a(f"| {c.id} | {c.area_ha:,.2f} | {c.centroid_lonlat[1]:.4f}, {c.centroid_lonlat[0]:.4f} "
              f"| {c.mean_delta:+.3f} | {c.persistence * 100:.0f}% | {c.onset_year or '—'} "
              f"| {c.confidence * 100:.0f}% |")
        a("")
    a("---")
    a("")
    a("*EarthPulse presents satellite-derived indications. Confidence scores describe the "
      "strength of the observational evidence, not legal or regulatory status. Independent "
      "verification is recommended before any enforcement or investment decision.*")
    return "\n".join(lines)
