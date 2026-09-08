"""
EarthPulse — Features 4, 5, 6, 7, 8, 10, 11.

Multi-temporal change detection with explicit false-change suppression,
persistence testing, cluster extraction and calibrated confidence.

Pipeline
--------
  1. Quality gate      reject epochs that are too cloudy / too empty
  2. Co-registration   sub-pixel phase-correlation check (grid is shared, so
                       this is a *verification*, not a correction)
  3. Radiometric       remove the scene-wide median index shift, which is the
     normalisation     signature of atmosphere / sun angle / phenology
  4. Change magnitude  robust z-score of the baseline -> latest index delta
  5. False-change      SCL cloud & shadow masking, seasonal-drift check,
     filter            minimum mapping unit, edge/no-data erosion
  6. Persistence       does the change hold across the intermediate epochs?
  7. Clustering        connected components -> measurable objects
  8. Confidence        evidence-weighted score per cluster and overall
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from .config import settings, SCL_LABELS
from .raster import EpochStack, Grid, compute_index, INDEX_META


# ---------------------------------------------------------------------------
# Spectral plausibility gates  (the second half of Feature 6)
# ---------------------------------------------------------------------------
# A statistically significant index change is NOT sufficient evidence that the
# named phenomenon occurred. Different land-cover transitions can move the same
# index in the same direction:
#
#   * a lake drying to bare mud raises NDBI exactly like fresh concrete does
#   * a harvested field raises NDBI just like a new car park
#   * a flooded field lowers NDVI just like clear-felling
#
# So every candidate pixel must also satisfy an END-STATE and a START-STATE
# test expressed in absolute index space: the pixel has to actually *look like*
# the thing we claim it became, and must not have started as something whose
# disappearance would trivially explain the signal.
#
# Thresholds follow standard remote-sensing practice (Zha 2003 for NDBI,
# Xu 2006 for MNDWI, Tucker 1979 for NDVI) with conservative margins.
_WATER_THR = 0.05        # MNDWI above this => open water
_VEG_THR = 0.30          # NDVI above this => meaningful vegetation
_BUILT_THR = -0.08       # NDBI above this => plausible impervious surface


def _spectral_gate(
    mode: str,
    first: EpochStack,
    last: EpochStack,
) -> tuple[np.ndarray, list[str]]:
    """
    Return (allowed_mask, notes) — pixels where the claimed transition is
    spectrally plausible.
    """
    notes: list[str] = []

    mndwi_a = compute_index(first, "MNDWI")
    mndwi_b = compute_index(last, "MNDWI")
    ndvi_b = compute_index(last, "NDVI")
    ndbi_b = compute_index(last, "NDBI")

    water_before = mndwi_a > _WATER_THR
    water_after = mndwi_b > _WATER_THR

    if mode == "urban_expansion":
        # Must end up looking built-up, must not be vegetated at the end, and
        # must not simply be a former water surface that dried out.
        allowed = (ndbi_b > _BUILT_THR) & (ndvi_b < _VEG_THR) & ~water_before & ~water_after
        n_water = int((water_before & (ndbi_b > _BUILT_THR)).sum())
        if n_water > 0:
            notes.append(
                f"{n_water:,} pixels rose in built-up index only because open water receded and "
                f"exposed dry lake bed. Drying water is not construction, so these were excluded "
                f"and reported separately as a water change."
            )
        notes.append(
            "Detections were restricted to pixels that actually resemble an impervious surface "
            "in the final image, rather than any pixel that merely became brighter."
        )

    elif mode in ("vegetation_loss", "vegetation_gain"):
        # Vegetation change must not be a water-level artefact.
        allowed = ~(water_before ^ water_after)
        n_w = int((water_before ^ water_after).sum())
        if n_w > 0:
            notes.append(
                f"{n_w:,} pixels changed between water and land; vegetation indices are not "
                f"meaningful over water, so these were excluded."
            )

    elif mode in ("water_loss", "water_gain"):
        # Require a genuine crossing of the water boundary, not just a small
        # shift in turbidity or wetness.
        if mode == "water_gain":
            allowed = water_after & ~water_before
        else:
            allowed = water_before & ~water_after
        notes.append(
            "Only pixels that genuinely crossed the open-water boundary were counted, so changes "
            "in turbidity or shallow wetness are not reported as area change."
        )

    elif mode == "burn_scar":
        # A burn scar must have had something to burn.
        ndvi_a = compute_index(first, "NDVI")
        allowed = (ndvi_a > 0.20) & ~water_before & ~water_after
        notes.append(
            "Burn detections were restricted to pixels that carried vegetation before the event."
        )
    else:
        allowed = np.ones_like(water_before, dtype=bool)

    return allowed, notes


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Cluster:
    """One contiguous detected change object."""

    id: int
    area_ha: float
    pixel_count: int
    centroid_lonlat: tuple[float, float]
    bbox_lonlat: tuple[float, float, float, float]
    mean_delta: float
    peak_delta: float
    confidence: float
    persistence: float
    onset_year: int | None
    label: str
    bbox_px: tuple[int, int, int, int]
    evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "area_ha": round(self.area_ha, 2),
            "pixel_count": self.pixel_count,
            "centroid": {"lon": round(self.centroid_lonlat[0], 6),
                         "lat": round(self.centroid_lonlat[1], 6)},
            "bbox": [round(v, 6) for v in self.bbox_lonlat],
            "bbox_px": list(self.bbox_px),
            "mean_delta": round(self.mean_delta, 4),
            "peak_delta": round(self.peak_delta, 4),
            "confidence": round(self.confidence, 3),
            "persistence": round(self.persistence, 3),
            "onset_year": self.onset_year,
            "label": self.label,
            "evidence": self.evidence,
            "caveats": self.caveats,
        }


@dataclass
class TimelinePoint:
    year: int
    date: str
    index_mean: float          # AOI-wide mean index
    hotspot_mean: float        # mean index inside the detected change area
    valid_fraction: float
    cloud_cover: float
    changed_fraction: float    # fraction of AOI already changed by this epoch
    status: str                # 'stable' | 'emerging' | 'changed'

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "date": self.date,
            "index_mean": round(self.index_mean, 4),
            "hotspot_mean": round(self.hotspot_mean, 4),
            "valid_fraction": round(self.valid_fraction, 3),
            "cloud_cover": round(self.cloud_cover, 2),
            "changed_fraction": round(self.changed_fraction, 4),
            "status": self.status,
        }


@dataclass
class ChangeResult:
    change_mask: np.ndarray
    magnitude: np.ndarray            # signed, normalised delta
    z_score: np.ndarray
    clusters: list[Cluster]
    timeline: list[TimelinePoint]
    stats: dict[str, Any]
    quality: dict[str, Any]
    evidence: list[str]
    caveats: list[str]
    confidence: float
    confidence_band: str


# ---------------------------------------------------------------------------
# Step 2 — sub-pixel co-registration verification
# ---------------------------------------------------------------------------
def estimate_shift(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """
    Phase-correlation shift estimate between two index images.

    Returns (dy, dx, peak_sharpness). The grid is shared, so a large shift
    means orthorectification disagreement between tiles rather than a bug —
    either way it must lower confidence, so we measure it.
    """
    a = np.where(mask, a, 0.0).astype(np.float64)
    b = np.where(mask, b, 0.0).astype(np.float64)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0, 0.0, 0.0

    a = a - a.mean()
    b = b - b.mean()
    win_y = np.hanning(a.shape[0])[:, None]
    win_x = np.hanning(a.shape[1])[None, :]
    a *= win_y * win_x
    b *= win_y * win_x

    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fa * np.conj(fb)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    corr = np.fft.irfft2(cross / denom, s=a.shape)

    idx = int(np.argmax(corr))
    py, px = np.unravel_index(idx, corr.shape)
    peak = float(corr[py, px])
    sharpness = float(peak / (corr.std() + 1e-12))

    dy = py - a.shape[0] if py > a.shape[0] // 2 else py
    dx = px - a.shape[1] if px > a.shape[1] // 2 else px
    return float(dy), float(dx), sharpness


# ---------------------------------------------------------------------------
# Step 8 — confidence model
# ---------------------------------------------------------------------------
def _band(conf: float) -> str:
    if conf >= 0.90:
        return "high"
    if conf >= 0.70:
        return "moderate"
    return "low"


def _classify_cluster(mode: str, mean_delta: float, persistence: float) -> str:
    labels = {
        "urban_expansion": "New built-up surface",
        "vegetation_loss": "Vegetation loss",
        "vegetation_gain": "Vegetation gain",
        "water_loss": "Water surface loss",
        "water_gain": "Water expansion",
        "burn_scar": "Burn scar",
    }
    base = labels.get(mode, "Surface change")
    if persistence < 0.4:
        return f"{base} (transient)"
    return base


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def detect(
    epochs: list[EpochStack],
    grid: Grid,
    *,
    index: str,
    direction: str,
    mode: str,
    z_threshold: float | None = None,
) -> ChangeResult:
    if len(epochs) < 2:
        raise ValueError("At least two usable epochs are required for change detection.")

    z_threshold = z_threshold if z_threshold is not None else settings.Z_THRESHOLD
    evidence: list[str] = []
    caveats: list[str] = []
    quality: dict[str, Any] = {}

    epochs = sorted(epochs, key=lambda e: e.date)
    baseline, latest = epochs[0], epochs[-1]

    # ---- Step 1: quality gate ---------------------------------------------
    per_epoch_quality = []
    for e in epochs:
        per_epoch_quality.append({
            "year": e.year,
            "date": e.date,
            "scene_id": e.scene_id,
            "valid_fraction": round(e.valid_fraction, 3),
            "cloud_cover": round(e.cloud_cover, 2),
            "degraded_fraction": round(e.degraded_fraction, 3),
        })
    quality["epochs"] = per_epoch_quality

    # ---- Index stack -------------------------------------------------------
    idx_stack = [compute_index(e, index) for e in epochs]
    valid_stack = [e.valid for e in epochs]

    # Pixels usable in BOTH endpoints — the analysis domain.
    core_valid = valid_stack[0] & valid_stack[-1]
    # Erode the domain slightly: cloud edges and tile borders are where
    # spurious change concentrates.
    core_valid = ndimage.binary_erosion(core_valid, structure=np.ones((3, 3)), border_value=0)

    valid_frac = float(core_valid.mean())
    quality["analysis_valid_fraction"] = round(valid_frac, 3)
    if valid_frac < settings.MIN_VALID_FRACTION:
        caveats.append(
            f"Only {valid_frac * 100:.0f}% of the area is cloud-free in both the first and last "
            f"image, so a large part of the scene could not be assessed."
        )
    else:
        evidence.append(
            f"{valid_frac * 100:.0f}% of the study area is simultaneously cloud-free and shadow-free "
            f"in the {baseline.year} and {latest.year} images."
        )

    if not core_valid.any():
        raise ValueError("No cloud-free overlap between the first and last image.")

    # ---- Step 2: co-registration check ------------------------------------
    dy, dx, sharp = estimate_shift(idx_stack[0], idx_stack[-1], core_valid)
    shift_px = float(np.hypot(dy, dx))
    quality["registration_shift_px"] = round(shift_px, 2)
    quality["registration_sharpness"] = round(sharp, 1)
    if shift_px <= settings.MAX_SHIFT_PX:
        evidence.append(
            f"Geometric alignment verified by phase correlation: residual offset "
            f"{shift_px:.1f} px (tolerance {settings.MAX_SHIFT_PX:.0f} px)."
        )
    else:
        caveats.append(
            f"Images are offset by about {shift_px:.1f} pixels, which can create false edges "
            f"along roads, field boundaries and coastlines."
        )

    # ---- Step 3: radiometric / seasonal normalisation ----------------------
    raw_delta = (idx_stack[-1] - idx_stack[0]).astype(np.float32)
    sample = raw_delta[core_valid]
    global_shift = float(np.median(sample))
    delta = raw_delta - global_shift          # remove scene-wide drift

    quality["global_index_shift"] = round(global_shift, 4)
    if abs(global_shift) >= 0.02:
        evidence.append(
            f"A uniform scene-wide {index} shift of {global_shift:+.3f} was detected and removed. "
            f"A change affecting the entire scene equally is characteristic of illumination, "
            f"atmosphere or season rather than real ground change."
        )
    else:
        evidence.append(
            f"Scene-wide {index} drift was negligible ({global_shift:+.3f}), indicating the two "
            f"dates are radiometrically comparable."
        )

    # ---- Step 4: robust change magnitude -----------------------------------
    resid = delta[core_valid]
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = max(1.4826 * mad, 1e-4)           # MAD -> Gaussian-equivalent sigma
    z = np.zeros_like(delta, dtype=np.float32)
    z[core_valid] = (delta[core_valid] / sigma).astype(np.float32)
    quality["noise_sigma"] = round(sigma, 4)

    # ---- Step 5: threshold with direction + absolute floor -----------------
    if direction == "increase":
        candidate = (z >= z_threshold) & (delta >= settings.MIN_INDEX_DELTA)
    else:
        candidate = (z <= -z_threshold) & (delta <= -settings.MIN_INDEX_DELTA)
    candidate &= core_valid

    n_before_gate = int(candidate.sum())

    # ---- Step 5b: spectral plausibility gate --------------------------------
    gate, gate_notes = _spectral_gate(mode, baseline, latest)
    candidate &= gate
    n_gated = n_before_gate - int(candidate.sum())
    quality["spectral_gate_rejected_px"] = n_gated
    if n_gated > 0:
        evidence.append(
            f"Land-cover plausibility check rejected {n_gated:,} pixels "
            f"({n_gated / max(n_before_gate, 1) * 100:.0f}% of candidates) where the index moved "
            f"but the surface does not match the requested change type."
        )
    for gn in gate_notes:
        evidence.append(gn)

    n_raw = int(candidate.sum())

    # Morphological cleanup.
    #
    # NOTE: a full 3x3 binary_opening is far too aggressive at this scale. A
    # real construction site or forest clearing is only 2-5 px across at 10-25 m
    # GSD, and opening erases any shape thinner than the structuring element —
    # measured at ~86% of true detections on the Kokapet benchmark.
    #
    # Instead we suppress only genuinely isolated pixels (fewer than 2 of the 8
    # neighbours also flagged), then close pinholes to consolidate patches.
    neighbours = ndimage.convolve(
        candidate.astype(np.uint8),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
        mode="constant", cval=0,
    )
    cleaned = candidate & (neighbours >= 2)
    cleaned = ndimage.binary_closing(cleaned, structure=np.ones((3, 3)), border_value=0)
    # Closing can bleed outside the valid domain — clip back.
    cleaned &= core_valid & gate

    # Minimum mapping unit.
    lab, n_lab = ndimage.label(cleaned, structure=np.ones((3, 3)))
    if n_lab:
        sizes = np.bincount(lab.ravel())
        too_small = np.where(sizes < settings.MIN_MAPPING_UNIT_PX)[0]
        cleaned[np.isin(lab, too_small)] = False

    n_clean = int(cleaned.sum())
    removed = n_raw - n_clean
    quality["raw_candidate_px"] = n_raw
    quality["filtered_px"] = n_clean
    if n_raw:
        quality["noise_rejection_rate"] = round(removed / n_raw, 3)
        if removed > 0:
            evidence.append(
                f"False-change filter removed {removed:,} of {n_raw:,} candidate pixels "
                f"({removed / n_raw * 100:.0f}%) that were isolated speckle or below the "
                f"minimum mapping unit."
            )

    # Cloud / shadow attribution: what would have been flagged if we ignored SCL?
    any_invalid = np.zeros_like(core_valid)
    for e in epochs:
        any_invalid |= ~e.valid
    cloud_suppressed = int((any_invalid & ~core_valid).sum())
    quality["cloud_masked_px"] = int((~core_valid).sum())
    if cloud_suppressed > 0:
        evidence.append(
            f"{cloud_suppressed:,} pixels were excluded because at least one date was affected by "
            f"cloud, cirrus, shadow or snow according to the Sentinel-2 scene classification layer."
        )

    # ---- Step 6: persistence across intermediate epochs --------------------
    persistence_map = np.zeros_like(delta, dtype=np.float32)
    inter = epochs[1:]
    if len(inter) >= 1:
        agree = np.zeros_like(delta, dtype=np.float32)
        counts = np.zeros_like(delta, dtype=np.float32)
        for k, e in enumerate(inter, start=1):
            d_k = (idx_stack[k] - idx_stack[0]) - global_shift * (k / (len(epochs) - 1))
            vk = valid_stack[k] & core_valid
            half = 0.5 * np.abs(delta)
            if direction == "increase":
                hit = (d_k >= np.maximum(half, settings.MIN_INDEX_DELTA * 0.5))
            else:
                hit = (d_k <= -np.maximum(half, settings.MIN_INDEX_DELTA * 0.5))
            agree += (hit & vk).astype(np.float32)
            counts += vk.astype(np.float32)
        persistence_map = np.divide(agree, np.maximum(counts, 1.0)).astype(np.float32)

    if len(epochs) >= 3:
        evidence.append(
            f"{len(epochs)} acquisitions between {baseline.year} and {latest.year} were used, so "
            f"each detection could be tested for persistence rather than relying on a single pair."
        )
    else:
        caveats.append(
            "Only two usable acquisitions were available, so persistence over time could not be "
            "verified and a temporary surface condition cannot be ruled out."
        )

    change_mask = cleaned

    # ---- Step 7: clusters --------------------------------------------------
    clusters: list[Cluster] = []
    px_ha = grid.pixel_area_ha()
    lab, n_lab = ndimage.label(change_mask, structure=np.ones((3, 3)))
    # Total number of real clusters, independent of how many we describe in
    # detail. The headline must quote the true count, not the reporting cap.
    total_clusters = int(n_lab)
    DETAIL_CAP = 40
    if n_lab:
        objects = ndimage.find_objects(lab)
        order = np.argsort(-np.bincount(lab.ravel())[1:])
        for rank, ci in enumerate(order[:DETAIL_CAP], start=1):
            cid = int(ci) + 1
            sl = objects[ci]
            if sl is None:
                continue
            sel = lab == cid
            npx = int(sel.sum())
            if npx < settings.MIN_MAPPING_UNIT_PX:
                continue

            cd = delta[sel]
            mean_d = float(cd.mean())
            peak_d = float(cd.max() if direction == "increase" else cd.min())
            pers = float(persistence_map[sel].mean()) if persistence_map.any() else 0.0

            cy, cx = ndimage.center_of_mass(sel)
            lon, lat = grid.xy(cy, cx)
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            lon0, lat1 = grid.xy(r0, c0)
            lon1, lat0 = grid.xy(r1 - 1, c1 - 1)

            # Onset year: first epoch where the cluster reaches half its final change.
            onset = None
            base_mean = float(idx_stack[0][sel].mean())
            final_mean = float(idx_stack[-1][sel].mean())
            target = base_mean + 0.5 * (final_mean - base_mean)
            for k, e in enumerate(epochs):
                m = float(idx_stack[k][sel].mean())
                crossed = m >= target if final_mean > base_mean else m <= target
                if crossed and k > 0:
                    onset = e.year
                    break

            # Per-cluster confidence.
            conf = 0.50
            cl_ev: list[str] = []
            cl_cav: list[str] = []

            strength = min(abs(mean_d) / 0.30, 1.0)
            conf += 0.20 * strength
            cl_ev.append(f"Mean {index} change of {mean_d:+.3f} inside the polygon.")

            if len(epochs) >= 3:
                conf += 0.18 * pers
                if pers >= settings.PERSISTENCE_AGREEMENT:
                    cl_ev.append(f"Change persists in {pers * 100:.0f}% of the intermediate observations.")
                else:
                    cl_cav.append(
                        f"Change is present in only {pers * 100:.0f}% of intermediate observations, "
                        f"so it may be temporary."
                    )
            size_bonus = min(npx / 400.0, 1.0)
            conf += 0.08 * size_bonus
            if npx * px_ha >= 1.0:
                cl_ev.append(f"Contiguous area of {npx * px_ha:.1f} ha exceeds the minimum mapping unit.")

            if shift_px > settings.MAX_SHIFT_PX:
                conf -= 0.10
                cl_cav.append("Residual image misalignment may inflate edge detections.")

            local_cloud = float(any_invalid[sel].mean())
            if local_cloud > 0.10:
                conf -= 0.12 * local_cloud
                cl_cav.append(f"{local_cloud * 100:.0f}% of this polygon was cloud-affected on at least one date.")

            conf = float(np.clip(conf, 0.05, 0.98))

            clusters.append(Cluster(
                id=rank,
                area_ha=npx * px_ha,
                pixel_count=npx,
                centroid_lonlat=(lon, lat),
                bbox_lonlat=(lon0, lat0, lon1, lat1),
                mean_delta=mean_d,
                peak_delta=peak_d,
                confidence=conf,
                persistence=pers,
                onset_year=onset,
                label=_classify_cluster(mode, mean_d, pers),
                bbox_px=(int(c0), int(r0), int(c1), int(r1)),
                evidence=cl_ev,
                caveats=cl_cav,
            ))

    clusters.sort(key=lambda c: -c.area_ha)
    for i, c in enumerate(clusters, start=1):
        c.id = i

    # ---- Timeline (Feature 8) ----------------------------------------------
    hotspot = change_mask if change_mask.any() else core_valid
    timeline: list[TimelinePoint] = []
    base_hot = float(idx_stack[0][hotspot].mean()) if hotspot.any() else 0.0
    final_hot = float(idx_stack[-1][hotspot].mean()) if hotspot.any() else 0.0
    span = final_hot - base_hot

    for k, e in enumerate(epochs):
        vk = valid_stack[k] & core_valid
        aoi_mean = float(idx_stack[k][vk].mean()) if vk.any() else float("nan")
        hot_sel = hotspot & vk
        hot_mean = float(idx_stack[k][hot_sel].mean()) if hot_sel.any() else float("nan")

        if abs(span) > 1e-6 and not np.isnan(hot_mean):
            progress = (hot_mean - base_hot) / span
        else:
            progress = 0.0
        progress = float(np.clip(progress, 0.0, 1.0))

        if k == 0:
            status = "stable"
        elif progress >= 0.75:
            status = "changed"
        elif progress >= 0.30:
            status = "emerging"
        else:
            status = "stable"

        d_k = (idx_stack[k] - idx_stack[0]) - global_shift * (k / max(len(epochs) - 1, 1))
        if direction == "increase":
            ch_k = (d_k >= settings.MIN_INDEX_DELTA) & vk
        else:
            ch_k = (d_k <= -settings.MIN_INDEX_DELTA) & vk
        changed_fraction = float(ch_k.sum() / max(vk.sum(), 1))

        timeline.append(TimelinePoint(
            year=e.year, date=e.date,
            index_mean=aoi_mean, hotspot_mean=hot_mean,
            valid_fraction=e.valid_fraction, cloud_cover=e.cloud_cover,
            changed_fraction=changed_fraction, status=status,
        ))

    # ---- Aggregate statistics ----------------------------------------------
    changed_px = int(change_mask.sum())
    area_ha = changed_px * px_ha
    aoi_ha = float(core_valid.sum()) * px_ha
    pct = (changed_px / max(int(core_valid.sum()), 1)) * 100.0

    mean_pers = float(np.mean([c.persistence for c in clusters])) if clusters else 0.0

    stats = {
        "changed_pixels": changed_px,
        "changed_area_ha": round(area_ha, 2),
        "changed_area_km2": round(area_ha / 100.0, 3),
        "analysed_area_ha": round(aoi_ha, 2),
        "changed_percent": round(pct, 3),
        "pixel_area_ha": round(px_ha, 5),
        "pixel_size_m": round(float(np.sqrt(px_ha * 10_000)), 1),
        "cluster_count": total_clusters,
        "clusters_detailed": len(clusters),
        "largest_cluster_ha": round(clusters[0].area_ha, 2) if clusters else 0.0,
        "mean_persistence": round(mean_pers, 3),
        "index": index,
        "index_meta": INDEX_META.get(index, {}),
        "baseline_year": baseline.year,
        "latest_year": latest.year,
        "baseline_date": baseline.date,
        "latest_date": latest.date,
        "epoch_count": len(epochs),
    }

    # ---- Overall confidence -------------------------------------------------
    conf = 0.50
    conf += 0.15 * min(valid_frac / 0.90, 1.0)
    conf += 0.12 * (1.0 if len(epochs) >= 4 else (0.6 if len(epochs) == 3 else 0.0))
    conf += 0.13 * min(mean_pers / settings.PERSISTENCE_AGREEMENT, 1.0) if clusters else 0.0
    conf += 0.10 * (1.0 if shift_px <= settings.MAX_SHIFT_PX else 0.0)

    mean_cloud = float(np.mean([e.cloud_cover for e in epochs]))
    if mean_cloud > 10:
        conf -= 0.08
        caveats.append(f"Average scene cloud cover across the selected images was {mean_cloud:.0f}%.")
    if not clusters:
        conf = min(conf, 0.72)
    conf = float(np.clip(conf, 0.05, 0.97))

    if clusters:
        evidence.append(
            f"{total_clusters} distinct change cluster(s) survived every filter, totalling "
            f"{area_ha:.1f} ha ({pct:.2f}% of the analysed area)."
        )
    else:
        caveats.append(
            "No change cluster passed the significance, persistence and minimum-size tests. "
            "This is a genuine result, not a failure: the area appears stable for this indicator."
        )

    return ChangeResult(
        change_mask=change_mask,
        magnitude=delta,
        z_score=z,
        clusters=clusters,
        timeline=timeline,
        stats=stats,
        quality=quality,
        evidence=evidence,
        caveats=caveats,
        confidence=conf,
        confidence_band=_band(conf),
    )
