"""
EarthPulse — global configuration and GDAL tuning.

Importing this module configures the GDAL/rasterio environment for fast,
anonymous, range-request reads of Cloud-Optimized GeoTIFFs on S3.
It MUST be imported before rasterio anywhere in the app.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# GDAL / COG tuning. Set before rasterio import.
# ---------------------------------------------------------------------------
_GDAL_ENV = {
    # Do not list the whole S3 "directory" when opening a file — huge speedup.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # Sentinel-2 COGs on AWS are in a public requester-pays-free bucket.
    "AWS_NO_SIGN_REQUEST": "YES",
    "AWS_REGION": "us-west-2",
    # HTTP/2 multiplexing + caches make windowed reads much faster.
    #
    # MEMORY BUDGET: these caches are per-process and are NOT freed between
    # requests. The original 512 MB VSI + 512 MB block cache (1 GB of caches
    # alone) exceeded the container budget on a ~2 GB host and the server was
    # OOM-killed (exit 137) after two consecutive analyses.
    #
    # 64 MB + 128 MB keeps the steady-state footprint well inside the limit
    # with no measurable loss on windowed COG reads, because each read only
    # touches the few overview blocks intersecting the AOI.
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": os.getenv("EP_VSI_CACHE_BYTES", "67108864"),   # 64 MB
    "GDAL_CACHEMAX": os.getenv("EP_GDAL_CACHEMAX_MB", "128"),        # MB
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_NUM_THREADS": "ALL_CPUS",
}
for _k, _v in _GDAL_ENV.items():
    os.environ.setdefault(_k, _v)


class Settings:
    """Runtime settings (env-overridable)."""

    APP_NAME = "EarthPulse"
    VERSION = "0.9.0-mvp"

    # --- External services (all free, no API key required) ----------------
    STAC_ENDPOINT = os.getenv(
        "EP_STAC_ENDPOINT", "https://earth-search.aws.element84.com/v1/search"
    )
    STAC_COLLECTION = os.getenv("EP_STAC_COLLECTION", "sentinel-2-l2a")
    GEOCODER_ENDPOINT = os.getenv(
        "EP_GEOCODER", "https://nominatim.openstreetmap.org/search"
    )
    USER_AGENT = "EarthPulse/0.9 (SIH26227 prototype; geospatial change detection)"

    # --- Analysis grid ----------------------------------------------------
    # All epochs are warped onto ONE common EPSG:4326 grid -> co-registration
    # is guaranteed by construction (no per-pair resampling drift).
    GRID_SIZE = int(os.getenv("EP_GRID_SIZE", "480"))     # px on the long edge
    MAX_AOI_DEG = float(os.getenv("EP_MAX_AOI_DEG", "0.5"))  # ~55 km cap
    MIN_AOI_DEG = 0.012                                    # ~1.3 km floor

    # --- Scene selection --------------------------------------------------
    MAX_CLOUD_COVER = float(os.getenv("EP_MAX_CLOUD", "20"))   # % scene-level
    SEASON_WINDOW_DAYS = 45      # +/- around the anchor day-of-year
    STAC_PAGE_LIMIT = 100

    # --- Change detection -------------------------------------------------
    # Robust z-score threshold on the change magnitude image.
    Z_THRESHOLD = float(os.getenv("EP_Z_THRESHOLD", "2.2"))
    # Absolute floor so tiny index wiggles never count as change.
    MIN_INDEX_DELTA = 0.10
    # Minimum mapping unit: connected components smaller than this are dropped.
    MIN_MAPPING_UNIT_PX = int(os.getenv("EP_MMU_PX", "6"))
    # Fraction of intermediate epochs that must agree for "persistent".
    PERSISTENCE_AGREEMENT = 0.6

    # --- Quality gates ----------------------------------------------------
    MIN_VALID_FRACTION = 0.55    # below this, the epoch is rejected
    MAX_SHIFT_PX = 2.0           # sub-pixel registration tolerance

    # --- Networking -------------------------------------------------------
    HTTP_TIMEOUT = 60.0
    # Each worker holds a decoded band window in memory. 12 was tuned for
    # throughput on a large host; 6 halves peak RSS during the load stage
    # with negligible wall-clock cost since the stage is network-bound.
    READ_WORKERS = int(os.getenv("EP_READ_WORKERS", "6"))

    CACHE_DIR = os.getenv("EP_CACHE_DIR", "/tmp/earthpulse-cache")


settings = Settings()
os.makedirs(settings.CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Sentinel-2 Scene Classification Layer (SCL) semantics.
# Used by the cloud/shadow/false-change filter.
# ---------------------------------------------------------------------------
SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_BARE_SOIL = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MED_PROB = 8
SCL_CLOUD_HIGH_PROB = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW_ICE = 11

# Pixels that must never contribute to a change decision.
SCL_INVALID = {
    SCL_NO_DATA,
    SCL_SATURATED,
    SCL_CLOUD_SHADOW,
    SCL_CLOUD_MED_PROB,
    SCL_CLOUD_HIGH_PROB,
    SCL_THIN_CIRRUS,
    SCL_SNOW_ICE,
}
# Pixels that are valid but flagged as atmospherically degraded.
SCL_DEGRADED = {SCL_DARK_AREA, SCL_UNCLASSIFIED}

SCL_LABELS = {
    0: "no data", 1: "saturated", 2: "dark area", 3: "cloud shadow",
    4: "vegetation", 5: "bare soil", 6: "water", 7: "unclassified",
    8: "cloud (medium prob)", 9: "cloud (high prob)", 10: "thin cirrus",
    11: "snow / ice",
}
