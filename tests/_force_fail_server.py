"""Fault-injection harness: forces country-scale reduction to fail so the
required 'Country-scale analysis is too large' guard can be verified."""
import sys

sys.path.insert(0, "/home/user/earthpulse/backend")

from app import aoi  # noqa: E402


async def broken_resolve(*a, **k):
    raise RuntimeError("simulated DEM + boundary scan outage")


aoi.resolve_study_area = broken_resolve

from app.main import app  # noqa: E402
import uvicorn  # noqa: E402

uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
