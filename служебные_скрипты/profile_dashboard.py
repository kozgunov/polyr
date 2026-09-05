"""Замеряет тяжёлые блоки dashboard без HTTP."""
from __future__ import annotations
import time
from polybot.dashboard import app

c = app.connect()
for name, fn in (
    ("latest_sources", lambda: app.latest_sources(c)),
    ("latency_metrics", lambda: app.latency_metrics(c)),
    ("paper_overview", lambda: app.paper_overview(c)),
    ("dataset_overview", lambda: app.dataset_overview(c)),
    ("model_overview", lambda: app.model_overview(c)),
    ("live_overview", lambda: app.live_overview(c, 300)),
    ("cached_model_analytics", app.cached_model_analytics),
):
    started = time.perf_counter(); fn(); print(name, round(time.perf_counter() - started, 3), flush=True)
c.close()
