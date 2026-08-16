"""Start the local observability dashboard."""

try:
    import uvicorn
    from polybot.runtime import run_sync

    import app_config as settings
except Exception:
    print("error_in_run_dashboard.py", flush=True)
    raise


def main() -> None:
    print(f"DASHBOARD_READY http://{settings.DASHBOARD_HOST}:{settings.DASHBOARD_PORT}", flush=True)
    uvicorn.run("polybot.dashboard.app:app", host=settings.DASHBOARD_HOST, port=settings.DASHBOARD_PORT, reload=False)


if __name__ == "__main__":
    run_sync(__file__, main)
