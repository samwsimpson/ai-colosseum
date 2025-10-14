# entrypoint.py
import os, sys, traceback

print("ENTRYPOINT: starting import of api.py", flush=True)
try:
    import api  # this is your FastAPI module
    print("ENTRYPOINT: imported api module OK", flush=True)
    app = getattr(api, "app", None)
    if app is None:
        raise RuntimeError("ENTRYPOINT: api.app not found")
except Exception as e:
    print("ENTRYPOINT: FAILED DURING IMPORT", flush=True)
    traceback.print_exc()
    # Exit non-zero so Cloud Run knows it failed, but with full traceback logged.
    sys.exit(1)

# If we get here, run uvicorn *after* confirming import worked
from uvicorn import run
port = int(os.environ.get("PORT", "8080"))
print(f"ENTRYPOINT: launching uvicorn on 0.0.0.0:{port}", flush=True)
run(app, host="0.0.0.0", port=port)
