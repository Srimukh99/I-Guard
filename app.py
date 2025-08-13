"""Entry point for ThreatDetect.

This script reads the configuration file, initialises the inference
pipeline and web UI, and starts all required threads.  Press Ctrl‑C
to stop the application.  For production deployments consider
wrapping this script with a process supervisor (e.g. systemd or
supervisor) to ensure automatic restart on failure.
"""

import argparse
import logging
import signal
import threading
import time
from typing import Any, Dict

import yaml  # type: ignore

# Use absolute imports so the application can be started with `python app.py`.
from pipeline.inference_pipeline import InferencePipeline
from pipeline.event_queue import EventQueue
from ui.server import create_app


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="ThreatDetect entry point")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Create event queue
    event_queue = EventQueue(maxsize=config.get("advanced", {}).get("queue_maxsize", 64))

    # Initialise and start the inference pipeline
    pipeline = InferencePipeline(config=config, event_queue=event_queue)
    pipeline.start()

    # Create Flask app for the UI
    flask_app = create_app(event_queue, pipeline)
    host = config.get("server", {}).get("host", "127.0.0.1")
    port = config.get("server", {}).get("port", 5000)

    # Run Flask in a separate thread so that Ctrl‑C stops both
    def _run_flask() -> None:
        flask_app.run(host=host, port=port, threaded=True, use_reloader=False)

    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()
    logger.info("UI available at http://%s:%s", host, port)

    # Handle Ctrl‑C gracefully
    stop_event = threading.Event()

    def _signal_handler(signum: int, frame: Any) -> None:
        logger.info("Received signal %s; shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Wait until stopped
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        pipeline.stop()
        logger.info("ThreatDetect stopped")


if __name__ == "__main__":
    main()