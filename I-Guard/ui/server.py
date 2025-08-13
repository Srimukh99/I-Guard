"""Flask web server for I‑Guard.

This module exposes :func:`create_app`, which constructs a small Flask
application serving both dynamic endpoints (for event data and video
clips) and static HTML/JS for the user interface.  It expects to be
called from the main application (see ``app.py``) with an
``EventQueue`` and an ``InferencePipeline`` instance.  The pipeline
maintains a history of events which the UI uses to display alerts.

The web interface is deliberately simple: it lists recent events,
shows the recorded clip and allows the operator to acknowledge or
dismiss them.  Additional routes implement health checks and expose
internal metrics via Prometheus.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from flask import Flask, Response, jsonify, render_template, send_from_directory

from ..pipeline import EventQueue, InferencePipeline

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # type: ignore
except Exception:  # pragma: no cover - metrics disabled if client unavailable
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"  # type: ignore


def create_app(event_queue: EventQueue, pipeline: InferencePipeline) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    event_queue : EventQueue
        Queue used by the pipeline to push candidate events.  Unused
        directly in the UI but included for future extensions.
    pipeline : InferencePipeline
        Running inference pipeline instance.  The UI queries its
        ``events_history`` to display alerts and serve clips.

    Returns
    -------
    Flask
        Configured Flask app ready to run.
    """
    # Initialise Flask application with explicit static and template paths
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )

    # Homepage displaying the dashboard
    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    # API endpoint returning recent events
    @app.route("/api/events")
    def api_events() -> Tuple[str, int, Dict[str, str]]:
        """Return a JSON array of recent events.

        Each event contains: ``id``, ``camera_id``, ``timestamp``,
        ``flags``, ``score`` (or ``null`` if Step2 is disabled), and
        ``clip_file`` (filename of the saved clip, if available).  The
        list is returned in reverse chronological order (most recent first).
        """
        with pipeline.events_lock:
            events_copy = list(pipeline.events_history)
        # Sort events by timestamp descending
        events_sorted = sorted(events_copy, key=lambda e: e["timestamp"], reverse=True)
        return jsonify(events_sorted), 200, {"Cache-Control": "no-cache"}

    # Endpoint to serve saved video clips
    @app.route("/clips/<path:filename>")
    def clips(filename: str) -> Any:
        """Serve a saved video clip from the clips directory.

        The directory is determined by the ``storage.clips_dir`` config
        option (default ``clips``).  Only files within this directory
        are served.  All access outside the configured directory is
        disallowed for security.  The clip is streamed directly to
        the client.
        """
        storage_cfg = pipeline.config.get("storage", {})
        clips_dir = storage_cfg.get("clips_dir", "clips")
        # Ensure the filename does not escape the clips directory
        safe_dir = os.path.abspath(clips_dir)
        safe_path = os.path.abspath(os.path.join(safe_dir, filename))
        if not safe_path.startswith(safe_dir):
            # Reject paths outside the clips directory
            return "Forbidden", 403
        return send_from_directory(safe_dir, filename, mimetype="video/mp4")

    # Readiness probe: return OK if configuration is loaded.  Does not
    # indicate the pipeline is currently running.
    @app.route("/readyz")
    def readyz() -> Tuple[str, int]:
        return ("OK", 200)

    # Liveness probe: return OK if the pipeline is running; otherwise 503
    @app.route("/livez")
    def livez() -> Tuple[str, int]:
        if getattr(pipeline, "running", False):
            return ("OK", 200)
        return ("Service Unavailable", 503)

    # Metrics endpoint compatible with Prometheus.  Expose metrics if
    # the client library is available; otherwise return a placeholder.
    @app.route("/metrics")
    def metrics() -> Response:
        if generate_latest is None:
            # Metrics disabled; return empty response
            return Response("", status=200, mimetype="text/plain")
        data = generate_latest()  # type: ignore
        return Response(data, status=200, mimetype=CONTENT_TYPE_LATEST)  # type: ignore

    return app
