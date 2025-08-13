"""Web UI package for the I‑Guard project.

This package contains the Flask application that serves the
dashboard and API endpoints used to monitor and interact with
the inference pipeline.  It also exposes template and static
assets used by the browser-based user interface.
"""

from .server import create_app  # noqa: F401
