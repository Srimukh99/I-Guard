"""Pipeline package for the I‑Guard project.

The pipeline ties together the various components of the system
including camera adapters, ring buffers, event queues and the
inference coordinator.  It provides the entry point for the
``InferencePipeline`` class and related helpers used by
``app.py`` and the web UI.
"""

try:
	# Import common entry-points for convenience. Wrap in try/except so
	# lightweight scripts can import specific modules (for example the
	# async Stage-2 test) without triggering heavy or package-relative
	# imports that may fail when the package is used as a top-level
	# module in development scripts.
	from .inference_pipeline import InferencePipeline  # noqa: F401
	from .event_queue import EventQueue  # noqa: F401
	from .deepstream_adapter import DeepStreamAdapter  # noqa: F401
except Exception:
	# Defer imports to callers to avoid import-time side-effects.
	pass
