"""Pipeline package for the I‑Guard project.

The pipeline ties together the various components of the system
including camera adapters, ring buffers, event queues and the
inference coordinator.  It provides the entry point for the
``InferencePipeline`` class and related helpers used by
``app.py`` and the web UI.
"""

from .inference_pipeline import InferencePipeline  # noqa: F401
from .event_queue import EventQueue  # noqa: F401
from .deepstream_adapter import DeepStreamAdapter  # noqa: F401
