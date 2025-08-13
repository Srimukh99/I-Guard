"""Event queue for candidate events.

The pipeline uses a simple thread‑safe queue to pass candidate events
from the detection thread to the verification and UI threads.  Each
event stores metadata about what triggered it, which camera it came
from, and a reference to the frame index in the ring buffer.  Downstream
consumers can then fetch the corresponding pre/post clip and perform
further analysis or notify operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, Optional


@dataclass
class Event:
    """Represents a candidate threat event detected in Step 1.

    Attributes
    ----------
    camera_id : str
        Identifier of the camera where the event occurred.
    timestamp : float
        Time (seconds since epoch) when the event was triggered.
    frame_id : int
        Index of the frame in the global inference sequence.
    detections : Any
        Raw detection results (implementation‑specific).  This may be a
        list of detection objects or other data structures.
    flags : Dict[str, bool]
        Event flags returned by :meth:`FrameDetector.analyze_events`.
    """

    camera_id: str
    timestamp: float
    frame_id: int
    detections: Any
    flags: Dict[str, bool]
    extra: Dict[str, Any] = field(default_factory=dict)


class EventQueue(Queue):
    """Thread‑safe queue for `Event` objects.

    Extends the built‑in :class:`queue.Queue` class without modification.
    You may add helper methods here if needed (e.g. size limits or
    priority handling).
    """

    pass