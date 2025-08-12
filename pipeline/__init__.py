"""Core pipeline components.

This package contains the building blocks for connecting cameras to
detectors and handling event queues.  It does not know anything about
the UI or long‑term storage; those concerns are handled by other
packages.  Each module should be independently testable.

Components include:

* :mod:`ring_buffer` – an in‑memory circular buffer for storing
  pre‑trigger frames.
* :mod:`camera_adapter` – classes for capturing frames from various
  sources (RTSP, USB, file) using OpenCV and GStreamer.
* :mod:`event_queue` – a thread‑safe queue for passing candidate events
  from the detector to downstream consumers.
* :mod:`inference_pipeline` – orchestrates camera capture, detection
  and verification loops on separate threads.
"""

from .ring_buffer import RingBuffer
from .camera_adapter import CameraAdapter, VideoFrame
from .event_queue import Event, EventQueue
from .inference_pipeline import InferencePipeline

__all__ = [
    "RingBuffer",
    "CameraAdapter",
    "VideoFrame",
    "Event",
    "EventQueue",
    "InferencePipeline",
]