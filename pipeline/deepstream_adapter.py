"""DeepStream/GStreamer camera adapter abstraction.

This module defines :class:`DeepStreamAdapter`, a wrapper around
``cv2.VideoCapture`` that constructs a GStreamer pipeline for reading
frames from hardware decoders (NVDEC/NVMM).  It is intended as a
drop‑in replacement for :class:`CameraAdapter` when running on
Nvidia Jetson devices or other hosts with DeepStream installed.  While
this implementation still converts frames to CPU memory via OpenCV,
it provides a convenient place to integrate true zero‑copy retrieval
using the DeepStream Python bindings or custom CUDA kernels.

Usage
-----

Configure your cameras in ``config.yaml`` with

```
advanced:
  use_deepstream: true
```

and ensure that the ``source`` string is a valid GStreamer pipeline.
For example:

```
cameras:
  - id: cam0
    source: "rtspsrc location=rtsp://... ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM), format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

You can customise the pipeline to use NVDEC, nvvidconv and other
DeepStream elements.  See the Nvidia documentation for details.

Note
----
True zero‑copy capture is not implemented here.  To achieve
zero‑copy, you would need to map the ``NvBufSurface`` memory to
CUDA tensors directly.  This adapter serves as a placeholder to
demonstrate how you might instantiate a GStreamer source with
cv2.VideoCapture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2  # type: ignore

from .camera_adapter import VideoFrame

LOGGER = logging.getLogger(__name__)


class DeepStreamAdapter:
    """Wrap a GStreamer pipeline for camera input.

    Parameters
    ----------
    source : str
        A GStreamer pipeline string (e.g. containing ``nvv4l2decoder``
        and other DeepStream elements).
    fps : int, optional
        Target frames per second.  If set, the adapter will sleep
        between frames to throttle capture.
    """

    def __init__(self, source: str, fps: Optional[int] = None) -> None:
        self.source = source
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_period = 1.0 / float(fps) if fps else None
        self._last_frame_time: float = 0.0
        self._open()

    def _open(self) -> None:
        # When using DeepStream, we expect ``source`` to be a full
        # GStreamer pipeline.  We pass it directly to cv2.VideoCapture
        # with CAP_GSTREAMER.  If the pipeline fails, an exception is
        # raised.  You must ensure GStreamer and the necessary plugins
        # are installed (e.g. gst-plugins-good, nvvidconv, nvv4l2decoder).
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        if not self.cap or not self.cap.isOpened():
            LOGGER.error("Failed to open DeepStream source: %s", self.source)
            raise RuntimeError(f"Cannot open DeepStream source {self.source}")
        LOGGER.info("Opened DeepStream source %s", self.source)

    def read(self) -> Optional[VideoFrame]:
        """Read a single frame from the GStreamer pipeline.

        Returns
        -------
        Optional[VideoFrame]
            A :class:`VideoFrame` containing timestamp and image, or
            ``None`` if capture fails.
        """
        if not self.cap:
            return None
        if self._frame_period is not None:
            # Sleep to enforce target FPS
            elapsed = time.time() - self._last_frame_time
            delay = self._frame_period - elapsed
            if delay > 0:
                time.sleep(delay)
        ok, frame = self.cap.read()
        self._last_frame_time = time.time()
        if not ok:
            LOGGER.warning("Failed to read frame from DeepStream source %s", self.source)
            return None
        return VideoFrame(timestamp=self._last_frame_time, image=frame)

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
            LOGGER.info("Released DeepStream source %s", self.source)
