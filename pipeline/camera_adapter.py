"""Camera adapter abstraction.

This module defines :class:`CameraAdapter`, a lightweight wrapper around
``cv2.VideoCapture`` that hides the underlying details of camera sources
and ensures frames are delivered at the desired framerate.  It supports
USB/webcam indexes, RTSP/HTTP URLs and local video files.  If
GStreamer is installed, a pipeline can be passed directly as the
``source`` string (e.g. using `rtspsrc` and `nvv4l2decoder`).

Each call to :meth:`read` returns a :class:`VideoFrame` containing the
frame and a timestamp (seconds since epoch).  You can use this
timestamp with :class:`RingBuffer` to store and retrieve clips.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    timestamp: float
    image: Any  # numpy.ndarray


class CameraAdapter:
    """Wraps OpenCV video capture objects.

    Parameters
    ----------
    source : Any
        Source specification.  It can be:

        * an integer index (e.g. 0) for a USB camera,
        * a string containing an RTSP/HTTP URL,
        * a string specifying a GStreamer pipeline.
    fps : int, optional
        Target frames per second.  If set, the adapter will sleep as
        needed to enforce this rate.
    """

    def __init__(self, source: Any, fps: Optional[int] = None) -> None:
        self.source = source
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_period = 1.0 / float(fps) if fps else None
        self._last_frame_time: float = 0.0
        self._open()

    def _open(self) -> None:
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        if not self.cap or not self.cap.isOpened():
            LOGGER.error("Failed to open video source: %s", self.source)
            raise RuntimeError(f"Cannot open video source {self.source}")
        LOGGER.info("Opened video source %s", self.source)

    def read(self) -> Optional[VideoFrame]:
        """Read a single frame from the camera.

        Returns ``None`` if the frame could not be read.  If ``fps`` was
        provided at construction time, the call will block to maintain
        roughly that framerate.
        """
        if not self.cap:
            return None
        if self._frame_period is not None:
            # Sleep to enforce target FPS.
            elapsed = time.time() - self._last_frame_time
            delay = self._frame_period - elapsed
            if delay > 0:
                time.sleep(delay)
        ok, frame = self.cap.read()
        self._last_frame_time = time.time()
        if not ok:
            LOGGER.warning("Failed to read frame from source %s", self.source)
            return None
        return VideoFrame(timestamp=self._last_frame_time, image=frame)

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
            LOGGER.info("Released video source %s", self.source)