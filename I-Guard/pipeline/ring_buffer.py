"""Circular buffer for video frames.

The :class:`RingBuffer` stores a fixed number of the most recent video
frames in memory.  When a new frame arrives and the buffer is full, the
oldest frame is automatically dropped.  This allows us to capture
pre‑event context (e.g. 10 seconds before a trigger) without writing
frames to disk until necessary.

Frames are stored alongside timestamps so that downstream code can
extract precise time ranges.  The buffer does not perform any
compression; you should ensure that you have enough RAM for the
configured duration and resolution.  As a rule of thumb, one second of
720p BGR video at 15 FPS requires roughly 30 MB of memory.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np


class RingBuffer:
    """A simple ring buffer for storing video frames and timestamps.

    Parameters
    ----------
    capacity : int
        Maximum number of frames to keep in the buffer.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: Deque[Tuple[float, np.ndarray]] = deque(maxlen=capacity)

    def push(self, timestamp: float, frame: np.ndarray) -> None:
        """Append a frame with its timestamp to the buffer.

        If the buffer is full, the oldest frame is automatically discarded.
        """
        self._data.append((timestamp, frame))

    def get_clip(self, start_time: float, end_time: float) -> List[np.ndarray]:
        """Retrieve a list of frames between the given timestamps.

        Parameters
        ----------
        start_time : float
            Start time (inclusive) of the desired clip in seconds since epoch.
        end_time : float
            End time (exclusive) of the desired clip in seconds since epoch.

        Returns
        -------
        List[np.ndarray]
            List of frames whose timestamps lie within the specified
            interval.  Frames are returned in chronological order.
        """
        return [frame for ts, frame in self._data if start_time <= ts < end_time]

    def snapshot(self) -> List[np.ndarray]:
        """Return a snapshot of all frames currently stored, without timestamps.
        Useful for saving the entire buffer when an event is triggered.
        """
        return [frame for _, frame in self._data]