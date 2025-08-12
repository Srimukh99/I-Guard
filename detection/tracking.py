"""Simple tracking and re‑identification utilities.

For the full ThreatDetect system it is useful to maintain persistent IDs
for objects across frames.  This file defines a minimal :class:`TrackManager`
class that assigns unique IDs to detections using a simple nearest
neighbour matching on bounding box centres.  It is not a replacement
for sophisticated trackers like ByteTrack or DeepSORT, but can be used
as a drop‑in placeholder until such models are integrated.

If you need more advanced tracking, consider using the `deep_sort_realtime`
package (BSD‑licensed) or implementing your own Kalman filter + Hungarian
assignment.  The `TrackManager` API here is intentionally minimal to
encourage experimentation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .frame_detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    # You can store additional attributes such as last update time.


class TrackManager:
    """Assigns persistent IDs to detections across frames.

    This simplistic tracker matches detections frame‑to‑frame based on
    minimum Euclidean distance between bounding box centres.  If a
    detection is not matched to an existing track, a new track is
    created.  Tracks that have not been updated for `max_age` frames
    are purged.

    Parameters
    ----------
    max_age : int
        Maximum number of frames a track can go without an update before
        being removed.
    iou_threshold : float
        Intersection‑over‑union threshold for matching.  Currently unused
        in this naive implementation.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self.tracks: Dict[int, Tuple[Tuple[int, int, int, int], int]] = {}

    def update_tracks(self, detections: List[Detection]) -> List[Tuple[int, Detection]]:
        """Update tracks with the latest detections.

        Parameters
        ----------
        detections : List[Detection]
            The detections in the current frame.

        Returns
        -------
        List[Tuple[int, Detection]]
            A list of `(track_id, detection)` pairs.
        """
        # Compute centres of current detections.
        centres = [((d.bbox[0] + d.bbox[2] / 2.0), (d.bbox[1] + d.bbox[3] / 2.0)) for d in detections]
        matched = set()
        results: List[Tuple[int, Detection]] = []

        # Attempt to match each detection to an existing track.
        for tid, (bbox, age) in list(self.tracks.items()):
            # Increase age and remove if too old.
            new_age = age + 1
            if new_age > self.max_age:
                del self.tracks[tid]
                continue
            self.tracks[tid] = (bbox, new_age)

        for idx, d in enumerate(detections):
            cx, cy = centres[idx]
            best_tid = None
            best_dist = float("inf")
            for tid, (bbox, age) in self.tracks.items():
                tx, ty = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
                dist = (cx - tx) ** 2 + (cy - ty) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            # If we found a track, update it.
            if best_tid is not None and best_dist < (50 ** 2):  # threshold of 50 pixels radius
                # Update track position and age reset.
                self.tracks[best_tid] = (d.bbox, 0)
                results.append((best_tid, d))
            else:
                # Create new track.
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = (d.bbox, 0)
                results.append((tid, d))
        return results