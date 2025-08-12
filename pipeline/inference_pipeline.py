"""Inference pipeline orchestrating cameras, detection and verification.

The :class:`InferencePipeline` ties together the camera adapters,
ring buffers, detectors, verifiers and event queues.  For each camera
configured in the system, it spawns a worker thread that captures
frames, maintains a pre/post buffer, runs the Step 1 detector and
enqueues candidate events.  A separate consumer thread pulls events
from the queue and (optionally) runs Step 2 verification before
handing the event off to whatever handler is registered (e.g. the UI or
alerting service).

This pipeline is intentionally simple; more sophisticated scheduling
(e.g. GPU resource pooling or batching) can be added as needed.  Each
camera has its own ring buffer but shares the Step 1 detector and
Step 2 verifier to reuse GPU memory.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..detection import Step1Detector, Step2Verifier, TrackManager
from .camera_adapter import CameraAdapter
from .ring_buffer import RingBuffer
from .event_queue import Event, EventQueue

LOGGER = logging.getLogger(__name__)


class InferencePipeline:
    """Coordinate camera capture, detection and verification.

    Parameters
    ----------
    config : Dict
        Configuration dictionary loaded from `config.yaml`.
    event_queue : EventQueue
        Queue used to push candidate events for downstream processing.
    """

    def __init__(self, config: Dict, event_queue: EventQueue) -> None:
        self.config = config
        self.event_queue = event_queue
        # Initialise Step1 detector
        step1_cfg = config.get("step1", {})
        self.detector = Step1Detector(
            model_path=step1_cfg.get("model_path", "yolov8n.pt"),
            input_size=step1_cfg.get("input_size", 640),
            classes=step1_cfg.get("classes", ["person", "gun", "knife"]),
            confidence_threshold=step1_cfg.get("confidence_threshold", 0.5),
            events_config=step1_cfg.get("events", {}),
        )
        # Initialise Step2 verifier if enabled
        step2_cfg = config.get("step2", {})
        self.step2: Optional[Step2Verifier] = None
        if step2_cfg.get("enabled"):
            self.step2 = Step2Verifier(
                model_path=step2_cfg.get("model_path", ""),
                threshold=step2_cfg.get("verification_threshold", 0.7),
            )
        # Tracking manager (optional)
        self.tracker = TrackManager()
        # Create ring buffers per camera
        self.ring_buffers: Dict[str, RingBuffer] = {}
        # Camera configurations
        self.cameras = config.get("cameras", [])
        # Pre/post buffer seconds
        self.pre_sec = step1_cfg.get("pre_buffer_sec", 10)
        self.post_sec = step1_cfg.get("post_buffer_sec", 10)
        # Map camera ID to adapter instance
        self.adapters: Dict[str, CameraAdapter] = {}
        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False

        # Event history and lock
        #
        # The UI queries this list to display recent events.  Each entry is
        # a dict with keys like `id`, `camera_id`, `timestamp`, `flags`,
        # `score` and `clip_file`.  Access to the list is protected by
        # a lock because events are appended from a background thread while
        # the web server may iterate over it.  The history is never
        # truncated in this skeleton; you may wish to add pruning logic
        # based on age or maximum length.
        self.events_history: List[Dict[str, Any]] = []
        self.events_lock = threading.Lock()

    def start(self) -> None:
        """Start all camera and event processing threads."""
        self.running = True
        # Start camera workers
        for cam_cfg in self.cameras:
            cam_id = cam_cfg.get("id") or f"cam_{len(self.adapters)}"
            source = cam_cfg.get("source", 0)
            fps = cam_cfg.get("fps")
            adapter = CameraAdapter(source=source, fps=fps)
            self.adapters[cam_id] = adapter
            # Determine ring buffer length in frames
            buffer_capacity = int((self.pre_sec + self.post_sec) * (fps or 15)) + 1
            self.ring_buffers[cam_id] = RingBuffer(capacity=buffer_capacity)
            thread = threading.Thread(target=self._camera_loop, args=(cam_id,), daemon=True)
            thread.start()
            self.workers.append(thread)
        # Start event consumer thread
        consumer_thread = threading.Thread(target=self._event_consumer, daemon=True)
        consumer_thread.start()
        self.workers.append(consumer_thread)
        LOGGER.info("Inference pipeline started with %d cameras", len(self.cameras))

    def stop(self) -> None:
        """Stop all worker threads and release resources."""
        self.running = False
        for adapter in self.adapters.values():
            adapter.release()
        for thread in self.workers:
            thread.join(timeout=1.0)
        LOGGER.info("Inference pipeline stopped")

    def _camera_loop(self, cam_id: str) -> None:
        """Worker loop for a single camera.

        Continuously reads frames, stores them in the ring buffer,
        performs Step1 detection and enqueues events when necessary.
        """
        adapter = self.adapters[cam_id]
        ring = self.ring_buffers[cam_id]
        frame_counter = 0
        while self.running:
            video_frame = adapter.read()
            if video_frame is None:
                # Sleep a bit before retrying on error.
                time.sleep(0.1)
                continue
            ts = video_frame.timestamp
            frame = video_frame.image
            ring.push(ts, frame)
            frame_counter += 1
            # Run Step1 detection
            detections = self.detector.detect(frame)
            flags = self.detector.analyze_events(detections)
            if any(flags.values()):
                # Candidate event detected; record start time for pre/post clip
                event = Event(
                    camera_id=cam_id,
                    timestamp=ts,
                    frame_id=frame_counter,
                    detections=detections,
                    flags=flags,
                )
                try:
                    self.event_queue.put_nowait(event)
                except Exception:
                    LOGGER.warning("Event queue full; dropping event from %s", cam_id)

    def _event_consumer(self) -> None:
        """Consume events from the queue and perform verification.

        This method is responsible for processing candidate events one by one.
        It extracts the relevant frames from the ring buffer, runs Step 2
        verification if enabled, and then hands the event to whatever
        downstream handler is appropriate.  In this skeleton we simply
        log the event and the verification score.  Replace this logic
        with calls to your alerting/UI system.
        """
        while self.running:
            try:
                event: Event = self.event_queue.get(timeout=1.0)
            except Exception:
                continue
            cam_id = event.camera_id
            ts = event.timestamp
            ring = self.ring_buffers.get(cam_id)
            if ring is None:
                LOGGER.error("No ring buffer found for camera %s", cam_id)
                continue
            # Extract pre/post clip frames
            start_time = ts - self.pre_sec
            end_time = ts + self.post_sec
            clip_frames = ring.get_clip(start_time, end_time)
            # Derive list of labels per frame from detections (naive)
            if self.step2:
                # For demonstration, we simply assume the same detections apply to all frames in the clip.
                # In a real system you would run detection on each frame of the clip or reuse stored detections.
                labels_per_frame = [[d.label for d in event.detections] for _ in clip_frames]
                result = self.step2.verify(labels_per_frame)
                score = result.get("score", 0.0)
                LOGGER.info("Event on %s at %.2f verified with score %.2f", cam_id, ts, score)
            else:
                LOGGER.info("Event on %s at %.2f: flags=%s", cam_id, ts, event.flags)
            # Save clip to disk
            if clip_frames:
                try:
                    clip_filename = self._save_clip(clip_frames, cam_id, ts)
                except Exception as exc:
                    LOGGER.error("Failed to save clip for event on %s: %s", cam_id, exc)
                    clip_filename = None
            else:
                clip_filename = None
            # Record event in history for UI
            event_record: Dict[str, Any] = {
                "id": len(self.events_history),
                "camera_id": cam_id,
                "timestamp": ts,
                "flags": event.flags,
                "score": float(score) if self.step2 else None,
                "clip_file": clip_filename,
            }
            with self.events_lock:
                self.events_history.append(event_record)
            # Mark the task done
            self.event_queue.task_done()

    def _save_clip(self, frames: List[np.ndarray], cam_id: str, timestamp: float) -> str:
        """Write a sequence of frames to disk as an MP4 file.

        The clip is saved into the directory specified by the
        ``storage.clips_dir`` configuration key (default ``clips``).  The
        file name is derived from the camera ID and event timestamp.
        The method returns the relative file name (not the full path).

        Parameters
        ----------
        frames : List[np.ndarray]
            List of frames (BGR images) constituting the clip.
        cam_id : str
            Identifier of the camera from which the frames originated.
        timestamp : float
            Event timestamp in seconds since epoch.

        Returns
        -------
        str
            The relative path to the saved MP4 file.
        """
        import os
        import cv2  # type: ignore

        storage_cfg = self.config.get("storage", {})
        clips_dir = storage_cfg.get("clips_dir", "clips")
        os.makedirs(clips_dir, exist_ok=True)
        ts_int = int(timestamp)
        filename = f"{cam_id}_{ts_int}.mp4"
        path = os.path.join(clips_dir, filename)
        if not frames:
            raise ValueError("Cannot save empty clip")
        # Determine frame size
        height, width = frames[0].shape[:2]
        # Determine FPS from camera config or default
        fps = None
        cam_cfg_list = [c for c in self.cameras if (c.get("id") or f"cam_{self.cameras.index(c)}") == cam_id]
        if cam_cfg_list:
            fps = cam_cfg_list[0].get("fps")
        if not fps:
            fps = 15  # default
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return filename