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
Notes:
- The pipeline intentionally separates capture, detection and verification
    to keep the CPU/GPU workloads isolated. Async Stage-2 verification
    is available to avoid blocking the consumer thread when running heavy
    3D CNN models.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional
import os

import numpy as np

from detection import FrameDetector, ClipVerifier, TrackManager
from .deepstream_pipeline import DeepStreamPipeline
from .camera_adapter import CameraAdapter
from .deepstream_adapter import DeepStreamAdapter
from .ring_buffer import RingBuffer
from .event_queue import Event, EventQueue
from .async_stage2 import AsyncStage2Pipeline
from .async_result_handler import AsyncResultHandler
from .backends import BackendFactory, BaseBackend

# Metrics instrumentation using the Prometheus client.  These metrics
# provide insight into the health and performance of the pipeline when
# exported via the `/metrics` endpoint in the web server.
try:
    from prometheus_client import Gauge, Counter, Histogram
except ImportError:  # pragma: no cover - fallback if not installed
    Gauge = Counter = Histogram = None  # type: ignore

LOGGER = logging.getLogger(__name__)

if Gauge and Counter and Histogram:
    # Current size of the event queue
    event_queue_size_gauge = Gauge(
        "iguard_event_queue_size", "Current number of pending events in the queue"
    )
    # Number of events recorded in history
    event_history_size_gauge = Gauge(
        "iguard_event_history_size", "Total number of events stored in history"
    )
    # Detector latency per frame
    detection_latency_histogram = Histogram(
        "iguard_detection_latency_seconds", "Latency of the step1 detector per frame"
    )
    # Pipeline running state (1 if running, 0 otherwise)
    pipeline_running_gauge = Gauge(
        "iguard_pipeline_running", "Whether the inference pipeline is currently running"
    )
    # Total number of events detected by step1
    events_detected_counter = Counter(
        "iguard_events_detected_total", "Total number of candidate events detected"
    )
else:
    # Provide dummy metrics if the Prometheus client is unavailable
    class DummyMetric:
        def observe(self, *args: Any, **kwargs: Any) -> None:
            pass

        def inc(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set(self, *args: Any, **kwargs: Any) -> None:
            pass

    event_queue_size_gauge = DummyMetric()
    event_history_size_gauge = DummyMetric()
    detection_latency_histogram = DummyMetric()
    pipeline_running_gauge = DummyMetric()
    events_detected_counter = DummyMetric()


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
        
        # Initialize hybrid backend system
        self.backend_factory = BackendFactory()
        self.backend: Optional[BaseBackend] = None
        self.backend_name: Optional[str] = None
        
        # Legacy mode support for backward compatibility
        self.mode = config.get("inference_mode", "python")
        
        # Legacy detector (kept for backward compatibility)
        step1_cfg = config.get("step1", {})
        self.detector: Optional[FrameDetector]
        if self.mode == "python":
            self.detector = FrameDetector(
                model_path=step1_cfg.get("model_path", "yolov8n.pt"),
                input_size=step1_cfg.get("input_size", 640),
                classes=step1_cfg.get("classes", ["person", "gun", "knife"]),
                confidence_threshold=step1_cfg.get("confidence_threshold", 0.5),
                events_config=step1_cfg.get("events", {}),
            )
        else:
            self.detector = None
        # Initialise clip verifier if enabled
        step2_cfg = config.get("step2", {})
        self.step2: Optional[ClipVerifier] = None
        self.async_stage2: Optional[AsyncStage2Pipeline] = None
        self.async_result_handler: Optional[AsyncResultHandler] = None
        
        if step2_cfg.get("enabled"):
            # Check if async processing is enabled
            async_enabled = step2_cfg.get("async_enabled", False)
            
            if async_enabled:
                # Use async Stage-2 pipeline
                self.async_stage2 = AsyncStage2Pipeline(
                    model_path=step2_cfg.get("model_path", ""),
                    threshold=step2_cfg.get("verification_threshold", 0.7),
                    max_workers=step2_cfg.get("max_workers", 2),
                    queue_size=step2_cfg.get("queue_size", 50),
                    result_queue_size=step2_cfg.get("result_queue_size", 100),
                    worker_timeout=step2_cfg.get("worker_timeout_sec", 10.0),
                    model_type=step2_cfg.get("model_type", "tao"),
                    input_size=step2_cfg.get("input_size", [224, 224]),
                    temporal_size=step2_cfg.get("temporal_size", 16),
                )
                LOGGER.info("Using asynchronous Stage-2 verification with %d workers", 
                           step2_cfg.get("max_workers", 2))
            else:
                # Use synchronous ClipVerifier
                self.step2 = ClipVerifier(
                    model_path=step2_cfg.get("model_path", ""),
                    threshold=step2_cfg.get("verification_threshold", 0.7),
                )
                LOGGER.info("Using synchronous Stage-2 verification")
        # Tracking manager (optional)
        self.tracker = TrackManager()
        # Create ring buffers per camera
        self.ring_buffers: Dict[str, RingBuffer] = {}
        # Camera configurations
        self.cameras = config.get("cameras", [])
        # Pre/post buffer seconds
        self.pre_sec = step1_cfg.get("pre_buffer_sec", 10)
        self.post_sec = step1_cfg.get("post_buffer_sec", 10)
        # Frame skipping for Step1 detection.  If set to N, we run detection on
        # every (N+1)th frame.  This can reduce load when multiple cameras are
        # active.  Default is 0 (no skipping).  When dynamic backpressure is
        # enabled, this value serves as the baseline for each camera before
        # adjustment.
        self.frame_skip = step1_cfg.get("frame_skip", 0)
        # Micro‑batch size for detection.  When greater than 1, the pipeline
        # accumulates this many frames before running detection.  The
        # detection will be performed on the final frame in the batch, and
        # the result applied to the entire batch window.  This can improve
        # throughput by amortising preprocessing overhead.  Default 1.
        self.frame_batch_size = step1_cfg.get("frame_batch_size", 1)
        # Whether to enable basic tracking.  Tracking assigns persistent
        # IDs to detections across frames and can be used to suppress
        # duplicate alerts.  Disable by default for minimal overhead.
        adv_cfg = config.get("advanced", {})
        self.tracking_enabled = adv_cfg.get("tracking_enabled", False)
        # Dynamic backpressure configuration.  When enabled, the per‑camera
        # frame skip is adjusted based on the event queue size.  This helps
        # prevent overload under high load by increasing the skip when the
        # queue is near capacity, and decreasing it when the queue drains.
        self.dynamic_backpressure = adv_cfg.get("dynamic_backpressure", False)
        self.min_frame_skip = adv_cfg.get("min_frame_skip", 0)
        self.max_frame_skip = adv_cfg.get("max_frame_skip", 4)
        self.backpressure_high_watermark = adv_cfg.get("backpressure_high_watermark", 0.75)
        self.backpressure_low_watermark = adv_cfg.get("backpressure_low_watermark", 0.25)
        # Map camera ID to current dynamic frame skip.  Filled during start().
        self._dynamic_skip: Dict[str, int] = {}
        # Map camera ID to adapter instance
        self.adapters: Dict[str, CameraAdapter] = {}
        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False

        # Event history and lock
        #
        # The UI queries this list to display recent events.  Each entry is
        # a dict with keys like `id`, `camera_id`, `timestamp`,
        # `flags`, `score` and `clip_file`.  Access to the list is protected by
        # a lock because events are appended from a background thread while
        # the web server may iterate over it.  The history is never
        # truncated in this skeleton; you may wish to add pruning logic
        # based on age or maximum length.
        self.events_history: List[Dict[str, Any]] = []
        self.events_lock = threading.Lock()
        # DeepStream pipeline handle (when in deepstream mode)
        self._ds: Optional[DeepStreamPipeline] = None

    def start(self) -> None:
        """Start all camera and event processing threads."""
        self.running = True
        # Set running gauge to 1
        pipeline_running_gauge.set(1)
        
        # Initialize hybrid backend
        try:
            self.backend, self.backend_name = self.backend_factory.create_backend_with_fallback(self.config)
            LOGGER.info(f"Using {self.backend_name} backend for processing")
            
            # Log backend capabilities
            capabilities = self.backend.capabilities
            LOGGER.info(f"Backend capabilities: max_streams={capabilities.max_streams}, "
                       f"performance_tier={capabilities.performance_tier}, "
                       f"gpu_required={capabilities.gpu_required}")
                       
        except Exception as e:
            LOGGER.error(f"Failed to initialize backend: {e}")
            raise RuntimeError(f"No suitable backend could be initialized: {e}")
        
        # Start async Stage-2 pipeline if enabled
        if self.async_stage2:
            self.async_stage2.start()
            
            # Start result handler for async processing
            self.async_result_handler = AsyncResultHandler(
                async_stage2=self.async_stage2,
                events_history=self.events_history,
                events_lock=self.events_lock,
                poll_interval=1.0,
            )
            self.async_result_handler.start()
            
            LOGGER.info("Async Stage-2 pipeline and result handler started")
        
        # Check if using legacy DeepStream mode or new backend system
        if self.backend_name == 'deepstream' and hasattr(self.backend, 'start_streaming'):
            # Use new DeepStream backend with streaming
            sources: List[str] = [str(c.get("source", "")) for c in self.cameras]
            if self.backend.start_streaming(sources):
                LOGGER.info("DeepStream backend streaming started")
            else:
                LOGGER.error("Failed to start DeepStream streaming")
                raise RuntimeError("DeepStream backend streaming failed")
        elif self.mode == "deepstream" or self.config.get("advanced", {}).get("use_deepstream", False):
            # Legacy DeepStream path for backward compatibility
            pgie_cfg_path = self.config.get("pgie_config")
            if not pgie_cfg_path or not isinstance(pgie_cfg_path, str) or not os.path.isfile(pgie_cfg_path):
                raise FileNotFoundError(
                    f"DeepStream mode enabled but pgie_config not found: {pgie_cfg_path}"
                )
            engine_path = self.config.get("step1", {}).get("model_path")
            if not engine_path or not os.path.isfile(engine_path):
                raise FileNotFoundError(
                    f"DeepStream mode requires a TensorRT engine at step1.model_path; missing: {engine_path}"
                )
            sources: List[str] = [str(c.get("source", "")) for c in self.cameras]
            camera_ids: List[str] = [c.get("id") or f"cam{i}" for i, c in enumerate(self.cameras)]
            batch_size = max(1, len(sources))
            self._ds = DeepStreamPipeline(
                sources=sources,
                model_engine=engine_path,
                config_file=pgie_cfg_path,
                event_queue=self.event_queue,
                camera_ids=camera_ids,
                batch_size=batch_size,
            )
            self._ds.start()
        else:
            # Start camera workers (for frame-by-frame processing backends)
            for cam_cfg in self.cameras:
                cam_id = cam_cfg.get("id") or f"cam_{len(self.adapters)}"
                source = cam_cfg.get("source", 0)
                fps = cam_cfg.get("fps")
                adapter = CameraAdapter(source=source, fps=fps)
                self.adapters[cam_id] = adapter
                
                # Determine ring buffer length in frames
                buffer_capacity = int((self.pre_sec + self.post_sec) * (fps or 15)) + 1
                self.ring_buffers[cam_id] = RingBuffer(capacity=buffer_capacity)
                
                # Initialize dynamic skip for this camera
                initial_skip = self.frame_skip
                if self.dynamic_backpressure:
                    initial_skip = max(self.min_frame_skip, min(initial_skip, self.max_frame_skip))
                self._dynamic_skip[cam_id] = initial_skip
                
                thread = threading.Thread(target=self._camera_loop, args=(cam_id,), daemon=True)
                thread.start()
                self.workers.append(thread)
        
        # Start event consumer thread (common to all backends)
        consumer_thread = threading.Thread(target=self._event_consumer, daemon=True)
        consumer_thread.start()
        self.workers.append(consumer_thread)
        
        LOGGER.info("Inference pipeline started with %d cameras using %s backend", 
                   len(self.cameras), self.backend_name)

    def stop(self) -> None:
        """Stop all worker threads and release resources."""
        self.running = False
        # Set running gauge to 0
        pipeline_running_gauge.set(0)
        
        # Stop async Stage-2 pipeline if running
        if self.async_result_handler:
            self.async_result_handler.stop()
            
        if self.async_stage2:
            self.async_stage2.stop()
            LOGGER.info("Async Stage-2 pipeline stopped")
        
        # Clean up backend
        if self.backend:
            try:
                if hasattr(self.backend, 'stop_streaming'):
                    self.backend.stop_streaming()
                self.backend.cleanup()
                LOGGER.info(f"{self.backend_name} backend cleaned up")
            except Exception as e:
                LOGGER.error(f"Error cleaning up backend: {e}")
            
        # Stop legacy DeepStream pipeline if running
        if self._ds:
            try:
                self._ds.stop()
            except Exception:
                pass
                
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
        # Micro‑batch buffer for this camera
        batch_frames: List[np.ndarray] = []
        batch_timestamps: List[float] = []
        while self.running:
            video_frame = adapter.read()
            if video_frame is None:
                # Sleep a bit before retrying on error.
                time.sleep(0.1)
                continue
            ts = video_frame.timestamp
            frame = video_frame.image
            # Push to ring buffer immediately so the clip includes all frames
            ring.push(ts, frame)
            frame_counter += 1
            # Append to micro‑batch lists
            batch_frames.append(frame)
            batch_timestamps.append(ts)
            # Determine effective frame skip.  Use dynamic skip if enabled
            effective_skip = self.frame_skip
            if self.dynamic_backpressure:
                # Compute absolute queue watermarks
                # Use configured queue_maxsize or event_queue maxsize
                try:
                    qmax = int(self.config.get("advanced", {}).get("queue_maxsize", 0))
                    if qmax <= 0:
                        qmax = self.event_queue.maxsize
                    if not qmax or qmax <= 0:
                        qmax = 64
                except Exception:
                    qmax = 64
                high_thresh = int(qmax * self.backpressure_high_watermark)
                low_thresh = int(qmax * self.backpressure_low_watermark)
                current_q = self.event_queue.qsize()
                current_skip = self._dynamic_skip.get(cam_id, self.frame_skip)
                # Increase skip if queue is above high watermark
                if current_q >= high_thresh and current_skip < self.max_frame_skip:
                    current_skip = min(self.max_frame_skip, current_skip + 1)
                # Decrease skip if queue is below low watermark
                elif current_q <= low_thresh and current_skip > self.min_frame_skip:
                    current_skip = max(self.min_frame_skip, current_skip - 1)
                # Store updated skip
                self._dynamic_skip[cam_id] = current_skip
                effective_skip = current_skip
            # Determine if we should run detection on this frame batch
            should_detect = False
            # Enough frames accumulated for batch
            if len(batch_frames) >= self.frame_batch_size:
                # Respect frame skipping: run only when counter modulo skip+1 is zero
                if effective_skip == 0 or (frame_counter % (effective_skip + 1) == 0):
                    should_detect = True
            if not should_detect:
                # Update queue size gauge and continue to next frame
                event_queue_size_gauge.set(self.event_queue.qsize())
                continue
            # Pop the frame to detect (use last one) and its timestamp
            detect_frame = batch_frames[-1]
            detect_ts = batch_timestamps[-1]
            # Clear batch buffers
            batch_frames.clear()
            batch_timestamps.clear()
            # Run detection using hybrid backend system
            start_time_det = time.perf_counter()
            
            # Use backend if available, fallback to legacy detector
            if self.backend:
                try:
                    backend_detections = self.backend.process_frame(detect_frame, frame_counter, detect_ts)
                    # Convert backend detections to legacy format for compatibility
                    detections = []
                    for det in backend_detections:
                        detections.append({
                            'class_id': det.class_id,
                            'confidence': det.confidence,
                            'bbox': det.bbox,
                            'class_name': det.class_name,
                            'label': det.class_name  # For backward compatibility
                        })
                except Exception as e:
                    LOGGER.error(f"Backend processing failed on {cam_id}: {e}")
                    detections = []
            else:
                # Fallback to legacy detector
                detections = self.detector.detect(detect_frame)
            
            latency = time.perf_counter() - start_time_det
            detection_latency_histogram.observe(latency)
            # Optionally update tracker to assign persistent IDs
            if self.tracking_enabled:
                try:
                    self.tracker.update_tracks(detections)
                except Exception as exc:
                    LOGGER.warning("Tracking update failed on %s: %s", cam_id, exc)
            flags = self.detector.analyze_events(detections)
            if any(flags.values()):
                # Candidate event detected; record start time for pre/post clip
                event = Event(
                    camera_id=cam_id,
                    timestamp=detect_ts,
                    frame_id=frame_counter,
                    detections=detections,
                    flags=flags,
                )
                try:
                    self.event_queue.put_nowait(event)
                    events_detected_counter.inc()
                    event_queue_size_gauge.set(self.event_queue.qsize())
                except Exception:
                    LOGGER.warning("Event queue full; dropping event from %s", cam_id)
            else:
                # Update queue size gauge even when not enqueuing to reflect current state
                event_queue_size_gauge.set(self.event_queue.qsize())

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
                # Update gauges even when queue is empty
                event_queue_size_gauge.set(self.event_queue.qsize())
                continue
            cam_id = event.camera_id
            ts = event.timestamp
            ring = self.ring_buffers.get(cam_id)
            if ring is None:
                # DeepStream path may not maintain a CPU ring buffer; proceed without clip frames
                LOGGER.debug("No ring buffer for %s; proceeding without clip save", cam_id)
                clip_frames: List[np.ndarray] = []
            else:
                # Extract pre/post clip frames
                start_time = ts - self.pre_sec
                end_time = ts + self.post_sec
                clip_frames = ring.get_clip(start_time, end_time)
            # Derive list of labels per frame from detections (naive)
            score = 0.0
            verification_result = None
            
            if self.async_stage2:
                # Submit to async Stage-2 pipeline (non-blocking)
                # Extract labels from detections (handle both dict and object formats)
                labels = []
                for d in event.detections:
                    if hasattr(d, 'label'):  # Detection object
                        labels.append(d.label)
                    elif isinstance(d, dict):  # Dictionary format
                        labels.append(d.get('label') or d.get('class_name', ''))
                
                labels_per_frame = [labels for _ in clip_frames]
                request_id = self.async_stage2.submit_verification(labels_per_frame, clip_frames)
                
                # Try to get a result immediately (non-blocking poll)
                verification_result = self.async_stage2.get_result(request_id, timeout=0.0)
                
                if verification_result:
                    score = verification_result.get("score", 0.0)
                    LOGGER.info("Event on %s at %.2f verified async with score %.2f", cam_id, ts, score)
                else:
                    # No immediate result - will be processed asynchronously
                    # Register with result handler for background processing
                    if self.async_result_handler:
                        event_index = len(self.events_history)  # Will be the index after we append
                        self.async_result_handler.register_pending_request(request_id, event_index)
                    
                    LOGGER.info("Event on %s at %.2f submitted for async verification (request_id: %s)", 
                               cam_id, ts, request_id)
                    score = -1.0  # Indicator for pending verification
                    
            elif self.step2:
                # Synchronous verification (original behavior)
                # Extract labels from detections (handle both dict and object formats)
                labels = []
                for d in event.detections:
                    if hasattr(d, 'label'):  # Detection object
                        labels.append(d.label)
                    elif isinstance(d, dict):  # Dictionary format
                        labels.append(d.get('label') or d.get('class_name', ''))
                
                labels_per_frame = [labels for _ in clip_frames]
                verification_result = self.step2.verify(labels_per_frame)
                score = verification_result.get("score", 0.0)
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
                "score": float(score) if (self.step2 or self.async_stage2) else None,
                "clip_file": clip_filename,
                "verification_status": "completed" if verification_result else ("pending" if self.async_stage2 else "disabled"),
            }
            with self.events_lock:
                self.events_history.append(event_record)
                event_history_size_gauge.set(len(self.events_history))
            # Mark the task done and update gauges
            self.event_queue.task_done()
            event_queue_size_gauge.set(self.event_queue.qsize())

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