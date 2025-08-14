"""
Asynchronous Stage-2 verification pipeline for I-Guard.

This module provides non-blocking Stage-2 verification that integrates
with the existing ClipVerifier while maintaining backward compatibility.

Notes:
- The pipeline accepts verification requests and processes them in
    background worker threads to avoid blocking the main capture/alert
    path. Results are delivered either via a callback or an internal
    result queue.
- Keep changes small and maintain backwards compatibility with the
    previous sync-style `verify` API used elsewhere in the codebase.
"""

import threading
import time
import queue
from queue import Queue, Empty, Full
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, Optional, List, Callable
import logging
from dataclasses import dataclass
from pathlib import Path
import uuid

# Import existing components
from detection.clip_verifier import ClipVerifier

LOGGER = logging.getLogger(__name__)


@dataclass
class VerificationRequest:
    """Request for async verification."""
    event_id: str
    camera_id: str
    timestamp: float
    video_clip: Any
    detections_per_frame: Optional[List[List[str]]] = None
    priority: int = 0  # Higher = more urgent


@dataclass 
class VerificationResult:
    """Result from async verification."""
    event_id: str
    camera_id: str
    timestamp: float
    verification_timestamp: float
    score: float
    action: str
    action_confidence: float
    verifier_type: str
    processing_time_ms: float
    metadata: Dict[str, Any]


class AsyncStage2Pipeline:
    """
    Asynchronous Stage-2 verification pipeline.
    
    Provides non-blocking verification while maintaining compatibility
    with existing ClipVerifier interface.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        result_callback: Optional[Callable[[VerificationResult], None]] = None,
        **kwargs: Any,
    ):
        """
        Initialize async verification pipeline.
        
        Args:
            config: Configuration dictionary with step2 settings
            result_callback: Optional callback for results (alternative to polling)
        """
        # Support two construction styles for backward compatibility:
        # 1) Pass a full `config` dict (preferred)
        # 2) Pass keyword args like model_path, threshold, max_workers (legacy/tests)
        self.result_callback = result_callback

        if config is None:
            # Build a minimal step2 config from kwargs
            step2_config = {
                "model_path": kwargs.get("model_path", ""),
                "model_type": kwargs.get("model_type", kwargs.get("model", "tao")),
                "verification_threshold": kwargs.get("threshold", kwargs.get("verification_threshold", 0.7)),
                "max_workers": kwargs.get("max_workers", 2),
                "queue_size": kwargs.get("queue_size", 50),
                "result_queue_size": kwargs.get("result_queue_size", 100),
                "worker_timeout_sec": kwargs.get("worker_timeout", kwargs.get("worker_timeout_sec", 10.0)),
            }
            self.config = {"step2": step2_config}
        else:
            self.config = config
            step2_config = config.get("step2", {})

        # Extract configuration
        self.enabled = step2_config.get("enabled", True)
        if not self.enabled:
            LOGGER.info("Stage-2 verification disabled in config")
            return
        self.max_workers = step2_config.get("max_workers", 2)
        self.queue_size = step2_config.get("queue_size", 50)
        self.result_queue_size = step2_config.get("result_queue_size", 100)
        self.worker_timeout = step2_config.get("worker_timeout_sec", 10.0)
        
        # Initialize verifier
        self.verifier = None
        self._init_verifier(step2_config)
        
        # Thread-safe queues
        self.request_queue = Queue(maxsize=self.queue_size)
        self.result_queue = Queue(maxsize=self.result_queue_size)
        
        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="Stage2")
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        
        # Statistics
        self.stats = {
            "submitted": 0,
            "processed": 0,
            "failed": 0,
            "dropped_input": 0,
            "dropped_output": 0,
            "avg_processing_time_ms": 0.0
        }
        self.processing_times = []

        # Synchronization primitives and tracking structures used by monitoring
        self.stats_lock = threading.Lock()
        self.pending_requests = {}
        # Work/result queue aliases used by monitoring methods
        self.work_queue = self.request_queue
        # Cumulative counters for compatibility with monitoring calls
        self.total_submitted = 0
        self.total_completed = 0
        self.total_processing_time = 0.0
        self.workers_active = 0

        LOGGER.info(f"AsyncStage2Pipeline initialized with {self.max_workers} workers")
    
    def _init_verifier(self, step2_config: Dict[str, Any]):
        """Initialize the ClipVerifier with fallback handling."""
        try:
            model_path = step2_config.get("model_path", "")
            model_type = step2_config.get("model_type", "tao")
            threshold = step2_config.get("verification_threshold", 0.7)
            
            self.verifier = ClipVerifier(
                model_path=model_path,
                model_type=model_type,
                threshold=threshold
            )
            
            LOGGER.info(f"Initialized verifier: {model_type} with threshold {threshold}")
            
        except Exception as e:
            LOGGER.error(f"Failed to initialize verifier: {e}")
            # Create fallback simple verifier
            self.verifier = ClipVerifier(
                model_path="",
                model_type="simple", 
                threshold=0.5
            )
            LOGGER.warning("Using simple aggregation fallback")
    
    def start(self):
        """Start the async verification pipeline."""
        if not self.enabled or not self.verifier:
            LOGGER.warning("Cannot start pipeline: disabled or no verifier")
            return
            
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                name=f"Stage2-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
            
        LOGGER.info(f"Started {self.max_workers} async Stage-2 workers")
    
    def stop(self, timeout: float = 5.0):
        """Stop the async verification pipeline."""
        if not self.running:
            return
            
        LOGGER.info("Stopping async Stage-2 pipeline...")
        self.running = False
        
        # Wait for workers to finish current tasks
        for worker in self.worker_threads:
            worker.join(timeout=timeout / len(self.worker_threads) if self.worker_threads else timeout)
            
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Log final statistics
        self._log_final_stats()
        LOGGER.info("Async Stage-2 pipeline stopped")
    
    def submit_verification(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Submit event for async verification (NON-BLOCKING).
        
        Args:
            event_id: Unique identifier for the event
            camera_id: Camera that detected the event
            timestamp: Event timestamp
            video_clip: Video clip for verification
            detections_per_frame: Frame-level detections (for simple mode)
            priority: Priority level (higher = more urgent)
            
        Returns:
            bool: True if queued successfully, False if queue full
        """
        # Backwards-compatible calling conventions:
        # - submit_verification(labels_per_frame, clip_frames) -> returns request_id
        # - submit_verification(event_id, camera_id, timestamp, video_clip, detections_per_frame)
        if not self.enabled or not self.running:
            return False

        # Parse args
        if len(args) == 2 and isinstance(args[0], list):
            # Called as submit_verification(labels_per_frame, clip_frames)
            detections_per_frame = args[0]
            video_clip = args[1]
            event_id = str(uuid.uuid4())
            camera_id = kwargs.get("camera_id", "unknown")
            timestamp = kwargs.get("timestamp", time.time())
        else:
            # Expect explicit signature style
            try:
                event_id = args[0]
                camera_id = args[1]
                timestamp = args[2]
                video_clip = args[3]
                detections_per_frame = args[4] if len(args) > 4 else kwargs.get("detections_per_frame")
            except Exception:
                raise TypeError("submit_verification called with unsupported arguments")

        request = VerificationRequest(
            event_id=event_id,
            camera_id=camera_id,
            timestamp=timestamp,
            video_clip=video_clip,
            detections_per_frame=detections_per_frame,
            priority=int(kwargs.get("priority", 0)),
        )

        try:
            self.request_queue.put(request, block=False)
            self.stats["submitted"] += 1
            self.total_submitted += 1
            # Track pending
            with self.stats_lock:
                self.pending_requests[request.event_id] = request
            return request.event_id

        except Full:
            # Queue full - try to drop oldest low-priority item
            try:
                dropped_request = self.request_queue.get(block=False)
                self.request_queue.put(request, block=False)
                self.stats["dropped_input"] += 1
                LOGGER.warning(f"Dropped event {dropped_request.event_id} to make room for {event_id}")
                with self.stats_lock:
                    self.pending_requests[request.event_id] = request
                return request.event_id
            except (Empty, Full):
                self.stats["dropped_input"] += 1
                LOGGER.warning(f"Failed to queue event {event_id}: queue full")
                return False
    
    def get_result(self, request_id: Optional[str] = None, timeout: float = 0.0) -> Optional[VerificationResult]:
        """Get next verification result (NON-BLOCKING).

        Parameters:
        - request_id: If provided, look for this specific request
        - timeout: Maximum time to wait for result
        """
        if request_id is not None:
            return self.get_result_by_id(request_id, timeout)
        
        try:
            if timeout > 0.0:
                return self.result_queue.get(timeout=timeout)
            else:
                return self.result_queue.get(block=False)
        except Empty:
            return None

    def get_result_by_id(self, request_id: str, timeout: float = 0.0) -> Optional[VerificationResult]:
        """Poll the result queue for a specific request_id."""
        end = time.time() + timeout
        temp: List[VerificationResult] = []
        found: Optional[VerificationResult] = None
        # If timeout == 0 do a single non-blocking scan
        while True:
            try:
                res: VerificationResult = self.result_queue.get(block=(timeout > 0.0), timeout=timeout if timeout > 0.0 else 0)
            except Empty:
                break
            if res.event_id == request_id:
                found = res
                break
            else:
                temp.append(res)
            if timeout == 0.0:
                break
            if time.time() > end:
                break

        # Put back non-matching results
        for r in temp:
            try:
                self.result_queue.put_nowait(r)
            except Full:
                LOGGER.warning("Result queue full while restoring results")

        return found
    
    def get_all_results(self) -> List[VerificationResult]:
        """Get all available results (NON-BLOCKING)."""
        results = []
        while True:
            result = self.get_result()
            if result is None:
                break
            results.append(result)
        return results
    
    def _worker_loop(self, worker_id: int):
        """Background worker for processing verification requests."""
        LOGGER.debug(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get work item with timeout
                try:
                    request = self.request_queue.get(timeout=1.0)
                except Empty:
                    continue  # Check if still running
                
                # Process the request
                start_time = time.time()
                result = self._process_request(request)
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                if result:
                    result.processing_time_ms = processing_time
                    self._handle_result(result)
                    self.stats["processed"] += 1
                    
                    # Update average processing time
                    self.processing_times.append(processing_time)
                    if len(self.processing_times) > 100:  # Keep rolling window
                        self.processing_times.pop(0)
                    self.stats["avg_processing_time_ms"] = sum(self.processing_times) / len(self.processing_times)
                else:
                    self.stats["failed"] += 1
                    
            except Exception as e:
                LOGGER.error(f"Worker {worker_id} error: {e}")
                self.stats["failed"] += 1
        
        LOGGER.debug(f"Worker {worker_id} stopped")
    
    def _process_request(self, request: VerificationRequest) -> Optional[VerificationResult]:
        """Process a single verification request."""
        try:
            # Run verification based on verifier type
            if self.verifier.model_type == "simple":
                # Simple aggregation mode
                verify_result = self.verifier.verify(
                    video_clip=None,
                    detections_per_frame=request.detections_per_frame
                )
            else:
                # 3D CNN modes
                verify_result = self.verifier.verify(request.video_clip)
            
            # Create result object
            result = VerificationResult(
                event_id=request.event_id,
                camera_id=request.camera_id,
                timestamp=request.timestamp,
                verification_timestamp=time.time(),
                score=verify_result.get("score", 0.0),
                action=verify_result.get("action", "unknown"),
                action_confidence=verify_result.get("action_confidence", 0.0),
                verifier_type=self.verifier.model_type,
                processing_time_ms=0.0,  # Will be set by caller
                metadata=verify_result
            )
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Verification failed for event {request.event_id}: {e}")
            return None
    
    def _handle_result(self, result: VerificationResult):
        """Handle verification result - either callback or queue."""
        if self.result_callback:
            try:
                self.result_callback(result)
            except Exception as e:
                LOGGER.error(f"Result callback failed: {e}")
        else:
            # Put in result queue
            try:
                self.result_queue.put(result, block=False)
            except Full:
                # Drop oldest result to make room
                try:
                    dropped = self.result_queue.get(block=False)
                    self.result_queue.put(result, block=False)
                    self.stats["dropped_output"] += 1
                    LOGGER.warning(f"Dropped result for event {dropped.event_id}")
                except (Empty, Full):
                    self.stats["dropped_output"] += 1
    
    def get_all_available_results(self, timeout: float = 0.0) -> Dict[str, Dict[str, Any]]:
        """Get all available completed results.
        
        This method returns all completed verification results that are
        currently available in the result queue. It's useful for batch
        processing of results.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for at least one result. Default 0.0 (non-blocking).
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Map of request_id -> verification result for all available results.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # Optionally wait for a single result to appear
        try:
            if timeout > 0.0:
                # Wait for at least one
                try:
                    first: VerificationResult = self.result_queue.get(timeout=timeout)
                    results[first.event_id] = first
                except Empty:
                    return results
        except Exception:
            pass

        # Drain remaining results
        while True:
            try:
                res: VerificationResult = self.result_queue.get_nowait()
                results[res.event_id] = res
            except Empty:
                break

        return results
    
    def get_pending_request_ids(self) -> List[str]:
        """Get list of request IDs that are currently pending.
        
        Returns
        -------
        List[str]
            List of request IDs that are currently being processed or queued.
        """
        with self.stats_lock:
            return list(self.pending_requests.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing performance statistics including queue sizes,
            processing times, and worker status.
        """
        with self.stats_lock:
            return {
                "pending_requests": len(self.pending_requests),
                "work_queue_size": self.work_queue.qsize(),
                "result_queue_size": self.result_queue.qsize(),
                "total_submitted": self.total_submitted,
                "total_completed": self.total_completed,
                "avg_processing_time": self.total_processing_time / max(1, self.total_completed),
                "workers_active": self.workers_active,
                "max_workers": self.max_workers,
            }
    
    def get_pending_request_ids(self) -> List[str]:
        """Get list of request IDs that are currently pending.
        
        Returns
        -------
        List[str]
            List of request IDs that are currently being processed or queued.
        """
        with self.stats_lock:
            return list(self.pending_requests.keys())
    
    def _log_final_stats(self):
        """Log final statistics when stopping."""
        LOGGER.info("=== Final Stage-2 Statistics ===")
        LOGGER.info(f"  Submitted: {self.stats['submitted']}")
        LOGGER.info(f"  Processed: {self.stats['processed']}")
        LOGGER.info(f"  Failed: {self.stats['failed']}")
        LOGGER.info(f"  Dropped (input): {self.stats['dropped_input']}")
        LOGGER.info(f"  Dropped (output): {self.stats['dropped_output']}")
        LOGGER.info(f"  Avg processing time: {self.stats['avg_processing_time_ms']:.1f}ms")
        
        if self.stats['submitted'] > 0:
            success_rate = (self.stats['processed'] / self.stats['submitted']) * 100
            LOGGER.info(f"  Success rate: {success_rate:.1f}%")


class Stage2Manager:
    """
    Manager class that provides both sync and async interfaces.
    
    This allows existing code to work unchanged while enabling async mode
    for improved performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.async_mode = config.get("step2", {}).get("async_enabled", True)
        
        if self.async_mode:
            self.async_pipeline = AsyncStage2Pipeline(config)
            self.sync_verifier = None
        else:
            # Traditional sync mode
            step2_config = config.get("step2", {})
            self.sync_verifier = ClipVerifier(
                model_path=step2_config.get("model_path", ""),
                model_type=step2_config.get("model_type", "tao"),
                threshold=step2_config.get("verification_threshold", 0.7)
            )
            self.async_pipeline = None
        
        LOGGER.info(f"Stage2Manager initialized in {'async' if self.async_mode else 'sync'} mode")
    
    def start(self):
        """Start the Stage-2 manager."""
        if self.async_pipeline:
            self.async_pipeline.start()
    
    def stop(self):
        """Stop the Stage-2 manager.""" 
        if self.async_pipeline:
            self.async_pipeline.stop()
    
    def verify_sync(self, video_clip: Any, detections_per_frame: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        """Synchronous verification (backward compatibility)."""
        if self.sync_verifier:
            return self.sync_verifier.verify(video_clip, detections_per_frame)
        else:
            LOGGER.warning("Sync verification called in async mode")
            return {"score": 0.0, "action": "unavailable", "action_confidence": 0.0}
    
    def verify_async(
        self,
        event_id: str,
        camera_id: str,
        timestamp: float,
        video_clip: Any,
        detections_per_frame: Optional[List[List[str]]] = None
    ) -> bool:
        """Asynchronous verification submission."""
        if self.async_pipeline:
            return self.async_pipeline.submit_verification(
                event_id, camera_id, timestamp, video_clip, detections_per_frame
            )
        else:
            LOGGER.warning("Async verification called in sync mode")
            return False
    
    def get_results(self) -> List[VerificationResult]:
        """Get all available results from async pipeline."""
        if self.async_pipeline:
            return self.async_pipeline.get_all_results()
        else:
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if self.async_pipeline:
            return self.async_pipeline.get_stats()
        else:
            return {"mode": "sync", "enabled": self.sync_verifier is not None}
