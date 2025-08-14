"""Background result handler for async Stage-2 verification.

This module provides a background thread that continuously polls
for completed async verification results and updates the event
history accordingly. This ensures that events with pending
verification get their scores updated when the verification
completes.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional

LOGGER = logging.getLogger(__name__)


class AsyncResultHandler:
    """Background handler for processing async verification results.
    
    This class runs a background thread that continuously polls
    the AsyncStage2Pipeline for completed results and updates
    the event history with verification scores.
    
    Parameters
    ----------
    async_stage2 : AsyncStage2Pipeline
        The async Stage-2 pipeline to poll for results.
    events_history : List[Dict[str, Any]]
        Reference to the main events history list.
    events_lock : threading.Lock
        Lock protecting access to the events history.
    poll_interval : float, optional
        How often to poll for results in seconds. Default 1.0.
    """
    
    def __init__(
        self,
        async_stage2,
        events_history: List[Dict[str, Any]],
        events_lock: threading.Lock,
        poll_interval: float = 1.0,
    ):
        self.async_stage2 = async_stage2
        self.events_history = events_history
        self.events_lock = events_lock
        self.poll_interval = poll_interval
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Track pending request IDs to event history indices
        self.pending_requests: Dict[str, int] = {}
    
    def start(self) -> None:
        """Start the background result handler thread."""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._result_loop, daemon=True)
        self.worker_thread.start()
        LOGGER.info("Async result handler started")
    
    def stop(self) -> None:
        """Stop the background result handler thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        LOGGER.info("Async result handler stopped")
    
    def register_pending_request(self, request_id: str, event_index: int) -> None:
        """Register a pending verification request.
        
        Parameters
        ----------
        request_id : str
            The request ID returned by AsyncStage2Pipeline.submit_verification()
        event_index : int
            The index in events_history corresponding to this request.
        """
        self.pending_requests[request_id] = event_index
    
    def _result_loop(self) -> None:
        """Background loop that polls for completed results."""
        while self.running:
            try:
                # Get all available results (non-blocking)
                completed_results = self.async_stage2.get_all_available_results()
                
                if completed_results:
                    self._process_completed_results(completed_results)
                
                # Clean up old completed requests
                self._cleanup_completed_requests()
                
            except Exception as exc:
                LOGGER.error("Error in async result handler loop: %s", exc)
            
            # Sleep until next poll
            time.sleep(self.poll_interval)
    
    def _process_completed_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Process a batch of completed verification results.
        
        Parameters
        ----------
        results : Dict[str, Dict[str, Any]]
            Map of request_id -> verification result.
        """
        with self.events_lock:
            for request_id, result in results.items():
                event_index = self.pending_requests.get(request_id)
                if event_index is None:
                    continue
                
                # Check if event still exists in history
                if 0 <= event_index < len(self.events_history):
                    event_record = self.events_history[event_index]
                    
                    # Update with verification result
                    event_record["score"] = float(result.get("score", 0.0))
                    event_record["verification_status"] = "completed"
                    
                    LOGGER.info(
                        "Updated event %d with async verification score %.2f",
                        event_index,
                        event_record["score"]
                    )
                
                # Remove from pending list
                del self.pending_requests[request_id]
    
    def _cleanup_completed_requests(self) -> None:
        """Clean up tracking for requests that are no longer in the async pipeline."""
        # This helps prevent memory leaks if results get lost
        try:
            # Get list of request IDs that are still in the async pipeline
            active_requests = set(self.async_stage2.get_pending_request_ids())
            
            # Remove any pending requests that are no longer active
            completed_request_ids = []
            for request_id in self.pending_requests:
                if request_id not in active_requests:
                    completed_request_ids.append(request_id)
            
            for request_id in completed_request_ids:
                event_index = self.pending_requests.pop(request_id)
                LOGGER.warning(
                    "Cleaned up orphaned request %s for event %d", 
                    request_id, event_index
                )
                
        except Exception as exc:
            LOGGER.error("Error during cleanup: %s", exc)
