"""Example script demonstrating async Stage-2 verification.

This script shows how to use the AsyncStage2Pipeline in isolation
and demonstrates the performance benefits of asynchronous processing
for multiple simultaneous verification requests.
"""

import asyncio
import logging
import time
from typing import List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.async_stage2 import AsyncStage2Pipeline
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


def create_dummy_clip_data(num_frames: int = 16) -> List[np.ndarray]:
    """Create dummy video frames for testing.
    
    Parameters
    ----------
    num_frames : int
        Number of frames to generate.
        
    Returns
    -------
    List[np.ndarray]
        List of dummy frame arrays.
    """
    frames = []
    for i in range(num_frames):
        # Create a dummy 224x224x3 frame
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def test_async_verification():
    """Test async verification with multiple concurrent requests."""
    
    # Initialize async Stage-2 pipeline
    async_pipeline = AsyncStage2Pipeline(
        model_path="models/simple_classifier.pth",  # Will fall back to simple model
        threshold=0.7,
        max_workers=2,
        queue_size=10,
        result_queue_size=20,
        worker_timeout=5.0,
        model_type="simple",  # Use simple model for testing
        input_size=[224, 224],
        temporal_size=16,
    )
    
    try:
        # Start the pipeline
        async_pipeline.start()
        logger.info("Async Stage-2 pipeline started")
        
        # Submit multiple verification requests
        request_ids = []
        num_requests = 5
        
        logger.info(f"Submitting {num_requests} verification requests...")
        start_time = time.time()
        
        for i in range(num_requests):
            # Create dummy data
            labels_per_frame = [["person", "gun"] for _ in range(16)]
            clip_frames = create_dummy_clip_data(16)
            
            # Submit for verification
            request_id = async_pipeline.submit_verification(labels_per_frame, clip_frames)
            request_ids.append(request_id)
            logger.info(f"Submitted request {i+1}/{num_requests}: {request_id}")
        
        submit_time = time.time() - start_time
        logger.info(f"All requests submitted in {submit_time:.2f} seconds")
        
        # Collect results
        results = {}
        max_wait_time = 30.0  # Maximum time to wait for all results
        start_wait = time.time()
        
        while len(results) < num_requests and (time.time() - start_wait) < max_wait_time:
            # Check for completed results
            for request_id in request_ids:
                if request_id not in results:
                    result = async_pipeline.get_result(request_id, timeout=0.1)
                    if result:
                        results[request_id] = result
                        logger.info(f"Got result for {request_id}: score={result.score:.3f}")
            
            time.sleep(0.1)  # Brief pause before next check
        
        total_time = time.time() - start_time
        logger.info(f"Received {len(results)}/{num_requests} results in {total_time:.2f} seconds")
        
        # Print statistics
        stats = async_pipeline.get_stats()
        logger.info(f"Pipeline stats: {stats}")
        
        # Test batch result retrieval
        logger.info("Testing batch result retrieval...")
        
        # Submit a few more requests
        for i in range(3):
            labels_per_frame = [["person"] for _ in range(8)]
            clip_frames = create_dummy_clip_data(8)
            request_id = async_pipeline.submit_verification(labels_per_frame, clip_frames)
            logger.info(f"Submitted batch test request: {request_id}")
        
        # Wait a bit and get all available results
        time.sleep(2.0)
        batch_results = async_pipeline.get_all_available_results()
        logger.info(f"Got {len(batch_results)} results in batch: {list(batch_results.keys())}")
        
    finally:
        # Stop the pipeline
        async_pipeline.stop()
        logger.info("Async Stage-2 pipeline stopped")


def test_performance_comparison():
    """Compare performance of sync vs async verification."""
    
    logger.info("=== Performance Comparison: Sync vs Async ===")
    
    # Test data
    num_requests = 3
    labels_per_frame = [["person", "gun"] for _ in range(16)]
    clip_frames = create_dummy_clip_data(16)
    
    # Test synchronous verification (simulated)
    logger.info("Testing synchronous verification (simulated)...")
    start_time = time.time()
    
    for i in range(num_requests):
        # Simulate synchronous verification delay
        time.sleep(0.5)  # Simulate model inference time
        logger.info(f"Sync request {i+1}/{num_requests} completed")
    
    sync_time = time.time() - start_time
    logger.info(f"Synchronous processing took {sync_time:.2f} seconds")
    
    # Test asynchronous verification
    logger.info("Testing asynchronous verification...")
    
    async_pipeline = AsyncStage2Pipeline(
        model_path="models/simple_classifier.pth",
        threshold=0.7,
        max_workers=2,
        model_type="simple",
    )
    
    try:
        async_pipeline.start()
        
        start_time = time.time()
        request_ids = []
        
        # Submit all requests quickly
        for i in range(num_requests):
            request_id = async_pipeline.submit_verification(labels_per_frame, clip_frames)
            request_ids.append(request_id)
            logger.info(f"Async request {i+1}/{num_requests} submitted")
        
        # Wait for all results
        results = {}
        while len(results) < num_requests:
            for request_id in request_ids:
                if request_id not in results:
                    result = async_pipeline.get_result(request_id, timeout=0.1)
                    if result:
                        results[request_id] = result
                        logger.info(f"Async request completed: score={result.score:.3f}")
            time.sleep(0.1)
        
        async_time = time.time() - start_time
        logger.info(f"Asynchronous processing took {async_time:.2f} seconds")
        
        # Performance improvement
        improvement = ((sync_time - async_time) / sync_time) * 100
        logger.info(f"Async processing was {improvement:.1f}% faster")
        
    finally:
        async_pipeline.stop()


if __name__ == "__main__":
    logger.info("Starting async Stage-2 verification tests...")
    
    try:
        test_async_verification()
        print()
        test_performance_comparison()
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    
    logger.info("Tests completed")
