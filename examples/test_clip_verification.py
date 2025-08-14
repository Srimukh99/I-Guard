#!/usr/bin/env python3
"""
Example script to demonstrate Step 2 verification with different 3D CNN models.

This script shows how to use the ClipVerifier with TAO ActionRecognitionNet,
X3D, or simple aggregation fallback.
"""

import numpy as np
import logging
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from detection.clip_verifier import ClipVerifier

logging.basicConfig(level=logging.INFO)

def create_mock_video_clip(frames=16, height=224, width=224) -> np.ndarray:
    """Create a mock video clip for testing."""
    # Create random video clip (T, H, W, C)
    clip = np.random.randint(0, 255, size=(frames, height, width, 3), dtype=np.uint8)
    return clip

def test_tao_verifier():
    """Test TAO ActionRecognitionNet verifier."""
    print("\n=== Testing TAO ActionRecognitionNet ===")
    
    verifier = ClipVerifier(
        model_path="models/tao_action_recognition.engine",
        model_type="tao",
        threshold=0.7
    )
    
    # Create mock video clip
    video_clip = create_mock_video_clip()
    
    # Run verification
    result = verifier.verify(video_clip)
    
    print(f"TAO Results: {result}")
    print(f"Predicted Action: {result['action']}")
    print(f"Overall Threat Score: {result['score']:.3f}")
    
    return result

def test_x3d_verifier():
    """Test X3D verifier."""
    print("\n=== Testing X3D ===")
    
    verifier = ClipVerifier(
        model_path="models/x3d_s.engine",
        model_type="x3d", 
        threshold=0.6
    )
    
    # Create mock video clip
    video_clip = create_mock_video_clip()
    
    # Run verification
    result = verifier.verify(video_clip)
    
    print(f"X3D Results: {result}")
    print(f"Predicted Action: {result['action']}")
    print(f"Overall Threat Score: {result['score']:.3f}")
    
    return result

def test_simple_verifier():
    """Test simple aggregation fallback."""
    print("\n=== Testing Simple Aggregation Fallback ===")
    
    verifier = ClipVerifier(
        model_path="non_existent_model.engine",  # This will trigger fallback
        model_type="simple",
        threshold=0.5
    )
    
    # Mock detection results per frame
    detections_per_frame = [
        ["person"],
        ["person", "gun"],
        ["person", "gun"],
        ["person"],
        ["person", "knife"],
        ["person"],
        ["person", "gun"],
        ["person"]
    ]
    
    # Run verification
    result = verifier.verify(video_clip=None, detections_per_frame=detections_per_frame)
    
    print(f"Simple Results: {result}")
    print(f"Weapon Detection Ratio: {result.get('weapon_ratio', 0):.3f}")
    print(f"Overall Threat Score: {result['score']:.3f}")
    
    return result

def main():
    """Run all verification tests."""
    print("I-Guard Step 2 Verification Demo")
    print("=" * 40)
    
    # Test all verifier types
    tao_result = test_tao_verifier()
    x3d_result = test_x3d_verifier() 
    simple_result = test_simple_verifier()
    
    print("\n=== Summary ===")
    print(f"TAO Threat Score:    {tao_result['score']:.3f}")
    print(f"X3D Threat Score:    {x3d_result['score']:.3f}")  
    print(f"Simple Threat Score: {simple_result['score']:.3f}")
    
    # Determine if event should be escalated
    threshold = 0.7
    escalate = any(result['score'] > threshold for result in [tao_result, x3d_result, simple_result])
    
    print(f"\nEscalate to human review: {'YES' if escalate else 'NO'}")

if __name__ == "__main__":
    main()
