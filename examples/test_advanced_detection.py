#!/usr/bin/env python3
"""
Advanced Step 1 Event Detection Demo

This script demonstrates the enhanced event detection capabilities
in Step 1, including pose-based analysis for pointing, firing,
fall detection, and assault recognition.
"""

import numpy as np
import cv2
import logging
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from detection.frame_detector import FrameDetector, Detection

logging.basicConfig(level=logging.INFO)

def create_mock_frame_with_person_and_weapon(scenario="pointing"):
    """Create a mock frame for testing different scenarios."""
    # Create a 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some background
    frame[:] = (50, 100, 150)  # Bluish background
    
    if scenario == "pointing":
        # Draw a person shape (rectangle for simplicity)
        cv2.rectangle(frame, (200, 150), (280, 400), (255, 200, 200), -1)  # Person body
        cv2.rectangle(frame, (220, 120), (260, 160), (255, 180, 180), -1)  # Head
        # Extended arm (horizontal rectangle)
        cv2.rectangle(frame, (280, 200), (380, 230), (255, 150, 150), -1)  # Extended arm
        # Weapon at end of arm
        cv2.rectangle(frame, (380, 210), (420, 220), (100, 100, 100), -1)  # Gun
        
    elif scenario == "fallen":
        # Draw fallen person (horizontal)
        cv2.rectangle(frame, (150, 300), (400, 350), (255, 200, 200), -1)  # Horizontal body
        cv2.rectangle(frame, (140, 310), (170, 340), (255, 180, 180), -1)  # Head
        
    elif scenario == "assault":
        # Draw two people close together
        cv2.rectangle(frame, (200, 150), (260, 400), (255, 200, 200), -1)  # Person 1
        cv2.rectangle(frame, (220, 120), (240, 160), (255, 180, 180), -1)  # Head 1
        cv2.rectangle(frame, (280, 150), (340, 400), (200, 255, 200), -1)  # Person 2
        cv2.rectangle(frame, (300, 120), (320, 160), (180, 255, 180), -1)  # Head 2
        # Extended arms from both (fighting pose)
        cv2.rectangle(frame, (260, 200), (320, 220), (255, 150, 150), -1)  # Arms extending toward each other
        
    elif scenario == "mass_shooter":
        # Multiple people and weapons
        cv2.rectangle(frame, (100, 150), (160, 400), (255, 200, 200), -1)  # Person 1
        cv2.rectangle(frame, (300, 150), (360, 400), (200, 255, 200), -1)  # Person 2
        cv2.rectangle(frame, (500, 150), (560, 400), (200, 200, 255), -1)  # Person 3
        # Multiple weapons
        cv2.rectangle(frame, (80, 210), (120, 220), (100, 100, 100), -1)   # Gun 1
        cv2.rectangle(frame, (480, 210), (520, 220), (100, 100, 100), -1)  # Gun 2
        
    return frame

def create_mock_detections(scenario="pointing"):
    """Create mock detection results for different scenarios."""
    if scenario == "pointing":
        return [
            Detection(label="person", confidence=0.9, bbox=(200, 120, 80, 280)),
            Detection(label="gun", confidence=0.8, bbox=(380, 210, 40, 10))
        ]
    elif scenario == "fallen":
        return [
            Detection(label="person", confidence=0.85, bbox=(150, 310, 250, 40))
        ]
    elif scenario == "assault":
        return [
            Detection(label="person", confidence=0.9, bbox=(200, 120, 60, 280)),
            Detection(label="person", confidence=0.88, bbox=(280, 120, 60, 280))
        ]
    elif scenario == "mass_shooter":
        return [
            Detection(label="person", confidence=0.9, bbox=(100, 150, 60, 250)),
            Detection(label="person", confidence=0.87, bbox=(300, 150, 60, 250)),
            Detection(label="person", confidence=0.92, bbox=(500, 150, 60, 250)),
            Detection(label="gun", confidence=0.85, bbox=(80, 210, 40, 10)),
            Detection(label="gun", confidence=0.82, bbox=(480, 210, 40, 10))
        ]
    else:
        return []

def test_advanced_event_detection():
    """Test advanced event detection with different scenarios."""
    print("Advanced Step 1 Event Detection Demo")
    print("=" * 50)
    
    # Initialize detector
    detector = FrameDetector(
        model_path="models/yolo11s.engine",  # This will fall back to simple mode
        input_size=640,
        classes=["person", "gun", "knife"],
        confidence_threshold=0.5,
        inference_mode="simple"  # Use simple mode for demo
    )
    
    scenarios = ["pointing", "fallen", "assault", "mass_shooter", "normal"]
    
    for scenario in scenarios:
        print(f"\n=== Testing {scenario.upper()} Scenario ===")
        
        # Create mock frame and detections
        frame = create_mock_frame_with_person_and_weapon(scenario)
        detections = create_mock_detections(scenario)
        
        # Analyze events with frame for pose analysis
        event_flags = detector.analyze_events(detections, frame)
        
        print(f"Detections found: {len(detections)}")
        for det in detections:
            print(f"  - {det.label}: {det.confidence:.2f} at {det.bbox}")
        
        print("Event flags:")
        for event, flag in event_flags.items():
            status = "🚨 DETECTED" if flag else "✅ Clear"
            print(f"  - {event}: {status}")
        
        # Calculate overall threat level
        threat_score = sum(event_flags.values()) / len(event_flags)
        threat_level = "HIGH" if threat_score > 0.5 else "MEDIUM" if threat_score > 0.2 else "LOW"
        
        print(f"Overall threat level: {threat_level} ({threat_score:.2f})")
        
        # Save frame for visual inspection (optional)
        output_path = f"demo_output_{scenario}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Saved demo frame: {output_path}")

def test_pose_analysis_components():
    """Test individual pose analysis components."""
    print("\n" + "=" * 50)
    print("Testing Individual Pose Analysis Components")
    print("=" * 50)
    
    detector = FrameDetector(
        model_path="dummy",
        input_size=640,
        classes=["person", "gun"],
        inference_mode="simple"
    )
    
    # Test extended arms detection
    pointing_frame = create_mock_frame_with_person_and_weapon("pointing")
    person_roi = pointing_frame[120:400, 200:280]  # Extract person region
    
    arms_extended = detector._detect_extended_arms(person_roi)
    print(f"Extended arms detected: {'YES' if arms_extended else 'NO'}")
    
    # Test body orientation
    orientation = detector._estimate_body_orientation(person_roi)
    print(f"Body orientation: {orientation}")
    
    # Test posture analysis
    posture = detector._analyze_vertical_posture(person_roi, 280, 80)
    print(f"Posture analysis: {posture}")
    
    # Test fall detection on horizontal person
    fallen_frame = create_mock_frame_with_person_and_weapon("fallen")
    fallen_roi = fallen_frame[310:350, 150:400]
    fallen_posture = detector._analyze_vertical_posture(fallen_roi, 40, 250)
    print(f"Fallen person posture: {fallen_posture}")

def main():
    """Run all advanced detection tests."""
    test_advanced_event_detection()
    test_pose_analysis_components()
    
    print(f"\n{'='*50}")
    print("Demo Complete!")
    print("Check the generated demo_output_*.jpg files to see the test scenarios.")
    print("In a real deployment, these would be actual camera frames with YOLO detections.")

if __name__ == "__main__":
    main()
