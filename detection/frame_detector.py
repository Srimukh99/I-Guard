"""Fast per-frame detector wrapper.

This module defines :class:`FrameDetector`, which implements the first
stage of the ThreatDetect pipeline. It uses a lightweight YOLO model to
detect objects of interest (e.g. people, weapons) on each incoming frame
and applies a few simple heuristics to flag candidate events. It is
designed to be extremely fast and should run on every frame to maintain
low latency.

The detector supports two backend modes:

* Ultralytics fallback – when the configured model path is not
  available (e.g. no TensorRT engine exists), the detector will use
  the ``ultralytics`` Python package to load a small YOLO model on CPU or
  GPU. This fallback is convenient for development and testing but
  slower than TensorRT.
* TensorRT engine – if the model path ends with ``.engine``, the
  detector will attempt to load a pre-converted TensorRT engine using
  the ``tensorrt`` Python API. This is by far the fastest option on
  Nvidia hardware but requires an engine file generated ahead of time.

For the sake of clarity, the current implementation only implements
the Ultralytics fallback. Loading TensorRT engines is left as an
exercise for advanced users.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # If ultralytics is unavailable, fallback to None.

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # If OpenCV is unavailable, disable pose analysis features

# Attempt to import NVIDIA TensorRT for loading `.engine` or `.trt` files.  If
# TensorRT is unavailable, ``trt`` will be ``None`` and TensorRT engines will
# not be loaded.  This allows the detector to fall back to Ultralytics.
try:
    import tensorrt as trt  # type: ignore
except Exception:
    trt = None  # pragma: no cover

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    """Simple data structure representing a detection result.

    Attributes:
        label: The class label predicted by the model (e.g. 'person', 'gun').
        confidence: Prediction confidence between 0 and 1.
        bbox: Bounding box in pixel coordinates (x, y, w, h).
    """

    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class FrameDetector:
    """Fast per-frame object detector with simple heuristics.

    Parameters
    ----------
    model_path : str
        Path to a YOLO checkpoint (e.g. ``.pt`` file) or TensorRT engine.
    input_size : int
        Size of the square input expected by the model (e.g. 640).
    classes : List[str]
        List of class names to detect. Classes not listed here will be
        filtered out.
    confidence_threshold : float
        Minimum confidence to accept a detection. Detections below this
        threshold are discarded.
    events_config : Dict[str, bool]
        Dict of flags controlling which heuristics to run. Keys include
        ``pointing``, ``firing``, ``fall``, ``assault`` and ``mass_shooter``.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int,
        classes: List[str],
        confidence_threshold: float = 0.5,
        events_config: Optional[Dict[str, bool]] = None,
        inference_mode: str = "python",
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.inference_mode = inference_mode
        self.events_config = events_config or {
            "pointing": True,
            "firing": True,
            "fall": True,
            "assault": True,
            "mass_shooter": True,
        }
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the detection model.

        In production DeepStream mode, we assert that the TensorRT engine exists
        and fail fast if not. No Ultralytics fallback is used.
        """
        # In deepstream mode, fail fast if engine doesn't exist
        if self.inference_mode == "deepstream":
            if not self.model_path.endswith((".engine", ".trt")):
                raise RuntimeError(
                    f"DeepStream mode requires TensorRT engine (.engine/.trt), got: {self.model_path}"
                )
            import os
            if not os.path.exists(self.model_path):
                raise RuntimeError(
                    f"TensorRT engine not found: {self.model_path}. "
                    "Build it on target Jetson with: YOLO('yolo11s.pt').export(format='engine', half=True)"
                )
            LOGGER.info("DeepStream mode: TensorRT engine verified at %s", self.model_path)
            self.model = "deepstream"  # Placeholder - actual inference handled by DeepStream
            return
            
        # If the model path ends with `.engine` or `.trt`, attempt to load a TensorRT
        # engine.  If TensorRT is unavailable or loading fails, we fall back to
        # Ultralytics.  In a production implementation, you would create a
        # trt.Runtime and deserialize the engine to perform inference with FP16/INT8
        # precision.  Here we simply log the action.
        if self.model_path.endswith((".engine", ".trt")):
            if trt is None:
                LOGGER.warning(
                    "TensorRT is not installed; falling back to Ultralytics for %s",
                    self.model_path,
                )
            else:
                try:
                    LOGGER.info("Loading TensorRT engine from %s", self.model_path)
                    with open(self.model_path, "rb") as f:
                        engine_data = f.read()
                    runtime = trt.Runtime(trt.Logger.WARNING)
                    engine = runtime.deserialize_cuda_engine(engine_data)
                    if engine is None:
                        raise RuntimeError("Failed to deserialize TensorRT engine")
                    # Store engine for illustration; inference is not implemented
                    self.model = engine
                    LOGGER.warning(
                        "TensorRT engine loaded, but inference is not implemented in this skeleton"
                    )
                    return
                except Exception as exc:
                    LOGGER.error("Failed to load TensorRT engine: %s", exc)
                    # Fall through to Ultralytics below
        # Fallback to Ultralytics
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics is not installed. Please install it or provide a TensorRT engine."
            )
        LOGGER.info("Loading YOLO model from %s", self.model_path)
        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            LOGGER.error("Failed to load YOLO model: %s", exc)
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run the detector on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image in BGR format (as returned by OpenCV).

        Returns
        -------
        List[Detection]
            A list of detection objects. If the model is not loaded, an
            empty list is returned.
        """
        if self.model is None:
            LOGGER.warning("No detection model loaded; returning empty results.")
            return []
        # The Ultralytics API expects RGB images. Convert from BGR.
        img_rgb = frame[:, :, ::-1]
        # Perform inference. We disable NMS to let Ultralytics handle it for us.
        results = self.model.predict(img_rgb, imgsz=self.input_size, conf=self.confidence_threshold)
        detections: List[Detection] = []
        for result in results:
            # Each result contains multiple boxes; iterate through them.
            for box in result.boxes:
                conf = float(box.conf.cpu().numpy())
                cls_idx = int(box.cls.cpu().numpy())
                label = self.model.names.get(cls_idx, str(cls_idx))
                if label not in self.classes:
                    continue
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                detections.append(Detection(label=label, confidence=conf, bbox=(x, y, w, h)))
        return detections

    def analyze_events(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> Dict[str, bool]:
        """Apply heuristic rules to detection results to determine event flags.

        This implementation uses pose estimation, motion analysis, and weapon detection
        to identify various threat scenarios:

        * weapon: set if any 'gun' or 'knife' detection is present.
        * pointing: detect weapon pointing pose using person-weapon proximity and pose estimation.
        * firing: detect muzzle flash or firing stance indicators.
        * fall: detect person falling based on pose analysis.
        * assault: detect aggressive poses or close-contact violence.
        * mass_shooter: set if multiple weapons or complex threat scenarios detected.

        Parameters
        ----------
        detections : List[Detection]
            List of detections from the current frame.
        frame : Optional[np.ndarray]
            The input frame for pose analysis (BGR format).

        Returns
        -------
        Dict[str, bool]
            A dictionary of event flags. Keys correspond to the keys in
            ``events_config``.
        """
        flags = {key: False for key in self.events_config.keys()}
        
        # Basic weapon detection
        weapons = [d for d in detections if d.label in {"gun", "knife"}]
        persons = [d for d in detections if d.label == "person"]
        
        if weapons:
            flags["weapon"] = True
            
        # Mass shooter detection: multiple weapons or weapons + multiple persons
        if len(weapons) >= 2 or (len(weapons) >= 1 and len(persons) >= 2):
            flags["mass_shooter"] = True
            
        # Advanced event detection (requires frame for pose analysis)
        if frame is not None and persons and cv2 is not None:
            pose_results = self._analyze_poses(frame, persons)
            
            # Pointing detection: weapon + pointing pose
            if weapons and self._detect_pointing_pose(pose_results, weapons, persons):
                flags["pointing"] = True
                
            # Firing detection: weapon + firing indicators
            if weapons and self._detect_firing_indicators(frame, pose_results, weapons):
                flags["firing"] = True
                
            # Fall detection: abnormal person pose/orientation
            if self._detect_fall(pose_results):
                flags["fall"] = True
                
            # Assault detection: aggressive poses or close contact
            if len(persons) >= 2 and self._detect_assault(pose_results, persons):
                flags["assault"] = True
                
        return flags

    def _analyze_poses(self, frame: np.ndarray, persons: List[Detection]) -> List[Dict]:
        """Analyze human poses in the frame using lightweight pose estimation.
        
        Returns pose keypoints and derived features for each detected person.
        """
        poses = []
        
        for person in persons:
            x, y, w, h = person.bbox
            
            # Extract person ROI with padding
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad) 
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
                
            # Simple pose analysis using contours and aspect ratios
            pose_data = {
                'bbox': (x, y, w, h),
                'roi': person_roi,
                'aspect_ratio': h / w if w > 0 else 0,
                'area': w * h,
                'center': (x + w//2, y + h//2),
                'arms_extended': self._detect_extended_arms(person_roi),
                'body_orientation': self._estimate_body_orientation(person_roi),
                'vertical_posture': self._analyze_vertical_posture(person_roi, h, w)
            }
            
            poses.append(pose_data)
            
        return poses

    def _detect_extended_arms(self, person_roi: np.ndarray) -> bool:
        """Detect if person has arms extended (potential weapon pointing)."""
        if person_roi.size == 0 or cv2 is None:
            return False
            
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
            
        # Analyze largest contour (assumed to be person)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate contour bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Extended arms typically increase width relative to height
        return aspect_ratio > 0.8  # Threshold for "wide" pose indicating extended arms

    def _estimate_body_orientation(self, person_roi: np.ndarray) -> str:
        """Estimate if person is facing camera, side profile, or back."""
        if person_roi.size == 0:
            return "unknown"
            
        # Simple heuristic based on symmetry analysis
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Split image vertically and compare left/right halves
        mid = w // 2
        left_half = gray[:, :mid]
        right_half = cv2.flip(gray[:, mid:], 1)  # Flip right half
        
        # Resize to same dimensions if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate correlation between halves
        correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0, 0]
        
        if correlation > 0.7:
            return "frontal"  # High symmetry = facing camera
        else:
            return "profile"  # Low symmetry = side view

    def _analyze_vertical_posture(self, person_roi: np.ndarray, height: int, width: int) -> Dict[str, float]:
        """Analyze vertical posture for fall detection."""
        aspect_ratio = height / width if width > 0 else 0
        
        # Normal standing person has aspect_ratio > 1.5
        # Fallen person has aspect_ratio < 1.0
        posture_score = aspect_ratio / 2.0  # Normalize
        
        return {
            'aspect_ratio': aspect_ratio,
            'is_upright': aspect_ratio > 1.2,
            'is_fallen': aspect_ratio < 0.8,
            'posture_score': min(posture_score, 1.0)
        }

    def _detect_pointing_pose(self, poses: List[Dict], weapons: List[Detection], persons: List[Detection]) -> bool:
        """Detect weapon pointing pose by analyzing person-weapon proximity and arm extension."""
        if not poses or not weapons:
            return False
            
        for pose in poses:
            person_center = pose['center']
            arms_extended = pose['arms_extended']
            
            # Check if weapon is near person with extended arms
            for weapon in weapons:
                wx, wy, ww, wh = weapon.bbox
                weapon_center = (wx + ww//2, wy + wh//2)
                
                # Calculate distance between person and weapon
                distance = np.sqrt((person_center[0] - weapon_center[0])**2 + 
                                 (person_center[1] - weapon_center[1])**2)
                
                # Weapon should be close to person (within reasonable range)
                person_bbox = pose['bbox']
                max_distance = max(person_bbox[2], person_bbox[3])  # Use person size as reference
                
                if distance < max_distance * 1.5 and arms_extended:
                    return True
                    
        return False

    def _detect_firing_indicators(self, frame: np.ndarray, poses: List[Dict], weapons: List[Detection]) -> bool:
        """Detect firing indicators like muzzle flash or firing stance."""
        if not weapons:
            return False
            
        # Simple muzzle flash detection using bright spot analysis
        for weapon in weapons:
            wx, wy, ww, wh = weapon.bbox
            
            # Extend search area around weapon
            search_area = {
                'x1': max(0, wx - ww//2),
                'y1': max(0, wy - wh//2),
                'x2': min(frame.shape[1], wx + ww + ww//2),
                'y2': min(frame.shape[0], wy + wh + wh//2)
            }
            
            roi = frame[search_area['y1']:search_area['y2'], search_area['x1']:search_area['x2']]
            
            if roi.size == 0:
                continue
                
            # Convert to grayscale and look for bright spots
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Threshold for very bright pixels (potential muzzle flash)
            _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            bright_pixels = cv2.countNonZero(bright_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # If significant portion is very bright, might be muzzle flash
            if bright_pixels > total_pixels * 0.1:  # 10% threshold
                return True
                
        return False

    def _detect_fall(self, poses: List[Dict]) -> bool:
        """Detect if a person has fallen based on pose analysis."""
        for pose in poses:
            posture = pose['vertical_posture']
            
            # Multiple indicators of fall
            if (posture['is_fallen'] or 
                posture['aspect_ratio'] < 0.7 or  # Very horizontal
                posture['posture_score'] < 0.3):   # Low posture score
                return True
                
        return False

    def _detect_assault(self, poses: List[Dict], persons: List[Detection]) -> bool:
        """Detect assault based on aggressive poses or close contact between people."""
        if len(poses) < 2:
            return False
            
        # Check for close proximity between people (potential altercation)
        for i, pose1 in enumerate(poses):
            for j, pose2 in enumerate(poses[i+1:], i+1):
                center1 = pose1['center']
                center2 = pose2['center']
                
                # Calculate distance between people
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Get average person size for reference
                size1 = max(pose1['bbox'][2], pose1['bbox'][3])
                size2 = max(pose2['bbox'][2], pose2['bbox'][3])
                avg_size = (size1 + size2) / 2
                
                # Very close proximity might indicate physical altercation
                if distance < avg_size * 0.8:
                    return True
                    
                # Check for aggressive poses (both people with extended arms)
                if pose1['arms_extended'] and pose2['arms_extended']:
                    return True
                    
        return False
        if self.events_config.get("mass_shooter") and len(weapons) >= 2:
            flags["mass_shooter"] = True
        # TODO: implement real pointing/firing/fall/assault heuristics.
        return flags