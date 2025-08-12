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
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.classes = classes
        self.confidence_threshold = confidence_threshold
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

        For now only the Ultralytics fallback is implemented. If the
        configured model path ends with ``.engine``, you should implement
        loading via TensorRT. Until then, the Ultralytics YOLO API will
        automatically download a small model if ``model_path`` is ``yolov8n.pt``
        or similar.
        """
        if self.model_path.endswith(".engine"):
            LOGGER.warning(
                "TensorRT engine loading is not implemented in this open-source skeleton. "
                "Please convert your model to ONNX and load it via ultralytics instead."
            )
            self.model = None
        else:
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

    def analyze_events(self, detections: List[Detection]) -> Dict[str, bool]:
        """Apply heuristic rules to detection results to determine event flags.

        This basic implementation uses simple presence checks:

        * weapon: set if any 'gun' or 'knife' detection is present.
        * pointing: placeholder; always False.
        * firing: placeholder; always False.
        * fall: placeholder; always False.
        * assault: placeholder; always False.
        * mass_shooter: set if two or more weapons are present.

        You are encouraged to implement your own heuristics here. For
        example, use a pose estimator to detect outstretched arms, or
        combine detections across multiple frames to detect muzzle flashes.

        Parameters
        ----------
        detections : List[Detection]
            List of detections from the current frame.

        Returns
        -------
        Dict[str, bool]
            A dictionary of event flags. Keys correspond to the keys in
            ``events_config``.
        """
        flags = {key: False for key in self.events_config.keys()}
        weapons = [d for d in detections if d.label in {"gun", "knife"}]
        if weapons:
            flags["weapon"] = True
        if self.events_config.get("mass_shooter") and len(weapons) >= 2:
            flags["mass_shooter"] = True
        # TODO: implement real pointing/firing/fall/assault heuristics.
        return flags
