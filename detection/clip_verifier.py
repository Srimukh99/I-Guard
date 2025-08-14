"""Secondary verification stage (clip-based) with 3D CNN support.

Simple version for testing purposes.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Union
import numpy as np

LOGGER = logging.getLogger(__name__)


class ClipVerifier:
    """Secondary clip-based verification with 3D CNN support."""

    def __init__(
        self, 
        model_path: str, 
        model_type: str = "simple",
        threshold: float = 0.7,
        **kwargs
    ) -> None:
        self.model_path = model_path
        self.model_type = model_type.lower() 
        self.threshold = threshold
        LOGGER.info(f"ClipVerifier initialized with {model_type} model")

    def verify(self, video_clip=None, detections_per_frame: Optional[Iterable[List[str]]] = None) -> Dict[str, float]:
        """Verify a candidate event using simple aggregation."""
        if detections_per_frame is None:
            return {"score": 0.0, "action": "no_data", "action_confidence": 0.0}
            
        total = 0
        weapon_frames = 0
        for labels in detections_per_frame:
            total += 1
            if any(label in {"gun", "knife", "weapon"} for label in labels):
                weapon_frames += 1
                
        score = float(weapon_frames) / float(total) if total else 0.0
        action = "weapon_detected" if score > self.threshold else "normal"
        
        return {
            "score": score,
            "action": action,
            "action_confidence": score,
            "weapon_ratio": score
        }
