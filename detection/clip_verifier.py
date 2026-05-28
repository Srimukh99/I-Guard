"""Secondary verification stage (clip-based) with 3D CNN support.

Simple version for testing purposes.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Union, Any
import numpy as np

# Conditional import of GemmaVerifier
try:
    from .gemma_verifier import GemmaVerifier
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

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
        self.kwargs = kwargs

        # Initialize specialized verifiers if requested
        self.gemma_verifier = None
        if self.model_type == "gemma":
            if GEMMA_AVAILABLE:
                self.gemma_verifier = GemmaVerifier(
                    model_id=model_path if model_path else "google/paligemma-3b-pt-224",
                    threshold=threshold,
                    **kwargs
                )
            else:
                LOGGER.error("Gemma requested but GemmaVerifier not available")

        LOGGER.info(f"ClipVerifier initialized with {model_type} model")

    def verify(self, video_clip: Any = None, detections_per_frame: Optional[Iterable[List[str]]] = None) -> Dict[str, Any]:
        """Verify a candidate event using selected model."""
        if self.model_type == "gemma" and self.gemma_verifier:
            return self.gemma_verifier.verify(video_clip, detections_per_frame)

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
