"""Detection module package.

This package exposes two stages of detection:

* :mod:`step1_detector` provides a fast, per‑frame detector for weapons and
  suspicious behaviour.  It should be used on every frame to flag
  candidate events.
* :mod:`step2_verifier` refines candidate events by analysing a short
  sequence of frames with a larger model.  This stage is optional and can
  be disabled in the configuration.

Tracking or re‑identification utilities live in :mod:`tracking` and can be
used to assign unique IDs to detected objects across frames.

The code is intended to run on Nvidia GPUs with TensorRT for best
performance, but pure PyTorch or ONNXRuntime fallbacks are possible.
"""

from .step1_detector import Step1Detector
from .step2_verifier import Step2Verifier
from .tracking import TrackManager

__all__ = [
    "Step1Detector",
    "Step2Verifier",
    "TrackManager",
]