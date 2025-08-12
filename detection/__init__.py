"""Detection module package.

This package exposes two stages of detection:

* :mod:`frame_detector` provides a fast, per‑frame detector for weapons and
  suspicious behaviour.  It should be used on every frame to flag
  candidate events.
* :mod:`clip_verifier` refines candidate events by analysing a short
  sequence of frames with a larger model.  This stage is optional and can
  be disabled in the configuration.

Tracking or re‑identification utilities live in :mod:`tracking` and can be
used to assign unique IDs to detected objects across frames.

The code is intended to run on Nvidia GPUs with TensorRT for best
performance, but pure PyTorch or ONNXRuntime fallbacks are possible.
"""

from .frame_detector import FrameDetector
from .clip_verifier import ClipVerifier
from .tracking import TrackManager

# Backwards compatible aliases are available via deprecated shim modules
# detection.step1_detector.Step1Detector -> detection.frame_detector.FrameDetector
# detection.step2_verifier.Step2Verifier -> detection.clip_verifier.ClipVerifier

__all__ = [
    "FrameDetector",
    "ClipVerifier",
    "TrackManager",
]