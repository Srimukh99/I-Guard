"""Detection package for the I‑Guard project.

This package contains the modules responsible for performing
object detection and basic heuristics (Step 1) as well as
optional clip verification and tracking (Step 2 and beyond).

The classes defined in this package are imported in the
``pipeline`` package to orchestrate the inference process.
"""

from .frame_detector import FrameDetector  # noqa: F401
from .clip_verifier import ClipVerifier  # noqa: F401
from .tracking import TrackManager  # noqa: F401
