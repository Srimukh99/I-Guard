"""
Backend abstraction layer for I-Guard pipeline.

Supports both Python-based and DeepStream-based processing backends
with automatic fallback and configuration-driven selection.
"""

from .base_backend import BaseBackend, BackendCapabilities, Detection
from .python_backend import PythonBackend
from .backend_factory import BackendFactory

# DeepStream backend with graceful import
try:
    from .deepstream_backend import DeepStreamBackend
    __all__ = [
        'BaseBackend',
        'BackendCapabilities', 
        'Detection',
        'PythonBackend',
        'DeepStreamBackend',
        'BackendFactory'
    ]
except ImportError:
    __all__ = [
        'BaseBackend',
        'BackendCapabilities',
        'Detection', 
        'PythonBackend',
        'BackendFactory'
    ]
