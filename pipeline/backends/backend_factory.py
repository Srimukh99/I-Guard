"""
Backend factory for automatic selection and fallback management.

Handles intelligent selection between Python and DeepStream backends based on
system capabilities, configuration preferences, and runtime requirements.
"""

import os
import platform
from typing import Dict, Any, Optional, Type
import logging

from .base_backend import BaseBackend
from .python_backend import PythonBackend

# Conditional DeepStream import
try:
    from .deepstream_backend import DeepStreamBackend
    DEEPSTREAM_AVAILABLE = True
except ImportError:
    DEEPSTREAM_AVAILABLE = False
    DeepStreamBackend = None


class BackendFactory:
    """Factory for creating and managing backend instances."""
    
    def __init__(self):
        """Initialize backend factory."""
        self.logger = logging.getLogger(__name__)
        self._available_backends = self._discover_backends()
    
    def _discover_backends(self) -> Dict[str, Type[BaseBackend]]:
        """Discover available backends on the system."""
        backends = {
            'python': PythonBackend
        }
        
        # Check DeepStream availability
        if DEEPSTREAM_AVAILABLE and self._check_deepstream_requirements():
            backends['deepstream'] = DeepStreamBackend
            self.logger.info("DeepStream backend available")
        else:
            self.logger.info("DeepStream backend not available")
        
        return backends
    
    def _check_deepstream_requirements(self) -> bool:
        """Check if DeepStream requirements are met."""
        # Check platform
        if platform.system() != 'Linux':
            return False
        
        # Check for NVIDIA GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return False
            
            # Check for DeepStream installation
            deepstream_paths = [
                '/opt/nvidia/deepstream',
                '/usr/local/deepstream',
                os.path.expanduser('~/deepstream')
            ]
            
            for path in deepstream_paths:
                if os.path.exists(path):
                    return True
            
            return False
            
        except ImportError:
            return False
        except Exception:
            return False
    
    def get_available_backends(self) -> list[str]:
        """Get list of available backend names."""
        return list(self._available_backends.keys())
    
    def create_backend(self, config: Dict[str, Any]) -> BaseBackend:
        """
        Create backend instance based on configuration and system capabilities.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Backend instance
            
        Raises:
            RuntimeError: If no suitable backend is available
        """
        backend_config = config.get('backend', {})
        preferred_backend = backend_config.get('type', 'auto')
        
        if preferred_backend == 'auto':
            backend_name = self._select_best_backend(config)
        else:
            backend_name = preferred_backend
        
        # Validate backend availability
        if backend_name not in self._available_backends:
            # Fallback logic
            fallback_backend = self._get_fallback_backend(backend_name)
            if fallback_backend:
                self.logger.warning(
                    f"Requested backend '{backend_name}' not available, "
                    f"falling back to '{fallback_backend}'"
                )
                backend_name = fallback_backend
            else:
                raise RuntimeError(f"No suitable backend available (requested: {backend_name})")
        
        # Create backend instance
        backend_class = self._available_backends[backend_name]
        backend = backend_class(config)
        
        self.logger.info(f"Created {backend_name} backend")
        return backend
    
    def _select_best_backend(self, config: Dict[str, Any]) -> str:
        """
        Automatically select the best backend based on requirements.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Backend name
        """
        # Check requirements
        stream_count = config.get('streams', {}).get('max_concurrent', 4)
        performance_priority = config.get('performance', {}).get('priority', 'balanced')
        
        # Decision logic
        if stream_count > 8 and 'deepstream' in self._available_backends:
            # High stream count - prefer DeepStream if available
            return 'deepstream'
        
        if performance_priority == 'high' and 'deepstream' in self._available_backends:
            # High performance priority - prefer DeepStream
            return 'deepstream'
        
        if performance_priority == 'compatibility':
            # Compatibility priority - prefer Python
            return 'python'
        
        # Default selection based on availability
        if 'deepstream' in self._available_backends:
            return 'deepstream'
        else:
            return 'python'
    
    def _get_fallback_backend(self, requested_backend: str) -> Optional[str]:
        """
        Get fallback backend if requested backend is not available.
        
        Args:
            requested_backend: Name of requested backend
            
        Returns:
            Fallback backend name or None
        """
        # Fallback priority order
        fallback_map = {
            'deepstream': 'python',  # DeepStream falls back to Python
            'python': None,  # Python has no fallback
        }
        
        fallback = fallback_map.get(requested_backend)
        if fallback and fallback in self._available_backends:
            return fallback
        
        # If specific fallback not available, try any available backend
        if self._available_backends:
            return list(self._available_backends.keys())[0]
        
        return None
    
    def validate_backend_config(self, backend_name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for specific backend.
        
        Args:
            backend_name: Name of backend to validate
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        if backend_name not in self._available_backends:
            return False
        
        backend_class = self._available_backends[backend_name]
        
        try:
            # Create temporary instance to check capabilities
            temp_backend = backend_class(config)
            capabilities = temp_backend.capabilities
            
            # Check stream count requirements
            max_streams = config.get('streams', {}).get('max_concurrent', 4)
            if max_streams > capabilities.max_streams:
                self.logger.warning(
                    f"Requested {max_streams} streams exceeds {backend_name} "
                    f"capability of {capabilities.max_streams}"
                )
                return False
            
            # Check platform requirements
            current_platform = platform.system().lower()
            required_platforms = [p.lower() for p in capabilities.platform_requirements]
            if required_platforms and current_platform not in required_platforms:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating {backend_name} config: {e}")
            return False
    
    def get_backend_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comparison of available backends.
        
        Returns:
            Dictionary with backend capabilities comparison
        """
        comparison = {}
        
        for name, backend_class in self._available_backends.items():
            try:
                # Create temporary instance to get capabilities
                temp_backend = backend_class({})
                capabilities = temp_backend.capabilities
                
                comparison[name] = {
                    'max_streams': capabilities.max_streams,
                    'gpu_required': capabilities.gpu_required,
                    'platform_requirements': capabilities.platform_requirements,
                    'performance_tier': capabilities.performance_tier,
                    'memory_usage': capabilities.memory_usage,
                    'cpu_intensive': capabilities.cpu_intensive,
                    'available': capabilities.is_available()
                }
                
            except Exception as e:
                comparison[name] = {
                    'error': str(e),
                    'available': False
                }
        
        return comparison
    
    def create_backend_with_fallback(self, config: Dict[str, Any]) -> tuple[BaseBackend, str]:
        """
        Create backend with automatic fallback and return both backend and selected type.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (backend_instance, backend_name)
            
        Raises:
            RuntimeError: If no backend can be created
        """
        backend_config = config.get('backend', {})
        preferred_backend = backend_config.get('type', 'auto')
        
        # Try preferred backend first
        try:
            backend = self.create_backend(config)
            if backend.initialize():
                # Determine which backend was actually selected
                actual_backend_name = backend.__class__.__name__.lower().replace('backend', '')
                return backend, actual_backend_name
            else:
                backend.cleanup()
                raise RuntimeError("Backend initialization failed")
                
        except Exception as e:
            self.logger.warning(f"Failed to create preferred backend: {e}")
        
        # Try fallback backends
        for backend_name in self._available_backends.keys():
            if backend_name == preferred_backend:
                continue  # Already tried
            
            try:
                fallback_config = config.copy()
                fallback_config['backend'] = {'type': backend_name}
                
                backend = self.create_backend(fallback_config)
                if backend.initialize():
                    self.logger.info(f"Successfully fell back to {backend_name} backend")
                    return backend, backend_name  # Return the actual backend name used
                else:
                    backend.cleanup()
                    
            except Exception as e:
                self.logger.warning(f"Fallback to {backend_name} failed: {e}")
                continue
        
        raise RuntimeError("No backend could be initialized successfully")
