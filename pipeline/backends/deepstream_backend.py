"""
DeepStream backend implementation for high-performance NVIDIA GPU processing.

Leverages NVIDIA DeepStream SDK with GStreamer pipelines for maximum
throughput on supported hardware (Tesla, RTX, Jetson platforms).
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import Gst, GLib, GstRtspServer
    import pyds
    DEEPSTREAM_AVAILABLE = True
except ImportError:
    DEEPSTREAM_AVAILABLE = False

from .base_backend import BaseBackend, BackendCapabilities, Detection


class DeepStreamBackend(BaseBackend):
    """DeepStream-based backend for high-performance GPU processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DeepStream backend."""
        if not DEEPSTREAM_AVAILABLE:
            raise ImportError("DeepStream dependencies not available")
        
        super().__init__(config)
        
        # GStreamer pipeline components
        self.pipeline = None
        self.loop = None
        self.source = None
        self.streammux = None
        self.pgie = None
        self.nvvidconv = None
        self.nvosd = None
        self.sink = None
        
        # Detection results
        self._current_detections = []
        self._frame_meta_list = []
        
        # Performance tracking
        self._processing_times = []
        self._frame_count = 0
        self._stream_count = 0
    
    def _get_capabilities(self) -> BackendCapabilities:
        """Return DeepStream backend capabilities."""
        return BackendCapabilities(
            max_streams=20,  # High throughput with GPU acceleration
            gpu_required=True,
            platform_requirements=['linux', 'nvidia-gpu'],
            performance_tier='high',
            memory_usage='high',
            cpu_intensive=False
        )
    
    def _check_system_requirements(self) -> bool:
        """Check if system meets DeepStream requirements."""
        if not DEEPSTREAM_AVAILABLE:
            return False
        
        # Check for NVIDIA GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            return device_count > 0
        except:
            return False
    
    def initialize(self) -> bool:
        """Initialize DeepStream pipeline."""
        if not self._check_system_requirements():
            print("DeepStream system requirements not met")
            return False
        
        try:
            # Initialize GStreamer
            Gst.init(None)
            
            # Create pipeline
            self.pipeline = Gst.Pipeline()
            if not self.pipeline:
                raise RuntimeError("Unable to create Pipeline")
            
            # Create pipeline elements
            self._create_pipeline_elements()
            
            # Link pipeline elements
            self._link_pipeline_elements()
            
            # Set up probe functions
            self._setup_probes()
            
            # Create event loop
            self.loop = GLib.MainLoop()
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize DeepStream backend: {e}")
            return False
    
    def _create_pipeline_elements(self):
        """Create GStreamer pipeline elements."""
        # Create elements
        self.source = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        
        # Check element creation
        elements = [self.source, self.streammux, self.pgie, self.nvvidconv, self.nvosd, self.sink]
        if not all(elements):
            raise RuntimeError("Unable to create some pipeline elements")
        
        # Configure elements
        self._configure_elements()
        
        # Add elements to pipeline
        for element in elements:
            self.pipeline.add(element)
    
    def _configure_elements(self):
        """Configure pipeline elements with settings."""
        # Configure stream muxer
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', self.config.get('batch_size', 4))
        self.streammux.set_property('batched-push-timeout', 4000000)
        
        # Configure primary inference engine
        model_config = self.config.get('model', {})
        config_file = model_config.get('config_file', 'configs/deepstream_yolo_config.txt')
        self.pgie.set_property('config-file-path', config_file)
        
        # Configure video converter
        self.nvvidconv.set_property('nvbuf-memory-type', 0)
        
        # Configure on-screen display
        self.nvosd.set_property('process-mode', 1)
        self.nvosd.set_property('display-text', 1)
    
    def _link_pipeline_elements(self):
        """Link pipeline elements together."""
        # Link source to stream muxer
        sinkpad = self.streammux.get_request_pad("sink_0")
        if not sinkpad:
            raise RuntimeError("Unable to get sink pad")
        
        srcpad = self.source.get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Unable to link source to stream muxer")
        
        # Link remaining elements
        if not self.streammux.link(self.pgie):
            raise RuntimeError("Unable to link stream muxer to pgie")
        
        if not self.pgie.link(self.nvvidconv):
            raise RuntimeError("Unable to link pgie to nvvidconv")
        
        if not self.nvvidconv.link(self.nvosd):
            raise RuntimeError("Unable to link nvvidconv to nvosd")
        
        if not self.nvosd.link(self.sink):
            raise RuntimeError("Unable to link nvosd to sink")
    
    def _setup_probes(self):
        """Set up probe functions for detection extraction."""
        # Add probe to extract detections
        pgie_src_pad = self.pgie.get_static_pad("src")
        if not pgie_src_pad:
            raise RuntimeError("Unable to get src pad from pgie")
        
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._pgie_src_pad_buffer_probe, 0)
    
    def _pgie_src_pad_buffer_probe(self, pad, info, user_data):
        """Extract detection metadata from GStreamer buffer."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        # Get batch metadata
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        current_detections = []
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                frame_number = frame_meta.frame_num
                
                # Get object metadata
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        
                        # Convert to standardized detection format
                        detection = Detection(
                            class_id=obj_meta.class_id,
                            confidence=obj_meta.confidence,
                            bbox=(
                                obj_meta.rect_params.left / frame_meta.source_frame_width,
                                obj_meta.rect_params.top / frame_meta.source_frame_height,
                                (obj_meta.rect_params.left + obj_meta.rect_params.width) / frame_meta.source_frame_width,
                                (obj_meta.rect_params.top + obj_meta.rect_params.height) / frame_meta.source_frame_height
                            ),
                            class_name=obj_meta.obj_label if obj_meta.obj_label else f"class_{obj_meta.class_id}",
                            timestamp=time.time(),
                            frame_id=frame_number
                        )
                        current_detections.append(detection)
                        
                    except StopIteration:
                        break
                    
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                
            except StopIteration:
                break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        # Store detections for retrieval
        self._current_detections = current_detections
        self._frame_count += len([d for d in current_detections])
        
        return Gst.PadProbeReturn.OK
    
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        """
        Process single frame through DeepStream pipeline.
        Note: DeepStream is optimized for streaming, so this is less efficient.
        """
        # For single frame processing, we'll need to push frame into pipeline
        # This is a simplified implementation - real implementation would require
        # more complex buffer management
        
        if not self.is_initialized:
            raise RuntimeError("Backend not initialized")
        
        if not self.validate_frame(frame):
            return []
        
        # Clear previous detections
        self._current_detections = []
        
        # TODO: Implement frame injection into pipeline
        # This would require creating an appsrc element and pushing the frame
        
        # For now, return empty list as this requires more complex implementation
        return []
    
    def process_batch(self, frames: List[np.ndarray], frame_ids: List[int], 
                     timestamps: List[float]) -> List[List[Detection]]:
        """Process batch of frames through DeepStream pipeline."""
        if not self.is_initialized:
            raise RuntimeError("Backend not initialized")
        
        # DeepStream is designed for batch processing
        # This would require implementing batch buffer injection
        
        results = []
        for frame, frame_id, timestamp in zip(frames, frame_ids, timestamps):
            frame_detections = self.process_frame(frame, frame_id, timestamp)
            results.append(frame_detections)
        
        return results
    
    def start_streaming(self, sources: List[str]) -> bool:
        """Start DeepStream pipeline for streaming processing."""
        if not self.is_initialized:
            return False
        
        try:
            # Set pipeline to playing state
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Unable to set pipeline to playing state")
            
            return True
            
        except Exception as e:
            print(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop DeepStream pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
    
    def cleanup(self) -> None:
        """Clean up DeepStream resources."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        
        if self.loop:
            if self.loop.is_running():
                self.loop.quit()
            self.loop = None
        
        self._is_initialized = False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get DeepStream performance metrics."""
        base_metrics = super().get_performance_metrics()
        
        deepstream_metrics = {
            'frames_processed': self._frame_count,
            'active_streams': self._stream_count,
            'pipeline_state': 'running' if self.pipeline else 'stopped',
            'gpu_accelerated': True,
            'batch_processing': True
        }
        
        return {**base_metrics, **deepstream_metrics}
