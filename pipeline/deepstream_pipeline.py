"""DeepStream inference pipeline for I‑Guard.

This module provides a reference implementation of an end‑to‑end
NVIDIA DeepStream pipeline using the Python bindings (`gst-python`
and `pyds`).  It demonstrates how to process multiple RTSP or file
streams on Jetson in a zero‑copy fashion, run a TensorRT engine via
DeepStream's `nvinfer` plugin and emit weapon detection events back
into the I‑Guard event queue.  The goal of this script is to replace
the per‑frame Python detector in `FrameDetector` with a high
performance, fully offloaded pipeline that remains compatible with
the existing UI and event handling code.

Usage
-----

Run this module as a script, passing the path to your I‑Guard
configuration YAML.  The script reads the camera sources and model
engine paths from the YAML and constructs a DeepStream pipeline
accordingly.  If you wish to run it outside of the I‑Guard control
plane, you may instantiate the ``DeepStreamPipeline`` class and
register your own callback to handle detection events.

Prerequisites
-------------

* Jetson device with JetPack and DeepStream installed.
* Python bindings: ``python3-gst`` and ``python3-pyds`` installed
  via your package manager.
* A TensorRT engine for your detector (e.g. ``yolov8n.engine``)
  generated via Ultralytics or TAO and referenced in your
  ``config.yaml`` under ``step1.model_path``.

Note: DeepStream configuration and model calibration are outside the
scope of this script.  See ``docs/deepstream-integration.md`` for
guidance.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from typing import Callable, Dict, List, Optional

try:
    import gi  # type: ignore
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # type: ignore
    HAS_GSTREAMER = True
except ImportError:
    gi = None
    Gst = None 
    GLib = None
    HAS_GSTREAMER = False

try:
    import pyds  # type: ignore
    HAS_PYDS = True
except ImportError:
    pyds = None
    HAS_PYDS = False

from .event_queue import EventQueue, Event

# Initialise GStreamer once if available
if HAS_GSTREAMER:
    Gst.init(None)

LOGGER = logging.getLogger(__name__)


class DeepStreamPipeline:
    """Construct and run a DeepStream pipeline for multiple streams.

    Parameters
    ----------
    sources : List[str]
        List of RTSP/HTTP URLs or file paths describing the input streams.
        You may also specify a full GStreamer pipeline string per source
        (e.g. including rtspsrc, depay and decoder elements).  If a
        plain URL is provided, a simple decode chain is constructed.
    model_engine : str
        Path to a TensorRT engine file (.engine) for the weapon detector.
    config_file : str, optional
        Path to a DeepStream config file for the `nvinfer` plugin.  If
        provided, the plugin will be configured via this file instead
        of inline properties.  See ``configs/deepstream_yolo_config.txt``
        for an example.  If not provided, minimal properties must
        still be set on the plugin (via ``model_engine`` and
        ``batch-size`` attributes).
    event_queue : EventQueue
        Shared event queue to which detection events will be pushed.
    camera_ids : Optional[List[str]]
        Optional list of camera identifiers; defaults to sequential
        names (e.g. ``cam0``, ``cam1`` ...).  Each detection event
        includes the camera id for UI display.
    postprocess : Optional[Callable[[Dict], None]]
        Optional callback invoked for each detection event dictionary
        prior to adding it to the queue.  You may use this to
        implement custom filtering or transformation.

    Notes
    -----
    This class manages its own GStreamer loop in a background thread.
    When ``start()`` is called, the pipeline enters the PLAYING state
    and a GLib main loop is started in another thread.  Call
    ``stop()`` to gracefully tear down the pipeline and join the
    threads.  Event handling is performed in a pad probe attached to
    the ``nvinfer`` element.
    """

    def __init__(
        self,
        sources: List[str],
        model_engine: str,
        config_file: Optional[str],
        event_queue: EventQueue,
        camera_ids: Optional[List[str]] = None,
        postprocess: Optional[Callable[[Dict], None]] = None,
        batch_size: int = 1,
    ) -> None:
        self.sources = sources
        self.model_engine = model_engine
        self.config_file = config_file
        self.event_queue = event_queue
        self.camera_ids = camera_ids or [f"cam{i}" for i in range(len(sources))]
        self.postprocess = postprocess
        self.batch_size = batch_size
        # Fail fast if engine path is required and missing
        if not config_file and (not model_engine or not os.path.exists(model_engine)):
            raise FileNotFoundError(
                f"DeepStreamPipeline: model engine not found: {model_engine}. "
                "Build the TensorRT engine on the target Jetson and update config."
            )
        self.pipeline: Optional[Gst.Pipeline] = None
        self.loop: Optional[GLib.MainLoop] = None
        self.thread: Optional[threading.Thread] = None

    def _build_pipeline(self) -> Gst.Pipeline:
        # Create an empty pipeline
        pipeline = Gst.Pipeline.new("iguard-deepstream-pipeline")
        if not pipeline:
            raise RuntimeError("Failed to create GStreamer Pipeline")

        # Create streammux
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            raise RuntimeError("Unable to create nvstreammux element")
        streammux.set_property("batch-size", self.batch_size)
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("live-source", 1)
        streammux.set_property("batched-push-timeout", 40000)
        pipeline.add(streammux)

        # Build sources and link to streammux
        for i, uri in enumerate(self.sources):
            # If URI is a full pipeline string, use it as is by wrapping
            # with uridecodebin; otherwise create a simple decode chain
            source_bin = self._create_source_bin(i, uri)
            pipeline.add(source_bin)
            pad_name = f"sink_{i}".encode("utf-8")
            sink_pad = streammux.get_request_pad(f"sink_{i}")
            src_pad = source_bin.get_static_pad("src")
            if not src_pad or not sink_pad:
                raise RuntimeError(f"Failed to link source {i} to streammux")
            src_pad.link(sink_pad)

        # Create nvinfer element for inference
        infer = Gst.ElementFactory.make("nvinfer", "primary-infer")
        if not infer:
            raise RuntimeError("Could not create nvinfer element")
        if self.config_file:
            infer.set_property("config-file-path", self.config_file)
        else:
            infer.set_property("model-engine-file", self.model_engine)
            infer.set_property("batch-size", self.batch_size)
            # Additional properties (e.g., input-dims) may be set here
        pipeline.add(infer)

        # Optionally add nvtracker for object tracking
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if tracker:
            # Prefer external tracker config if provided via environment or config file path
            tracker_cfg_env = os.getenv("IGUARD_TRACKER_CONFIG")
            tracker_config = tracker_cfg_env or os.path.join(os.getcwd(), "configs", "nvtracker_iou.yml")
            if os.path.exists(tracker_config):
                tracker.set_property("ll-config-file", tracker_config)
            pipeline.add(tracker)

        # Add a converter and OSD for debugging (can be removed in headless)
        convert = Gst.ElementFactory.make("nvvideoconvert", "convert")
        osd = Gst.ElementFactory.make("nvdsosd", "osd")
        if not convert or not osd:
            raise RuntimeError("Failed to create nvvideoconvert or nvdsosd")
        pipeline.add(convert)
        pipeline.add(osd)

        # Create fakesink
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")
        sink.set_property("sync", False)
        pipeline.add(sink)

        # Link elements: streammux -> infer -> tracker (optional) -> convert -> osd -> sink
        if not streammux.link(infer):
            raise RuntimeError("Failed to link streammux to infer")
        if tracker:
            if not infer.link(tracker):
                raise RuntimeError("Failed to link infer to tracker")
            if not tracker.link(convert):
                raise RuntimeError("Failed to link tracker to convert")
        else:
            if not infer.link(convert):
                raise RuntimeError("Failed to link infer to convert")
        if not convert.link(osd):
            raise RuntimeError("Failed to link convert to osd")
        if not osd.link(sink):
            raise RuntimeError("Failed to link osd to sink")

        # Attach pad probe to primary infer to read detection metadata
        sink_pad = infer.get_static_pad("sink")
        if not sink_pad:
            raise RuntimeError("Unable to get infer sink pad")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_infer_buffer_probe)

        return pipeline

    def _create_source_bin(self, index: int, uri: str) -> Gst.Bin:
        """Create a source bin for the given URI.  The bin exposes a
        `src` pad which outputs NVMM buffers.  If the URI is a
        pipeline string (contains `!`), it is wrapped with
        `gst-launch-1.0` style syntax; otherwise a simple `uridecodebin`
        is used.
        """
        bin_name = f"source-bin-{index}"
        bin = Gst.Bin.new(bin_name)
        if not bin:
            raise RuntimeError("Failed to create source bin")

        if "!" in uri:
            # Parse the user‑supplied pipeline description
            src = Gst.parse_bin_from_description(uri, True)
        else:
            # Create uridecodebin which handles RTSP/file decoding
            src = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
            src.set_property("uri", uri)
            src.connect("pad-added", self._on_decode_pad_added, bin)
        bin.add(src)
        # Add ghost pad
        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        if not ghost_pad:
            raise RuntimeError("Failed to create ghost pad")
        bin.add_pad(ghost_pad)
        return bin

    def _on_decode_pad_added(self, decodebin: Gst.Element, pad: Gst.Pad, bin: Gst.Bin) -> None:
        """Callback when uridecodebin adds a pad.  We link to our
        downstream convertor to ensure output is in NVMM memory."""
        caps = pad.get_current_caps()
        if not caps:
            return
        structure_name = caps.get_structure(0).get_name()
        if structure_name.startswith("video"):
            queue = Gst.ElementFactory.make("queue", None)
            if not queue:
                return
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", None)
            caps_filter = Gst.ElementFactory.make("capsfilter", None)
            caps_filter.set_property(
                "caps",
                Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"),
            )
            bin.add(queue)
            bin.add(nvvidconv)
            bin.add(caps_filter)
            queue.sync_state_with_parent()
            nvvidconv.sync_state_with_parent()
            caps_filter.sync_state_with_parent()
            pad.link(queue.get_static_pad("sink"))
            queue.link(nvvidconv)
            nvvidconv.link(caps_filter)
            ghost_pad = bin.get_static_pad("src")
            caps_filter_src_pad = caps_filter.get_static_pad("src")
            ghost_pad.set_target(caps_filter_src_pad)

    def _on_infer_buffer_probe(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        """Pad probe function called for each batch pushed into the
        primary infer.  We extract detection metadata from the NvDsBatchMeta
        structures and push weapon events into the event queue."""
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        # Retrieve batch metadata from the Gst buffer
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        l_frame = batch_meta.frame_meta_list
        frame_index = 0
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            # Camera id from stream id (pad index matches camera index)
            cam_id = self.camera_ids[frame_meta.pad_index]
            # Iterate through object metadata
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                # Label ID corresponds to YOLO class index.  You must
                # ensure your labels file matches the engine.  Here we
                # treat class_id 1 as 'person' and class_id 2 as 'weapon'
                # for illustration; adjust according to your labels file.
                label_id = obj_meta.class_id
                # Example: treat class IDs > 0 as weapons
                if label_id > 0:
                    # Build event dictionary
                    event_dict: Dict[str, float] = {
                        "camera_id": cam_id,
                        "timestamp": frame_meta.ntp_timestamp / 1e9 if frame_meta.ntp_timestamp else 0.0,
                        "label": obj_meta.obj_label.decode("utf-8") if obj_meta.obj_label else str(label_id),
                        "confidence": float(obj_meta.confidence),
                        "bbox": [obj_meta.rect_params.left, obj_meta.rect_params.top, obj_meta.rect_params.width, obj_meta.rect_params.height],
                    }
                    # Optional postprocess hook
                    if self.postprocess:
                        try:
                            self.postprocess(event_dict)
                        except Exception as exc:
                            LOGGER.warning("Postprocess callback raised: %s", exc)
                    # Convert to Event and enqueue
                    event = Event(
                        camera_id=cam_id,
                        timestamp=event_dict["timestamp"],
                        frame_id=frame_index,
                        detections=event_dict,
                        flags={"weapon": True},
                    )
                    try:
                        self.event_queue.put_nowait(event)
                    except Exception:
                        LOGGER.warning("DeepStream event queue full; dropping event from %s", cam_id)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            frame_index += 1
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def start(self) -> None:
        """Build and run the pipeline in a separate thread."""
        if self.thread and self.thread.is_alive():
            LOGGER.warning("DeepStream pipeline is already running")
            return

        def _run() -> None:
            try:
                self.pipeline = self._build_pipeline()
                # Create GLib MainLoop
                self.loop = GLib.MainLoop()
                # Bus to handle messages
                bus = self.pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect("message", self._bus_call)
                # Set pipeline state
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    LOGGER.error("Unable to set the pipeline to the playing state")
                    return
                LOGGER.info("DeepStream pipeline started")
                self.loop.run()
            except Exception as exc:
                LOGGER.exception("DeepStream pipeline encountered an error: %s", exc)
            finally:
                # Clean up pipeline and loop
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
                if self.loop:
                    self.loop.quit()
                LOGGER.info("DeepStream pipeline stopped")

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the pipeline and join the thread."""
        if self.loop:
            self.loop.quit()
        if self.thread:
            self.thread.join(timeout=5.0)

    def _bus_call(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle GStreamer bus messages."""
        t = message.type
        if t == Gst.MessageType.EOS:
            LOGGER.info("End‑of‑stream reached on DeepStream pipeline")
            if self.loop:
                self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            LOGGER.error("Error on DeepStream pipeline: %s", err)
            if debug:
                LOGGER.error("Debug info: %s", debug)
            if self.loop:
                self.loop.quit()


if __name__ == "__main__":  # pragma: no cover
    import yaml
    from pipeline.event_queue import EventQueue

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.deepstream_pipeline <config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Determine camera sources
    cameras = cfg.get("cameras", [])
    sources = []
    cam_ids = []
    for cam in cameras:
        sources.append(str(cam.get("source", "")))
        cam_ids.append(cam.get("id") or f"cam{len(cam_ids)}")
    # Determine model engine
    model_engine = cfg.get("step1", {}).get("model_path")
    if not model_engine or not os.path.isfile(model_engine):
        raise RuntimeError(
            "step1.model_path must point to a TensorRT engine file for DeepStream usage"
        )
    # Optional config file for nvinfer
    ds_cfg = os.path.join(os.path.dirname(model_engine), "deepstream_yolo_config.txt")
    if not os.path.isfile(ds_cfg):
        ds_cfg = None
    # Create event queue
    ev_queue = EventQueue(maxsize=cfg.get("advanced", {}).get("queue_maxsize", 64))
    # Instantiate pipeline
    dsp = DeepStreamPipeline(
        sources=sources,
        model_engine=model_engine,
        config_file=ds_cfg,
        event_queue=ev_queue,
        camera_ids=cam_ids,
        batch_size=cfg.get("step1", {}).get("frame_batch_size", 8),
    )
    # Run pipeline; stop on SIGINT
    def sig_handler(signum, frame):
        dsp.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    dsp.start()
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass