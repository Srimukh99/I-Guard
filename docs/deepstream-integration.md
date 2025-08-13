# DeepStream Integration Notes

This document describes how to integrate I‑Guard with NVIDIA DeepStream to achieve zero‑copy, batched inference on Jetson devices.

## Current Status

The current Python implementation includes a `DeepStreamAdapter` which wraps OpenCV's GStreamer support to build a basic pipeline. This is a **preview** and does not yet achieve true zero‑copy NVMM buffers or full DeepStream features.

## Planned Enhancements

Future iterations should:

* Replace the Python detection loop with DeepStream's `nvinfer` plugin to run TensorRT engines directly on the GPU/DLA.
* Use `nvstreammux` to batch frames from multiple cameras.
* Implement `nvtracker` for built‑in object tracking.
* Use DeepStream's Python bindings (`pyds`) for event callbacks.

## Example GStreamer Source

When `advanced.use_deepstream` is enabled, each camera source in `config.yaml` should be specified as a complete GStreamer pipeline string. For example:

```
rtspsrc location=rtsp://<camera-ip>/stream ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM),format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink
```

This pipeline uses NVIDIA's hardware decoder (`nvv4l2decoder`) and converts the NVMM buffer to BGR for consumption by OpenCV. Replace the `location` with your camera's RTSP URL.