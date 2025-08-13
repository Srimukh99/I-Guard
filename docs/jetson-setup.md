# Jetson Setup Guide

This document explains how to set up an NVIDIA Jetson device to run I‑Guard, including installation of JetPack, DeepStream and TensorRT.

## Install JetPack

1. Flash or update your Jetson device with the latest JetPack release. This installs Ubuntu, CUDA, cuDNN and other necessary drivers.
2. Install the JetPack SDK by following the official NVIDIA documentation.

## Install DeepStream

DeepStream provides a hardware‑accelerated multimedia pipeline with plugins for decoding, inference and display.

1. Follow the NVIDIA DeepStream installation guide for your Jetson platform.
2. Install the Python bindings (`python3-pyds`) if you plan to use the DeepStream Python API.
3. Install system OpenCV via apt; do not use the `opencv-python` pip wheel on Jetson.

## Install TensorRT

TensorRT is included with JetPack, but you may need to install additional Python bindings:

```bash
sudo apt install python3-libnvinfer
```

## Prepare Models

Use Ultralytics or the TAO Toolkit to export your detection models to ONNX and then convert them to TensorRT engines (FP16 or INT8). Place the resulting `.engine` files in a directory referenced by `model_path` in your `config.yaml`.