# I-Guard DeepStream Setup Guide

This guide covers setting up I-Guard with NVIDIA DeepStream on Jetson hardware for optimal performance.

## Prerequisites

- NVIDIA Jetson (Orin Nano/NX/AGX or Xavier series)
- JetPack 5.x+ with DeepStream 6.x+ installed
- Python 3.8+

## Step 1: Build TensorRT Engine on Target Jetson

**Important:** Always build the engine on the target Jetson to avoid plan incompatibility.

```bash
# Install Ultralytics if not already installed
pip install ultralytics

# Export YOLO11s to TensorRT engine with FP16 precision
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='engine', half=True)"

# This creates yolo11s.engine in the current directory
# Move it to your models folder
mkdir -p models
mv yolo11s.engine models/
```

## Step 2: Configure DeepStream Pipeline

The repository includes optimized configurations:

- `configs/deepstream_yolo_config.txt` - PGIE configuration for YOLO11s
- `configs/nvtracker_iou.yml` - Fast IOU tracker configuration  
- `configs/labels.txt` - Class labels including weapons

Key settings for different Jetson models:

### Orin Nano (6-8 cameras)
- Batch size: 8
- Network mode: FP16 (mode=2)
- Tracker: IOU (fast)
- Frame skip: 0

### Orin NX/AGX (10-20 cameras)  
- Batch size: 16
- Network mode: FP16 or INT8 (mode=0)
- Tracker: NvDCF (robust) or IOU (fast)
- Consider DLA for Stage-2 models

## Step 3: Update Configuration

Edit `config.yaml`:

```yaml
inference_mode: deepstream  # Force DeepStream path only
pgie_config: "configs/deepstream_yolo_config.txt"
tracker_config: "configs/nvtracker_iou.yml"
enable_stage2: true

step1:
  model_path: "models/yolo11s.engine"
  frame_batch_size: 8  # Adjust based on hardware
```

## Step 4: Install Dependencies

```bash
# DeepStream Python bindings
sudo apt install python3-gi python3-gst-1.0 gir1.2-gst-1.0 gir1.2-gstreamer-1.0

# For development/testing (optional)
pip install opencv-python numpy pyyaml
```

## Step 5: Run I-Guard

```bash
python app.py
```

The system will:
1. Assert TensorRT engine exists (fail-fast if missing)
2. Initialize DeepStream pipeline with optimized settings
3. Process multiple camera streams with batching
4. Run Stage-2 verification on detected events

## Performance Optimization

### Memory Management
- Use zero-copy capture when possible
- Configure appropriate buffer sizes in streammux
- Monitor GPU memory usage

### Scaling Guidelines
- Start with conservative batch sizes
- Increase gradually while monitoring latency
- Use `nvidia-smi` to monitor GPU utilization
- Consider INT8 quantization for maximum throughput

### Debugging
- Enable DeepStream debug logs: `export GST_DEBUG=3`
- Monitor pipeline performance with `nvtop`
- Check for dropped frames in DeepStream analytics

## Advanced Configuration

For production deployments, consider:
- Custom NvDCF tracker configuration for better accuracy
- Multi-stream tiling for visualization  
- DLA acceleration for secondary models
- Custom CUDA kernels for specialized preprocessing
