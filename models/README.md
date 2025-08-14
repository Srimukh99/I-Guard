# 3D CNN Action Recognition Models for I-Guard Step 2

This directory contains configuration and setup for 3D CNN models used in Step 2 verification.

## Supported Models

### 1. TAO ActionRecognitionNet (Default - Recommended)

**Advantages:**
- Native NVIDIA DeepStream integration
- Optimized for Jetson hardware
- Easy TensorRT conversion via TAO Toolkit
- Production-ready with minimal setup

**Setup:**
```bash
# Option A: Train custom model with TAO Toolkit
tao action_recognition train -e specs/action_recognition_train.txt

# Option B: Use pre-trained TAO model
# Download from NGC catalog and convert to TensorRT
tao action_recognition export -m model.etlt -k your_key -e specs/export.txt

# Move to models directory
mv action_recognition.engine models/tao_action_recognition.engine
```

**DeepStream Integration:**
The TAO model integrates as a secondary GIE in DeepStream:
```
configs/sgie_action_recognition.txt  # Secondary inference config
```

### 2. X3D (Alternative - Research/Development)

**Advantages:**
- Facebook's state-of-the-art 3D CNN
- Multiple size variants (X3D-S, X3D-M, X3D-L)
- Good accuracy/efficiency balance
- Active research community

**Setup:**
```bash
# Install PyTorchVideo
pip install pytorchvideo

# Export to TensorRT (requires custom conversion)
python scripts/export_x3d_to_tensorrt.py --model x3d_s --output models/x3d_s.engine
```

**Configuration:**
```yaml
step2:
  model_type: "x3d"
  model_path: "models/x3d_s.engine"
```

### 3. Simple Aggregation (Fallback)

When 3D CNN models are not available, falls back to counting weapon detections across frames.

## Model Training Data

For custom training, prepare video clips with these action classes:

**TAO ActionRecognitionNet Classes:**
- `normal`: Regular, non-threatening behavior
- `threatening_with_weapon`: Person holding/brandishing weapon
- `pointing_weapon`: Aiming weapon at target
- `firing_weapon`: Discharging weapon
- `assault`: Physical violence without weapons
- `fighting`: Mutual combat/altercation

**X3D Classes (Simplified):**
- `normal`: Regular behavior
- `weapon_threat`: Any weapon-related threat
- `violent_action`: Physical violence
- `suspicious_behavior`: Unusual/concerning behavior

## Performance Benchmarks

| Model | Jetson Orin Nano | Jetson Orin NX | Jetson AGX Orin |
|-------|------------------|----------------|-----------------|
| TAO ActionRec | ~30 FPS | ~60 FPS | ~120 FPS |
| X3D-S | ~25 FPS | ~50 FPS | ~100 FPS |
| X3D-M | ~15 FPS | ~30 FPS | ~60 FPS |

*Benchmarks with FP16 precision, 16-frame clips, 224x224 resolution*

## Production Deployment

For production on Jetson:

1. **Use TAO ActionRecognitionNet** for best DeepStream integration
2. **Convert to TensorRT FP16** for optimal performance
3. **Validate on target hardware** - always build engines on deployment device
4. **Monitor inference latency** - adjust clip length/resolution as needed

## Fallback Strategy

The system gracefully degrades:
1. Try TAO/X3D 3D CNN inference
2. Fall back to simple frame aggregation if model loading fails
3. Log warnings for debugging model issues

This ensures the pipeline remains functional even with model problems.
