# I‑Guard: Real‑Time Weapon and Violence Detection

I‑Guard is an open‑source edge AI framework for detecting weapons and violent behaviour in real‑time video streams. It is designed to run on Nvidia Jetson devices or other GPU‑equipped hosts, but the core components can also run on a CPU for testing and development. The project implements a **three‑stage pipeline**:

1. **Step 1 – Fast detection**: A lightweight object detector scans incoming frames for weapons, pointing poses, muzzle flashes and falls. It runs on every frame to ensure low latency. False positives are tolerated at this stage because further checks follow.
2. **Step 2 – Verification**: When Step 1 flags a candidate event, a secondary verifier analyses a short clip (e.g. 10 seconds pre/post) using a larger model or an action classifier. This stage trades speed for accuracy and reduces false alarms.
3. **Step 3 – Human in the loop**: Verified events are surfaced to a local operator via a simple web interface. Operators can acknowledge, dismiss or trigger escalation. The design also supports optional auto‑escalation when confidence is very high.

I‑Guard provides a modular codebase consisting of camera adapters, a ring buffer for pre/post recording, detection modules, and a small Flask web server for monitoring alerts. It aims to be a starting point for researchers, engineers and volunteers who wish to build more robust safety systems for schools, small businesses or public spaces.

## Features

* **Multi‑camera support** – handle multiple RTSP/USB streams concurrently and time‑multiplex inference on a single GPU.
* **Plug‑and‑play models** – use the bundled YOLOv8n model for quick start or swap in your own TensorRT engines. Step 2 verification is optional and pluggable.
* **Edge‑friendly** – built to run entirely on the edge; no cloud connectivity is required. Hardware decoding/encoding via GStreamer and NVENC/NVDEC keeps latency low.
* **Pre/post recording** – ring buffer captures seconds of video before and after an event; clips are encoded and stored locally for review.
* **Local web UI** – simple interface showing live detections, saved clips and action buttons for acknowledgement and escalation.

## Repository structure

```
I-Guard/
├── README.md               # this file
├── LICENSE                 # project licence (Apache 2.0)
├── requirements.txt        # Python dependencies
├── config.yaml             # example configuration file
├── app.py                  # entry point to start the pipeline and UI
├── detection/
│   ├── __init__.py
│   ├── frame_detector.py   # fast per‑frame detector wrapper (was Step 1)
│   ├── clip_verifier.py    # slower verification stage (was Step 2)
│   └── tracking.py         # optional tracking/Re‑ID utilities
├── pipeline/
│   ├── __init__.py
│   ├── ring_buffer.py      # in‑memory circular buffer for frames
│   ├── camera_adapter.py   # wrappers for RTSP or webcam input
│   ├── event_queue.py      # thread‑safe queue for candidate events
│   └── inference_pipeline.py  # orchestrates detection and verification
├── ui/
│   ├── server.py           # Flask application serving the UI
│   ├── templates/
│   │   ├── layout.html     # base HTML template
│   │   └── index.html      # UI for monitoring events
│   └── static/
│       └── main.js         # client‑side script for updates
└── tests/                  # unit tests (optional)
```

## Quick start

1. Install dependencies into a Python virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

2. Edit `config.yaml` to point to your camera streams. The file contains sensible defaults for 10 seconds of pre/post buffering and detection thresholds.

3. Download or convert your detection models to TensorRT engines and place them into a directory of your choice. Update `config.yaml` with the paths.

4. Launch the application:

    ```bash
    python app.py
    ```

   Open your browser at [http://localhost:5000](http://localhost:5000) to see the dashboard.

## Testing

I‑Guard includes a comprehensive end‑to‑end test suite that validates the complete pipeline flow without requiring GPU resources or model dependencies.

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run basic test suite (no additional dependencies required)
python run_tests.py

# For advanced testing with pytest (optional)
pip install -r requirements-test.txt
pytest tests/ -v
```

### Test Coverage

The test suite includes:

* **Mock Components** – Synthetic camera, detector, and verifier for dependency-free testing
* **Pipeline Integration** – Complete flow from frame capture through threat verification
* **Event Processing** – Queue management and event history tracking
* **Performance Validation** – Async Stage-2 pipeline performance improvements

See `tests/README.md` for detailed testing documentation.

## Contributing

We welcome pull requests and issues. Please see `LICENSE` for licence details. When contributing code, try to follow the PEP 8 style guide and include docstrings. For long‑running enhancements (e.g. new models or platforms), please open an issue first to discuss direction.
