# I-Guard End-to-End Testing

This directory contains comprehensive end-to-end tests for the I-Guard threat detection system.

## Overview

The test suite validates the complete application flow:

1. **Camera Simulation** - Mock camera generates synthetic video frames with controllable threat patterns
2. **Detection Pipeline** - Mock detector identifies objects and flags threat events  
3. **Event Processing** - Event queue handles candidate threat events
4. **Verification** - Mock verifier provides threat scores based on detection patterns
5. **Integration** - Complete pipeline orchestration and event history tracking

## Test Structure

### Core Test Files

- `test_end_to_end.py` - Main test suite with mock components and integration tests
- `run_tests.py` - Simple test runner that works without pytest installation

### Mock Components

- **MockCameraAdapter** - Generates synthetic 640x480 frames with threat patterns at frame 10
- **MockFrameDetector** - Detects person + weapon based on frame content (red pixels = weapon)  
- **MockClipVerifier** - Returns threat scores based on weapon detection ratios

### Test Categories

1. **Import Tests** - Verify all core modules can be imported
2. **Component Tests** - Test individual mock components work correctly
3. **Queue Tests** - Validate event queue put/get operations
4. **Pipeline Tests** - End-to-end pipeline flow with and without verification
5. **Metrics Tests** - Event history and performance tracking

## Running Tests

### Simple Test Runner (No Dependencies)

```bash
python run_tests.py
```

### Full Test Suite with pytest

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_end_to_end.py::TestEndToEndPipeline::test_complete_pipeline_flow -v
```

## Expected Test Output

A successful test run should show:

- ✅ All core imports successful
- ✅ Individual mock components working
- ✅ Event queue operations functional
- ✅ Pipeline processing events (with queue warnings normal)
- ✅ Event history tracking working
- ✅ 18+ event records created during pipeline test

## Dependencies

### Required

- numpy (for synthetic frame generation)
- Standard library modules (threading, queue, time, etc.)

### Optional

- pytest (for advanced test features)
- cv2 (OpenCV - gracefully skipped if unavailable)
- GStreamer/DeepStream bindings (skipped if unavailable)

## CI/CD Integration

Add to your CI pipeline:

```yaml
# Example GitHub Actions
- name: Run End-to-End Tests
  run: |
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python run_tests.py
```

## Test Design Philosophy

- **Dependency-Free** - Core tests run without GPU, models, or heavy dependencies
- **Mock Everything** - Use lightweight mocks to avoid external dependencies  
- **Fast Execution** - Complete test suite runs in ~10 seconds
- **CI-Friendly** - No special hardware or services required
- **Real Patterns** - Mock components simulate actual application behavior

## Extending Tests

To add new test scenarios:

1. Modify mock components to generate different patterns
2. Add new test methods to `TestEndToEndPipeline` class
3. Update `run_tests.py` to include new tests
4. Consider adding parametrized tests for different configurations

## Troubleshooting

### Common Issues

- **Import Errors** - Check virtual environment activation and dependency installation
- **Queue Full Messages** - Normal during pipeline tests (indicates high event generation)
- **Skipped Pipeline Tests** - Occurs when optional dependencies unavailable (expected)

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
