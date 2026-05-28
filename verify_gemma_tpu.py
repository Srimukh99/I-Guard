
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from pipeline.backends.backend_factory import BackendFactory
from detection.clip_verifier import ClipVerifier
from detection.gemma_verifier import GemmaVerifier

logging.basicConfig(level=logging.INFO)

def test_backend_discovery():
    print("\n--- Testing Backend Discovery ---")
    factory = BackendFactory()
    available = factory.get_available_backends()
    print(f"Available backends: {available}")
    # Even if hardware is missing, it should be in the known list if the file is there,
    # but might not be in 'available' if requirements check fails.
    # Actually _discover_backends only adds to the dict if it's available.

def test_gemma_verifier_init():
    print("\n--- Testing GemmaVerifier Initialization ---")
    try:
        verifier = GemmaVerifier(model_type="gemma", device="cpu")
        print("GemmaVerifier initialized successfully (likely in mock mode)")

        # Test mock verify
        res = verifier.verify([None]*16, detections_per_frame=[["gun"]]*5 + [[]]*11)
        print(f"Mock verification result: {res}")
    except Exception as e:
        print(f"GemmaVerifier init failed: {e}")

def test_clip_verifier_with_gemma():
    print("\n--- Testing ClipVerifier with Gemma ---")
    try:
        verifier = ClipVerifier(model_path="", model_type="gemma")
        print(f"ClipVerifier initialized with type: {verifier.model_type}")

        # It should have gemma_verifier initialized (possibly mock)
        if verifier.gemma_verifier:
            print("Internal gemma_verifier is present")
        else:
            print("Internal gemma_verifier is MISSING")

    except Exception as e:
        print(f"ClipVerifier with Gemma failed: {e}")

if __name__ == "__main__":
    test_backend_discovery()
    test_gemma_verifier_init()
    test_clip_verifier_with_gemma()
