# TPU and Gemma Setup Guide for I-Guard

This guide describes how to set up I-Guard to run on Google Cloud TPUs using the latest Gemma (PaliGemma) models.

## Prerequisites

1.  **Google Cloud Project**: With TPU API enabled.
2.  **TPU VM**: Created and accessible via SSH.
3.  **Kaggle Account**: To download PaliGemma models (requires accepting the license).
4.  **Python 3.8+**: Recommended environment.

## Environment Setup

### 1. Install Dependencies

On your TPU VM, install the necessary libraries for TPU acceleration and Gemma support:

```bash
# Install JAX with TPU support
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install PyTorch XLA
pip install torch-xla

# Install Transformers and related tools
pip install transformers accelerate sentencepiece

# Install I-Guard dependencies
pip install -r requirements.txt
```

### 2. Configure Kaggle API

To download the model directly from Kaggle:

1.  Go to your Kaggle account settings and create a new API token.
2.  Save the `kaggle.json` file to `~/.kaggle/kaggle.json`.
3.  Set permissions: `chmod 600 ~/.kaggle/kaggle.json`.

## Model Download

You can use the `transformers` library to download the model automatically, or download it manually from Kaggle/HuggingFace.

**HuggingFace Example:**
```python
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
```

## Running I-Guard on TPU

### 1. Configuration

Create or modify your configuration file (e.g., `config_tpu.yaml`):

```yaml
backend:
  type: tpu

step2:
  enabled: true
  model_type: "gemma"
  model_path: "google/paligemma-3b-pt-224"
  device: "xla"  # Use XLA for TPU acceleration
```

### 2. Execution

Launch the application with the TPU configuration:

```bash
python app.py --config config_tpu.yaml
```

## Optimization Tips

-   **Precision**: Use `bfloat16` for model loading on TPU to save memory and improve performance.
-   **Batching**: TPU is highly efficient with larger batches. Adjust `frame_batch_size` in your config.
-   **Async Stage 2**: Ensure `async_enabled: true` is set for Step 2 to prevent verification from blocking the main pipeline.

## Troubleshooting

-   **libtpu not found**: Ensure `jax[tpu]` is installed correctly and `LD_LIBRARY_PATH` includes the TPU libraries.
-   **Out of Memory (OOM)**: TPU memory is limited. Try reducing resolution or batch size.
-   **Model Access Denied**: Ensure you have accepted the Gemma license on HuggingFace or Kaggle.
