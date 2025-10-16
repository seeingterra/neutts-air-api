# NeuTTS Air — CUDA/GPU Setup

Use this guide to install CUDA-enabled PyTorch and run NeuTTS Air on your NVIDIA GPU.

## 1) Pick your Python and CUDA

- Recommended Python: 3.11 or 3.12
- Determine your target CUDA Toolkit version (e.g., 12.1, 12.4). You do NOT need to install the full CUDA Toolkit; PyTorch wheels include the necessary CUDA runtime.

Check the official matrix for your platform: https://pytorch.org/get-started/locally/

## 2) Create and activate a virtual environment

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
```

Linux/macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

## 3) Install CUDA-enabled PyTorch

Pick the command from https://pytorch.org/get-started/locally/. Examples for Windows with CUDA 12.1:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Linux/macOS examples will be similar (choose your CUDA index URL accordingly).

Verify GPU availability:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
```

Expected output: `CUDA: True`.

## 4) Install the rest of the requirements

Install project dependencies that exclude torch:

Windows (PowerShell):
```powershell
pip install -r requirements-cuda.txt
```

Linux/macOS:
```bash
pip install -r requirements-cuda.txt
```

Optional extras:

```powershell
# GGUF backends
pip install llama-cpp-python

# ONNX decoder
pip install onnxruntime

# Streaming example audio output
pip install pyaudio
```

## 5) Run on GPU

You can pass `backbone_device="cuda"` and `codec_device="cuda"` where supported. For example, update your scripts or create a variant that selects CUDA when available. The backbone HF model supports CUDA; GGUF/ONNX options vary (llama-cpp may use GPU via CUDA if compiled appropriately; the ONNX codec decoder is CPU-only per README).

Example (Python):

```python
from neuttsair.neutts import NeuTTSAir

tts = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air",  # HF model
    backbone_device="cuda",
    codec_repo="neuphonic/neucodec",
    codec_device="cuda"
)
```

If you see CUDA OOM errors, try `backbone_device="cuda"` and `codec_device="cpu"`, or reduce batch sizes/sequence lengths if you adapt the code.
