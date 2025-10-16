# NeuTTS Air on Windows

This guide walks you through installing and running NeuTTS Air on Windows using PowerShell.

## Prerequisites

- Windows 10/11 x64
- Python 3.11 or 3.12 recommended (3.13 may work for CPU, but some packages can lag)
- eSpeak NG (required by `phonemizer`)

## 1) Install eSpeak NG

Download the Windows installer from the eSpeak NG releases page:
https://github.com/espeak-ng/espeak-ng/releases

Install to the default location (C:\\Program Files\\eSpeak NG). After installation, set environment variables so `phonemizer` can find it. In PowerShell:

```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY = "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"
$env:PHONEMIZER_ESPEAK_PATH    = "C:\\Program Files\\eSpeak NG"
setx PHONEMIZER_ESPEAK_LIBRARY "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"
setx PHONEMIZER_ESPEAK_PATH    "C:\\Program Files\\eSpeak NG"
```

Note: The library will attempt to auto-detect this path, but setting these variables ensures consistent behavior.

## 2) Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

Upgrade tooling and install dependencies:

```powershell
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If you plan to use GGUF backends or streaming:

```powershell
# GGUF backends
pip install llama-cpp-python

# ONNX decoder
pip install onnxruntime

# Streaming example audio output
pip install pyaudio
```

For GPU-enabled PyTorch on Windows, follow official instructions:
https://pytorch.org/get-started/locally/

Then re-run installation if needed.

## 3) Run an example

```powershell
python -m examples.basic_example `
  --input_text "My name is Dave, and um, I'm from London" `
  --ref_audio samples/dave.wav `
  --ref_text samples/dave.txt
```

Output will be written to `output.wav` by default.

## Troubleshooting

Phonemizer cannot find eSpeak:
- Verify `libespeak-ng.dll` exists under `C:\\Program Files\\eSpeak NG` and environment variables are set (see step 1). Restart PowerShell or log out/in after `setx`.

ImportError for torch or transformers:
- Ensure the virtual environment is active and that `pip install -r requirements.txt` completed without errors.

No audio in streaming example:
- Install `pyaudio` and ensure your device's audio output is accessible.

Slow or choppy generation:
- Use GGUF backbones with `llama-cpp-python`, pre-encode references, and try the ONNX codec decoder for lower latency.
