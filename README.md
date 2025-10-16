# NeuTTS Air ☁️

<p align="left">
   <a href="https://buymeacoffee.com/starmoose" target="_blank">
      <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="40" />
   </a>
</p>

HuggingFace 🤗: [Model](https://huggingface.co/neuphonic/neutts-air), [Q8 GGUF](https://huggingface.co/neuphonic/neutts-air-q8-gguf), [Q4 GGUF](https://huggingface.co/neuphonic/neutts-air-q4-gguf) [Spaces](https://huggingface.co/spaces/neuphonic/neutts-air)

[Demo Video](https://github.com/user-attachments/assets/020547bc-9e3e-440f-b016-ae61ca645184)

*Created by [Neuphonic](http://neuphonic.com/) - building faster, smaller, on-device voice AI*

State-of-the-art Voice AI has been locked behind web APIs for too long. NeuTTS Air is the world’s first super-realistic, on-device, TTS speech language model with instant voice cloning. Built off a 0.5B LLM backbone, NeuTTS Air brings natural-sounding speech, real-time performance, built-in security and speaker cloning to your local device - unlocking a new category of embedded voice agents, assistants, toys, and compliance-safe apps.

## Key Features

- 🗣Best-in-class realism for its size - produces natural, ultra-realistic voices that sound human
- 📱Optimised for on-device deployment - provided in GGML format, ready to run on phones, laptops, or even Raspberry Pis
- 👫Instant voice cloning - create your own speaker with as little as 3 seconds of audio
- 🚄Simple LM + codec architecture built off a 0.5B backbone - the sweet spot between speed, size, and quality for real-world applications

> [!CAUTION]
> Websites like neutts.com are popping up and they're not affliated with Neuphonic, our github or this repo.
>
> We are on neuphonic.com only. Please be careful out there! 🙏

## Model Details

NeuTTS Air is built off Qwen 0.5B - a lightweight yet capable language model optimised for text understanding and generation - as well as a powerful combination of technologies designed for efficiency and quality:
- **Supported Languages**: English
- **Audio Codec**: [NeuCodec](https://huggingface.co/neuphonic/neucodec) - our 50hz neural audio codec that achieves exceptional audio quality at low bitrates using a single codebook
- **Context Window**: 2048 tokens, enough for processing ~30 seconds of audio (including prompt duration)
- **Format**: Available in GGML format for efficient on-device inference
- **Responsibility**: Watermarked outputs
- **Inference Speed**: Real-time generation on mid-range devices
- **Power Consumption**: Optimised for mobile and embedded devices

## Get Started

1. **Clone Git Repo**
   ```bash
   git clone https://github.com/neuphonic/neutts-air.git
   ```
   ```bash
   cd neutts-air
   ```

2. **Install `eSpeak NG` (required dependency)**

   Please refer to the following link for instructions on how to install `espeak`:

   https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

   ```bash
   # macOS
   brew install espeak

   # Ubuntu/Debian
   sudo apt install espeak
   ```

   Mac users may need to put the following lines at the top of the neutts.py file.
   ```python
   from phonemizer.backend.espeak.wrapper import EspeakWrapper
   _ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'  #use the Path to the library.
   EspeakWrapper.set_library(_ESPEAK_LIBRARY)
   ```

   Windows users may need to run (see https://github.com/bootphon/phonemizer/issues/163)
   ```pwsh
   $env:PHONEMIZER_ESPEAK_LIBRARY = "c:\Program Files\eSpeak NG\libespeak-ng.dll"
   $env:PHONEMIZER_ESPEAK_PATH = "c:\Program Files\eSpeak NG"
   setx PHONEMIZER_ESPEAK_LIBRARY "c:\Program Files\eSpeak NG\libespeak-ng.dll"
   setx PHONEMIZER_ESPEAK_PATH "c:\Program Files\eSpeak NG"
   ```

   Note: On Windows, this library will attempt to auto-detect eSpeak NG from the default install location (C:\Program Files\eSpeak NG). If you installed it elsewhere, set the environment variables above before running.

   See WINDOWS.md in this repo for a complete Windows setup guide.

3. **Install Python dependencies**

   The requirements file includes the dependencies needed to run the model with PyTorch.
   When using an ONNX decoder or a GGML model, some dependencies (such as PyTorch) are no longer required.

   The inference is compatible and tested on `python>=3.11`.

    ```
    pip install -r requirements.txt
    ```

   On Windows, we recommend creating a virtual environment first:

   ```pwsh
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -U pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **(Optional) Install Llama-cpp-python to use the `GGUF` models.**
   ```
   pip install llama-cpp-python
   ```
   To run llama-cpp with GPU suport (CUDA, MPS) support please refer to:
   https://pypi.org/project/llama-cpp-python/

5. **(Optional) Install onnxruntime to use the `.onnx` decoder.**
   If you want to run the onnxdecoder
   ```
   pip install onnxruntime
   ```

6. **(Optional) Install CUDA-enabled PyTorch and GPU deps**

   If you want to run with an NVIDIA GPU, follow CUDA.md for installing a CUDA-enabled PyTorch wheel. Then install the rest of the deps with `requirements-cuda.txt`:

   ```
   pip install -r requirements-cuda.txt
   ```

## Running the Model

Run the basic example script to synthesize speech (GGUF recommended on Windows):
```bash
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio samples/dave.wav \
   --ref_text samples/dave.txt \
   --backbone neuphonic/neutts-air-q4-gguf
```

On Windows PowerShell the command is the same. Make sure your virtual environment is activated and eSpeak NG is installed. See WINDOWS.md for more details.

To specify a particular model repo for the backbone or codec, add the `--backbone` argument. Available backbones are listed in [NeuTTS-Air huggingface collection](https://huggingface.co/collections/neuphonic/neutts-air-68cc14b7033b4c56197ef350). If you choose HF (PyTorch) mode, ensure the repo contains weights (pytorch_model.bin or model.safetensors); otherwise prefer GGUF.

To use GPU (when available), pass device flags in examples:

```bash
python -m examples.basic_example \
   --input_text "..." \
   --ref_audio samples/dave.wav \
   --ref_text samples/dave.txt \
   --backbone neuphonic/neutts-air \
   --backbone_device cuda \
   --codec_device cuda
```

Note: GGUF backends via llama-cpp may require a CUDA-enabled build; the ONNX codec decoder runs on CPU.

Several examples are available, including a Jupyter notebook in the `examples` folder.

### GUI App (Streamlit)

You can launch a simple GUI to test text-to-speech, switch between modes (HF/GGUF/ONNX), and manage models.

```
pip install -r requirements-gui.txt
streamlit run app/app.py
```

Notes:
- GGUF mode requires `llama-cpp-python` installed. GPU acceleration for llama.cpp may require a CUDA-enabled build.
- ONNX codec runs on CPU; backbone can still run on GPU.
- Windows users: ensure eSpeak NG is installed and environment variables are set (see WINDOWS.md). The app will try to auto-detect the default install.

Features:
- Mode switching: HF (PyTorch), GGUF (llama.cpp), HF + ONNX codec
- GGUF Model Manager: pre-download/update checkpoints from Hugging Face, list locally cached snapshots with sizes, and delete snapshots
- Config save/load: save current selections to `app/config.json` and reload later

### One-Code Block Usage

```python
from neuttsair.neutts import NeuTTSAir
import soundfile as sf

tts = NeuTTSAir(
   backbone_repo="neuphonic/neutts-air", # or 'neutts-air-q4-gguf' with llama-cpp-python installed
   backbone_device="cpu",
   codec_repo="neuphonic/neucodec",
   codec_device="cpu"
)
input_text = "My name is Dave, and um, I'm from London."

ref_text = "samples/dave.txt"
ref_audio_path = "samples/dave.wav"

ref_text = open(ref_text, "r").read().strip()
ref_codes = tts.encode_reference(ref_audio_path)

wav = tts.infer(input_text, ref_codes, ref_text)
sf.write("test.wav", wav, 24000)
```

## Preparing References for Cloning

NeuTTS Air requires two inputs:

1. A reference audio sample (`.wav` file)
2. A text string

The model then synthesises the text as speech in the style of the reference audio. This is what enables NeuTTS Air’s instant voice cloning capability.

### Example Reference Files

You can find some ready-to-use samples in the `examples` folder:

- `samples/dave.wav`
- `samples/jo.wav`

### Guidelines for Best Results

For optimal performance, reference audio samples should be:

1. **Mono channel**
2. **16-44 kHz sample rate**
3. **3–15 seconds in length**
4. **Saved as a `.wav` file**
5. **Clean** — minimal to no background noise
6. **Natural, continuous speech** — like a monologue or conversation, with few pauses, so the model can capture tone effectively

## Guidelines for minimizing Latency

For optimal performance on-device:

1. Use the GGUF model backbones
2. Pre-encode references
3. Use the [onnx codec decoder](https://huggingface.co/neuphonic/neucodec-onnx-decoder)

Take a look at this example [examples README](examples/README.md###minimal-latency-example) to get started.

## Responsibility

Every audio file generated by NeuTTS Air includes [Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth).

## Disclaimer

Don't use this model to do bad things… please.

## Developer Requirements

To run the pre commit hooks to contribute to this project run:

```bash
pip install pre-commit
```
Then:
```bash
pre-commit install
```

## Windows-specific notes

- eSpeak NG is required for phonemization. Install from https://github.com/espeak-ng/espeak-ng/releases and ensure `libespeak-ng.dll` is present in the install folder. This package attempts to auto-detect it, but you can override via `PHONEMIZER_ESPEAK_LIBRARY` and `PHONEMIZER_ESPEAK_PATH` environment variables.
- Audio I/O uses `soundfile` and resampling uses `scipy.signal.resample_poly` to avoid `librosa`'s optional heavy dependencies on Windows.
- For GPU inference with PyTorch on Windows, install a CUDA-enabled torch build following https://pytorch.org/get-started/locally/ and then re-run `pip install -r requirements.txt` (you may need to edit the torch line accordingly).
