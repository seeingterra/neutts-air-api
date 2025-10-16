# Windows Installation Guide for NeuTTS Air API

This guide provides step-by-step instructions for setting up NeuTTS Air with the API server on Windows.

## Prerequisites

### 1. Install Python

1. Download Python 3.11 or later from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```powershell
   python --version
   ```

### 2. Install eSpeak NG (Required)

eSpeak NG is required for phonemization:

1. Download the eSpeak NG installer from: https://github.com/espeak-ng/espeak-ng/releases
2. Install to the default location: `C:\Program Files\eSpeak NG`
3. Set environment variables (run in PowerShell as Administrator):
   ```powershell
   # Set for current session
   $env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
   $env:PHONEMIZER_ESPEAK_PATH = "C:\Program Files\eSpeak NG"
   
   # Set permanently
   [System.Environment]::SetEnvironmentVariable('PHONEMIZER_ESPEAK_LIBRARY', 'C:\Program Files\eSpeak NG\libespeak-ng.dll', 'Machine')
   [System.Environment]::SetEnvironmentVariable('PHONEMIZER_ESPEAK_PATH', 'C:\Program Files\eSpeak NG', 'Machine')
   ```

4. Alternatively, use the older `setx` command:
   ```powershell
   setx PHONEMIZER_ESPEAK_LIBRARY "C:\Program Files\eSpeak NG\libespeak-ng.dll"
   setx PHONEMIZER_ESPEAK_PATH "C:\Program Files\eSpeak NG"
   ```

5. Restart your terminal/PowerShell for changes to take effect.

## Installation Steps

### 1. Clone the Repository

```powershell
git clone https://github.com/seeingterra/neutts-air-api.git
cd neutts-air-api
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Note**: If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies

Install base requirements:
```powershell
pip install -r requirements.txt
```

Install API server requirements:
```powershell
pip install -r requirements-api.txt
```

### 4. (Optional) Install CUDA Support

For GPU acceleration (if you have an NVIDIA GPU):

1. Install CUDA Toolkit 11.8 or 12.1 from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

2. Install PyTorch with CUDA support:
   ```powershell
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. For GGUF models with GPU support, install llama-cpp-python with CUDA:
   ```powershell
   # Set CUDA environment variables
   $env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
   $env:FORCE_CMAKE = "1"
   
   pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```

### 5. (Optional) Install ONNX Runtime

For faster inference with ONNX decoder:
```powershell
pip install onnxruntime
```

Or for GPU support:
```powershell
pip install onnxruntime-gpu
```

## Running the API Server

### Start the Server

```powershell
python api_server.py --host 127.0.0.1 --port 8011
```

Options:
- `--host`: Host address (default: 127.0.0.1)
- `--port`: Port number (default: 8011)
- `--reload`: Enable auto-reload for development

### Access the Web GUI

Once the server is running, open your browser and navigate to:
```
http://127.0.0.1:8011/gui
```

### Check Server Health

```powershell
curl http://127.0.0.1:8011/health
```

Or open in browser: http://127.0.0.1:8011/health

## Basic Usage

### 1. Using the Web GUI

The web interface provides an easy way to:
- Generate speech from text
- Upload and manage voice samples
- Test different voices
- Download generated audio

### 2. Using the API Directly

#### Health Check
```powershell
curl http://127.0.0.1:8011/health
```

#### List Available Voices
```powershell
curl http://127.0.0.1:8011/voices
```

#### Synthesize Speech
```powershell
curl -X POST http://127.0.0.1:8011/synthesize `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Hello world\", \"voice_id\": \"dave\"}' `
  --output output.wav
```

#### Upload Voice Sample
```powershell
curl -X POST http://127.0.0.1:8011/upload_voice `
  -F "name=my_voice" `
  -F "ref_text=This is my voice sample text" `
  -F "audio_file=@path/to/your/audio.wav"
```

## Troubleshooting

### Common Issues

#### 1. eSpeak Not Found
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'espeak'`

**Solution**: Make sure eSpeak NG is installed and environment variables are set correctly. Restart your terminal after setting the variables.

#### 2. Module Import Errors
**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Ensure you've activated your virtual environment and installed all requirements:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-api.txt
```

#### 3. Port Already in Use
**Error**: `[Errno 10048] Only one usage of each socket address is normally permitted`

**Solution**: The port is already in use. Either:
- Stop the other application using port 8011
- Use a different port: `python api_server.py --port 8012`

#### 4. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Use CPU instead or use quantized models:
```powershell
# In api_server.py, change device to "cpu"
# Or use GGUF quantized models which are more memory efficient
```

#### 5. Slow Generation on CPU
**Solution**: 
- Use GGUF quantized models for faster CPU inference
- Pre-encode reference audio samples
- Use ONNX decoder for faster decoding

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Ensure all prerequisites are installed
3. Verify environment variables are set correctly
4. Try running in a fresh virtual environment
5. Check the main README.md for additional information

## Advanced Configuration

### Using GGUF Models (Recommended for CPU)

GGUF models provide faster inference on CPU:

```python
# Modify api_server.py startup_event:
tts_instance = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air-q4-gguf",  # Use quantized model
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec",
    codec_device="cpu"
)
```

### Using ONNX Decoder

For faster decoding:

```python
# Modify api_server.py startup_event:
tts_instance = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec-onnx-decoder",  # Use ONNX decoder
    codec_device="cpu"
)
```

### Running as a Service

To run the API server as a Windows service, you can use tools like:
- NSSM (Non-Sucking Service Manager)
- Windows Task Scheduler
- PM2 for Node.js (if you prefer)

Example with NSSM:
```powershell
nssm install NeuTTSAir "C:\path\to\venv\Scripts\python.exe" "C:\path\to\neutts-air-api\api_server.py"
nssm start NeuTTSAir
```

## Next Steps

- Explore the example scripts in the `examples/` directory
- Read the main README.md for more detailed usage information
- Customize the API server for your specific needs
- Check out the Hugging Face model cards for more information about the models
