# NeuTTS Air API Server

A FastAPI-based REST API server for NeuTTS Air, providing text-to-speech synthesis with instant voice cloning capabilities.

## Features

- 🎙️ RESTful API for TTS synthesis
- 🌐 Modern web GUI for testing and voice management
- 🎤 Voice sample upload and management
- ⚡ Real-time speech generation
- 🔊 Multiple voice support
- 📦 Easy integration with other applications

## Quick Start

### Installation

1. Install base requirements:
```bash
pip install -r requirements.txt
```

2. Install API server requirements:
```bash
pip install -r requirements-api.txt
```

3. (Optional) Install Whisper for auto-transcription:
```bash
pip install -r requirements-whisper.txt
```

4. Start the server:
```bash
# Linux/Mac
./start_server.sh

# Windows
start_server.bat

# Or manually
python api_server.py
```

The server will start on `http://127.0.0.1:8011`

### Access the Web GUI

Open your browser and navigate to:
```
http://127.0.0.1:8011/gui
```

## API Endpoints

### Health Check

Check if the server is running and the model is initialized.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "NeuTTS Air"
}
```

### List Voices

Get all available voice samples.

```http
GET /voices
```

**Response:**
```json
{
  "voices": [
    {
      "id": "dave",
      "ref_text": "My name is Dave, and um, I'm from London..."
    }
  ]
}
```

### Synthesize Speech

Generate speech from text using a specific voice.

```http
POST /synthesize
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Hello, world!",
  "voice_id": "dave",
  "ref_text": null
}
```

**Parameters:**
- `text` (string, required): Text to synthesize
- `voice_id` (string, optional): Voice ID to use (default: "default")
- `ref_text` (string, optional): Custom reference text (uses voice's default if not provided)

**Response:**
- Content-Type: `audio/wav`
- Binary audio data in WAV format

**Example (curl):**
```bash
curl -X POST http://127.0.0.1:8011/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "dave"}' \
  --output output.wav
```

**Example (Python):**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8011/synthesize",
    json={
        "text": "Hello, this is a test.",
        "voice_id": "dave"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Upload Voice Sample

Upload a new voice sample for cloning.

```http
POST /upload_voice
Content-Type: multipart/form-data
```

**Form Data:**
- `name` (string, required): Name for the voice sample
- `ref_text` (string, optional): Transcription of the audio (can be auto-transcribed)
- `audio_file` (file, required): WAV audio file (3-15 seconds recommended)
- `auto_transcribe` (boolean, optional): Use Whisper to auto-transcribe (default: false)

**Response:**
```json
{
  "message": "Voice sample 'my_voice' uploaded successfully",
  "voice_id": "my_voice",
  "ref_text": "This is my voice sample",
  "auto_transcribed": false
}
```

**Example (curl):**
```bash
# With manual transcription
curl -X POST http://127.0.0.1:8011/upload_voice \
  -F "name=my_voice" \
  -F "ref_text=This is my voice sample" \
  -F "audio_file=@/path/to/audio.wav"

# With auto-transcription (requires Whisper)
curl -X POST http://127.0.0.1:8011/upload_voice \
  -F "name=my_voice" \
  -F "auto_transcribe=true" \
  -F "audio_file=@/path/to/audio.wav"
```

### Transcribe Audio

Transcribe an audio file using Whisper (requires `openai-whisper` installed).

```http
POST /transcribe
Content-Type: multipart/form-data
```

**Form Data:**
- `audio_file` (file, required): Audio file to transcribe
- `model` (string, optional): Whisper model size (tiny, base, small, medium, large)

**Response:**
```json
{
  "text": "This is the transcribed text",
  "language": "en"
}
```

**Example (curl):**
```bash
curl -X POST http://127.0.0.1:8011/transcribe \
  -F "audio_file=@/path/to/audio.wav" \
  -F "model=base"
```

### Delete Voice

Remove a voice sample.

```http
DELETE /voices/{voice_id}
```

**Response:**
```json
{
  "message": "Voice 'my_voice' deleted successfully"
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `200`: Success
- `400`: Bad request (invalid input)
- `404`: Not found (voice not found)
- `500`: Internal server error
- `503`: Service unavailable (model not initialized)

**Error Response Format:**
```json
{
  "detail": "Error message description"
}
```

## Configuration

### Command Line Arguments

```bash
python api_server.py --host 127.0.0.1 --port 8011 --reload
```

**Arguments:**
- `--host`: Host address to bind to (default: 127.0.0.1)
- `--port`: Port number to bind to (default: 8011)
- `--reload`: Enable auto-reload for development

### Model Configuration

Edit `api_server.py` to customize the model configuration:

```python
tts_instance = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air",  # or "neuphonic/neutts-air-q4-gguf"
    backbone_device="cpu",  # or "cuda" for GPU
    codec_repo="neuphonic/neucodec",
    codec_device="cpu"
)
```

## Integration Examples

### JavaScript/TypeScript

```javascript
async function synthesizeSpeech(text, voiceId) {
  const response = await fetch('http://127.0.0.1:8011/synthesize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      voice_id: voiceId
    })
  });
  
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
}
```

### Python

```python
import requests
import soundfile as sf
import io

def synthesize(text, voice_id="dave"):
    response = requests.post(
        "http://127.0.0.1:8011/synthesize",
        json={"text": text, "voice_id": voice_id}
    )
    
    if response.status_code == 200:
        # Save to file
        with open("output.wav", "wb") as f:
            f.write(response.content)
        
        # Or load directly
        audio, sr = sf.read(io.BytesIO(response.content))
        return audio, sr
    else:
        raise Exception(f"Synthesis failed: {response.json()['detail']}")
```

### Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

async function synthesize(text, voiceId = 'dave') {
  const response = await axios.post(
    'http://127.0.0.1:8011/synthesize',
    { text, voice_id: voiceId },
    { responseType: 'arraybuffer' }
  );
  
  fs.writeFileSync('output.wav', Buffer.from(response.data));
}
```

## Best Practices

### Voice Sample Guidelines

For optimal voice cloning results:
1. **Duration**: 3-15 seconds of audio
2. **Format**: WAV file, mono channel
3. **Sample Rate**: 16-44 kHz
4. **Quality**: Clean audio with minimal background noise
5. **Content**: Natural, continuous speech with few pauses
6. **Reference Text**: Accurate transcription of the audio

### Performance Optimization

1. **Use GGUF Models**: For CPU inference, use quantized GGUF models:
   ```python
   backbone_repo="neuphonic/neutts-air-q4-gguf"
   ```

2. **Pre-encode References**: Encode voice samples once during upload rather than on each synthesis

3. **Use ONNX Decoder**: For faster decoding:
   ```python
   codec_repo="neuphonic/neucodec-onnx-decoder"
   ```

4. **GPU Acceleration**: Use CUDA if available:
   ```python
   backbone_device="cuda"
   codec_device="cuda"
   ```

## Troubleshooting

### Common Issues

1. **"TTS model not initialized"**
   - Wait for the model to load on startup
   - Check console for error messages during initialization

2. **"Synthesis failed: number of lines in input and output must be equal"**
   - This error is related to text formatting
   - Ensure input text is properly formatted without unusual line breaks
   - Try shorter text segments

3. **"Voice not found"**
   - Check available voices with `GET /voices`
   - Ensure the voice ID is spelled correctly

4. **Slow generation**
   - Use GGUF quantized models for faster CPU inference
   - Consider using GPU if available
   - Use ONNX decoder for faster decoding

## Security Considerations

⚠️ **Important**: This API server is designed for local development and testing. For production use:

1. Add authentication/authorization
2. Implement rate limiting
3. Validate and sanitize all inputs
4. Use HTTPS
5. Restrict CORS origins
6. Add request size limits
7. Implement proper logging and monitoring

## License

See the main LICENSE file in the repository root.

## Contributing

Contributions are welcome! Please follow the existing code style and add tests for new features.

## Support

For issues and questions:
- Check the main README.md
- See WINDOWS_INSTALL.md for Windows-specific setup
- Open an issue on GitHub
