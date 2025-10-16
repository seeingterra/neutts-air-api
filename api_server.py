"""
FastAPI server for NeuTTS Air API
Provides endpoints for TTS synthesis, voice sample management, and health checks.
"""
import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Optional
import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from neuttsair.neutts import NeuTTSAir

# Try to import whisper for optional transcription support
try:
    import whisper
    WHISPER_AVAILABLE = True
    whisper_model = None
except ImportError:
    WHISPER_AVAILABLE = False
    whisper_model = None

app = FastAPI(title="NeuTTS Air API", version="1.0.0")

# Enable CORS for web GUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global TTS instance
tts_instance: Optional[NeuTTSAir] = None
voice_samples = {}  # Store voice samples and their encoded references


class SynthesisRequest(BaseModel):
    text: str
    voice_id: str = "default"
    ref_text: Optional[str] = None


class VoiceSample(BaseModel):
    name: str
    ref_text: str


@app.on_event("startup")
async def startup_event():
    """Initialize TTS model on startup"""
    global tts_instance
    try:
        print("Initializing NeuTTS Air...")
        tts_instance = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )
        
        # Load default voice samples
        samples_dir = Path("samples")
        if samples_dir.exists():
            for wav_file in samples_dir.glob("*.wav"):
                txt_file = wav_file.with_suffix(".txt")
                if txt_file.exists():
                    voice_name = wav_file.stem
                    ref_text = txt_file.read_text().strip()
                    try:
                        ref_codes = tts_instance.encode_reference(str(wav_file))
                        voice_samples[voice_name] = {
                            "ref_codes": ref_codes,
                            "ref_text": ref_text,
                            "audio_path": str(wav_file)
                        }
                        print(f"Loaded voice sample: {voice_name}")
                    except Exception as e:
                        print(f"Failed to load voice sample {voice_name}: {e}")
        
        print("NeuTTS Air initialized successfully")
    except Exception as e:
        print(f"Failed to initialize NeuTTS Air: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="TTS model not initialized")
    return {"status": "healthy", "model": "NeuTTS Air"}


@app.get("/voices")
async def list_voices():
    """List available voice samples"""
    return {
        "voices": [
            {
                "id": voice_id,
                "ref_text": info["ref_text"]
            }
            for voice_id, info in voice_samples.items()
        ]
    }


@app.post("/synthesize")
async def synthesize_speech(request: SynthesisRequest):
    """
    Synthesize speech from text using a specified voice
    """
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="TTS model not initialized")
    
    # Get voice sample
    voice_id = request.voice_id if request.voice_id else "default"
    
    # Use first available voice if default not found
    if voice_id == "default" and voice_id not in voice_samples:
        if voice_samples:
            voice_id = list(voice_samples.keys())[0]
        else:
            raise HTTPException(status_code=404, detail="No voice samples available")
    
    if voice_id not in voice_samples:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    voice_info = voice_samples[voice_id]
    ref_codes = voice_info["ref_codes"]
    ref_text = request.ref_text if request.ref_text else voice_info["ref_text"]
    
    try:
        # Clean and validate input text
        input_text = request.text.strip()
        if not input_text:
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
        # Generate speech
        wav = tts_instance.infer(input_text, ref_codes, ref_text)
        
        # Convert to WAV format in memory
        buffer = io.BytesIO()
        sf.write(buffer, wav, 24000, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=synthesis.wav"
            }
        )
    except ValueError as e:
        error_msg = str(e)
        if "number of lines" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Synthesis failed: Text formatting error. Ensure input text and reference text are properly formatted. {error_msg}"
            )
        raise HTTPException(status_code=400, detail=f"Synthesis failed: {error_msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/upload_voice")
async def upload_voice_sample(
    name: str = Form(...),
    ref_text: str = Form(None),
    audio_file: UploadFile = File(...),
    auto_transcribe: bool = Form(False)
):
    """
    Upload a new voice sample for cloning
    
    Args:
        name: Name for the voice sample
        ref_text: Reference text (optional if auto_transcribe is True)
        audio_file: WAV audio file
        auto_transcribe: Use Whisper to auto-transcribe (requires whisper installed)
    """
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="TTS model not initialized")
    
    # Validate name
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    
    tmp_path = None
    
    # Save uploaded audio to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Auto-transcribe if requested and ref_text not provided
        if auto_transcribe and not ref_text:
            if not WHISPER_AVAILABLE:
                raise HTTPException(
                    status_code=400,
                    detail="Whisper not installed. Install with: pip install openai-whisper"
                )
            
            global whisper_model
            if whisper_model is None:
                print("Loading Whisper model for transcription...")
                whisper_model = whisper.load_model("base")
            
            print(f"Transcribing audio for voice '{name}'...")
            result = whisper_model.transcribe(tmp_path)
            ref_text = result["text"].strip()
            print(f"Transcribed: {ref_text}")
        
        # Validate ref_text
        if not ref_text or not ref_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Reference text is required (or enable auto_transcribe)"
            )
        
        # Encode reference audio
        ref_codes = tts_instance.encode_reference(tmp_path)
        
        # Store voice sample
        voice_samples[name] = {
            "ref_codes": ref_codes,
            "ref_text": ref_text.strip(),
            "audio_path": tmp_path
        }
        
        return {
            "message": f"Voice sample '{name}' uploaded successfully",
            "voice_id": name,
            "ref_text": ref_text.strip(),
            "auto_transcribed": auto_transcribe and WHISPER_AVAILABLE
        }
    except HTTPException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload voice sample: {str(e)}")


@app.post("/transcribe")
async def transcribe_audio_file(
    audio_file: UploadFile = File(...),
    model: str = Form("base")
):
    """
    Transcribe an audio file using Whisper (requires openai-whisper installed)
    
    Args:
        audio_file: Audio file to transcribe
        model: Whisper model size (tiny, base, small, medium, large)
    """
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Whisper not installed. Install with: pip install openai-whisper"
        )
    
    tmp_path = None
    try:
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load model if needed
        global whisper_model
        if whisper_model is None:
            print(f"Loading Whisper model '{model}'...")
            whisper_model = whisper.load_model(model)
        
        # Transcribe
        result = whisper_model.transcribe(tmp_path)
        
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice sample"""
    if voice_id not in voice_samples:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    # Remove temporary audio file if it exists
    voice_info = voice_samples[voice_id]
    audio_path = voice_info.get("audio_path")
    if audio_path and os.path.exists(audio_path) and audio_path.startswith(tempfile.gettempdir()):
        try:
            os.unlink(audio_path)
        except:
            pass
    
    del voice_samples[voice_id]
    return {"message": f"Voice '{voice_id}' deleted successfully"}


@app.get("/gui")
async def gui():
    """Serve the web GUI"""
    gui_path = Path(__file__).parent / "static" / "index.html"
    if gui_path.exists():
        return FileResponse(gui_path)
    raise HTTPException(status_code=404, detail="GUI not found")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "NeuTTS Air API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "synthesize": "/synthesize",
            "voices": "/voices",
            "upload_voice": "/upload_voice",
            "transcribe": "/transcribe",
            "gui": "/gui"
        },
        "features": {
            "whisper_available": WHISPER_AVAILABLE
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuTTS Air API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8011, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
