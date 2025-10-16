"""
Example script demonstrating how to use the NeuTTS Air API.

This script shows how to:
1. Check API health
2. List available voices
3. Synthesize speech
4. Upload a new voice sample
5. Use auto-transcription (if Whisper is available)

Requirements:
    pip install requests

Usage:
    # Make sure the API server is running first
    python api_server.py
    
    # Then run this example
    python examples/api_example.py
"""

import requests
import soundfile as sf
from pathlib import Path

API_BASE = "http://127.0.0.1:8011"


def check_health():
    """Check if the API server is healthy"""
    print("Checking API health...")
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        print(f"✅ API is healthy: {response.json()}")
        return True
    else:
        print(f"❌ API is not healthy: {response.status_code}")
        return False


def list_voices():
    """List all available voices"""
    print("\nListing available voices...")
    response = requests.get(f"{API_BASE}/voices")
    if response.status_code == 200:
        data = response.json()
        voices = data.get('voices', [])
        print(f"Found {len(voices)} voice(s):")
        for voice in voices:
            print(f"  - {voice['id']}: {voice['ref_text'][:50]}...")
        return voices
    else:
        print(f"❌ Failed to list voices: {response.status_code}")
        return []


def synthesize_speech(text, voice_id="dave", output_file="output.wav"):
    """Synthesize speech from text"""
    print(f"\nSynthesizing speech with voice '{voice_id}'...")
    print(f"Text: {text}")
    
    response = requests.post(
        f"{API_BASE}/synthesize",
        json={
            "text": text,
            "voice_id": voice_id
        }
    )
    
    if response.status_code == 200:
        # Save the audio
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Speech synthesized successfully: {output_file}")
        
        # Also load and display info
        audio, sr = sf.read(output_file)
        duration = len(audio) / sr
        print(f"   Duration: {duration:.2f} seconds, Sample rate: {sr} Hz")
        return True
    else:
        try:
            error = response.json()
            print(f"❌ Synthesis failed: {error.get('detail', 'Unknown error')}")
        except:
            print(f"❌ Synthesis failed: {response.status_code}")
        return False


def upload_voice_sample(name, audio_path, ref_text=None, auto_transcribe=False):
    """Upload a new voice sample"""
    print(f"\nUploading voice sample '{name}'...")
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return False
    
    data = {
        'name': name,
        'auto_transcribe': auto_transcribe
    }
    
    if ref_text:
        data['ref_text'] = ref_text
    
    with open(audio_path, 'rb') as f:
        files = {
            'audio_file': f
        }
        
        response = requests.post(
            f"{API_BASE}/upload_voice",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Voice uploaded successfully: {result['voice_id']}")
        if result.get('auto_transcribed'):
            print(f"   Auto-transcribed: {result['ref_text']}")
        return True
    else:
        try:
            error = response.json()
            print(f"❌ Upload failed: {error.get('detail', 'Unknown error')}")
        except:
            print(f"❌ Upload failed: {response.status_code}")
        return False


def transcribe_audio(audio_path, model="base"):
    """Transcribe an audio file using Whisper"""
    print(f"\nTranscribing audio: {audio_path}")
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    data = {
        'model': model
    }
    
    with open(audio_path, 'rb') as f:
        files = {
            'audio_file': f
        }
        
        response = requests.post(
            f"{API_BASE}/transcribe",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        text = result['text']
        print(f"✅ Transcription successful:")
        print(f"   Text: {text}")
        print(f"   Language: {result.get('language', 'unknown')}")
        return text
    elif response.status_code == 501:
        print(f"⚠️  Whisper not installed (install with: pip install -r requirements-whisper.txt)")
        return None
    else:
        try:
            error = response.json()
            print(f"❌ Transcription failed: {error.get('detail', 'Unknown error')}")
        except:
            print(f"❌ Transcription failed: {response.status_code}")
        return None


def main():
    """Run the example"""
    print("=" * 70)
    print("NeuTTS Air API Example")
    print("=" * 70)
    
    # Check health
    if not check_health():
        print("\n❌ API server is not running or not healthy!")
        print("Start the server with: python api_server.py")
        return
    
    # List available voices
    voices = list_voices()
    if not voices:
        print("\n⚠️  No voices available. The default samples may not be loaded yet.")
    
    # Example 1: Synthesize with default voice
    print("\n" + "=" * 70)
    print("Example 1: Basic Speech Synthesis")
    print("=" * 70)
    synthesize_speech(
        text="Hello, this is a demonstration of the NeuTTS Air text to speech system.",
        voice_id=voices[0]['id'] if voices else "dave",
        output_file="example_output1.wav"
    )
    
    # Example 2: Synthesize with custom text
    print("\n" + "=" * 70)
    print("Example 2: Custom Text")
    print("=" * 70)
    synthesize_speech(
        text="The quick brown fox jumps over the lazy dog. This is a test of voice cloning.",
        voice_id=voices[0]['id'] if voices else "dave",
        output_file="example_output2.wav"
    )
    
    # Example 3: Upload a voice sample (if sample files exist)
    sample_path = Path("samples/dave.wav")
    if sample_path.exists():
        print("\n" + "=" * 70)
        print("Example 3: Upload Voice Sample")
        print("=" * 70)
        
        # Read the reference text
        ref_text_path = Path("samples/dave.txt")
        if ref_text_path.exists():
            ref_text = ref_text_path.read_text().strip()
            upload_voice_sample(
                name="dave_copy",
                audio_path=str(sample_path),
                ref_text=ref_text
            )
        else:
            # Try auto-transcription
            print("Reference text not found, trying auto-transcription...")
            upload_voice_sample(
                name="dave_copy",
                audio_path=str(sample_path),
                auto_transcribe=True
            )
    
    # Example 4: Transcribe audio (if Whisper is available)
    if sample_path.exists():
        print("\n" + "=" * 70)
        print("Example 4: Audio Transcription")
        print("=" * 70)
        transcribe_audio(str(sample_path))
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
    print("\n📝 Check the generated files:")
    print("  - example_output1.wav")
    print("  - example_output2.wav")
    print("\n🌐 You can also use the web GUI at: http://127.0.0.1:8011/gui")


if __name__ == "__main__":
    main()
