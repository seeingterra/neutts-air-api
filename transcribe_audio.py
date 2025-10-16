"""
Helper script to transcribe audio files using Whisper.
This is optional and requires openai-whisper to be installed.

Usage:
    pip install openai-whisper
    python transcribe_audio.py --audio path/to/audio.wav
"""

import argparse
import sys

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not installed. Install with: pip install openai-whisper")


def transcribe_audio(audio_path: str, model_name: str = "base") -> str:
    """
    Transcribe audio file using Whisper
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        Transcribed text
    """
    if not WHISPER_AVAILABLE:
        raise ImportError(
            "Whisper is not installed. Install it with:\n"
            "    pip install openai-whisper\n"
            "Or for faster GPU support:\n"
            "    pip install openai-whisper[gpu]"
        )
    
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    
    return result["text"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper for use with NeuTTS Air"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output file for transcription (default: prints to stdout)"
    )
    
    args = parser.parse_args()
    
    try:
        transcription = transcribe_audio(args.audio, args.model)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"Transcription saved to: {args.output}")
        else:
            print("\nTranscription:")
            print("-" * 50)
            print(transcription)
            print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
