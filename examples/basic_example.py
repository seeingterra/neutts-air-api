import os
from datetime import datetime
import soundfile as sf
from neuttsair.neutts import NeuTTSAir


def main(input_text, ref_audio_path, ref_text, backbone, output_path="output.wav", backbone_device="cpu", codec_device="cpu"):
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    # Initialize NeuTTSAir with the desired model and codec
    # Prefer ONNX codec by default for Windows portability
    codec_repo = "neuphonic/neucodec-onnx-decoder" if codec_device == "cpu" else "neuphonic/neucodec"
    tts = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        codec_device=codec_device,
    )

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    print("Encoding reference audio")
    ref_codes = tts.encode_reference(ref_audio_path)

    print(f"Generating audio for input text: {input_text}")
    wav = tts.infer(input_text, ref_codes, ref_text)

    # Avoid overwriting the default output name
    if output_path == "output.wav":
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}-{ts}{ext}"
    print(f"Saving output to {output_path}")
    sf.write(output_path, wav, 24000)


if __name__ == "__main__":
    # get arguments from command line
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir Example")
    parser.add_argument(
        "--input_text", 
        type=str, 
        required=True, 
        help="Input text to be converted to speech"
    )
    parser.add_argument(
        "--ref_audio", 
        type=str, 
        default="./samples/dave.wav", 
        help="Path to reference audio file"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/dave.txt", 
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output.wav", 
        help="Path to save the output audio"
    )
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="neuphonic/neutts-air-q4-gguf", 
        help="Backbone model repo. GGUF recommended on Windows (e.g., q4 or q8). For HF PyTorch, ensure the repo contains weights."
    )
    parser.add_argument(
        "--backbone_device",
        type=str,
        default="cpu",
        help="Device for backbone model: cpu or cuda"
    )
    parser.add_argument(
        "--codec_device",
        type=str,
        default="cpu",
        help="Device for codec model: cpu or cuda"
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        backbone=args.backbone,
        output_path=args.output_path,
        backbone_device=args.backbone_device,
        codec_device=args.codec_device,
    )
