# This file contains an example of how to use the NeuTTSAir class to generate codes

import torch
import soundfile as sf
import numpy as np
from neucodec import NeuCodec


def main(ref_audio_path, output_path="output.pt"):
    print("Encoding reference audio")

    # Make sure output path ends with .pt
    if not output_path.endswith(".pt"):
        print("Output path should end with .pt to save the codes.")
        return

    # Initialize codec
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cpu")

    # Load and encode reference audio
    wav, sr = sf.read(str(ref_audio_path), always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32, copy=False)
    target_sr = 16000
    if sr != target_sr:
        duration = len(wav) / float(sr)
        new_length = int(round(duration * target_sr))
        x_old = np.linspace(0.0, duration, num=len(wav), endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, duration, num=new_length, endpoint=False, dtype=np.float64)
        wav = np.interp(x_new, x_old, wav).astype(np.float32, copy=False)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)

    # Save the codes
    torch.save(ref_codes, output_path)


if __name__ == "__main__":
    # get arguments from command line
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir Reference Encoding Example")
    parser.add_argument(
        "--ref_audio", type=str, default="./samples/dave.wav", help="Path to reference audio"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="encoded_reference.pt",
        help="Path to save the output codes",
    )
    args = parser.parse_args()
    main(
        ref_audio_path=args.ref_audio,
        output_path=args.output_path,
    )
