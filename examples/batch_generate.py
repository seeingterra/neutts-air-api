"""
Batch multi-speaker generation (CLI)

Usage (JSON config):
  python -m examples.batch_generate --config configs/batch.json

Config schema (JSON or YAML):
{
  "backbone_repo": "neuphonic/neutts-air-q4-gguf",
  "backbone_device": "cpu",                 # cpu | cuda | gpu (llama-cpp)
  "codec_repo": "neuphonic/neucodec-onnx-decoder",  # onnx recommended on Windows
  "codec_device": "cpu",
  "output_dir": "outputs",
  "filename_base": "run",
  "speakers": [
    {
      "name": "dave",
      "text": "Hello from Dave.",
      "ref_audio": "samples/dave.wav",
      "ref_text": "samples/dave.txt"   # can be a path or inline text
      # Optional: "ref_codes": "samples/dave.pt"   # pre-encoded reference codes
    }
  ]
}

Notes:
- If codec_repo is ONNX and a speaker only provides ref_audio (no ref_codes), this script will locally encode
  the reference using the torch codec (neuphonic/neucodec) on CPU to be ONNX-friendly for decoding.
  This avoids loading the TTS backbone just for encoding.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from neuttsair.neutts import NeuTTSAir


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_text(text_or_path: Optional[str]) -> str:
    if not text_or_path:
        return ""
    p = Path(text_or_path)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return str(text_or_path).strip()


def _encode_with_neucodec(ref_audio_path: str | Path) -> torch.Tensor:
    """Encode reference audio to codes using neucodec directly (CPU)."""
    from neucodec import NeuCodec
    import soundfile as _sf
    import numpy as _np

    wav, sr = _sf.read(str(ref_audio_path), always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.dtype != _np.float32:
        wav = wav.astype(_np.float32, copy=False)
    target_sr = 16000
    if sr != target_sr:
        duration = len(wav) / float(sr)
        new_length = int(round(duration * target_sr))
        if new_length <= 0:
            raise ValueError("Invalid resample length computed.")
        x_old = _np.linspace(0.0, duration, num=len(wav), endpoint=False, dtype=_np.float64)
        x_new = _np.linspace(0.0, duration, num=new_length, endpoint=False, dtype=_np.float64)
        wav = _np.interp(x_new, x_old, wav).astype(_np.float32, copy=False)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cpu")
    with torch.no_grad():
        codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
    return codes


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requested but PyYAML is not installed. Install 'pyyaml'.") from e
        return yaml.safe_load(text)  # type: ignore
    return json.loads(text)


def run_batch(cfg: Dict[str, Any]) -> List[Path]:
    backbone_repo = cfg.get("backbone_repo", "neuphonic/neutts-air-q4-gguf")
    backbone_device = cfg.get("backbone_device", "cpu")
    codec_repo = cfg.get("codec_repo", "neuphonic/neucodec-onnx-decoder")
    codec_device = cfg.get("codec_device", "cpu")
    out_dir = Path(cfg.get("output_dir", "outputs"))
    base = str(cfg.get("filename_base", "output"))
    speakers = cfg.get("speakers", [])

    out_dir.mkdir(parents=True, exist_ok=True)

    tts = NeuTTSAir(
        backbone_repo=backbone_repo,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        codec_device=codec_device,
    )

    results: List[Path] = []
    for spk in speakers:
        name = str(spk.get("name", "speaker"))
        text = _read_text(spk.get("text", ""))
        if not text:
            print(f"[skip] speaker '{name}': empty text")
            continue

        ref_codes_path = spk.get("ref_codes")
        ref_audio = spk.get("ref_audio")
        ref_text = _read_text(spk.get("ref_text"))

        codes = None
        if ref_codes_path and Path(ref_codes_path).exists():
            codes = torch.load(ref_codes_path)
        elif ref_audio and Path(ref_audio).exists():
            if codec_repo == "neuphonic/neucodec-onnx-decoder":
                # Encode via neucodec on CPU to be ONNX-friendly for decoding
                codes = _encode_with_neucodec(ref_audio)
            else:
                codes = tts.encode_reference(ref_audio)
        else:
            print(f"[skip] speaker '{name}': no ref_codes or ref_audio found")
            continue

        print(f"[gen] {name}...")
        wav = tts.infer(text, codes, ref_text)
        ts = _timestamp()
        out_path = out_dir / f"{base}-{name}-{ts}.wav"
        sf.write(out_path, wav, 24000)
        results.append(out_path)
        print(f"[ok] -> {out_path}")

    return results


def main():
    ap = argparse.ArgumentParser(description="Batch multi-speaker TTS generation")
    ap.add_argument("--config", required=True, help="Path to batch config JSON/YAML")
    ap.add_argument("--zip", action="store_true", help="Zip outputs after generation")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = _load_config(cfg_path)
    outputs = run_batch(cfg)

    if args.zip and outputs:
        import zipfile

        ts = _timestamp()
        zip_path = Path(cfg.get("output_dir", "outputs")) / f"batch-{ts}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in outputs:
                zf.write(p, arcname=p.name)
        print(f"[zip] -> {zip_path}")


if __name__ == "__main__":
    main()
