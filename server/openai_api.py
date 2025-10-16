import io
import os
from pathlib import Path
from typing import Optional, List, Dict
import threading
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, Request, Depends, Query
from pydantic import BaseModel

from neuttsair.neutts import NeuTTSAir


# Configuration via environment variables with safe Windows defaults
BACKBONE_REPO = os.getenv("NEUTTS_BACKBONE_REPO", "neuphonic/neutts-air-q4-gguf")
CODEC_REPO = os.getenv("NEUTTS_CODEC_REPO", "neuphonic/neucodec-onnx-decoder")
BACKBONE_DEVICE = os.getenv("NEUTTS_BACKBONE_DEVICE", "cpu")
CODEC_DEVICE = os.getenv("NEUTTS_CODEC_DEVICE", "cpu")
KEEP_WARM = os.getenv("NEUTTS_KEEP_WARM", "0") in ("1", "true", "True")
KEEP_WARM_SECS = max(5, int(os.getenv("NEUTTS_KEEP_WARM_SECS", "120")))
WARMUP_MODE = os.getenv("NEUTTS_WARMUP_MODE", "light").lower()  # light|full
WARM_ON_SELECT = os.getenv("NEUTTS_WARM_ON_SELECT", "1") in ("1", "true", "True")


class AudioSpeechRequest(BaseModel):
    model: Optional[str] = None
    voice: Optional[str] = None
    input: str
    response_format: Optional[str] = "wav"
    format: Optional[str] = None  # accept alternate field name
    language: Optional[str] = None
    emotion: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    voice_mode: Optional[str] = None  # expected 'predefined'
    predefined_voice_id: Optional[str] = None
    reference_audio_filename: Optional[str] = None
    output_format: Optional[str] = "wav"
    split_text: Optional[bool] = None
    chunk_size: Optional[int] = None
    temperature: Optional[float] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    seed: Optional[int] = None
    speed_factor: Optional[float] = None
    culture: Optional[str] = None
    language: Optional[str] = None


app = FastAPI(title="NeuTTS Air OpenAI-Compatible API", version="0.1.0")


_tts: Optional[NeuTTSAir] = None
_voice_cache: dict[str, tuple[np.ndarray, str]] = {}
_API_KEY: Optional[str] = os.getenv("NEUTTS_API_KEY") or None
_warm_thread: Optional[threading.Thread] = None
_warm_stop = threading.Event()
_warm_info: dict[str, dict] = {}
_warm_inflight: set[str] = set()
_lat_ema: dict[str, float] = {}
_start_time = time.time()
_startup_error: Optional[str] = None


def _norm_voice_id(name: str) -> str:
    try:
        stem = Path(str(name)).stem
    except Exception:
        stem = str(name)
    return stem.strip().lower().replace(" ", "_")


def _write_wav_bytes(wav: np.ndarray, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _load_sample_voice(name: str) -> Optional[tuple[np.ndarray, str]]:
    name = _norm_voice_id(name)
    base = Path("samples")
    pt = base / f"{name}.pt"
    txt = base / f"{name}.txt"
    if not pt.exists():
        return None
    try:
        import torch
        codes = torch.load(str(pt))
    except Exception:
        return None
    ref_text = txt.read_text(encoding="utf-8").strip() if txt.exists() else "This is my voice."
    if not ref_text:
        ref_text = "This is my voice."
    return codes, ref_text


def _scan_sample_voices() -> List[Dict[str, str]]:
    base = Path("samples")
    voices: List[Dict[str, str]] = []
    if not base.exists():
        return voices
    for pt in base.glob("*.pt"):
        name = _norm_voice_id(pt.stem)
        display = name.replace("_", " ").title()
        voices.append({
            "display_name": display,
            "filename": name,
        })
    return voices


def _auth_guard(request: Request):
    """Optional API key auth: if NEUTTS_API_KEY is set, require it.
    Accept either Authorization: Bearer <key> or X-API-Key: <key>.
    """
    if not _API_KEY:
        return  # no auth enforced
    auth = request.headers.get("authorization")
    xkey = request.headers.get("x-api-key")
    ok = False
    if xkey and xkey == _API_KEY:
        ok = True
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1]
        if token == _API_KEY:
            ok = True
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
def _startup():
    global _tts, _voice_cache, _startup_error
    # Initialize TTS once, but don't crash the server if it fails; record error instead
    os.environ.setdefault("TRANSFORMERS_NO_TORCHAO", "1")
    try:
        _tts = NeuTTSAir(
            backbone_repo=BACKBONE_REPO,
            backbone_device=BACKBONE_DEVICE,
            codec_repo=CODEC_REPO,
            codec_device=CODEC_DEVICE,
        )
    except Exception as e:
        _tts = None
        _startup_error = f"TTS init failed: {e}"
        # Continue startup so /health can report the issue
        return

    # Preload known sample voices if available
    for v in ("dave", "jo"):
        vdat = _load_sample_voice(v)
        if vdat is not None:
            _voice_cache[v] = vdat

    # Also scan the samples folder for any additional voices
    for entry in _scan_sample_voices():
        name = entry.get("filename")
        if name and name not in _voice_cache:
            vdat = _load_sample_voice(name)
            if vdat is not None:
                _voice_cache[name] = vdat

    # Immediate one-time warmup to prime ONNX + backbone
    try:
        _do_warmup(mode=WARMUP_MODE)
    except Exception:
        pass

    # Optional keep-warm thread (only if TTS initialized)
    if KEEP_WARM and _tts is not None:
        _warm_stop.clear()
        def _loop():
            while not _warm_stop.is_set():
                try:
                    _do_warmup(mode="light")
                except Exception:
                    pass
                # Sleep with early exit
                _warm_stop.wait(KEEP_WARM_SECS)
        global _warm_thread
        _warm_thread = threading.Thread(target=_loop, name="neutts-keep-warm", daemon=True)
        _warm_thread.start()


@app.get("/health")
def health():
    if _startup_error:
        return {"status": "error", "detail": _startup_error}
    return {"status": "ok"}


@app.get("/admin/warmup")
def admin_warmup(
    voice: Optional[str] = Query(default=None, description="Voice id (stem of .pt) to warm"),
    mode: Optional[str] = Query(default=None, description="Warmup mode: light|full"),
    _: None = Depends(_auth_guard),
):
    try:
        _do_warmup(voice=voice, mode=(mode or WARMUP_MODE))
        return {"status": "warmed", "voice": voice or "auto", "mode": mode or WARMUP_MODE}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")


@app.get("/v1/models")
def list_models(_: None = Depends(_auth_guard)):
    return {"data": [{"id": BACKBONE_REPO}]}


@app.get("/v1/voices")
@app.get("/v1/voxta/voices")
def list_voices(_: None = Depends(_auth_guard)):
    items = []
    for key, (codes, _) in _voice_cache.items():
        items.append({
            "label": key.capitalize(),
            "voice": key,
            "parameters": {"voice": key},
        })
    return items


@app.get("/get_predefined_voices")
def get_predefined_voices(_: None = Depends(_auth_guard)):
    """Return voices in chatterbox-format: list of {display_name, filename}."""
    return _scan_sample_voices()


@app.post("/v1/audio/speech")
def audio_speech(body: AudioSpeechRequest, _: None = Depends(_auth_guard)):
    global _tts
    if _tts is None:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    # Determine response format
    fmt = (body.format or body.response_format or "wav").lower()
    if fmt not in ("wav",):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    # Pick voice
    voice = _norm_voice_id(body.voice or "dave")
    if voice not in _voice_cache:
        # Try to load on-the-fly from samples
        vdat = _load_sample_voice(voice)
        if vdat is None:
            raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")
        _voice_cache[voice] = vdat
    codes, ref_text = _voice_cache[voice]

    # Background warm-on-select (if configured)
    _maybe_background_warm(voice)

    try:
        t0 = time.perf_counter()
        # Normalize whitespace/newlines to avoid multi-line alignment issues
        user_text = " ".join((body.input or "").replace("\r\n", "\n").replace("\r", "\n").split())
        ref_text_norm = " ".join((ref_text or "").replace("\r\n", "\n").replace("\r", "\n").split())
        wav = _tts.infer(user_text, codes, ref_text_norm)
        dt = time.perf_counter() - t0
        _update_latency("/v1/audio/speech", dt)
        audio_bytes = _write_wav_bytes(wav, 24000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    return Response(content=audio_bytes, media_type="audio/wav")


def _do_warmup(voice: Optional[str] = None, mode: str = "light"):
    # Pick a voice: prefer 'dave', else any cached, else try to load first available sample
    v = voice
    if not v:
        if "dave" in _voice_cache:
            v = "dave"
        elif _voice_cache:
            v = next(iter(_voice_cache.keys()))
    if not v:
        # attempt scan/load
        scanned = _scan_sample_voices()
        if scanned:
            candidate = scanned[0]["filename"]
            vdat = _load_sample_voice(candidate)
            if vdat is not None:
                _voice_cache[candidate] = vdat
                v = candidate
    if not v or _tts is None:
        return
    codes, ref_text = _voice_cache[v]
    text = "Hi." if mode == "light" else "Hello there. This is a warmup to prime caches."
    # Run a very small inference to force-load kernels and caches
    t0 = time.perf_counter()
    _ = _tts.infer(text, codes, ref_text)
    dt = time.perf_counter() - t0
    # Record warm state
    _warm_info[v] = {
        "last": time.time(),
        "count": _warm_info.get(v, {}).get("count", 0) + 1,
        "mode": mode,
        "last_dt": round(dt, 4),
    }


@app.on_event("shutdown")
def _shutdown():
    # Stop keep-warm thread
    try:
        _warm_stop.set()
    except Exception:
        pass


def _maybe_background_warm(voice: str):
    try:
        # If explicit warm-on-select disabled, only do this when KEEP_WARM is enabled
        if not (WARM_ON_SELECT or KEEP_WARM):
            return
        info = _warm_info.get(voice, {})
        last = float(info.get("last", 0))
        if (time.time() - last) < KEEP_WARM_SECS:
            return
        if voice in _warm_inflight:
            return
        _warm_inflight.add(voice)
        def _bg():
            try:
                _do_warmup(voice=voice, mode="light")
            finally:
                _warm_inflight.discard(voice)
        threading.Thread(target=_bg, name=f"warm-{voice}", daemon=True).start()
    except Exception:
        pass


def _update_latency(name: str, dt: float):
    alpha = 0.2
    prev = _lat_ema.get(name, dt)
    _lat_ema[name] = alpha * dt + (1 - alpha) * prev


@app.get("/metrics")
def metrics(_: None = Depends(_auth_guard)):
    return {
        "uptime_s": round(time.time() - _start_time, 3),
        "latency_ema_s": {k: round(v, 4) for k, v in _lat_ema.items()},
        "warm": _warm_info,
        "voices_cached": list(_voice_cache.keys()),
    }


@app.post("/tts")
def tts(body: TTSRequest, _: None = Depends(_auth_guard)):
    global _tts
    if _tts is None:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    # Determine voice id
    voice_id = _norm_voice_id(body.predefined_voice_id or body.reference_audio_filename or "dave")
    # Ensure in cache or load from samples
    if voice_id not in _voice_cache:
        vdat = _load_sample_voice(voice_id)
        if vdat is None:
            raise HTTPException(status_code=400, detail=f"Unknown voice: {voice_id}")
        _voice_cache[voice_id] = vdat

    _maybe_background_warm(voice_id)
    codes, ref_text = _voice_cache[voice_id]
    try:
        t0 = time.perf_counter()
        user_text = " ".join((body.text or "").replace("\r\n", "\n").replace("\r", "\n").split())
        ref_text_norm = " ".join((ref_text or "").replace("\r\n", "\n").replace("\r", "\n").split())
        wav = _tts.infer(user_text, codes, ref_text_norm)
        dt = time.perf_counter() - t0
        _update_latency("/tts", dt)
        audio_bytes = _write_wav_bytes(wav, 24000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    return Response(content=audio_bytes, media_type="audio/wav")
