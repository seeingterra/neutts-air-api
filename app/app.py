import io
import os
import sys
import json
import shutil
import tempfile
import importlib.util
import importlib.metadata as importlib_metadata
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import streamlit as st
from datetime import datetime
import zipfile

from huggingface_hub import snapshot_download, list_repo_files
import requests

from neuttsair.neutts import NeuTTSAir


st.set_page_config(page_title="NeuTTS Air", page_icon="☁️", layout="centered")

# Streamlit compatibility helpers (older Streamlit may lack toast or container(border=True))
from contextlib import contextmanager

def _toast(msg: str, icon: str | None = None):
    try:
        fn = getattr(st, "toast", None)
        if callable(fn):
            fn(msg, icon=icon)
        else:
            st.info(msg)
    except Exception:
        try:
            st.info(msg)
        except Exception:
            pass

@contextmanager
def bordered_container():
    try:
        # Newer Streamlit supports border kw
        with st.container(border=True):
            yield
    except TypeError:
        # Fallback without border
        with st.container():
            st.markdown("---")
            yield
            st.markdown("---")

# Streamlit rerun compatibility: ensure experimental_rerun exists if only rerun is available
try:
    if not hasattr(st, "experimental_rerun") and hasattr(st, "rerun"):
        st.experimental_rerun = st.rerun  # type: ignore[attr-defined]
except Exception:
    pass

# Ensure torchao quantizers are disabled in any downstream Transformers imports
os.environ.setdefault("TRANSFORMERS_NO_TORCHAO", "1")

# Global Torch dtype shims to avoid AttributeError during imports that reference low-bit dtypes
try:
    import torch as _torch_glob
    import sys as _sys
    import types as _types
    # Stub torch._inductor modules if missing
    if "torch._inductor" not in _sys.modules:
        _sys.modules["torch._inductor"] = _types.ModuleType("torch._inductor")
    if "torch._inductor.custom_graph_pass" not in _sys.modules:
        _sys.modules["torch._inductor.custom_graph_pass"] = _types.ModuleType("torch._inductor.custom_graph_pass")
    # Provide required attributes on the stub
    _cgp = _sys.modules.get("torch._inductor.custom_graph_pass")
    if _cgp is not None:
        if not hasattr(_cgp, "CustomGraphPass"):
            class CustomGraphPass:  # type: ignore
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, *args, **kwargs):
                    return args[0] if args else None
            _cgp.CustomGraphPass = CustomGraphPass  # type: ignore[attr-defined]
        if not hasattr(_cgp, "register_custom_graph_pass"):
            def register_custom_graph_pass(*args, **kwargs):  # type: ignore
                return None
            _cgp.register_custom_graph_pass = register_custom_graph_pass  # type: ignore[attr-defined]
        if not hasattr(_cgp, "get_hash_for_files"):
            import hashlib as _hashlib
            def get_hash_for_files(files):  # type: ignore
                try:
                    h = _hashlib.sha1()
                    for f in files or []:
                        if hasattr(f, "read"):
                            data = f.read()
                        else:
                            try:
                                with open(str(f), "rb") as _fh:
                                    data = _fh.read()
                            except Exception:
                                data = b""
                        h.update(data)
                    return h.hexdigest()
                except Exception:
                    return "0" * 40
            _cgp.get_hash_for_files = get_hash_for_files  # type: ignore[attr-defined]
    for _n in range(1, 8):
        _iname = f"int{_n}"
        _uname = f"uint{_n}"
        if not hasattr(_torch_glob, _iname):
            setattr(_torch_glob, _iname, _torch_glob.int8)  # type: ignore[attr-defined]
        if not hasattr(_torch_glob, _uname):
            setattr(_torch_glob, _uname, _torch_glob.uint8)  # type: ignore[attr-defined]
except Exception:
    pass


@st.cache_resource(show_spinner=False)
def load_tts(backbone_repo: str, codec_repo: str, backbone_device: str, codec_device: str) -> NeuTTSAir:
    return NeuTTSAir(
        backbone_repo=backbone_repo,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        codec_device=codec_device,
    )


def infer_once(
    tts: NeuTTSAir,
    input_text: str,
    ref_text: str,
    ref_wav_path: Path | None = None,
    ref_codes_path: Path | None = None,
    ref_codes_tensor: np.ndarray | None = None,
) -> np.ndarray:
    # Resolve reference codes from one of: tensor -> path -> wav
    codes = None
    if ref_codes_tensor is not None:
        codes = ref_codes_tensor
    elif ref_codes_path is not None:
        try:
            import torch as _torch
            codes = _torch.load(str(ref_codes_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load reference codes: {e}")
    elif ref_wav_path is not None:
        # This will require neucodec when using ONNX decoder; sample paths will be short-circuited above.
        codes = tts.encode_reference(str(ref_wav_path))
    else:
        raise ValueError("No reference provided. Provide ref_codes or ref_wav_path.")

    fade_in_ms = int(st.session_state.get("fade_in_ms", 0))
    wav = tts.infer(input_text, codes, ref_text, fade_in_ms=fade_in_ms)
    return wav


def write_wav_bytes(wav: np.ndarray, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _unique_output_path(prefix: str = "output", directory: Path | None = None) -> Path:
    """Return a unique WAV path like output-YYYYmmdd-HHMMSS.wav in the given directory."""
    directory = directory or Path(".")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return directory / f"{prefix}-{ts}.wav"


def _unique_codes_path(prefix: str = "ref-codes", directory: Path | None = None) -> Path:
    """Return a unique .pt path like ref-codes-YYYYmmdd-HHMMSS.pt in the given directory."""
    directory = directory or Path(".")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return directory / f"{prefix}-{ts}.pt"


def sidebar_config() -> Tuple[str, str, str, str, str]:
    st.sidebar.header("Configuration")

    mode = st.sidebar.radio(
        "Mode",
        [
            "HF (PyTorch)",
            "GGUF (llama.cpp)",
            "HF + ONNX codec",
        ],
        index=1,
        help="Choose the backbone/codec stack",
    )

    prefer_onnx = st.sidebar.checkbox(
        "Prefer ONNX codec where possible",
        value=True,
    )

    if mode == "GGUF (llama.cpp)":
        backbone_repo = st.sidebar.selectbox(
            "Backbone (GGUF)",
            [
                "neuphonic/neutts-air-q4-gguf",
                "neuphonic/neutts-air-q8-gguf",
            ],
        )
        codec_repo = st.sidebar.selectbox(
            "Codec",
            [
                "neuphonic/neucodec",  # torch codec
                "neuphonic/neucodec-onnx-decoder",  # onnx decoder
            ],
            index=1,
        )
        if prefer_onnx:
            codec_repo = "neuphonic/neucodec-onnx-decoder"
    else:
        backbone_repo = st.sidebar.text_input(
            "Backbone (HF repo)", value="neuphonic/neutts-air",
            help="This must be a repo that contains actual weights (pytorch_model.bin or model.safetensors). If not, use GGUF mode."
        )
        # Optional validation to avoid surprise failures
        with st.sidebar.expander("Validate HF repo", expanded=False):
            try:
                files = list_repo_files(backbone_repo)
                has_w = any(
                    f.endswith(("pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt", "flax_model.msgpack"))
                    for f in files
                )
                if has_w:
                    st.success("Weights found in repo.")
                else:
                    st.warning("No HF weights detected in this repo. Use GGUF mode or a repo with weights.")
            except Exception as e:
                st.info("Click to check now")
                if st.button("Check for HF weights", key="check_hf_weights"):
                    try:
                        files = list_repo_files(backbone_repo)
                        has_w = any(
                            f.endswith(("pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt", "flax_model.msgpack"))
                            for f in files
                        )
                        if has_w:
                            st.success("Weights found in repo.")
                        else:
                            st.warning("No HF weights detected in this repo. Use GGUF mode or a repo with weights.")
                    except Exception as e2:
                        st.error(f"Failed to check repo: {e2}")

        codec_repo = (
            "neuphonic/neucodec-onnx-decoder"
            if (mode.endswith("ONNX codec") or prefer_onnx)
            else "neuphonic/neucodec"
        )

    backbone_device = st.sidebar.selectbox("Backbone device", ["cpu", "cuda"], index=0)

    codec_device_choices = ["cpu", "cuda"] if codec_repo == "neuphonic/neucodec" else ["cpu"]
    codec_device = st.sidebar.selectbox("Codec device", codec_device_choices, index=0)

    st.sidebar.caption(
        "Tip: For minimal latency, use ONNX codec with pre-encoded reference (see examples)."
    )

    # Output settings
    with st.sidebar.expander("Output settings", expanded=False):
        default_dir = str(Path(".").resolve())
        out_dir_text = st.text_input(
            "Output directory",
            value=st.session_state.get("output_dir", default_dir),
            help="Folder where generated WAV files will be saved.",
        )
        st.session_state["output_dir"] = out_dir_text
        filename_base = st.text_input(
            "Filename base",
            value=st.session_state.get("filename_base", "output"),
            help="Base name used when saving files (timestamp will be appended).",
        )
        st.session_state["filename_base"] = filename_base
        fade_in_ms = st.number_input(
            "Fade-in (ms)", min_value=0, max_value=500, value=int(st.session_state.get("fade_in_ms", 0)), step=5,
            help="Optional micro fade-in to smooth any residual start artifact after trimming."
        )
        st.session_state["fade_in_ms"] = fade_in_ms

    with st.sidebar.expander("GGUF model management", expanded=False):
        st.write("Manage GGUF checkpoints: download, inspect, or remove from cache.")
        default_repos = [
            "neuphonic/neutts-air-q4-gguf",
            "neuphonic/neutts-air-q8-gguf",
        ]
        gguf_repo = st.selectbox("Select GGUF repo", default_repos, index=0)
        col_d1, col_d2 = st.columns(2)
        if col_d1.button("Download/Update", key="dl_btn"):
            with st.spinner("Downloading GGUF model from Hugging Face..."):
                local_dir = snapshot_download(repo_id=gguf_repo)
            st.success(f"Downloaded to: {local_dir}")
        if col_d2.button("Delete from cache", key="del_btn"):
            # Try to locate the cached snapshot and remove it
            try:
                local_dir = snapshot_download(repo_id=gguf_repo, local_files_only=True)
                shutil.rmtree(local_dir, ignore_errors=True)
                st.success("Removed cached snapshot: " + local_dir)
            except Exception as e:
                st.warning(f"No local snapshot found or failed to delete: {e}")

        st.markdown("---")
        st.caption("Installed GGUF snapshots (detected in local cache):")

        def list_installed(repos: List[str]):
            items = []
            for repo in repos:
                try:
                    local_dir = snapshot_download(repo_id=repo, local_files_only=True)
                except Exception:
                    continue
                # find .gguf files and compute size
                total = 0
                gguf_files = []
                for root, _, files in os.walk(local_dir):
                    for f in files:
                        if f.lower().endswith(".gguf"):
                            p = os.path.join(root, f)
                            gguf_files.append(p)
                            try:
                                total += os.path.getsize(p)
                            except OSError:
                                pass
                if gguf_files:
                    items.append((repo, local_dir, total, gguf_files))
            return items

        def human_bytes(n: int) -> str:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if n < 1024.0:
                    return f"{n:3.1f} {unit}"
                n /= 1024.0
            return f"{n:.1f} PB"

        installed = list_installed(default_repos)
        if not installed:
            st.write("No GGUF snapshots detected yet.")
        else:
            for repo, local_dir, total, gguf_files in installed:
                with st.container(border=True):
                    st.write(f"Repo: {repo}")
                    st.write(f"Path: {local_dir}")
                    st.write(f"Size: {human_bytes(total)}")
                    col_o1, col_o2 = st.columns(2)
                    if col_o1.button("Open in Explorer", key=f"open_{repo}"):
                        try:
                            os.startfile(local_dir)  # Windows only
                        except Exception as e:
                            st.error(f"Failed to open: {e}")
                    if st.button("Delete this snapshot", key=f"del_{repo}"):
                        try:
                            shutil.rmtree(local_dir, ignore_errors=True)
                            st.success("Deleted: " + local_dir)
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

        # Also provide a quick link to the HF cache root when possible
        cache_root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if os.path.isdir(cache_root):
            if st.button("Open HF cache root"):
                try:
                    os.startfile(cache_root)
                except Exception as e:
                    st.error(f"Failed to open: {e}")

    with st.sidebar.expander("Config (save/load)", expanded=False):
        cfg_path = Path("app/config.json")
        if st.button("Save current config"):
            data = {
                "mode": mode,
                "backbone_repo": backbone_repo,
                "backbone_device": backbone_device,
                "codec_repo": codec_repo,
                "codec_device": codec_device,
                "prefer_onnx": prefer_onnx,
                "output_dir": st.session_state.get("output_dir"),
                "filename_base": st.session_state.get("filename_base"),
                "api_autostart": st.session_state.get("api_autostart", False),
            }
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(data, indent=2))
            st.success(f"Saved to {cfg_path}")
        if st.button("Load config"):
            try:
                data = json.loads(cfg_path.read_text())
                for k, v in data.items():
                    st.session_state[k] = v
                st.success("Config loaded. Rerendering with loaded values...")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load config: {e}")

    return mode, backbone_repo, backbone_device, codec_repo if mode != "HF + ONNX codec" else "neuphonic/neucodec-onnx-decoder", codec_device


def main():
    st.title("NeuTTS Air ☁️")
    st.caption("Realistic on-device TTS with instant voice cloning")

    # Utility: scan samples folder for predefined voices (.pt)
    def scan_predefined_voices() -> list[tuple[str, Path, Optional[Path]]]:
        # returns list of (display_name, pt_path, txt_path)
        out: list[tuple[str, Path, Optional[Path]]] = []
        samples_dir = Path("samples")
        if samples_dir.is_dir():
            for pt in sorted(samples_dir.glob("*.pt")):
                base = pt.stem
                # derive display name from filename; allow sidecar .txt for description
                disp = base.replace("_", " ")
                txt = pt.with_suffix(".txt")
                out.append((disp, pt, txt if txt.exists() else None))
        return out

    mode, backbone_repo, backbone_device, codec_repo, codec_device = sidebar_config()

    # Inline banner if HF mode and repo seems to lack weights
    if mode.startswith("HF"):
        try:
            files = list_repo_files(backbone_repo)
            has_w = any(
                f.endswith(("pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt", "flax_model.msgpack"))
                for f in files
            )
            if not has_w:
                st.warning("HF repo appears to have no weights. Use GGUF mode or a repo with weights.")
        except Exception:
            pass

    st.subheader("Inputs")
    input_text = st.text_area("Input text", value="My name is Dave, and um, I'm from London.")
    multi = st.checkbox("Multi-speaker (batch)", value=False, help="Generate multiple speakers in one go.")
    # Top-level generate trigger under TTS area
    top_gen_clicked = st.button("Generate" if not multi else "Generate all", type="primary")

    sample_wav_path = Path("samples/dave.wav")
    predefined = scan_predefined_voices()
    has_predef = len(predefined) > 0
    reference_options: list[str] = []
    if sample_wav_path.exists():
        reference_options.append("Use sample (Dave)")
    if has_predef:
        reference_options.append("Choose predefined voice")
    reference_options.extend(["Upload .wav", "Upload codes (.pt)"])
    default_index = 0
    sample_choice = st.radio("Reference", reference_options, index=default_index, horizontal=True, disabled=multi)
    # Refresh voices control
    if has_predef:
        col_r1, col_r2 = st.columns([3, 1])
    else:
        col_r1, col_r2 = st.columns([1, 1])
    with col_r2:
        if st.button("Refresh voices", help="Rescan samples/ for new .pt files"):
            st.rerun()

    ref_text = ""
    ref_wav_path: Optional[Path] = None
    ref_codes_tensor = None
    ref_codes_path: Optional[Path] = None
    if not multi:
        if sample_choice == "Use sample (Dave)":
            ref_wav_path = sample_wav_path
            txt_path = Path("samples/dave.txt")
            if txt_path.exists():
                with open(txt_path, "r") as f:
                    ref_text = f.read().strip()
            st.audio(str(ref_wav_path))
            st.caption("Sample: Dave")
            # If ONNX codec is selected, prefer bundled pre-encoded codes to avoid requiring neucodec locally
            try:
                if codec_repo == "neuphonic/neucodec-onnx-decoder":
                    import torch as _torch
                    ref_codes_tensor = _torch.load("samples/dave.pt")
            except Exception:
                # Fallback: we'll encode from WAV (requires neucodec)
                ref_codes_tensor = None
        elif sample_choice == "Choose predefined voice":
            if not has_predef:
                st.warning("No .pt voices found in samples/. Use the pre-encode tool below to add some.")
            else:
                # Build selection list with display names
                labels = [f"{name} ({pt_path.name})" for name, pt_path, _ in predefined]
                sel = st.selectbox("Select voice", options=labels, index=0)
                idx = labels.index(sel)
                _, pt_path, txt_path = predefined[idx]
                ref_codes_path = pt_path
                if txt_path is not None:
                    try:
                        ref_text = txt_path.read_text(encoding="utf-8").strip()
                    except Exception:
                        ref_text = "This is my voice."
                else:
                    ref_text = "This is my voice."
                st.caption(f"Voice: {pt_path.name}")
        else:
            if sample_choice == "Upload .wav":
                uploaded = st.file_uploader("Upload reference .wav", type=["wav"])
                ref_text = st.text_area("Reference text", value="This is a short reference of my voice.")
                if uploaded is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded.read())
                        tmp.flush()
                        ref_wav_path = Path(tmp.name)
                    st.audio(str(ref_wav_path))
            else:
                uploaded_codes = st.file_uploader("Upload reference codes (.pt)", type=["pt"]) 
                ref_text = st.text_area("Reference text", value="This is a short reference of my voice.")
                if uploaded_codes is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                        tmp.write(uploaded_codes.read())
                        tmp.flush()
                        ref_codes_path = Path(tmp.name)
    else:
        st.markdown("Define speakers:")
        n_speakers = st.number_input("Number of speakers", min_value=1, max_value=8, value=2, step=1)
        speaker_rows = []
        for i in range(int(n_speakers)):
            with st.container(border=True):
                name = st.text_input(f"Speaker {i+1} name", value=f"speaker_{i+1}", key=f"spk_name_{i}")
                spk_text = st.text_area(f"Speaker {i+1} input text", value=input_text, key=f"spk_text_{i}")
                choice = st.radio(
                    f"Speaker {i+1} reference",
                    reference_options,
                    index=0,
                    horizontal=True,
                    key=f"spk_choice_{i}"
                )
                spk_ref_text = ""
                spk_ref_wav: Optional[Path] = None
                spk_ref_codes_path: Optional[Path] = None
                if choice == "Use sample (Dave)":
                    spk_ref_wav = sample_wav_path
                    txt_path = Path("samples/dave.txt")
                    if txt_path.exists():
                        spk_ref_text = txt_path.read_text().strip()
                    st.audio(str(spk_ref_wav))
                elif choice == "Choose predefined voice":
                    if not has_predef:
                        st.warning("No .pt voices found in samples/.")
                    else:
                        labels = [f"{nm} ({pt_path.name})" for nm, pt_path, _ in predefined]
                        sel = st.selectbox(f"Select voice (speaker {i+1})", options=labels, index=0, key=f"spk_sel_{i}")
                        sidx = labels.index(sel)
                        _, pt_path, txt_path = predefined[sidx]
                        spk_ref_codes_path = pt_path
                        if txt_path is not None:
                            try:
                                spk_ref_text = txt_path.read_text(encoding="utf-8").strip()
                            except Exception:
                                spk_ref_text = "This is my voice."
                        else:
                            spk_ref_text = "This is my voice."
                else:
                    up = st.file_uploader(f"Upload .wav (speaker {i+1})", type=["wav"], key=f"spk_up_{i}")
                    spk_ref_text = st.text_area(f"Reference text (speaker {i+1})", value="This is my voice.", key=f"spk_reftext_{i}")
                    if up is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(up.read())
                            tmp.flush()
                            spk_ref_wav = Path(tmp.name)
                        st.audio(str(spk_ref_wav))
                speaker_rows.append((name, spk_text, spk_ref_wav, spk_ref_text, spk_ref_codes_path))

    # Tools: Pre-encode reference (.wav -> .pt)
    st.divider()
    # Keep expander open after encode actions
    if "_preencode_expanded" not in st.session_state:
        st.session_state["_preencode_expanded"] = False
    with st.expander("Tools: Pre-encode references for API (.wav → .pt)", expanded=bool(st.session_state.get("_preencode_expanded", False))):
        st.caption(
            "Create reusable reference .pt files from .wav for fastest generation with the ONNX decoder.\n"
            "Tip: Drop multiple .wav files to batch-encode. Place resulting .pt files into the 'samples/' folder so the API exposes them as voices."
        )

        # Detect whether neucodec is installed
        has_neucodec = importlib.util.find_spec("neucodec") is not None
        neucodec_ver = None
        if has_neucodec:
            try:
                neucodec_ver = importlib_metadata.version("neucodec")
            except Exception:
                neucodec_ver = None
        st.info(f"neucodec: {'installed' if has_neucodec else 'not installed'}" + (f" (v{neucodec_ver})" if neucodec_ver else ""))

        # Provide in-app installer for neucodec
        if not has_neucodec:
            st.write("Install neucodec to enable .wav → .pt encoding (CPU-only). This uses your current Python environment.")
            col_i1, col_i2 = st.columns([1, 1])
            install_clicked = col_i1.button("Install neucodec")
            logs_area = st.empty()
            progress_bar = st.progress(0)
            if install_clicked:
                import subprocess, time
                logs = ""
                prog_val = 0.0
                try:
                    with st.spinner("Installing neucodec via pip..."):
                        # Run pip install in the same interpreter/venv
                        proc = subprocess.Popen(
                            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )
                        for line in proc.stdout:  # type: ignore[attr-defined]
                            logs += line
                            logs_area.code(logs[-8000:], language="bash")
                            prog_val = min(0.25, prog_val + 0.02)
                            progress_bar.progress(prog_val)
                        proc.wait()

                        proc2 = subprocess.Popen(
                            [sys.executable, "-m", "pip", "install", "neucodec"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )
                        for line in proc2.stdout:  # type: ignore[attr-defined]
                            logs += line
                            logs_area.code(logs[-8000:], language="bash")
                            # Heuristic progress update
                            prog_val = min(0.95, max(0.25, prog_val) + 0.02)
                            progress_bar.progress(prog_val)
                        ret = proc2.wait()
                        if ret == 0:
                            progress_bar.progress(1.0)
                            st.success("neucodec installed successfully. The app will reload.")
                            time.sleep(1.0)
                            st.rerun()
                        else:
                            st.error("Installation failed. See logs above.")
                except Exception as e:
                    st.error(f"Installer error: {e}")

        # All pre-encode controls and results are rendered inside this expander
        tool_choice = st.radio("Source", ["Use sample (Dave)", "Upload .wav file(s)"], index=0, horizontal=True)

        # Optional transcript generation with faster-whisper
        has_fw = importlib.util.find_spec("faster_whisper") is not None
        transcribe_enabled = st.checkbox("Auto-generate transcript (.txt) using Whisper", value=True)
        whisper_models = [
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "medium", "medium.en",
            "large-v1", "large-v2", "large-v3",
            "distil-small.en", "distil-large-v2",
        ]
        fw_model_size = st.selectbox("Whisper model size", whisper_models, index=2, disabled=not transcribe_enabled)
        fw_device = st.selectbox("Whisper device", ["cpu", "cuda"], index=0, disabled=not transcribe_enabled)
        fw_language = st.text_input("Language code (blank = auto-detect)", value="", disabled=not transcribe_enabled)
        if transcribe_enabled and not has_fw:
            st.warning("faster-whisper is not installed. Install it below to enable transcription.")
            col_fw1, col_fw2 = st.columns([1, 1])
            if col_fw1.button("Install faster-whisper"):
                import subprocess
                logs_area = st.empty()
                progress_bar = st.progress(0)
                try:
                    with st.spinner("Installing faster-whisper via pip..."):
                        prog_val = 0.0
                        proc = subprocess.Popen(
                            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )
                        logs = ""
                        for line in proc.stdout:  # type: ignore[attr-defined]
                            logs += line
                            logs_area.code(logs[-8000:], language="bash")
                            prog_val = min(0.3, prog_val + 0.02)
                            progress_bar.progress(prog_val)
                        proc.wait()
                        proc2 = subprocess.Popen(
                            [sys.executable, "-m", "pip", "install", "faster-whisper"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )
                        for line in proc2.stdout:  # type: ignore[attr-defined]
                            logs += line
                            logs_area.code(logs[-8000:], language="bash")
                            prog_val = min(0.95, max(0.3, prog_val) + 0.02)
                            progress_bar.progress(prog_val)
                        ret = proc2.wait()
                        if ret == 0:
                            progress_bar.progress(1.0)
                            st.success("faster-whisper installed. Reloading app...")
                            st.rerun()
                        else:
                            st.error("Installation failed. See logs above.")
                except Exception as e:
                    st.error(f"Installer error: {e}")

        wavs_to_encode: list[tuple[str, Path]] = []
        if tool_choice == "Use sample (Dave)":
            if sample_wav_path.exists():
                wavs_to_encode = [(sample_wav_path.name, sample_wav_path)]
                st.audio(str(sample_wav_path))
            else:
                st.warning("Sample not found at samples/dave.wav")
        else:
            up_list = st.file_uploader(
                "Upload one or more .wav files to encode (drag-and-drop supported)",
                type=["wav"],
                key="codes_up_wav_multi",
                accept_multiple_files=True,
                help="These will be converted to .pt reference codes. You can then copy them into the 'samples/' folder for API use."
            )
            if up_list:
                for up in up_list:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(up.read())
                        tmp.flush()
                        wavs_to_encode.append((up.name, Path(tmp.name)))
                # Preview first file only to avoid clutter
                st.audio(str(wavs_to_encode[0][1]))

        col_e1, col_e2, col_e3 = st.columns([1, 1, 1])
        encode_disabled = (len(wavs_to_encode) == 0 or not has_neucodec)
        if col_e1.button("Encode to .pt" if len(wavs_to_encode) <= 1 else f"Encode {len(wavs_to_encode)} file(s)", disabled=encode_disabled):
            try:
                # Lazy import to avoid heavy deps unless used
                os.environ.setdefault("TRANSFORMERS_NO_TORCHAO", "1")
                import torch as _torch
                # Compatibility shims for non-standard low-bit dtypes
                try:
                    for _n in range(1, 8):
                        _iname = f"int{_n}"
                        _uname = f"uint{_n}"
                        if not hasattr(_torch, _iname):
                            setattr(_torch, _iname, _torch.int8)  # type: ignore[attr-defined]
                        if not hasattr(_torch, _uname):
                            setattr(_torch, _uname, _torch.uint8)  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Ensure Hubert modeling imports after shims are active
                try:
                    import importlib as _importlib
                    _importlib.import_module("transformers.models.hubert.modeling_hubert")
                except Exception:
                    pass
                from neucodec import NeuCodec as _NeuCodec

                # Prepare optional transcription model
                fw_model = None
                if transcribe_enabled:
                    if not has_fw:
                        st.error("Transcription is enabled but faster-whisper is not installed.")
                        st.stop()
                    try:
                        import importlib as _il
                        _fw_mod = _il.import_module("faster_whisper")
                        _FWModel = getattr(_fw_mod, "WhisperModel")
                        compute_type = "int8" if fw_device == "cpu" else "float16"
                        fw_model = _FWModel(fw_model_size, device=fw_device, compute_type=compute_type)
                    except Exception as e:
                        st.error(f"Failed to initialize faster-whisper: {e}")
                        fw_model = None

                codec = _NeuCodec.from_pretrained("neuphonic/neucodec").to("cpu").eval()
                results: list[tuple[str, bytes, Path, Optional[str]]] = []  # (suggested_name, bytes, tmp_path, transcript)
                prog = st.progress(0)
                status = st.empty()
                total_n = max(1, len(wavs_to_encode))
                for idx, (orig_name, wav_path) in enumerate(wavs_to_encode, start=1):
                    status.write(f"Encoding {idx}/{total_n}: {Path(orig_name).name}")
                    wav, sr = sf.read(str(wav_path), always_2d=False)
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
                    wav_tensor = _torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
                    with _torch.no_grad():
                        codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)

                    import io as _io
                    buf = _io.BytesIO()
                    _torch.save(codes, buf)
                    pt_bytes = buf.getvalue()
                    base = Path(orig_name).stem
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    fname = f"{base}-{ts}.pt"
                    transcript_text: Optional[str] = None
                    if transcribe_enabled and fw_model is not None:
                        try:
                            with st.spinner(f"Transcribing {orig_name}..."):
                                language = None if (fw_language.strip() == "") else fw_language.strip()
                                segments, info = fw_model.transcribe(str(wav_path), language=language, beam_size=1)
                                parts = []
                                for seg in segments:
                                    try:
                                        parts.append(seg.text)
                                    except Exception:
                                        pass
                                transcript_text = (" ".join(parts)).strip()
                        except Exception as e:
                            st.warning(f"Transcription failed for {orig_name}: {e}")
                            transcript_text = None
                    results.append((fname, pt_bytes, wav_path, transcript_text))
                    prog.progress(min(1.0, idx/total_n))

                # Offer individual downloads and ZIP
                if not results:
                    st.warning("Nothing encoded.")
                elif len(results) == 1:
                    st.success("Reference encoded successfully.")
                    st.toast("Encoded 1 reference", icon="✅")
                    fname, data, _, transcript = results[0]
                    st.download_button("Download .pt", data=data, file_name=fname, mime="application/octet-stream", key=f"dl_pt_{fname}")
                    if transcript:
                        txt_name = Path(fname).with_suffix(".txt").name
                        st.text_area("Transcript", value=transcript, height=120, key=f"view_tr_{fname}")
                        st.download_button("Download transcript (.txt)", data=transcript, file_name=txt_name, mime="text/plain", key=f"dl_txt_{fname}")
                else:
                    st.success(f"Encoded {len(results)} references.")
                    st.toast(f"Encoded {len(results)} references", icon="✅")
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fname, data, _, transcript in results:
                            zf.writestr(fname, data)
                            if transcript:
                                zf.writestr(Path(fname).with_suffix(".txt").name, transcript)
                    zip_buf.seek(0)
                    st.download_button(
                        "Download all as ZIP",
                        data=zip_buf.getvalue(),
                        file_name=f"refs-pt-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
                        mime="application/zip",
                    )
                # Persist results and keep this expander open
                st.session_state["_preencode_expanded"] = True
                st.session_state["_last_encoded_results"] = [
                    {"fname": fname, "data": data, "wav_path": str(wav_path), "transcript": transcript or ""}
                    for (fname, data, wav_path, transcript) in results
                ]

            except ModuleNotFoundError:
                st.error("neucodec not installed. Please use the Install button above to add it to your environment.")
            except Exception as e:
                st.error(f"Encoding failed: {e}")

        # If we have previous results, present persistent download and finish/copy UI
        entries: list[dict[str, object]] = st.session_state.get("_last_encoded_results", [])  # type: ignore[assignment]
        if entries:
            st.markdown("### Encoded references")
            # Per-file downloads and transcript editor
            for i, item in enumerate(entries):
                fname = str(item["fname"])  # type: ignore[index]
                data: bytes = item["data"]  # type: ignore[index]
                tr_key = f"tr_edit_{fname}_{i}"
                with st.container():
                    cols = st.columns([2, 1, 1])
                    cols[0].write(fname)
                    cols[1].download_button("Download .pt", data=data, file_name=fname, mime="application/octet-stream", key=f"dlp_{fname}_{i}")
                    txt_name = Path(fname).with_suffix(".txt").name
                    cols[2].download_button("Download .txt", data=str(item.get("transcript", "")), file_name=txt_name, mime="text/plain", key=f"dlt_{fname}_{i}")
                    # Editable transcript
                    item["transcript"] = st.text_area("Transcript", value=str(item.get("transcript", "")), key=tr_key, height=100)

            # Bulk ZIP download
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in entries:
                    fname = str(item["fname"])  # type: ignore[index]
                    data: bytes = item["data"]  # type: ignore[index]
                    tr_text = str(item.get("transcript", ""))
                    zf.writestr(fname, data)
                    if tr_text:
                        zf.writestr(Path(fname).with_suffix(".txt").name, tr_text)
            zip_buf.seek(0)
            st.download_button("Download all as ZIP", data=zip_buf.getvalue(), file_name=f"refs-pt-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip", mime="application/zip", key="zip_all")

            # Finish & Copy to samples button
            if st.button("Finish & Copy to samples", type="primary"):
                try:
                    samples_dir = Path("samples"); samples_dir.mkdir(exist_ok=True)
                    prog = st.progress(0)
                    total = max(1, len(entries))
                    for idx, item in enumerate(entries, start=1):
                        fname = str(item["fname"])  # type: ignore[index]
                        data: bytes = item["data"]  # type: ignore[index]
                        tr_text = str(item.get("transcript", ""))
                        with open(samples_dir / fname, "wb") as f:
                            f.write(data)
                        if tr_text:
                            (samples_dir / Path(fname).with_suffix(".txt")).write_text(tr_text, encoding="utf-8")
                        prog.progress(min(1.0, idx/total))
                    st.success(f"Added {len(entries)} file(s) to samples/.")
                    _toast(f"Copied {len(entries)} file(s) into samples/", icon="📁")
                except Exception as e:
                    st.error(f"Failed to copy to samples/: {e}")

            # Clear results without closing expander
            if st.button("Clear list"):
                st.session_state["_last_encoded_results"] = []

    # OpenAI-compatible API + Voxta config
    st.divider()
    with st.expander("OpenAI-compatible API + Voxta", expanded=False):
        st.caption("Run a local OpenAI-compatible API for audio/speech and generate a Voxta provider config.")

        # API controls
        col_api1, col_api2 = st.columns([1, 1])
        api_host = col_api1.text_input("Host", value=os.environ.get("NEUTTS_API_HOST", "127.0.0.1"))
        api_port = col_api2.number_input("Port", value=int(os.environ.get("NEUTTS_API_PORT", "8011")), min_value=1, max_value=65535, step=1)

        # Optional API key auth
        api_auth_enabled = st.checkbox("Require API key (Authorization: Bearer / X-API-Key)", value=False)
        api_key = st.text_input(
            "API key (optional)",
            value="" if not api_auth_enabled else os.environ.get("NEUTTS_API_KEY", ""),
            type="password",
            help="Leave blank to disable auth.",
        )

        # Keep-warm settings (reduce latency by keeping model hot between requests)
        keep_warm = st.checkbox("Keep warm between requests", value=True, help="Runs a tiny periodic warmup to keep caches hot.")
        keep_warm_secs = st.number_input("Keep-warm interval (seconds)", value=120, min_value=5, max_value=3600, step=5)
        warmup_mode = st.selectbox("Warmup mode", ["light", "full"], index=0)
        warm_on_select = st.checkbox(
            "Warm selected voice in background",
            value=True,
            help="When a new voice is used, warm it in background to reduce first-hit latency.",
        )

        # Auto-start option (persisted to session and saved via Config section)
        st.checkbox("Auto-start API server on app launch", value=st.session_state.get("api_autostart", False), key="api_autostart")

        # Server run controls
        api_run_col1, api_run_col2 = st.columns([1, 1])
        run_key = "api_server_running"
        if run_key not in st.session_state:
            st.session_state[run_key] = False
            st.session_state["api_server_pid"] = None

        # Auto-start once per session if enabled
        if st.session_state.get("api_autostart") and not st.session_state.get(run_key) and not st.session_state.get("_api_autostart_done"):
            try:
                import subprocess
                env = os.environ.copy()
                env.setdefault("NEUTTS_BACKBONE_REPO", st.session_state.get("backbone_repo", "neuphonic/neutts-air-q4-gguf"))
                env.setdefault("NEUTTS_CODEC_REPO", st.session_state.get("codec_repo", "neuphonic/neucodec-onnx-decoder"))
                env.setdefault("NEUTTS_BACKBONE_DEVICE", st.session_state.get("backbone_device", "cpu"))
                env.setdefault("NEUTTS_CODEC_DEVICE", st.session_state.get("codec_device", "cpu"))
                repo_root = str(Path(".").resolve())
                existing_pp = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = repo_root + (os.pathsep + existing_pp if existing_pp else "")
                if api_auth_enabled and api_key:
                    env["NEUTTS_API_KEY"] = api_key
                else:
                    env.pop("NEUTTS_API_KEY", None)
                if keep_warm:
                    env["NEUTTS_KEEP_WARM"] = "1"
                    env["NEUTTS_KEEP_WARM_SECS"] = str(int(keep_warm_secs))
                else:
                    env.pop("NEUTTS_KEEP_WARM", None)
                    env.pop("NEUTTS_KEEP_WARM_SECS", None)
                env["NEUTTS_WARMUP_MODE"] = warmup_mode
                env["NEUTTS_WARM_ON_SELECT"] = "1" if warm_on_select else "0"
                runtime_dir = Path("runtime"); runtime_dir.mkdir(exist_ok=True)
                log_path = runtime_dir / "api_server.log"
                try:
                    log_file = open(log_path, "w", encoding="utf-8")
                except Exception:
                    log_file = None
                cmd = [sys.executable, "-m", "uvicorn", "server.openai_api:app", "--host", str(api_host), "--port", str(int(api_port))]
                subprocess.Popen(cmd, stdout=log_file or subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True, env=env)
                st.session_state[run_key] = True
                st.session_state["api_server_pid"] = None
                st.session_state["api_server_log"] = str(log_path)
                st.session_state["_api_autostart_done"] = True
                st.toast(f"API server auto-started on http://{api_host}:{api_port}", icon="🚀")
            except Exception as ex:
                st.error(f"Auto-start failed: {ex}")

        if not st.session_state[run_key]:
            if api_run_col1.button("Start API server"):
                import subprocess
                env = os.environ.copy()
                env.setdefault("NEUTTS_BACKBONE_REPO", st.session_state.get("backbone_repo", "neuphonic/neutts-air-q4-gguf"))
                env.setdefault("NEUTTS_CODEC_REPO", st.session_state.get("codec_repo", "neuphonic/neucodec-onnx-decoder"))
                env.setdefault("NEUTTS_BACKBONE_DEVICE", st.session_state.get("backbone_device", "cpu"))
                env.setdefault("NEUTTS_CODEC_DEVICE", st.session_state.get("codec_device", "cpu"))
                # Ensure our workspace is importable by uvicorn
                repo_root = str(Path(".").resolve())
                existing_pp = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = repo_root + (os.pathsep + existing_pp if existing_pp else "")
                if api_auth_enabled and api_key:
                    env["NEUTTS_API_KEY"] = api_key
                elif "NEUTTS_API_KEY" in env:
                    del env["NEUTTS_API_KEY"]
                if keep_warm:
                    env["NEUTTS_KEEP_WARM"] = "1"
                    env["NEUTTS_KEEP_WARM_SECS"] = str(int(keep_warm_secs))
                else:
                    env.pop("NEUTTS_KEEP_WARM", None)
                    env.pop("NEUTTS_KEEP_WARM_SECS", None)
                env["NEUTTS_WARMUP_MODE"] = warmup_mode
                env["NEUTTS_WARM_ON_SELECT"] = "1" if warm_on_select else "0"
                # Launch uvicorn in background, write logs to file so users can inspect
                runtime_dir = Path("runtime")
                runtime_dir.mkdir(exist_ok=True)
                log_path = runtime_dir / "api_server.log"
                try:
                    log_file = open(log_path, "w", encoding="utf-8")
                except Exception:
                    log_file = None
                cmd = [sys.executable, "-m", "uvicorn", "server.openai_api:app", "--host", str(api_host), "--port", str(int(api_port))]
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_file or subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    st.session_state[run_key] = True
                    st.session_state["api_server_pid"] = proc.pid
                    st.session_state["api_server_log"] = str(log_path)
                    st.success(f"API server starting on http://{api_host}:{api_port}")
                except Exception as ex:
                    st.error(f"Failed to start API server: {ex}")
        else:
            api_run_col1.success(f"Running at http://{api_host}:{api_port}")
            if api_run_col2.button("Stop API server"):
                try:
                    pid = st.session_state.get("api_server_pid")
                    if pid:
                        # Try cross-platform termination
                        try:
                            import signal
                            os.kill(pid, signal.SIGTERM)
                        except Exception:
                            # Windows fallback
                            os.system(f"taskkill /PID {pid} /T /F >nul 2>&1")
                    st.session_state[run_key] = False
                    st.session_state["api_server_pid"] = None
                except Exception as ex:
                    st.error(f"Failed to stop: {ex}")

        # Health check and log viewer
        col_h1, col_h2 = st.columns([1, 1])
        if col_h1.button("Check API health"):
            try:
                url = f"http://{api_host}:{int(api_port)}/health"
                headers = {}
                if api_auth_enabled and api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                r = requests.get(url, timeout=2.5, headers=headers)
                if r.ok:
                    st.success(f"Health OK: {r.json()}")
                else:
                    st.warning(f"Health check returned {r.status_code}")
            except Exception as ex:
                st.error(f"Health check failed: {ex}")
        log_file_shown = st.session_state.get("api_server_log")
        if log_file_shown and col_h2.button("Show latest logs"):
            try:
                with open(log_file_shown, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()[-16000:]
                st.code(content or "(log is empty)", language="bash")
            except Exception as ex:
                st.error(f"Could not read log: {ex}")

        # One-shot warmup action (auth header passed if provided)
        if st.session_state.get("api_server_pid") and st.button("Warm up now"):
            try:
                url = f"http://{api_host}:{int(api_port)}/admin/warmup?mode={warmup_mode}"
                headers = {}
                if api_auth_enabled and api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                r = requests.get(url, timeout=10, headers=headers)
                if r.ok:
                    st.success("Warmup completed.")
                else:
                    st.warning(f"Warmup returned {r.status_code}: {r.text}")
            except Exception as ex:
                st.error(f"Warmup failed: {ex}")

        # Metrics viewer
        if st.session_state.get("api_server_pid") and st.button("Show metrics"):
            try:
                url = f"http://{api_host}:{int(api_port)}/metrics"
                headers = {}
                if api_auth_enabled and api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                r = requests.get(url, timeout=5, headers=headers)
                if r.ok:
                    st.json(r.json())
                else:
                    st.warning(f"Metrics returned {r.status_code}")
            except Exception as ex:
                st.error(f"Metrics failed: {ex}")

        st.markdown("---")
        st.subheader("Generate Voxta Provider Config")
        # Fields based on example, adapted for our API
        label = st.text_input("Provider label", value="NeuTTS Air (local)")
        voice_default = st.text_input("Default voice", value="dave")
        docs_url = f"http://{api_host}:{api_port}/docs"
        provider_notes = st.text_input("Provider notes", value=f"Ensure the local API runs on port {api_port}. Docs: {docs_url}")

        headers_val = ""
        if api_auth_enabled and api_key:
            headers_val = f"Authorization: Bearer {api_key}"

        voxta_config = {
            "label": label,
            "values": {
                "ContentType": "audio/mpeg",
                "ForceConversion": "true",
                "UrlTemplate": f"http://{api_host}:{api_port}/tts",
                "Request ContentType": "application/json",
                "RequestBody": (
                    "{\n"
                    "\"text\": \"{{ text }}\",\n"
                    "\"voice_mode\": \"predefined\",\n"
                    "\"predefined_voice_id\": \"{{ voice_id }}\",\n"
                    "\"reference_audio_filename\": \"\",\n"
                    "\"output_format\": \"wav\",\n"
                    "\"split_text\": false,\n"
                    "\"chunk_size\": 120,\n"
                    "\"temperature\": 0.8,\n"
                    "\"exaggeration\": 0.8,\n"
                    "\"cfg_weight\": 0.5,\n"
                    "\"seed\": 0,\n"
                    "\"speed_factor\": 1,\n"
                    "\"culture\": \"{{ culture }}\",\n"
                    "\"language\": \"{{ language }}\"\n"
                    "}"
                ),
                "VoicesFormat": "{\"label\":\"{{display_name}}\", \"parameters\": { \"voice_id\": \"{{ filename }}\" } }\n",
                "ProviderNotes": provider_notes,
                "DocsIndex": docs_url,
                "GPURequirements": "CPU works via GGUF+ONNX; CUDA optional.",
                "DebugEndpoints": "/health, /get_predefined_voices",
                "ThinkingSpeech": "..",
                "AudioGap": "0",
                "VoicesUrl": f"http://{api_host}:{api_port}/get_predefined_voices",
                "AuthorizationHeader": headers_val
            }
        }

        cfg_str = json.dumps(voxta_config, indent=2)
        st.code(cfg_str, language="json")
        col_c1, col_c2 = st.columns([1, 1])
        col_c1.download_button("Download Voxta provider JSON", data=cfg_str, file_name="voxta_neutts_provider.json", mime="application/json")
        if col_c2.button("Copy to clipboard"):
            try:
                # Streamlit clipboard support is limited; show a hint for manual copy
                st.info("Select the JSON code block and press Ctrl+C to copy.")
            except Exception:
                pass

    st.divider()
    col_gen, col_clear = st.columns([1, 1])
    # Single or batch generate
    gen_disabled = False if multi else ((ref_wav_path is None and ref_codes_tensor is None and ref_codes_path is None) or len(input_text.strip()) == 0)
    if top_gen_clicked and not gen_disabled:
        try:
            with st.spinner("Loading models and generating..."):
                # Preflight check for HF modes: ensure repo contains weights to avoid import failures
                if mode.startswith("HF"):
                    try:
                        files = list_repo_files(backbone_repo)
                        has_w = any(
                            f.endswith(("pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt", "flax_model.msgpack"))
                            for f in files
                        )
                    except Exception:
                        has_w = False
                    if not has_w:
                        st.error("Selected HF repo appears to have no weights. Please use GGUF mode or choose a repo that contains model weights.")
                        st.stop()
                tts = load_tts(backbone_repo, codec_repo, backbone_device, codec_device)
                out_dir = Path(st.session_state.get("output_dir", "."))
                out_dir.mkdir(parents=True, exist_ok=True)
                base = st.session_state.get("filename_base", "output")

                if not multi:
                    # Session-level cache for uploaded reference WAV encodes
                    if "_codes_cache" not in st.session_state:
                        st.session_state["_codes_cache"] = {}
                    codes_cache: dict[str, object] = st.session_state["_codes_cache"]

                    # If a WAV is provided and not already pre-encoded, compute a digest and reuse encodes
                    local_ref_codes_tensor = ref_codes_tensor
                    local_ref_wav_path = ref_wav_path
                    if local_ref_wav_path is not None and local_ref_codes_tensor is None and ref_codes_path is None:
                        try:
                            import hashlib
                            data = Path(local_ref_wav_path).read_bytes()
                            key = hashlib.sha1(data).hexdigest()
                            cached = codes_cache.get(key)
                            if cached is None:
                                # Encode once and cache
                                local_ref_codes_tensor = tts.encode_reference(str(local_ref_wav_path))
                                codes_cache[key] = local_ref_codes_tensor
                            else:
                                local_ref_codes_tensor = cached
                            # Switch to codes path (avoid re-encoding inside infer_once)
                            local_ref_wav_path = None
                        except Exception:
                            pass

                    wav = infer_once(
                        tts,
                        input_text.strip(),
                        ref_text.strip(),
                        ref_wav_path=local_ref_wav_path,
                        ref_codes_path=ref_codes_path,
                        ref_codes_tensor=local_ref_codes_tensor,
                    )
                    audio_bytes = write_wav_bytes(wav, 24000)
                    st.success("Done!")
                    st.audio(audio_bytes, format="audio/wav")
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    dl_name = f"neutts_air-{ts}.wav"
                    st.download_button("Download WAV", data=audio_bytes, file_name=dl_name, mime="audio/wav")
                    # Save unique file
                    out_path = _unique_output_path(prefix=base, directory=out_dir)
                    try:
                        with open(out_path, "wb") as f:
                            f.write(audio_bytes)
                        col_s1, col_s2 = st.columns(2)
                        col_s1.caption(f"Saved to: {out_path.resolve()}")
                        if col_s2.button("Open folder"):
                            try:
                                os.startfile(str(out_dir.resolve()))
                            except Exception as e:
                                st.error(f"Failed to open folder: {e}")
                    except Exception:
                        pass
                else:
                    # Batch generate for multiple speakers
                    results: List[tuple[str, bytes]] = []
                    if "_codes_cache" not in st.session_state:
                        st.session_state["_codes_cache"] = {}
                    codes_cache: dict[str, object] = st.session_state["_codes_cache"]
                    for (name, spk_text, spk_ref_wav, spk_ref_text, spk_ref_codes_path) in speaker_rows:
                        if (spk_ref_wav is None and spk_ref_codes_path is None) or len(spk_text.strip()) == 0:
                            continue
                        # Cache encode per unique WAV content to avoid repeated encodes during the session
                        local_codes = None
                        if spk_ref_codes_path is not None:
                            try:
                                import torch as _torch
                                local_codes = _torch.load(str(spk_ref_codes_path))
                            except Exception:
                                local_codes = None
                        elif spk_ref_wav is not None:
                            try:
                                import hashlib
                                data = Path(spk_ref_wav).read_bytes()
                                key = hashlib.sha1(data).hexdigest()
                                cached = codes_cache.get(key)
                                if cached is None:
                                    local_codes = tts.encode_reference(str(spk_ref_wav))
                                    codes_cache[key] = local_codes
                                else:
                                    local_codes = cached
                            except Exception:
                                pass
                        wav = infer_once(
                            tts,
                            spk_text.strip(),
                            spk_ref_text.strip(),
                            ref_wav_path=None if (local_codes is not None or spk_ref_codes_path is not None) else spk_ref_wav,
                            ref_codes_tensor=local_codes,
                            ref_codes_path=spk_ref_codes_path,
                        )
                        audio_bytes = write_wav_bytes(wav, 24000)
                        results.append((name, audio_bytes))

                    if not results:
                        st.warning("No valid speaker rows to generate.")
                    else:
                        st.success(f"Generated {len(results)} speaker(s).")
                        # Offer per-file downloads and save to disk
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                            for name, data in results:
                                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                                fname = f"{base}-{name}-{ts}.wav"
                                # Save individual
                                out_path = out_dir / fname
                                try:
                                    with open(out_path, "wb") as f:
                                        f.write(data)
                                except Exception:
                                    pass
                                # Add to zip
                                zf.writestr(fname, data)
                                # Show inline player + download
                                with st.container(border=True):
                                    st.write(f"Speaker: {name}")
                                    st.audio(data, format="audio/wav")
                                    st.download_button(
                                        f"Download {fname}", data=data, file_name=fname, mime="audio/wav"
                                    )
                        zip_buf.seek(0)
                        st.download_button(
                            "Download all as ZIP",
                            data=zip_buf.getvalue(),
                            file_name=f"{base}-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
                            mime="application/zip",
                        )
                        if st.button("Open output folder"):
                            try:
                                os.startfile(str(out_dir.resolve()))
                            except Exception as e:
                                st.error(f"Failed to open folder: {e}")
        except Exception as e:
            st.error(f"Generation failed: {e}")

    if col_clear.button("Clear cache"):
        load_tts.clear()
        st.toast("Model cache cleared", icon="🧹")


if __name__ == "__main__":
    main()
