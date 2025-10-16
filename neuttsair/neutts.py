from typing import Generator, Optional
from pathlib import Path
import os
import platform
import soundfile as sf
import numpy as np
import torch
import re
import perth
import importlib

# Avoid importing torchao quantizers which can be incompatible with some Torch builds.
# This must be set BEFORE importing `transformers`.
os.environ.setdefault("TRANSFORMERS_NO_TORCHAO", "1")

# Compatibility shim: some modules reference non-standard low-bit dtypes (intN/uintN where N<8).
# Map them to nearest 8-bit equivalents to avoid AttributeError during imports.
try:
    for _n in range(1, 8):
        _iname = f"int{_n}"
        _uname = f"uint{_n}"
        if not hasattr(torch, _iname):
            setattr(torch, _iname, torch.int8)  # type: ignore[attr-defined]
        if not hasattr(torch, _uname):
            setattr(torch, _uname, torch.uint8)  # type: ignore[attr-defined]
except Exception:
    pass

import sys
import types

# Stub torch._inductor modules if missing to satisfy some transformer/neucodec import paths
try:
    if "torch._inductor" not in sys.modules:
        sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
    if "torch._inductor.custom_graph_pass" not in sys.modules:
        sys.modules["torch._inductor.custom_graph_pass"] = types.ModuleType("torch._inductor.custom_graph_pass")
    # Ensure required attributes exist
    _cgp = sys.modules.get("torch._inductor.custom_graph_pass")
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
except Exception:
    pass

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from threading import Thread


def _linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    # original impl --> https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


class NeuTTSAir:

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-air",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
    ):

        # Consts
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        # ggml & onnx flags
        self._is_quantized_model = False
        self._is_onnx_codec = False

        # HF tokenizer
        self.tokenizer = None

        # Load phonemizer + models
        print("Loading phonemizer...")
        self._configure_espeak()
        try:
            self.phonemizer = EspeakBackend(
                language="en-us", preserve_punctuation=True, with_stress=True
            )
        except Exception as e:
            # Fallback to a naive phonemizer passthrough to keep API alive; better than crashing
            print(f"Phonemizer init failed ({e}); using fallback passthrough.")
            class _FallbackPhonemizer:
                def phonemize(self, lines):
                    # Return tokenized words as-is; not ideal but prevents crashes
                    out = []
                    for ln in lines:
                        ln = str(ln or "")
                        ln = ln.replace("\r\n", "\n").replace("\r", "\n")
                        out.append(" ".join(ln.split()))
                    return out
            self.phonemizer = _FallbackPhonemizer()

        self._load_backbone(backbone_repo, backbone_device)

        self._load_codec(codec_repo, codec_device)

        # Load watermarker (fallback to no-op if unavailable)
        self.watermarker = self._init_watermarker()

    def _init_watermarker(self):
        class _NoOpWatermarker:
            def apply_watermark(self, wav, sample_rate: int):
                return wav

        try:
            wm_cls = getattr(perth, "PerthImplicitWatermarker", None)
            if wm_cls is None or not callable(wm_cls):
                print("Perth watermarker unavailable; using no-op watermark.")
                return _NoOpWatermarker()
            return wm_cls()
        except Exception as e:
            print(f"Perth watermarker init failed ({e}); using no-op watermark.")
            return _NoOpWatermarker()

    def _configure_espeak(self):
        """Attempt to auto-configure eSpeak NG on Windows if installed in default location.

        On Windows, phonemizer requires the eSpeak NG DLL and installation path.
        If the environment variables are not set, try to set them programmatically
        to common install paths. Otherwise, leave as-is and let phonemizer raise
        a clear error during initialization.
        """
        if platform.system() != "Windows":
            return

        # If already configured via env, don't override
        es_lib_env = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
        es_path_env = os.environ.get("PHONEMIZER_ESPEAK_PATH")
        if es_lib_env and es_path_env:
            try:
                EspeakWrapper.set_library(es_lib_env)
            except Exception:
                pass
            return

        default_dirs = [
            os.path.join(os.environ.get("ProgramFiles", r"C:\\Program Files"), "eSpeak NG"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)"), "eSpeak NG"),
        ]
        for base in default_dirs:
            dll_path = os.path.join(base, "libespeak-ng.dll")
            if os.path.exists(dll_path):
                # Set for current process and inform phonemizer
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = dll_path
                os.environ["PHONEMIZER_ESPEAK_PATH"] = base
                try:
                    EspeakWrapper.set_library(dll_path)
                except Exception:
                    pass
                print(f"Detected eSpeak NG at: {base}")
                break

    def _load_backbone(self, backbone_repo, backbone_device):
        print(f"Loading backbone from: {backbone_repo} on {backbone_device} ...")

        # GGUF loading
        if backbone_repo.endswith("gguf"):

            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with:\n"
                    "    pip install llama-cpp-python"
                ) from e

            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo,
                filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if backbone_device == "gpu" else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if backbone_device == "gpu" else False,
            )
            self._is_quantized_model = True

        else:
            # Lazy import transformers only when needed (HF path)
            import transformers as _tf
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            # Prefer fast tokenizer; fall back to slow if incompatible, allow remote code
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    backbone_repo, trust_remote_code=True
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    backbone_repo, use_fast=False, trust_remote_code=True
                )
            dtype = torch.float16 if backbone_device == "cuda" else None
            device = torch.device(backbone_device)
            try:
                self.backbone = AutoModelForCausalLM.from_pretrained(
                    backbone_repo,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    use_safetensors=False,
                ).to(device)
            except OSError as oe:
                # Common when the repo doesn't contain HF checkpoint weights
                raise OSError(
                    f"Backbone repo '{backbone_repo}' does not contain HF weights (pytorch_model.bin/model.safetensors). "
                    "Use GGUF mode (e.g., 'neuphonic/neutts-air-q4-gguf' or 'neuphonic/neutts-air-q8-gguf') "
                    "or specify a valid HF repo that includes weights."
                ) from oe
            except Exception as e:
                # Fallback: some Transformers builds may not export Qwen2ForCausalLM at the package root
                # Try to import the modeling module directly when config indicates Qwen2
                try:
                    cfg = AutoConfig.from_pretrained(backbone_repo, trust_remote_code=True)
                    archs = (cfg.architectures or [])
                    is_qwen2 = (cfg.model_type == "qwen2") or any("Qwen2" in a for a in archs)
                except Exception:
                    is_qwen2 = False

                if is_qwen2:
                    try:
                        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as _Qwen2ForCausalLM
                    except Exception:
                        try:
                            # Older alias in some releases
                            from transformers.models.qwen2.modeling_qwen2 import Qwen2LMHeadModel as _Qwen2ForCausalLM
                        except Exception as e2:
                            raise ImportError(
                                "Qwen2 modeling class not available in this Transformers build. "
                                "Please upgrade/downgrade Transformers to a version that includes Qwen2ForCausalLM, "
                                "or use the GGUF or ONNX modes."
                            ) from e2
                    try:
                        self.backbone = _Qwen2ForCausalLM.from_pretrained(
                            backbone_repo,
                            trust_remote_code=True,
                            torch_dtype=dtype,
                            use_safetensors=False,
                        ).to(device)
                    except OSError as oe:
                        raise OSError(
                            f"Backbone repo '{backbone_repo}' does not contain HF weights. "
                            "Use GGUF mode (e.g., 'neuphonic/neutts-air-q4-gguf') or provide a HF repo with weights."
                        ) from oe
                else:
                    raise

    def _load_codec(self, codec_repo, codec_device):

        print(f"Loading codec from: {codec_repo} on {codec_device} ...")
        match codec_repo:
            case "neuphonic/neucodec":
                from neucodec import NeuCodec
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                from neucodec import DistillNeuCodec
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder":

                if codec_device != "cpu":
                    raise ValueError("Onnx decoder only currently runs on CPU.")

                try:
                    from .onnx_codec import OnnxCodecDecoder
                except Exception as e:
                    raise ImportError(
                        "Failed to import local ONNX codec decoder. Please ensure onnxruntime is installed."
                    ) from e

                self.codec = OnnxCodecDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True

            case _:
                raise ValueError(
                    "Invalid codec repo! Must be one of:"
                    " 'neuphonic/neucodec', 'neuphonic/distill-neucodec',"
                    " 'neuphonic/neucodec-onnx-decoder'."
                )

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str, fade_in_ms: int = 0) -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Returns:
            np.ndarray: Generated speech waveform.
        """

        # Normalize ref_codes to a Python list of ints when needed
        ref_codes_list = ref_codes
        try:
            if isinstance(ref_codes, torch.Tensor):
                ref_codes_list = ref_codes.detach().cpu().reshape(-1).tolist()
            elif isinstance(ref_codes, np.ndarray):
                ref_codes_list = ref_codes.reshape(-1).astype(np.int32).tolist()
        except Exception:
            ref_codes_list = ref_codes

        # Generate tokens
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes_list, ref_text, text)
        else:
            prompt_ids = self._apply_chat_template(ref_codes_list, ref_text, text)
            output_str = self._infer_torch(prompt_ids)

        # Decode
        wav = self._decode(output_str, ref_codes=ref_codes)
        # Optional micro fade-in to smooth boundaries
        if fade_in_ms and fade_in_ms > 0:
            n = int(self.sample_rate * (fade_in_ms / 1000.0))
            n = max(1, min(n, wav.shape[0]))
            ramp = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
            wav[:n] *= ramp
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=24_000)

        return watermarked_wav
    
    def infer_stream(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> Generator[np.ndarray, None, None]:
        """
        Perform streaming inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Yields:
            np.ndarray: Generated speech waveform.
        """ 

        if self._is_quantized_model:
            return self._infer_stream_ggml(ref_codes, ref_text, text)

        else:
            raise NotImplementedError("Streaming is not implemented for the torch backend!")

    def encode_reference(self, ref_audio_path: str | Path):
        """
        Encode a reference waveform into codec token IDs.

        Note: When using the ONNX decoder, we still need a Torch-based encoder to create
        reference codes. In that case, use neucodec on CPU just for encoding.
        """
        wav, sr = sf.read(str(ref_audio_path), always_2d=False)
        # Ensure mono
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        # Ensure float32
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32, copy=False)
        # Resample to 16k if needed using high-quality polyphase
        target_sr = 16000
        if sr != target_sr:
            # Lightweight linear resampling using NumPy
            duration = len(wav) / float(sr)
            new_length = int(round(duration * target_sr))
            if new_length <= 0:
                raise ValueError("Invalid resample length computed.")
            x_old = np.linspace(0.0, duration, num=len(wav), endpoint=False, dtype=np.float64)
            x_new = np.linspace(0.0, duration, num=new_length, endpoint=False, dtype=np.float64)
            wav = np.interp(x_new, x_old, wav).astype(np.float32, copy=False)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]

        if self._is_onnx_codec:
            # Lazy-load a CPU neucodec encoder to produce reference codes
            try:
                from neucodec import NeuCodec as _NeuCodec
            except Exception as e:
                raise ImportError(
                    "ONNX codec is for decoding only. To encode references, the 'neucodec' package is required. "
                    "Please install neucodec (already listed in requirements.txt)."
                ) from e
            encoder = getattr(self, "_ref_encoder", None)
            if encoder is None:
                encoder = _NeuCodec.from_pretrained("neuphonic/neucodec").to("cpu").eval()
                self._ref_encoder = encoder
            with torch.no_grad():
                ref_codes = encoder.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        else:
            with torch.no_grad():
                ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes: str, ref_codes: Optional[np.ndarray | torch.Tensor | list[int]] = None):

        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]

        if len(speech_ids) > 0:

            # Onnx decode
            if self._is_onnx_codec:
                gen_ids = speech_ids
                prefix_ids: list[int] = []
                if ref_codes is not None:
                    if isinstance(ref_codes, torch.Tensor):
                        prefix_ids = ref_codes.detach().cpu().numpy().astype(np.int32).tolist()
                    elif isinstance(ref_codes, np.ndarray):
                        prefix_ids = ref_codes.astype(np.int32).tolist()
                    else:
                        prefix_ids = [int(x) for x in ref_codes]

                def _decode_ids(ids: list[int]) -> np.ndarray:
                    arr = np.array(ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
                    out = self.codec.decode_code(arr)[0, 0, :]
                    if np.issubdtype(out.dtype, np.integer):
                        out = out.astype(np.float32) / 32768.0
                    else:
                        out = out.astype(np.float32, copy=False)
                    return out

                if prefix_ids:
                    # First, decode prefix alone to measure exact warm-up length
                    wav_prefix = _decode_ids(prefix_ids)
                    # Then decode prefix + generated and trim the exact prefix sample count
                    wav_total = _decode_ids(prefix_ids + gen_ids)
                    pL = wav_prefix.shape[0]
                    if pL < wav_total.shape[0]:
                        wav = wav_total[pL:]
                    else:
                        wav = wav_total[-1:]
                else:
                    # No prefix provided; decode generated tokens as-is
                    wav = _decode_ids(gen_ids)
                return wav

            # Torch decode
            else:
                with torch.no_grad():
                    codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                        self.codec.device
                    )
                    recon = self.codec.decode_code(codes).cpu().numpy()
                wav = recon[0, 0, :]
                if np.issubdtype(wav.dtype, np.integer):
                    wav = wav.astype(np.float32) / 32768.0
                else:
                    wav = wav.astype(np.float32, copy=False)
                return wav
        else:
            raise ValueError("No valid speech tokens found in the output.")

    def _to_phones(self, text: str) -> str:
        # Normalize and handle multi-line inputs robustly to avoid phonemizer line count mismatches
        if text is None:
            text = ""
        # Replace any Windows/Mac newlines with \n uniformly
        norm = str(text).replace("\r\n", "\n").replace("\r", "\n")
        # Split into non-empty lines; if no lines, keep a single empty string
        lines = [ln.strip() for ln in norm.split("\n") if ln.strip() != ""] or [norm.strip()]

        def _safe_phonemize(line_list: list[str]) -> list[str]:
            """Call phonemizer safely, enforcing output length to match input; if not, collapse to one line.
            Any exception results in a whitespace-collapsed passthrough.
            """
            try:
                out = self.phonemizer.phonemize(line_list)
                # Some backends may return a single string; normalize to list
                if isinstance(out, (str, bytes)):
                    out = [str(out)]
                # If the backend produced a different number of lines, try collapsing to a single line
                if not isinstance(out, (list, tuple)):
                    out = [str(out)]
                if len(out) != len(line_list):
                    joined = " ".join(line_list)
                    try:
                        out2 = self.phonemizer.phonemize([joined])
                        if isinstance(out2, (str, bytes)):
                            out2 = [str(out2)]
                        if isinstance(out2, (list, tuple)) and len(out2) == 1:
                            out = list(out2)
                        else:
                            out = [re.sub(r"\s+", " ", joined).strip()]
                    except Exception:
                        out = [re.sub(r"\s+", " ", joined).strip()]
            except Exception:
                joined = " ".join(line_list)
                out = [re.sub(r"\s+", " ", joined).strip()]
            # Final guarantee: return a list of strings with same length as input (collapse if needed)
            if len(out) != len(line_list):
                out = [re.sub(r"\s+", " ", " ".join(line_list)).strip()]
            return [str(x) for x in out]

        ph_list = _safe_phonemize(lines)
        # Tokenize each line's phones into words, then flatten and join with single spaces
        words: list[str] = []
        for ph in ph_list:
            words.extend(str(ph).split())
        return " ".join(words)

    def _apply_chat_template(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:

        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1 :]  # noqa
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)

        return ids

    def _infer_torch(self, prompt_ids: list[int]) -> str:
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                use_cache=True,
                min_new_tokens=50,
            )
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False
        )
        return output_str
    
    def _infer_ggml(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        output_str = output["choices"][0]["text"]
        return output_str

    def _infer_stream_ggml(self, ref_codes: torch.Tensor, ref_text: str, input_text: str) -> Generator[np.ndarray, None, None]:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)

        # Normalize tensor codes to ints
        try:
            if isinstance(ref_codes, torch.Tensor):
                ref_codes_iter = ref_codes.detach().cpu().reshape(-1).tolist()
            elif isinstance(ref_codes, np.ndarray):
                ref_codes_iter = ref_codes.reshape(-1).astype(np.int32).tolist()
            else:
                ref_codes_iter = ref_codes
        except Exception:
            ref_codes_iter = ref_codes
        codes_str = "".join([f"<|speech_{int(idx)}|>" for idx in ref_codes_iter])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True
        ):
            output_str = item["choices"][0]["text"]
            token_cache.append(output_str)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:

                # decode chunk
                tokens_start = max(
                    n_decoded_tokens
                    - self.streaming_lookback
                    - self.streaming_overlap_frames,
                    0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (
                    n_decoded_tokens - tokens_start
                ) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                )
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                # postprocess
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[
                    n_decoded_samples:new_samples_end
                ]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        # final decoding handled seperately as non-constant chunk size
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(
                len(token_cache)
                - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 
                0
            )
            sample_start = (
                len(token_cache) 
                - tokens_start 
                - remaining_tokens 
                - self.streaming_overlap_frames
            ) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
            recon = recon[sample_start:]
            audio_cache.append(recon)

            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon