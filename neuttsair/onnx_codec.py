from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from huggingface_hub import snapshot_download


@dataclass
class OnnxCodecDecoder:
    repo_id: str
    model_path: str
    _session: any

    @classmethod
    def from_pretrained(cls, repo_id: str, local_dir: Optional[str] = None) -> "OnnxCodecDecoder":
        local_dir = local_dir or snapshot_download(repo_id=repo_id)
        # Heuristically find the decoder ONNX file in the repo.
        # Prefer names indicating decoder functionality.
        candidates = []
        for root, _, files in os.walk(local_dir):
            for f in files:
                if f.lower().endswith(".onnx"):
                    candidates.append(os.path.join(root, f))

        def _score(path: str) -> int:
            name = os.path.basename(path).lower()
            score = 0
            # Strongly prefer files that look like decoders
            if "decoder" in name or "decode" in name:
                score += 10
            # Deprioritize obvious encoders
            if "encoder" in name or "encode" in name:
                score -= 5
            # Prefer smaller directory depth slightly (often the main file sits at top-level)
            score -= path.count(os.sep) // 2
            return score

        onnx_path = None
        if candidates:
            candidates.sort(key=_score, reverse=True)
            onnx_path = candidates[0]
        if onnx_path is None:
            raise FileNotFoundError("No .onnx model found in repository: " + repo_id)

        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        # Enable basic graph optimizations
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        return cls(repo_id=repo_id, model_path=onnx_path, _session=session)

    def decode_code(self, codes: np.ndarray) -> np.ndarray:
        # Expect codes shape [1, 1, T] int32
        if codes.dtype != np.int32:
            codes = codes.astype(np.int32, copy=False)
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        in_name = inputs[0].name
        out_name = outputs[0].name
        ort_out = self._session.run([out_name], {in_name: codes})[0]
        return ort_out
