#!/usr/bin/env python3
"""Sprint 11 — Language interface: text commands → task embeddings (SAR-only).

Pipeline:
    User text ("deliver medkit to room 2")
      → TextEncoder (frozen CLIP ViT-B/32-style / DistilBERT-style) → 512-dim
      → ProjectionHead (2-layer MLP) → 4-dim task_embedding (shared with local_policy)
      → MissionPlanner: task_id + params → GlobalPlanner → LocalPolicy → actions

SAR-only task map: 0=nav, 1=hover, 2=detect, 3=drop, 4=return.

The encoder is a frozen, deterministic hash-bag-of-words encoder with a
CLIP/DistilBERT-compatible interface (encode → 512-dim L2-normalized).
If ``transformers`` + weights are installed, set backend="hf" to use a real
DistilBERT/CLIP text model; otherwise the built-in backend is used (no downloads).
"""

from __future__ import annotations

import argparse
import hashlib
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

TEXT_EMBED_DIM = 512
TASK_EMBED_DIM = 4
NUM_TASKS = 5

TASK_NAMES = ["nav", "hover", "detect", "drop", "return"]

# Canonical reference phrase per task (used for CLIPScore ground truth).
TASK_CANONICAL = {
    0: "go to room navigate",
    1: "hover wait hold position",
    2: "find victim search scan detect",
    3: "deliver medkit to room",
    4: "return home base land",
}

PAYLOAD_TYPES = ["medkit", "water", "food", "package", "payload", "medicine", "kit"]

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric word tokens."""
    return _WORD_RE.findall(text.lower())


def _word_vector(word: str, dim: int = TEXT_EMBED_DIM) -> np.ndarray:
    """Deterministic pseudo-random unit vector per word (frozen encoder)."""
    seed = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float64)
    n = float(np.linalg.norm(vec)) + 1e-12
    return (vec / n).astype(np.float32)


class TextEncoder(nn.Module):
    """Frozen CLIP-style text encoder → 512-dim L2-normalized embedding.

    Args:
        backend: "hash" (built-in, default) or "hf" (transformers DistilBERT,
            requires ``transformers`` + ``torch`` weights; falls back to hash).
        model_name: HF model id used when backend="hf".
    """

    def __init__(self, embed_dim: int = TEXT_EMBED_DIM, backend: str = "hash",
                 model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.embed_dim = embed_dim
        self.backend = backend
        self._hf = None
        self._hf_tok = None
        if backend == "hf":
            try:
                from transformers import AutoModel, AutoTokenizer
                self._hf_tok = AutoTokenizer.from_pretrained(model_name)
                self._hf = AutoModel.from_pretrained(model_name)
                self._hf.eval()
                for p in self._hf.parameters():
                    p.requires_grad = False
                self._proj = nn.Linear(int(self._hf.config.hidden_size), embed_dim)
                nn.init.orthogonal_(self._proj.weight)
                nn.init.zeros_(self._proj.bias)
                for p in self._proj.parameters():
                    p.requires_grad = False
            except Exception:
                self._hf = None
                self._hf_tok = None
                self.backend = "hash"
        # frozen by contract
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def encode(self, texts) -> torch.Tensor:
        """Encode str or list[str] → (N, 512) L2-normalized tensor."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if self._hf is not None and self._hf_tok is not None:
            toks = self._hf_tok(list(texts), padding=True, truncation=True,
                                return_tensors="pt")
            out = self._hf(**toks).last_hidden_state[:, 0, :]
            emb = self._proj(out)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
            return emb[0] if single else emb
        vecs = []
        for t in texts:
            toks = tokenize(t)
            if not toks:
                vecs.append(np.zeros(self.embed_dim, dtype=np.float32))
                continue
            acc = np.zeros(self.embed_dim, dtype=np.float64)
            for w in toks:
                acc += _word_vector(w, self.embed_dim)
            n = float(np.linalg.norm(acc)) + 1e-12
            vecs.append((acc / n).astype(np.float32))
        emb = torch.as_tensor(np.stack(vecs), dtype=torch.float32)
        return emb[0] if single else emb

    def forward(self, texts) -> torch.Tensor:
        return self.encode(texts)


class ProjectionHead(nn.Module):
    """2-layer MLP: 512-dim text embedding → 4-dim task_embedding."""

    def __init__(self, in_dim: int = TEXT_EMBED_DIM, hidden: int = 128,
                 out_dim: int = TASK_EMBED_DIM, seed: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2.0)))
                    nn.init.zeros_(m.bias)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.net(text_emb)


# Shared default instances (frozen encoder; trainable projection in practice).
_default_encoder: Optional[TextEncoder] = None
_default_projection: Optional[ProjectionHead] = None


def get_encoder() -> TextEncoder:
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = TextEncoder()
    return _default_encoder


def get_projection() -> ProjectionHead:
    global _default_projection
    if _default_projection is None:
        _default_projection = ProjectionHead()
    return _default_projection


def encode_text(text: str) -> torch.Tensor:
    """Encode single command → (512,) tensor."""
    return get_encoder().encode(text)


def project_to_task_embedding(text_emb: torch.Tensor) -> torch.Tensor:
    """Project (512,) or (N,512) text embedding → 4-dim task_embedding."""
    return get_projection()(text_emb)


def text_to_task_embedding(text: str) -> torch.Tensor:
    """Full pipeline: text → 4-dim task_embedding."""
    return project_to_task_embedding(encode_text(text))


def task_embedding_for_id(task_id: int) -> torch.Tensor:
    """Reference task embedding = projection of the canonical phrase."""
    tid = int(task_id) % NUM_TASKS
    return text_to_task_embedding(TASK_CANONICAL[tid])


# -- Language → task_id + params -------------------------------------------

def _extract_room(text: str) -> Optional[int]:
    m = re.search(r"room\s*(\d+)", text.lower())
    if m:
        return int(m.group(1))
    return None


def _extract_payload(text: str) -> Optional[str]:
    low = text.lower()
    for p in PAYLOAD_TYPES:
        if p in low:
            return "medkit" if p in ("medkit", "medicine", "kit") else p
    return None


def parse_language(text: str) -> Tuple[int, Dict]:
    """Parse language → (task_id, params). Unknown → nav home fallback.

    task_ids: 0=nav, 1=hover, 2=detect, 3=drop, 4=return.
    """
    low = text.lower().strip()
    room = _extract_room(text)
    payload = _extract_payload(text)
    params: Dict = {}
    if room is not None:
        params["room"] = room
    if payload is not None:
        params["payload"] = payload

    if re.search(r"\b(deliver|drop|dropoff|drop-off|release|place|payload|medkit)\b", low):
        return 3, params
    if re.search(r"\b(return|home|base|land|recall|come back|rtb)\b", low):
        return 4, dict(params)
    if re.search(r"\b(hover|wait|hold|stay|hold position|station)\b", low):
        return 1, params
    if re.search(r"\b(detect|find|search|scan|locate|look for|victim|survivor)\b", low):
        return 2, params
    if re.search(r"\b(go|navigate|fly|move|head|proceed|take me|kitchen|room|waypoint)\b", low):
        if not params:
            params = {"location": text.strip() or "home"}
        return 0, params
    # Fallback: unknown command → nav to home.
    fb: Dict = {"location": "home"}
    if room is not None:
        fb["room"] = room
    return 0, fb


def language_to_command(text: str) -> Dict:
    """Parse text → dict with task_id, task_name, params, task_embedding."""
    task_id, params = parse_language(text)
    emb = text_to_task_embedding(text).detach().cpu().numpy().astype(np.float32)
    return {
        "task_id": task_id,
        "task_name": TASK_NAMES[task_id],
        "params": params,
        "task_embedding": emb,
    }


# -- Few-shot prompt + OpenAI-compatible LLM/SLM hook (local fallback) ----

# Few-shot examples grounding the SAR-only parser (nav/hover/detect/drop/return).
FEW_SHOT_EXAMPLES: List[Tuple[str, int, Dict]] = [
    ("go to room 3", 0, {"room": 3}),
    ("fly to kitchen", 0, {"location": "fly to kitchen"}),
    ("return to base", 4, {}),
    ("deliver medkit to room 2", 3, {"room": 2, "payload": "medkit"}),
    ("drop supplies at victim 1", 3, {"payload": "medkit"}),
    ("inspect hallway 3", 2, {}),
    ("search room 1 for victims", 2, {"room": 1}),
    ("hover here", 1, {}),
    ("return to base immediately", 4, {}),
    ("abort mission", 4, {}),
]

PROMPT_TEMPLATE = (
    "Parse the SAR drone command into JSON with keys "
    "task_id (0=nav, 1=hover, 2=detect, 3=drop, 4=return), params (room, payload). "
    "SAR-only skills: navigate_to/hover/drop_payload/return_home.\n"
    "Examples:\n{few_shot}\nCommand: {text}\nJSON:"
)


def build_few_shot_prompt(text: str) -> str:
    """Render the few-shot prompt for `text` (LLM/SLM backends)."""
    import json as _json
    lines = [
        f'- "{ex}" -> {{"task_id": {tid}, "params": {_json.dumps(p)}}}'
        for ex, tid, p in FEW_SHOT_EXAMPLES
    ]
    return PROMPT_TEMPLATE.format(few_shot="\n".join(lines), text=text)


def parse_with_llm(text: str, timeout_s: float = 10.0) -> Tuple[int, Dict]:
    """Parse via an OpenAI-compatible API; fall back to local parser.

    Honors OPENAI_API_URL/OPENAI_API_KEY (+ optional OPENAI_MODEL).
    Any failure (no creds, no network, bad JSON, non-SAR output) falls
    back to :func:`parse_language` so command latency stays < 2 s local.
    """
    import json as _json
    import os as _os
    import urllib.request as _req

    base = _os.environ.get("OPENAI_API_URL", "").strip()
    key = _os.environ.get("OPENAI_API_KEY", "").strip()
    model = _os.environ.get("OPENAI_MODEL", "slm-local").strip() or "slm-local"
    if not base or not key:
        return parse_language(text)
    try:
        payload = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": build_few_shot_prompt(text)}],
            "temperature": 0.0,
            "max_tokens": 128,
        }).encode("utf-8")
        url = base.rstrip("/") + "/chat/completions"
        r = _req.Request(url, data=payload,
                         headers={"Content-Type": "application/json",
                                  "Authorization": f"Bearer {key}"})
        with _req.urlopen(r, timeout=timeout_s) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}")
        data = _json.loads(content[start:end + 1] if start >= 0 and end > start else content)
        tid = int(data.get("task_id", 0)) % NUM_TASKS
        params = dict(data.get("params", {}) or {})
        return tid, params
    except Exception:
        return parse_language(text)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sprint 11 language interface (SAR-only).")
    ap.add_argument("--text", required=True, help='Command, e.g. "deliver medkit to room 2"')
    ap.add_argument("--backend", default="hash", help="encoder backend: hash|hf")
    args = ap.parse_args(argv)
    if args.backend == "hf":
        enc = TextEncoder(backend="hf")
    else:
        enc = get_encoder()
    task_id, params = parse_language(args.text)
    print(f"task_id={task_id}, params={params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
