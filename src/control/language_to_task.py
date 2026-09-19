#!/usr/bin/env python3
"""Sprint 11 — Language → task mapping (SAR-only).

Maps a parsed language intent (task_id + params) to a
``task_embedding_id`` + SAR-only skill params.

SAR-only skills: navigate_to / hover / drop_payload / return_home.
Task map: 0=nav, 1=hover, 2=detect, 3=drop, 4=return.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

SAR_SKILLS = ("navigate_to", "hover", "drop_payload", "return_home")

TASK_ID_TO_SKILL = {
    0: "navigate_to",
    1: "hover",
    2: "hover",  # detect/search runs on the hover skill (detector scans)
    3: "drop_payload",
    4: "return_home",
}


def intent_to_task(task_id: int, params: Dict) -> Dict:
    """Map (task_id, params) → {task_embedding_id, skill, params} (SAR-only)."""
    tid = int(task_id) % 5
    skill = TASK_ID_TO_SKILL[tid]
    assert skill in SAR_SKILLS, f"non-SAR skill: {skill!r}"
    clean = dict(params or {})
    return {
        "task_embedding_id": tid,
        "task_id": tid,
        "skill": skill,
        "params": clean,
    }


def text_to_task(text: str) -> Dict:
    """Parse text → {task_embedding_id, skill, params, task_embedding, latency_ms}.

    Local rule-based parser is the default (no network, < 2 s budget);
    the returned task_embedding is the 4-dim projection used by LocalPolicy.
    """
    from src.control.language_interface import (
        parse_language,
        text_to_task_embedding,
    )

    t0 = time.perf_counter()
    task_id, params = parse_language(text)
    mapped = intent_to_task(task_id, params)
    emb = text_to_task_embedding(text).detach().cpu().numpy().astype(np.float32)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    mapped["task_embedding"] = emb
    mapped["latency_ms"] = float(latency_ms)
    mapped["text"] = text
    return mapped


def batch_text_to_task(texts) -> list:
    """Map a list of commands → list of task dicts."""
    return [text_to_task(t) for t in list(texts)]


def skill_for_task(task_embedding_id: int) -> str:
    """Return the SAR-only skill for a task_embedding_id."""
    return TASK_ID_TO_SKILL[int(task_embedding_id) % 5]


def parse_and_map(text: str) -> Tuple[int, Dict]:
    """Legacy helper: text → (task_embedding_id, params)."""
    from src.control.language_interface import parse_language

    task_id, params = parse_language(text)
    mapped = intent_to_task(task_id, params)
    return mapped["task_embedding_id"], mapped["params"]
