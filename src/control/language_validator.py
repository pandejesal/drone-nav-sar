#!/usr/bin/env python3
"""Sprint 11 — Language validator: CLIPScore semantic similarity (SAR-only).

CLIPScore(text, reference) = cosine similarity of L2-normalized text
embeddings, rescaled to [0, 1] via (cos + 1) / 2. Identical phrases score 1.0.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from src.control.language_interface import (
    NUM_TASKS,
    TASK_CANONICAL,
    get_encoder,
    parse_language,
)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = float(a.norm() * b.norm()) + 1e-12
    return float((a @ b) / denom)


def clip_score(text: str, reference: str, encoder=None) -> float:
    """CLIPScore in [0, 1] between a command and a reference phrase."""
    enc = encoder or get_encoder()
    with torch.no_grad():
        e1 = enc.encode(text).reshape(-1)
        e2 = enc.encode(reference).reshape(-1)
    cos = cosine_sim(e1, e2)
    return float((cos + 1.0) / 2.0)


def validate_command(text: str, threshold: float = 0.8, encoder=None) -> Dict:
    """Validate a command: parse → task_id, score vs canonical phrase.

    Returns dict with task_id, score, passed (score >= threshold).
    """
    task_id, params = parse_language(text)
    score = clip_score(text, TASK_CANONICAL[task_id], encoder=encoder)
    return {
        "text": text,
        "task_id": task_id,
        "params": params,
        "reference": TASK_CANONICAL[task_id],
        "clip_score": score,
        "passed": bool(score >= threshold),
    }


def accuracy_on_suite(phrases: List[str], ground_truth_ids: List[int],
                      encoder=None) -> Dict:
    """Accuracy + mean CLIPScore of parse_language over labeled phrases."""
    assert len(phrases) == len(ground_truth_ids)
    enc = encoder or get_encoder()
    scores: List[float] = []
    correct = 0
    for text, gt in zip(phrases, ground_truth_ids):
        pred, _ = parse_language(text)
        if int(pred) == int(gt):
            correct += 1
        scores.append(clip_score(text, TASK_CANONICAL[int(gt)], encoder=enc))
    n = max(1, len(phrases))
    return {
        "accuracy": correct / n,
        "mean_clip_score": sum(scores) / n,
        "n": len(phrases),
    }


# 50-phrase SAR-only ground-truth suite (target: accuracy >= 0.8).
TEST_PHRASES: List[str] = [
    # nav (0)
    "go to room 1", "navigate to kitchen", "fly to room 2", "move to room 3",
    "head to room 0", "proceed to room 4", "take me to room 2", "go to room 5",
    "navigate to room 3", "fly to the kitchen",
    # hover (1)
    "hover here", "wait in place", "hold position", "stay where you are",
    "hover above the victim", "hold altitude", "wait for instructions",
    "stay hovering", "hover in place", "hold station",
    # detect (2)
    "find the victim", "search room 1", "scan for survivors", "locate victim",
    "detect people in room 2", "look for the victim", "search the kitchen",
    "scan room 3", "find survivors", "search for victim in room 0",
    # drop/deliver (3)
    "deliver medkit to room 2", "drop the payload", "drop medkit here",
    "deliver package to room 1", "release the medkit", "place payload in room 3",
    "drop water in room 0", "deliver medicine to room 4", "drop the kit",
    "deliver food to room 2",
    # return (4)
    "return home", "go back to base", "land now", "come back home",
    "recall to base", "return to home position", "fly back home",
    "land at base", "go home", "return to base",
]


def ground_truth_ids() -> List[int]:
    return [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
