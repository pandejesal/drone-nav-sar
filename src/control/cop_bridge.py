#!/usr/bin/env python3
"""Sprint 20: COP -> policy interface.

Converts COP entities into task embeddings the navigation policy /
mission planner can consume. SAR-only: navigate_to / hover /
drop_payload / return_home.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

SAR_SKILLS = ("navigate_to", "hover", "drop_payload", "return_home")

# Fixed task-embedding ids aligned with MissionPlanner task_embedding_id.
TYPE_TO_TASK = {
    "victim": 2,        # hover/detect cue
    "hazard": 1,        # caution hover
    "landing_zone": 0,  # navigate_to
    "teammate": 4,      # formation / return-home context
}

EMBED_DIM = 8


def entity_to_embedding(entity, drone_pos) -> np.ndarray:
    """Encode one COP entity relative to the drone into an 8-D embedding."""
    pos = np.asarray(entity.pos, dtype=np.float64).reshape(3)
    vel = np.asarray(entity.vel, dtype=np.float64).reshape(3)
    drone = np.asarray(drone_pos, dtype=np.float64).reshape(3)
    rel = pos - drone
    dist = float(np.linalg.norm(rel)) + 1e-6
    direction = rel / dist
    unc = float(np.sqrt(max(np.trace(np.asarray(entity.cov)) / 3.0, 0.0)))
    emb = np.array(
        [
            direction[0], direction[1], direction[2],
            min(dist / 20.0, 1.0),
            float(np.linalg.norm(vel)) / 5.0,
            float(entity.confidence),
            min(unc / 2.0, 1.0),
            float(TYPE_TO_TASK.get(entity.type, 1)) / 4.0,
        ],
        dtype=np.float64,
    )
    return emb


class COPBridge:
    """COP entities -> policy-ready task interface."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        self.embed_dim = int(embed_dim)

    def entities_to_tasks(self, entities, drone_pos) -> List[Dict]:
        """Map each entity to a SAR skill + task embedding id."""
        tasks = []
        for e in entities:
            etype = getattr(e, "type", "victim")
            if etype == "victim":
                skill = "hover" if float(np.linalg.norm(
                    np.asarray(e.pos) - np.asarray(drone_pos)[:3])) < 1.0 else "navigate_to"
            elif etype == "landing_zone":
                skill = "navigate_to"
            elif etype == "hazard":
                skill = "hover"
            else:
                skill = "return_home" if etype == "teammate" else "hover"
            if skill not in SAR_SKILLS:
                skill = "hover"
            emb = entity_to_embedding(e, drone_pos)
            tasks.append(
                {
                    "entity_id": getattr(e, "entity_id", "unknown"),
                    "skill": skill,
                    "task_embedding_id": int(TYPE_TO_TASK.get(etype, 1)),
                    "embedding": emb,
                    "confidence": float(getattr(e, "confidence", 0.9)),
                }
            )
        # Highest confidence first so the planner serves victims first.
        tasks.sort(key=lambda t: -t["confidence"])
        return tasks

    def to_policy_obs(self, entities, drone_pos, max_entities: int = 10) -> np.ndarray:
        """Fixed-size policy observation: stacked embeddings (max_entities x dim)."""
        tasks = self.entities_to_tasks(entities, drone_pos)[:max_entities]
        obs = np.zeros((max_entities, self.embed_dim))
        for i, t in enumerate(tasks):
            obs[i] = np.asarray(t["embedding"]).reshape(-1)[: self.embed_dim]
        return obs
