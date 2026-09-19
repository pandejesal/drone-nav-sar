#!/usr/bin/env python3
"""Lightweight multi-drone message bus (Sprint 9, SAR-only).

Broadcast-only bus for: victim_found (victim spots), intent (planned goal),
position (telemetry). SAR-only payloads: navigate/hover/drop/return_home
intents carry goal positions, never weapons/targeting data.

Usage:
    bus = MessageBus(n_drones=3)
    bus.broadcast_victim(sender_id=0, position=np.array([2.5, 2.5, 0.0]))
    msgs = bus.get_messages(msg_type="victim_found")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# SAR-only allowed message types.
MSG_TYPES = ("victim_found", "intent", "position")

# SAR-only allowed intent skills.
SAR_SKILLS = ("navigate_to", "hover", "drop_payload", "return_home")


@dataclass
class Message:
    """Single broadcast message."""

    sender_id: int
    msg_type: str  # one of MSG_TYPES
    payload: Dict = field(default_factory=dict)
    step: int = 0


class MessageBus:
    """In-process broadcast bus (no networking, deterministic, test-friendly)."""

    def __init__(self, n_drones: int = 3):
        self.n_drones = int(n_drones)
        self._messages: List[Message] = []
        self._step = 0
        # Shared victim belief: list of positions known team-wide.
        self.shared_victims: List[np.ndarray] = []

    # -- core ----------------------------------------------------------
    def broadcast(self, msg: Message) -> Message:
        """Append a message; returns it with the current step stamped."""
        if msg.msg_type not in MSG_TYPES:
            raise ValueError(f"unknown msg_type: {msg.msg_type!r}")
        msg.step = self._step
        self._messages.append(msg)
        if msg.msg_type == "victim_found":
            pos = np.asarray(msg.payload.get("position"), dtype=np.float32).reshape(3)
            if not any(np.allclose(pos, v) for v in self.shared_victims):
                self.shared_victims.append(pos)
        return msg

    def tick(self) -> int:
        """Advance the logical step counter."""
        self._step += 1
        return self._step

    def get_messages(
        self,
        for_agent: Optional[int] = None,
        msg_type: Optional[str] = None,
    ) -> List[Message]:
        """Return messages, excluding own sends when for_agent is set."""
        out = self._messages
        if msg_type is not None:
            out = [m for m in out if m.msg_type == msg_type]
        if for_agent is not None:
            out = [m for m in out if m.sender_id != int(for_agent)]
        return list(out)

    def clear(self) -> None:
        """Drop all buffered messages (beliefs retained)."""
        self._messages = []

    def reset(self) -> None:
        """Full reset: messages + shared beliefs + step."""
        self._messages = []
        self.shared_victims = []
        self._step = 0

    # -- convenience publishers ----------------------------------------
    def broadcast_victim(self, sender_id: int, position: np.ndarray) -> Message:
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        return self.broadcast(
            Message(sender_id=int(sender_id), msg_type="victim_found",
                    payload={"position": pos})
        )

    def broadcast_position(self, sender_id: int, position: np.ndarray) -> Message:
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        return self.broadcast(
            Message(sender_id=int(sender_id), msg_type="position",
                    payload={"position": pos})
        )

    def broadcast_intent(
        self, sender_id: int, skill: str, goal: np.ndarray
    ) -> Message:
        if skill not in SAR_SKILLS:
            raise ValueError(f"SAR-only intent skill required, got {skill!r}")
        g = np.asarray(goal, dtype=np.float32).reshape(3)
        return self.broadcast(
            Message(sender_id=int(sender_id), msg_type="intent",
                    payload={"skill": skill, "goal": g})
        )

    # -- shared belief ---------------------------------------------------
    def get_shared_victims(self) -> List[np.ndarray]:
        """Team-wide victim spots (copies)."""
        return [np.array(v, dtype=np.float32) for v in self.shared_victims]

    @property
    def message_count(self) -> int:
        return len(self._messages)
