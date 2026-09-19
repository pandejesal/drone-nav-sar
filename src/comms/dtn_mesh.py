#!/usr/bin/env python3
"""Sprint 23: DTN Bundle Protocol (RFC 5050) — SAR-only.

Implements custody transfer, TTL expiration, fragmentation, epidemic
routing, and BAB/PIB stubs for swarm store-and-forward in denied links.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

SAR_SKILLS = ("navigate_to", "hover", "drop_payload", "return_home")

# Fragment payload cap (bytes of repr) to force fragmentation on big payloads.
FRAGMENT_SIZE = 512


@dataclass
class PrimaryBlock:
    """RFC 5050 primary bundle block."""

    bundle_id: int
    src: str
    dst: str
    creation_ts: float
    lifetime_s: float
    priority: int = 1  # 0 bulk, 1 normal, 2 expedited
    custody_requested: bool = True

    def is_expired(self, now: float) -> bool:
        return (now - self.creation_ts) > self.lifetime_s


@dataclass
class Bundle:
    """Full bundle: primary + payload + optional extension blocks."""

    primary: PrimaryBlock
    payload: Dict
    fragments: List[bytes] = field(default_factory=list)
    bab: str = ""  # Bundle Authentication Block (HMAC stub)
    pib: str = ""  # Payload Integrity Block (SHA-256 stub)

    @property
    def bundle_id(self) -> int:
        return self.primary.bundle_id

    def is_expired(self, now: float) -> bool:
        return self.primary.is_expired(now)


def sign_bundle(bundle: Bundle, key: str = "sar-swarm-key") -> Bundle:
    """Attach BAB (HMAC stub) + PIB (payload hash)."""
    raw = repr(sorted(bundle.payload.items())).encode()
    bundle.pib = hashlib.sha256(raw).hexdigest()[:32]
    bundle.bab = hashlib.sha256((bundle.pib + key).encode()).hexdigest()[:32]
    return bundle


def verify_bundle(bundle: Bundle, key: str = "sar-swarm-key") -> bool:
    raw = repr(sorted(bundle.payload.items())).encode()
    pib = hashlib.sha256(raw).hexdigest()[:32]
    bab = hashlib.sha256((pib + key).encode()).hexdigest()[:32]
    return pib == bundle.pib and bab == bundle.bab


def fragment_payload(payload: Dict, frag_size: int = FRAGMENT_SIZE) -> List[bytes]:
    raw = repr(payload).encode()
    return [raw[i:i + frag_size] for i in range(0, len(raw), frag_size)] or [b""]


def reassemble_payload(frags: List[bytes]) -> Dict:
    import ast

    raw = b"".join(frags).decode()
    try:
        return dict(ast.literal_eval(raw))
    except Exception:
        return {"raw": raw}


class DTNNode:
    """Single DTN endpoint with custody store + epidemic routing."""

    def __init__(self, eid: str, link_availability: float = 0.5,
                 loss: float = 0.3, seed: int = 0) -> None:
        self.eid = eid
        self.link_availability = float(np.clip(link_availability, 0.0, 1.0))
        self.loss = float(np.clip(loss, 0.0, 0.99))
        self._rng = np.random.default_rng(seed)
        self._next_id = 0
        self.store: Dict[int, Bundle] = {}
        self.custody: Dict[int, str] = {}  # bundle_id -> holder eid
        self.delivered: Dict[int, Bundle] = {}
        self.seen: Dict[str, set] = {}  # neighbour eid -> seen bundle ids (epidemic)
        self.sent = 0
        self.acked = 0

    def create_bundle(self, dst: str, payload: Dict, lifetime_s: float = 300.0,
                      priority: int = 1) -> Bundle:
        skill = payload.get("skill", "navigate_to")
        if skill not in SAR_SKILLS:
            raise ValueError(f"SAR-only bundle required, got skill={skill!r}")
        self._next_id += 1
        primary = PrimaryBlock(
            bundle_id=self._next_id, src=self.eid, dst=dst,
            creation_ts=time.time(), lifetime_s=float(lifetime_s),
            priority=int(priority),
        )
        b = Bundle(primary=primary, payload=dict(payload))
        b.fragments = fragment_payload(b.payload)
        sign_bundle(b)
        self.store[b.bundle_id] = b
        self.custody[b.bundle_id] = self.eid
        self.sent += 1
        return b

    def expire_bundles(self, now: Optional[float] = None) -> int:
        """Drop expired bundles from store. Returns count expired."""
        now = time.time() if now is None else float(now)
        expired = [bid for bid, b in self.store.items() if b.is_expired(now)]
        for bid in expired:
            del self.store[bid]
            self.custody.pop(bid, None)
        return len(expired)

    def _contact_up(self) -> bool:
        return bool(self._rng.random() < self.link_availability)

    def _tx_ok(self) -> bool:
        return bool(self._rng.random() >= self.loss)

    def custody_transfer(self, bundle_id: int, peer: "DTNNode",
                         max_retries: int = 8) -> bool:
        """Custody transfer with ACK/NACK + retransmission.

        Returns True when peer ACKs (takes custody). Expiration checked first.
        """
        b = self.store.get(bundle_id)
        if b is None:
            return False
        if b.is_expired(time.time()):
            self.expire_bundles()
            return False
        if not verify_bundle(b):
            return False  # PIB/BAB failure -> NACK
        for _ in range(max_retries):
            if not self._contact_up():
                continue  # no contact: store-and-forward, retry later
            if not self._tx_ok():
                continue  # lost: retransmit
            # Peer receives; check duplicate (epidemic already-seen).
            seen = peer.seen.setdefault(self.eid, set())
            peer.store[b.bundle_id] = b
            peer.custody[b.bundle_id] = peer.eid
            seen.add(b.bundle_id)
            if b.primary.dst == peer.eid:
                peer.delivered[b.bundle_id] = b
            self.custody[b.bundle_id] = peer.eid
            self.acked += 1
            return True
        return False

    def epidemic_exchange(self, peer: "DTNNode", max_retries: int = 6) -> int:
        """Push unseen bundles to peer (epidemic routing). Returns # ACKed."""
        n = 0
        for bid in list(self.store.keys()):
            if bid in peer.store:
                continue
            if self.custody_transfer(bid, peer, max_retries=max_retries):
                n += 1
        return n

    def delivery_ratio(self) -> float:
        if self.sent == 0:
            return 1.0
        return self.acked / self.sent

    def report(self) -> Dict:
        return {"eid": self.eid, "sent": self.sent, "acked": self.acked,
                "delivery": self.delivery_ratio(), "stored": len(self.store),
                "delivered": len(self.delivered)}
