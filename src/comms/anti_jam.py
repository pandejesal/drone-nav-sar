#!/usr/bin/env python3
"""Sprint 23: Anti-jam layer — SAR-only.

FHSS (75 ch, 100 hops/s) + DSSS (Barker-11, PG 10.4 dB) + 4-element
adaptive nulling (LMS) + CJAM energy/cyclostationary detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.comms.lpi_lpd import DSSS_GAIN_DB, FHSS_GAIN_DB

N_ANTENNA = 4
REQUIRED_SNR_DB = 6.0
NULLING_GAIN_DB = 12.0  # 4-element array null depth (simulated)


@dataclass
class JammerSpec:
    power_dbm: float = 30.0
    bandwidth_frac: float = 0.3
    distance_m: float = 500.0
    js_db: float = 30.0  # jammer-to-signal ratio at victim


class AdaptiveNuller:
    """4-element array, LMS adaptive null steering (simulated)."""

    def __init__(self, n_elements: int = N_ANTENNA, mu: float = 0.05,
                 seed: int = 0) -> None:
        self.n = int(n_elements)
        self.mu = float(mu)
        self.w = np.zeros(self.n, dtype=complex)
        self.w[0] = 1.0 + 0j
        self._rng = np.random.default_rng(seed)

    def step(self, x: np.ndarray, d: complex) -> complex:
        """One LMS update; x = array snapshot, d = reference. Returns output."""
        x = np.asarray(x, dtype=complex).reshape(-1)
        y = complex(np.dot(self.w.conj(), x))
        e = d - y
        self.w = self.w + self.mu * e.conjugate() * x
        return y

    def null_gain_db(self, jammer_angle_deg: float = 30.0) -> float:
        """Simulated null depth toward jammer (dB of J suppression)."""
        # Deeper null as LMS converges; deterministic stub scaled by angle.
        depth = NULLING_GAIN_DB * (0.7 + 0.3 * abs(math.sin(math.radians(jammer_angle_deg))))
        return float(min(depth, 15.0))


class AntiJamLink:
    """Combined FHSS + DSSS + nulling anti-jam endpoint."""

    def __init__(self, n_channels: int = 75, seed: int = 0) -> None:
        self.n_channels = int(n_channels)
        self.nuller = AdaptiveNuller(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.tx_count = 0
        self.ok_count = 0

    @property
    def processing_gain_db(self) -> float:
        return FHSS_GAIN_DB + DSSS_GAIN_DB

    def jamming_margin_db(self, jammer_angle_deg: float = 30.0) -> float:
        """PG(FHSS+DSSS) + nulling - required SNR. Must exceed 30 dB."""
        return self.processing_gain_db + self.nuller.null_gain_db(jammer_angle_deg) - REQUIRED_SNR_DB

    def detect_jammer(self, spectrum: np.ndarray, threshold: float = 3.0) -> Dict:
        """Energy detection + cyclostationary stub.

        Returns {jammed, jammer_bins, metric}.
        """
        s = np.asarray(spectrum, dtype=float).reshape(-1)
        med = float(np.median(s)) + 1e-12
        metric = float(np.max(s) / med)
        jammed = metric > threshold
        bins = [int(i) for i in range(len(s)) if s[i] > threshold * med]
        # Cyclostationary confirm: variance-of-variance heuristic.
        cyc = float(np.var(s) / (np.mean(s) ** 2 + 1e-12))
        confirmed = jammed and cyc > 0.5
        return {"jammed": bool(confirmed), "jammer_bins": bins,
                "metric": metric, "cyclo": cyc}

    def packet_success_prob(self, js_db: float,
                            jammer_angle_deg: float = 30.0) -> float:
        """Success prob at given J/S (dB) after all AJ gains."""
        margin = self.jamming_margin_db(jammer_angle_deg) - float(js_db)
        p = 1.0 / (1.0 + math.exp(-0.6 * margin))
        return float(np.clip(p, 0.01, 0.999))

    def send_packet(self, js_db: float = 0.0,
                    jammer_angle_deg: float = 30.0) -> Dict:
        p = self.packet_success_prob(js_db, jammer_angle_deg)
        ok = bool(self._rng.random() < p)
        self.tx_count += 1
        self.ok_count += int(ok)
        return {"delivered": ok, "p": p,
                "margin_db": self.jamming_margin_db(jammer_angle_deg)}

    def link_maintained_at_js(self, js_db: float = 30.0, trials: int = 200,
                              need_pdr: float = 0.5) -> bool:
        oks = sum(1 for _ in range(trials)
                  if self._rng.random() < self.packet_success_prob(js_db))
        return (oks / trials) >= need_pdr

    def report(self) -> Dict:
        return {
            "pg_db": self.processing_gain_db,
            "null_db": self.nuller.null_gain_db(),
            "margin_db": self.jamming_margin_db(),
            "pdr": (self.ok_count / self.tx_count) if self.tx_count else 1.0,
        }
