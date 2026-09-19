#!/usr/bin/env python3
"""Sprint 23: LPI/LPD waveform simulation — SAR-only.

FHSS: 75 channels, 100 hops/sec, 1 MHz spacing.
DSSS: BPSK, 11-chip Barker, 11 Mcps (processing gain 10.4 dB).
Power control: adaptive -10 .. +20 dBm. LPI threshold < -100 dBm @ 1 km.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

N_CHANNELS = 75
HOP_RATE_HZ = 100.0
CHANNEL_SPACING_MHZ = 1.0
CENTER_FREQ_MHZ = 2400.0
BARKER_11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=float)
DSSS_GAIN_DB = 10.0 * math.log10(11)  # ~= 10.41 dB
FHSS_GAIN_DB = 10.0 * math.log10(N_CHANNELS)  # ~= 18.75 dB
MIN_TX_DBM = -10.0
MAX_TX_DBM = 20.0
LPI_THRESHOLD_DBM = -100.0


def fspl_db(distance_m: float, freq_mhz: float = CENTER_FREQ_MHZ) -> float:
    d_km = max(float(distance_m), 0.001) / 1000.0
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(freq_mhz) + 32.44


def dsss_spread(bits: np.ndarray) -> np.ndarray:
    """Spread BPSK bits (±1) with 11-chip Barker code."""
    bits = np.asarray(bits, dtype=float).reshape(-1)
    return np.repeat(bits, len(BARKER_11)) * np.tile(BARKER_11, len(bits))


def dsss_despread(chips: np.ndarray) -> np.ndarray:
    """Correlate received chips against Barker code (hard decision)."""
    chips = np.asarray(chips, dtype=float).reshape(-1)
    n = len(chips) // len(BARKER_11)
    out = np.zeros(n)
    for i in range(n):
        seg = chips[i * len(BARKER_11):(i + 1) * len(BARKER_11)]
        out[i] = 1.0 if float(seg @ BARKER_11) >= 0 else -1.0
    return out


@dataclass
class LpiLpdLink:
    """FHSS/DSSS link with adaptive power control."""

    tx_power_dbm: float = 0.0
    n_channels: int = N_CHANNELS
    hop_rate_hz: float = HOP_RATE_HZ
    hop_seed: int = 23
    spreading: bool = True

    def __post_init__(self) -> None:
        self.tx_power_dbm = float(np.clip(self.tx_power_dbm, MIN_TX_DBM, MAX_TX_DBM))
        self._hop_index = 0

    def hop_sequence(self, n: int) -> List[int]:
        seq = []
        for _ in range(n):
            self._hop_index += 1
            seq.append((self.hop_seed * 1103515245 + self._hop_index * 12345) % self.n_channels)
        return [int(c) for c in seq]

    @property
    def processing_gain_db(self) -> float:
        g = FHSS_GAIN_DB
        if self.spreading:
            g += DSSS_GAIN_DB
        return g

    def interceptor_power_dbm(self, distance_m: float) -> float:
        """Power seen by an interceptor (no despread/hop gain)."""
        return self.tx_power_dbm - fspl_db(distance_m)

    def is_detectable(self, distance_m: float,
                      threshold_dbm: float = LPI_THRESHOLD_DBM) -> bool:
        return self.interceptor_power_dbm(distance_m) >= threshold_dbm

    def adapt_power(self, distance_m: float, target_snr_db: float = 8.0,
                    noise_floor_dbm: float = -114.0) -> float:
        """Adapt TX power so friendly (despread) SNR hits target.

        Friendly rx = tx - pathloss + PG; solve tx = target + noise + PL - PG.
        Clamped to [-10, +20] dBm. Returns new tx power.
        """
        want = target_snr_db + noise_floor_dbm + fspl_db(distance_m) - self.processing_gain_db
        self.tx_power_dbm = float(np.clip(want, MIN_TX_DBM, MAX_TX_DBM))
        return self.tx_power_dbm

    def report(self, distance_m: float = 1000.0) -> Dict:
        return {
            "tx_power_dbm": self.tx_power_dbm,
            "processing_gain_db": self.processing_gain_db,
            "interceptor_1km_dbm": self.interceptor_power_dbm(1000.0),
            "detectable_1km": self.is_detectable(1000.0),
            "lpi_threshold_dbm": LPI_THRESHOLD_DBM,
            "fhss": {"channels": self.n_channels, "hop_rate_hz": self.hop_rate_hz},
            "dsss": {"code": "barker-11", "gain_db": DSSS_GAIN_DB if self.spreading else 0.0},
        }
