#!/usr/bin/env python3
"""Sprint 19: LPI/LPD resilient link layer (SAR-only).

Simulates frequency-hopping spread spectrum (FHSS) with:
  - pseudo-random hop sequence shared by the swarm (symmetric key ``hop_seed``)
  - processing gain from spreading over ``n_channels``
  - anti-jam margin vs. a broadband / partial-band jammer
  - LPI/LPD accounting: interceptor received power at range

SAR-only: carries mission-intent / telemetry payloads only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# 2.4 GHz ISM band model.
CENTER_FREQ_MHZ = 2400.0
# Thermal noise floor for 1 MHz hop channel (dBm).
NOISE_FLOOR_DBM = -114.0
# Required SNR for the (coded) SAR telemetry waveform.
REQUIRED_SNR_DB = 6.0


def fspl_db(distance_m: float, freq_mhz: float = CENTER_FREQ_MHZ) -> float:
    """Free-space path loss in dB."""
    d_km = max(float(distance_m), 0.001) / 1000.0
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(freq_mhz) + 32.44


@dataclass
class JammerSpec:
    """Broadband/partial-band jammer description."""

    power_dbm: float = 30.0       # jammer EIRP
    bandwidth_frac: float = 0.3   # fraction of band jammed (0..1)
    distance_m: float = 500.0     # jammer -> victim receiver range


class ResilientLink:
    """FHSS LPI/LPD link endpoint shared by every swarm member."""

    def __init__(
        self,
        tx_power_dbm: float = -3.0,
        n_channels: int = 50,
        hop_rate_hz: float = 100.0,
        hop_seed: int = 19,
        spreading_gain_db: Optional[float] = None,
    ):
        self.tx_power_dbm = float(tx_power_dbm)
        self.n_channels = max(1, int(n_channels))
        self.hop_rate_hz = float(hop_rate_hz)
        self.hop_seed = int(hop_seed)
        # Processing gain ~= 10*log10(channels) unless overridden.
        self.spreading_gain_db = (
            float(spreading_gain_db)
            if spreading_gain_db is not None
            else 10.0 * math.log10(self.n_channels)
        )
        self._rng = np.random.default_rng(self.hop_seed)
        self._hop_index = 0
        self.tx_count = 0
        self.rx_ok_count = 0
        self.jammed_count = 0

    # -- frequency hopping -------------------------------------------
    def hop_sequence(self, n: int) -> List[int]:
        """Next ``n`` hop channels (deterministic from hop_seed)."""
        seq = []
        for _ in range(n):
            # LCG-style deterministic hop for reproducibility across nodes.
            self._hop_index += 1
            ch = (self.hop_seed * 1103515245 + self._hop_index * 12345) % self.n_channels
            seq.append(int(ch))
        return seq

    def reset_hops(self) -> None:
        self._hop_index = 0

    # -- link budget ---------------------------------------------------
    def rx_power_dbm(self, distance_m: float) -> float:
        """Friendly receiver power (despread: +processing gain)."""
        return self.tx_power_dbm - fspl_db(distance_m) + self.spreading_gain_db

    def snr_db(self, distance_m: float) -> float:
        return self.rx_power_dbm(distance_m) - NOISE_FLOOR_DBM

    def interceptor_power_dbm(self, distance_m: float) -> float:
        """Interceptor power: no spreading gain (cannot despread).

        This is the LPI/LPD metric: must be < -100 dBm at 1 km.
        """
        return self.tx_power_dbm - fspl_db(distance_m)

    def is_detectable(self, distance_m: float, threshold_dbm: float = -100.0) -> bool:
        return self.interceptor_power_dbm(distance_m) >= threshold_dbm

    # -- anti-jam --------------------------------------------------------
    def jamming_margin_db(self) -> float:
        """Jamming margin = processing gain - required SNR."""
        return self.spreading_gain_db - REQUIRED_SNR_DB

    def packet_success_prob(
        self, distance_m: float, jammer: Optional[JammerSpec] = None
    ) -> float:
        """Packet success probability with optional jammer.

        Without jamming: logistic on SNR margin. With jamming: a hop is
        lost when it lands in the jammed fraction AND jammer overcomes
        the processing gain; FHSS then recovers via non-jammed hops.
        """
        margin = self.snr_db(distance_m) - REQUIRED_SNR_DB
        base = 1.0 / (1.0 + math.exp(-0.8 * margin))
        base = float(np.clip(base, 0.01, 0.999))
        if jammer is None:
            return base
        # Jammer power at receiver vs signal: does it beat the margin?
        j_rx = jammer.power_dbm - fspl_db(jammer.distance_m)
        s_rx = self.tx_power_dbm - fspl_db(distance_m)
        jam_wins_hop = (j_rx - s_rx) > self.jamming_margin_db()
        if not jam_wins_hop:
            return base  # jammer too weak: FHSS rides through
        # Partial-band: only jammed_frac hops are contested; each
        # contested hop survives with small prob, uncontested at base.
        frac = float(np.clip(jammer.bandwidth_frac, 0.0, 1.0))
        contested_survival = 0.05
        return float((1.0 - frac) * base + frac * contested_survival)

    def send_packet(
        self,
        distance_m: float,
        jammer: Optional[JammerSpec] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict:
        """Simulate one packet; returns {delivered, jammed, snr_db}."""
        r = rng if rng is not None else self._rng
        p = self.packet_success_prob(distance_m, jammer)
        delivered = bool(r.random() < p)
        jammed = bool(jammer is not None and not delivered)
        self.tx_count += 1
        if delivered:
            self.rx_ok_count += 1
        if jammed:
            self.jammed_count += 1
        # Consume one hop.
        self.hop_sequence(1)
        return {"delivered": delivered, "jammed": jammed,
                "snr_db": self.snr_db(distance_m)}

    def report(self) -> Dict:
        return {
            "tx": self.tx_count,
            "rx_ok": self.rx_ok_count,
            "jammed": self.jammed_count,
            "pdr": (self.rx_ok_count / self.tx_count) if self.tx_count else 1.0,
            "hop_rate_hz": self.hop_rate_hz,
            "n_channels": self.n_channels,
            "jamming_margin_db": self.jamming_margin_db(),
            "interceptor_1km_dbm": self.interceptor_power_dbm(1000.0),
        }
