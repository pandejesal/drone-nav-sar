#!/usr/bin/env python3
"""Sprint 24: safety invariants — barrier functions & CBF verification.

Zero-order CBF:   h(x) >= 0  ⇒  ḣ + α(h) >= 0,  α(h) = γ·h.
High-order CBF:   chain ψ_0=h, ψ_k=ψ̇_{k-1}+α_k(ψ_{k-1}), all >= 0.
Reciprocal CBF:   pairwise multi-agent B(xi,xj) = ‖xi-xj‖ − d_min.
NumPy-only, SAR-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np


@dataclass
class SafetyConfig:
    geofence_xy: float = 50.0
    min_altitude: float = 0.3
    max_altitude: float = 120.0
    max_speed: float = 15.0
    obstacle_margin: float = 1.0
    cbf_gamma: float = 1.0
    tol: float = 1e-6


def _num_grad(h: Callable[[np.ndarray], float], x: np.ndarray,
              eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (h(xp) - h(xm)) / (2 * eps)
    return g


class ZeroOrderCBF:
    """h(x) >= 0 safe; CBF condition ḣ + γ·h >= 0."""

    def __init__(self, h: Callable[[np.ndarray], float],
                 gamma: float = 1.0,
                 grad: Callable[[np.ndarray], np.ndarray] | None = None):
        self.h = h
        self.gamma = float(gamma)
        self._grad = grad

    def grad(self, x: np.ndarray) -> np.ndarray:
        if self._grad is not None:
            return np.asarray(self._grad(x), dtype=np.float64)
        return _num_grad(self.h, x)

    def margin(self, x: np.ndarray) -> float:
        return float(self.h(np.asarray(x, dtype=np.float64)))

    def condition(self, x: np.ndarray, x_dot: np.ndarray) -> float:
        """ḣ + α(h); >= 0 means the CBF condition holds."""
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(x_dot, dtype=np.float64).reshape(x.reshape(-1).shape)
        return float(self.grad(x).reshape(-1) @ v + self.gamma * self.h(x))

    def verify(self, states: np.ndarray, velocities: np.ndarray,
               tol: float = 1e-6) -> Tuple[bool, float]:
        S = np.asarray(states, dtype=np.float64).reshape(-1, 3)
        V = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
        vals = [self.condition(s, v) for s, v in zip(S, V)]
        return (all(c >= -tol for c in vals),
                float(min(vals)) if vals else float("inf"))


class HighOrderCBF:
    """Recursive HOCBF: ψ_0 = h, ψ_k = ψ̇_{k-1} + α_k(ψ_{k-1})."""

    def __init__(self, h: Callable[[np.ndarray], float],
                 relative_degree: int = 2,
                 gammas: List[float] | None = None):
        self.h = h
        self.r = int(relative_degree)
        self.gammas = list(gammas) if gammas else [1.0] * self.r

    def psi_chain(self, x: np.ndarray, derivs: List[np.ndarray]) -> List[float]:
        """derivs[k] = x^{(k+1)}; returns [ψ_0..ψ_{r-1}] values."""
        x = np.asarray(x, dtype=np.float64)
        psi = [float(self.h(x))]
        g = _num_grad(self.h, x)
        for k in range(1, self.r):
            v = np.asarray(derivs[k - 1]).reshape(-1) if k - 1 < len(derivs) else np.zeros_like(g.reshape(-1))
            psi_dot_prev = float(g.reshape(-1) @ v)
            psi.append(psi_dot_prev + self.gammas[k - 1] * psi[-1])
        return psi

    def verify(self, x: np.ndarray, derivs: List[np.ndarray],
               tol: float = 1e-6) -> Tuple[bool, float]:
        chain = self.psi_chain(x, derivs)
        return (all(p >= -tol for p in chain), float(min(chain)))


class ReciprocalCBF:
    """Pairwise multi-agent barrier B(xi,xj) = ‖xi − xj‖ − d_min."""

    def __init__(self, d_min: float = 2.0, gamma: float = 1.0):
        self.d_min = float(d_min)
        self.gamma = float(gamma)

    def barrier(self, xi: np.ndarray, xj: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(xi) - np.asarray(xj))) - self.d_min

    def condition(self, xi: np.ndarray, xj: np.ndarray,
                  vi: np.ndarray, vj: np.ndarray) -> float:
        xi, xj = np.asarray(xi, float), np.asarray(xj, float)
        d = xi - xj
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return -float("inf")
        n_vec = d / n
        h_dot = float(n_vec @ (np.asarray(vi, float) - np.asarray(vj, float)))
        return h_dot + self.gamma * (n - self.d_min)

    def verify_pair(self, xi: np.ndarray, xj: np.ndarray,
                    vi: np.ndarray, vj: np.ndarray,
                    tol: float = 1e-6) -> Tuple[bool, float]:
        c = self.condition(xi, xj, vi, vj)
        return (c >= -tol, float(c))


class SafetyInvariantChecker:
    """Geofence / altitude / speed / obstacle invariants over episodes."""

    def __init__(self, config: SafetyConfig | None = None,
                 obstacles: List[np.ndarray] | None = None,
                 obstacle_radius: float = 1.0):
        self.cfg = config or SafetyConfig()
        self.obstacles = [np.asarray(o, float).reshape(3) for o in (obstacles or [])]
        self.obstacle_radius = float(obstacle_radius)
        sep = self.cfg.obstacle_margin + self.obstacle_radius
        self._obstacle_cbfs = [
            ZeroOrderCBF(lambda p, o=o: float(np.linalg.norm(np.asarray(p).reshape(3) - o)) - sep,
                         gamma=self.cfg.cbf_gamma)
            for o in self.obstacles
        ]

    def check(self, states: np.ndarray) -> List[str]:
        S = np.asarray(states, dtype=np.float64)
        P = S[:, 0:3] if S.ndim == 2 and S.shape[1] >= 3 else S.reshape(-1, 3)
        V = S[:, 3:6] if S.ndim == 2 and S.shape[1] >= 6 else np.zeros_like(P)
        c = self.cfg
        bad: List[str] = []
        if np.any(np.abs(P[:, 0]) > c.geofence_xy) or np.any(np.abs(P[:, 1]) > c.geofence_xy):
            bad.append("geofence")
        if np.any(P[:, 2] < c.min_altitude):
            bad.append("min_altitude")
        if np.any(P[:, 2] > c.max_altitude):
            bad.append("max_altitude")
        if np.any(np.linalg.norm(V, axis=1) > c.max_speed):
            bad.append("max_speed")
        for cbf in self._obstacle_cbfs:
            if any(cbf.margin(p) < 0 for p in P):
                bad.append("obstacle_cbf")
                break
        return sorted(set(bad))

    def verify_cbf_conditions(self, states: np.ndarray,
                              dt: float = 0.1) -> Dict[str, object]:
        """Finite-difference ḣ + γh >= 0 check per obstacle CBF."""
        S = np.asarray(states, dtype=np.float64)
        P = S[:, 0:3] if S.ndim == 2 and S.shape[1] >= 3 else S.reshape(-1, 3)
        V = (P[1:] - P[:-1]) / dt if len(P) > 1 else np.zeros_like(P)
        worst = float("inf")
        ok_all = True
        for cbf in self._obstacle_cbfs:
            ok, w = cbf.verify(P[:-1], V, tol=self.cfg.tol)
            ok_all = ok_all and ok
            worst = min(worst, w)
        if not self._obstacle_cbfs:
            worst = float("inf")
        return {"cbf_holds": bool(ok_all), "worst_margin_rate": float(worst)}
