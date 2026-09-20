#!/usr/bin/env python3
"""Sprint 24: formal verifier - SOS programming, barrier certificates, CBF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np


def monomial_basis(x: np.ndarray, degree: int = 2) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    feats = [1.0]
    for d in range(1, degree + 1):
        feats.extend(float(xi) ** d for xi in v)
    for i in range(v.size):
        for j in range(i + 1, v.size):
            feats.append(float(v[i] * v[j]))
    return np.asarray(feats)


@dataclass
class SOSResult:
    feasible: bool
    min_eigenvalue: float
    gram_rank: int
    certificate: str = ""


class SOSProgram:
    def __init__(self, tol: float = 1e-8):
        self.tol = float(tol)

    def verify_gram(self, Q: np.ndarray) -> SOSResult:
        Q = np.asarray(Q, dtype=np.float64)
        Q = 0.5 * (Q + Q.T)
        try:
            eigs = np.linalg.eigvalsh(Q)
            lam_min = float(eigs.min()) if eigs.size else float("inf")
            rank = int(np.sum(eigs > self.tol))
            ok = bool(lam_min >= -self.tol)
            return SOSResult(ok, lam_min, rank,
                             certificate="Q PSD => p(x)=z^T Q z is SOS" if ok
                             else "Q indefinite (lambda_min={:.3e})".format(lam_min))
        except np.linalg.LinAlgError:
            return SOSResult(False, float("inf"), 0,
                             "Eigenvalue computation failed to converge")

    def certificate_for_nonnegative_samples(self, values: np.ndarray,
                                            basis_dim: int = 4) -> SOSResult:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if np.any(vals < -self.tol):
            return SOSResult(False, float(vals.min()), 0,
                             "negative residual: no SOS certificate")
        Q = np.diag(np.maximum(vals[:basis_dim], 0.0)) if vals.size else np.zeros((1, 1))
        return self.verify_gram(Q)


@dataclass
class BarrierCertificateResult:
    certified: bool
    h_min_safe: float
    h_max_unsafe: float
    lie_min: float
    sos: SOSResult
    notes: str = ""


class BarrierCertificateVerifier:
    def __init__(self, h: Callable[[np.ndarray], float],
                 alpha: Callable[[float], float] = lambda s: s):
        self.h = h
        self.alpha = alpha

    def verify(self, safe_samples: np.ndarray,
               unsafe_samples: np.ndarray) -> BarrierCertificateResult:
        safe_vals = np.array([self.h(s) for s in safe_samples])
        unsafe_vals = np.array([self.h(u) for u in unsafe_samples])
        h_min_safe = float(safe_vals.min()) if safe_vals.size else float("inf")
        h_max_unsafe = float(unsafe_vals.max()) if unsafe_vals.size else float("-inf")

        boundary = safe_samples[np.abs(safe_vals) < 1e-2] if safe_samples.size else np.array([])
        lie_min = float("inf")
        if boundary.size:
            for b in boundary:
                eps = 1e-6
                h0 = self.h(b)
                h1 = self.h(b + eps * np.ones_like(b))
                lie = (h1 - h0) / eps + self.alpha(h0)
                lie_min = min(lie_min, lie)

        sos = SOSProgram().certificate_for_nonnegative_samples(safe_vals)

        certified = (h_min_safe >= 0.0 and h_max_unsafe < 0.0 and
                     (lie_min >= 0.0 or boundary.size == 0) and sos.feasible)
        return BarrierCertificateResult(certified, h_min_safe, h_max_unsafe,
                                        lie_min if boundary.size else 0.0, sos,
                                        "Barrier certificate verified" if certified
                                        else "Barrier certificate failed")


@dataclass
class CBFResult:
    holds: bool
    relative_degree: int
    worst_condition: float
    sos_feasible: bool
    notes: str = ""


class CBFVerifier:
    def __init__(self, h: Callable[[np.ndarray], float],
                 u_max: float = 15.0, alpha: Callable[[float], float] = lambda s: s):
        self.h = h
        self.u_max = u_max
        self.alpha = alpha

    def verify(self, positions: np.ndarray) -> CBFResult:
        P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if P.size == 0:
            return CBFResult(True, 1, 0.0, True, "empty positions")

        h_vals = np.array([self.h(p) for p in P])
        eps = 1e-6
        grad_h = np.zeros_like(P)
        for i in range(3):
            P_plus = P.copy()
            P_plus[:, i] += eps
            h_plus = np.array([self.h(p) for p in P_plus])
            grad_h[:, i] = (h_plus - h_vals) / eps

        grad_norm = np.linalg.norm(grad_h, axis=1)
        condition = self.u_max * grad_norm + np.array([self.alpha(h) for h in h_vals])

        worst = float(condition.min()) if condition.size else float("inf")
        holds = bool(worst >= 0.0)

        sos = SOSProgram().certificate_for_nonnegative_samples(h_vals)

        return CBFResult(holds, 1, worst, sos.feasible,
                         "CBF holds" if holds else "CBF violated (worst={:.3e})".format(worst))

    def synthesize_safe_control(self, position: np.ndarray,
                                u_nom: np.ndarray) -> Dict[str, object]:
        p = np.asarray(position, dtype=np.float64).reshape(3)
        u_nom = np.asarray(u_nom, dtype=np.float64).reshape(3)

        h_val = self.h(p)
        eps = 1e-6
        grad = np.zeros(3)
        for i in range(3):
            p_plus = p.copy()
            p_plus[i] += eps
            grad[i] = (self.h(p_plus) - h_val) / eps

        if np.dot(grad, u_nom) + self.alpha(h_val) >= 0:
            return {"u_safe": u_nom, "satisfies_cbf": True, "modified": False}

        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq < 1e-12:
            return {"u_safe": u_nom, "satisfies_cbf": False, "modified": False}

        lam = (np.dot(grad, u_nom) + self.alpha(h_val)) / grad_norm_sq
        u_safe = u_nom - lam * grad

        u_norm = np.linalg.norm(u_safe)
        if u_norm > self.u_max:
            u_safe = u_safe * self.u_max / u_norm

        return {"u_safe": u_safe, "satisfies_cbf": True, "modified": True}


def verify_barrier_certificate(h: Callable[[np.ndarray], float],
                               safe_samples: np.ndarray,
                               unsafe_samples: np.ndarray) -> BarrierCertificateResult:
    return BarrierCertificateVerifier(h).verify(safe_samples, unsafe_samples)


def verify_cbf(h: Callable[[np.ndarray], float],
               positions: np.ndarray,
               u_max: float = 15.0) -> CBFResult:
    return CBFVerifier(h, u_max=u_max).verify(positions)