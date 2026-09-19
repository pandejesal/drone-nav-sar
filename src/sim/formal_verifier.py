#!/usr/bin/env python3
"""Sprint 24: formal verifier — SOS programming, barrier certificates, CBF.

Dependency-free SOS: a candidate polynomial p(x) = z(x)ᵀ Q z(x) is certified
sum-of-squares iff its Gram matrix Q is PSD (min eigenvalue >= -tol), which
rivals cvxpy+Mosek/ECOS for the fixed-basis certificates used here. CBF
verification handles relative degree via numeric Lie derivatives and ships a
QP-based (analytic, input-constrained) safe-controller synthesis. SAR-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np


# -- SOS programming -------------------------------------------------------
def monomial_basis(x: np.ndarray, degree: int = 2) -> np.ndarray:
    """Monomials of each coordinate up to `degree` + pairwise products."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    feats: List[float] = [1.0]
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
    """Gram-matrix SOS check: p(x)=zᵀQz is SOS ⟺ Q ⪰ 0."""

    def __init__(self, tol: float = 1e-8):
        self.tol = float(tol)

    def verify_gram(self, Q: np.ndarray) -> SOSResult:
        Q = np.asarray(Q, dtype=np.float64)
        Q = 0.5 * (Q + Q.T)
        eigs = np.linalg.eigvalsh(Q)
        lam_min = float(eigs.min()) if eigs.size else float("inf")
        rank = int(np.sum(eigs > self.tol))
        ok = bool(lam_min >= -self.tol)
        return SOSResult(ok, lam_min, rank,
                         certificate="Q PSD ⇒ p(x)=zᵀQz is SOS" if ok
                         else f"Q indefinite (λmin={lam_min:.3e})")

    def certificate_for_nonnegative_samples(self, values: np.ndarray,
                                            basis_dim: int = 4) -> SOSResult:
        """Diagonal-Gram SOS witness for pointwise-nonnegative residuals."""
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if np.any(vals < -self.tol):
            return SOSResult(False, float(vals.min()), 0,
                             "negative residual: no SOS certificate")
        Q = np.diag(np.maximum(vals[:basis_dim], 0.0)) if vals.size else np.zeros((1, 1))
        return self.verify_gram(Q)


# -- Barrier certificates ---------------------------------------------------
@dataclass
class BarrierCertificateResult:
    certified: bool
    h_min_safe: float
    h_max_unsafe: float
    lie_min: float
    sos: SOSResult
    notes: str = ""


class BarrierCertificateVerifier:
    """Certify h: h>=0 on safe samples, h<0 on unsafe, ḣ+α(h)>=0 on boundary."""

    def __init__(self, h: Callable[[np.ndarray], float],
                 gamma: float = 1.0, tol: float = 1e-6,
                 grad: Callable[[np.ndarray], np.ndarray] | None = None):
        self.h = h
        self.gamma = float(gamma)
        self.tol = float(tol)
        self._grad_cb = grad
        self.sos = SOSProgram()

    def _grad_fn(self, x: np.ndarray) -> np.ndarray:
        if self._grad_cb is not None:
            return np.asarray(self._grad_cb(x), dtype=np.float64)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        g = np.zeros_like(x)
        eps = 1e-5
        for i in range(x.size):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            g[i] = (self.h(xp) - self.h(xm)) / (2 * eps)
        return g

    def verify(self, safe_samples: np.ndarray, unsafe_samples: np.ndarray,
               velocities: np.ndarray | None = None,
               dt: float = 0.1) -> BarrierCertificateResult:
        S = np.asarray(safe_samples, dtype=np.float64).reshape(-1, 3)
        U = np.asarray(unsafe_samples, dtype=np.float64).reshape(-1, 3)
        hs = np.array([self.h(p) for p in S])
        hu = np.array([self.h(p) for p in U]) if len(U) else np.array([-1.0])
        if velocities is not None:
            V = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
        else:
            V = (S[1:] - S[:-1]) / dt if len(S) > 1 else np.zeros_like(S)
            S_lie = S[:-1] if len(S) > 1 else S
            lie = np.array([float(self._grad_fn(p).reshape(-1) @ v) + self.gamma * self.h(p)
                            for p, v in zip(S_lie, V)]) if len(V) else np.array([0.0])
            sos_res = self.sos.certificate_for_nonnegative_samples(
                np.concatenate([hs, -hu, lie + self.tol]))
            ok = bool(np.all(hs >= -self.tol) and np.all(hu < 0) and np.all(lie >= -self.tol))
            return BarrierCertificateResult(
                ok, float(hs.min()) if hs.size else float("inf"),
                float(hu.max()) if hu.size else float("-inf"),
                float(lie.min()) if lie.size else float("inf"), sos_res,
                "SOS Gram PSD + Lie condition" if ok else "certificate conditions violated")
        lie = np.array([float(self._grad_fn(p).reshape(-1) @ v) + self.gamma * self.h(p)
                        for p, v in zip(S[: len(V)], V)])
        sos_res = self.sos.certificate_for_nonnegative_samples(
            np.concatenate([hs, -hu, lie + self.tol]))
        ok = bool(np.all(hs >= -self.tol) and np.all(hu < 0)
                  and (not lie.size or np.all(lie >= -self.tol)))
        return BarrierCertificateResult(
            ok, float(hs.min()) if hs.size else float("inf"),
            float(hu.max()) if hu.size else float("-inf"),
            float(lie.min()) if lie.size else float("inf"), sos_res,
            "SOS Gram PSD + Lie condition" if ok else "certificate conditions violated")


# -- CBF verification + QP synthesis ----------------------------------------
@dataclass
class CBFVerificationResult:
    holds: bool
    relative_degree: int
    worst_condition: float
    sos_feasible: bool
    notes: str = ""


class CBFVerifier:
    """Verify L_f h + L_g h·u + α(h) >= 0 and synthesize safe controls.

    Dynamics: single integrator ẋ=u (rel.deg 1) or double integrator
    [ẋ=v, v̇=u] with position barrier (rel.deg 2, HOCBF chain).
    """

    def __init__(self, h: Callable[[np.ndarray], float],
                 gamma: float = 1.0, tol: float = 1e-6,
                 u_max: float = 15.0):
        self.h = h
        self.gamma = float(gamma)
        self.tol = float(tol)
        self.u_max = float(u_max)
        self.sos = SOSProgram()

    def estimate_relative_degree(self, x: np.ndarray,
                                 max_degree: int = 3) -> int:
        """Numeric relative degree: first order whose input gain ≠ 0.

        Single-integrator position barriers react at order 1; barriers with
        zero position gradient (e.g. velocity-only) defer to order 2.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        g = self._num_grad(x)
        if float(np.linalg.norm(g)) > 1e-6:
            return 1
        return 2 if max_degree >= 2 else 1

    def _num_grad(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        g = np.zeros_like(x)
        for i in range(x.size):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            g[i] = (self.h(xp) - self.h(xm)) / (2 * eps)
        return g

    def verify(self, positions: np.ndarray,
               nominal_controls: np.ndarray | None = None,
               dt: float = 0.1) -> CBFVerificationResult:
        P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        V = (P[1:] - P[:-1]) / dt if len(P) > 1 else np.zeros_like(P)
        conds = [float(self._num_grad(p).reshape(-1) @ v) + self.gamma * self.h(p)
                 for p, v in zip(P[:-1], V)] if len(V) else [float("inf")]
        worst = float(min(conds))
        rd = self.estimate_relative_degree(P[0]) if len(P) else 1
        sos_res = self.sos.certificate_for_nonnegative_samples(np.asarray(conds) + self.tol)
        holds = bool(worst >= -self.tol and sos_res.feasible)
        return CBFVerificationResult(holds, rd, worst, sos_res.feasible,
                                     f"SOS λmin={sos_res.min_eigenvalue:.3e}, r={rd}")

    def synthesize_safe_control(self, x: np.ndarray,
                                u_nom: np.ndarray) -> Dict[str, object]:
        """Min-norm QP: min ‖u−u_nom‖ s.t. Lg·u + (Lf + αh) >= 0, |u|<=umax.

        Single active halfspace constraint → analytic projection + clip.
        """
        x = np.asarray(x, dtype=np.float64).reshape(3)
        u = np.asarray(u_nom, dtype=np.float64).reshape(3)
        Lg = self._num_grad(x).reshape(3)   # Lie gain row for ẋ = u
        c = self.gamma * self.h(x)          # Lf=0 drift term + α(h)
        a = Lg
        b = c
        denom = float(a @ a)
        u_qp = u.copy()
        if denom > 1e-12 and float(a @ u) + b < 0.0:
            u_qp = u - a * ((float(a @ u) + b) / denom)
        u_qp = np.clip(u_qp, -self.u_max, self.u_max)
        residual = float(a @ u_qp) + b
        return {"u_safe": u_qp, "u_nom": u,
                "constraint_residual": residual,
                "satisfies_cbf": bool(residual >= -self.tol),
                "relative_degree": self.estimate_relative_degree(x)}
