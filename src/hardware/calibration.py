#!/usr/bin/env python3
"""Calibration for DroneNav-SAR hardware (Sprint 14).

- IMU bias: mean gyro at rest; pass if |bias| < 0.01 rad/s per axis.
- Motor thrust curve: linear fit thrust = a*pwm + b; pass if R^2 > 0.98.
- Vibration analysis: RMS / peak / dominant frequency from accel samples.

stdlib + numpy only.
"""

from typing import Any, Dict

import numpy as np

IMU_BIAS_TOL_RAD_S = 0.01
THRUST_R2_MIN = 0.98


def estimate_imu_bias(gyro_samples: np.ndarray) -> Dict[str, Any]:
    """Estimate gyro bias from Mx3 static samples (rad/s)."""
    g = np.asarray(gyro_samples, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"gyro_samples must be Mx3, got {g.shape}")
    if g.shape[0] < 10:
        raise ValueError("need >= 10 gyro samples")
    bias = g.mean(axis=0)
    std = g.std(axis=0)
    per_axis_ok = bool(np.all(np.abs(bias) < IMU_BIAS_TOL_RAD_S))
    return {"bias_rad_s": bias.astype(float).tolist(),
            "std_rad_s": std.astype(float).tolist(),
            "bias_norm": float(np.linalg.norm(bias)),
            "tol": IMU_BIAS_TOL_RAD_S,
            "pass": per_axis_ok,
            "n_samples": int(g.shape[0])}


def fit_thrust_curve(pwm: np.ndarray, thrust_n: np.ndarray) -> Dict[str, Any]:
    """Linear fit thrust(pwm); returns slope, intercept, R^2."""
    x = np.asarray(pwm, dtype=np.float64).ravel()
    y = np.asarray(thrust_n, dtype=np.float64).ravel()
    if x.shape != y.shape or x.size < 3:
        raise ValueError("pwm and thrust_n must match with >= 3 points")
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(a), "intercept": float(b),
            "r2": float(r2), "r2_min": THRUST_R2_MIN,
            "pass": bool(r2 > THRUST_R2_MIN), "n_points": int(x.size)}


def analyze_vibration(accel_samples: np.ndarray,
                      dt_s: float = 0.005) -> Dict[str, Any]:
    """RMS / peak / dominant-frequency vibration summary from Nx3 accel (m/s^2)."""
    a = np.asarray(accel_samples, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"accel_samples must be Nx3, got {a.shape}")
    dt = float(dt_s)
    if not 0 < dt < 1.0:
        raise ValueError("dt_s must be in (0, 1)")
    centered = a - a.mean(axis=0)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    peak = float(np.abs(centered).max())
    # Dominant frequency of the highest-energy axis via rFFT.
    energy = (centered ** 2).sum(axis=0)
    axis = int(np.argmax(energy))
    sig = centered[:, axis]
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(sig.size, d=dt)
    dom_idx = int(np.argmax(spec[1:]) + 1) if spec.size > 2 else 0
    dom_freq = float(freqs[dom_idx]) if spec.size > 1 else 0.0
    return {"rms": rms, "peak": peak, "dominant_freq_hz": dom_freq,
            "dominant_axis": axis, "n_samples": int(a.shape[0]),
            "pass": bool(rms < 1.5 and peak < 8.0)}


def run_calibration(gyro_samples: np.ndarray,
                    pwm: np.ndarray, thrust_n: np.ndarray,
                    accel_samples: np.ndarray,
                    dt_s: float = 0.005) -> Dict[str, Any]:
    """Full calibration report; pass requires IMU + thrust (+ vibration info)."""
    imu = estimate_imu_bias(gyro_samples)
    thrust = fit_thrust_curve(pwm, thrust_n)
    vib = analyze_vibration(accel_samples, dt_s)
    overall = bool(imu["pass"] and thrust["pass"])
    return {"imu": imu, "thrust_curve": thrust, "vibration": vib,
            "pass": overall}
