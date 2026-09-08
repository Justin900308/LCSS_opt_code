"""Adaptive integration of the six coupled flows in draft Eq. (23)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

ADAPTIVE_METHODS = ("RK45", "DOP853", "Radau", "BDF", "LSODA")
ALL_METHODS = ADAPTIVE_METHODS


@dataclass
class Trajectory:
    """Time/state output returned by ``solve_ivp``."""

    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    nfev: int
    runtime_seconds: float


def integrate(
    system,
    t_span: tuple[float, float],
    y0: np.ndarray,
    method: str = "DOP853",
    output_samples: int = 801,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = np.inf,
) -> Trajectory:
    """Integrate Eq. (23) with a SciPy adaptive ODE solver."""
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f"method must be one of {ADAPTIVE_METHODS}")
    if rtol <= 0.0 or atol <= 0.0 or max_step <= 0.0:
        raise ValueError("rtol, atol, and max_step must be positive")

    t0, tf = map(float, t_span)
    if tf <= t0:
        raise ValueError("Require tf > t0")

    t_eval = np.linspace(t0, tf, output_samples) if output_samples >= 2 else None
    start = time.perf_counter()
    solution = solve_ivp(
        system.rhs,
        (t0, tf),
        np.asarray(y0, dtype=float),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    return Trajectory(
        t=solution.t,
        y=solution.y,
        success=bool(solution.success),
        message=str(solution.message),
        nfev=int(solution.nfev),
        runtime_seconds=time.perf_counter() - start,
    )
