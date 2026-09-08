"""Score terms used by the finite-particle flow in draft Eq. (23)."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

INTERACTION_METHODS = ("gaussian", "diffusion_map")


def gaussian_density(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    """Evaluate a one-dimensional Gaussian density."""
    if variance <= 0.0:
        raise ValueError("variance must be positive")
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def _validated_mixture_parameters(
    weights: np.ndarray | tuple[float, ...],
    means: np.ndarray | tuple[float, ...],
    variances: np.ndarray | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize Gaussian-mixture parameters."""
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional array")
    if means.shape != weights.shape or variances.shape != weights.shape:
        raise ValueError("weights, means, and variances must have the same shape")
    if np.any(weights <= 0.0) or np.any(variances <= 0.0):
        raise ValueError("mixture weights and variances must be positive")
    return weights / np.sum(weights), means, variances


def gaussian_mixture_density(
    x: np.ndarray,
    weights: np.ndarray | tuple[float, ...],
    means: np.ndarray | tuple[float, ...],
    variances: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Evaluate a Gaussian-mixture density used in Sec. VI."""
    weights, means, variances = _validated_mixture_parameters(
        weights, means, variances
    )
    x = np.asarray(x, dtype=float)
    components = [
        w * gaussian_density(x, m, v)
        for w, m, v in zip(weights, means, variances)
    ]
    return np.sum(np.stack(components, axis=0), axis=0)


def gaussian_mixture_target_gradient(
    x: np.ndarray,
    weights: np.ndarray | tuple[float, ...],
    means: np.ndarray | tuple[float, ...],
    variances: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Return ``-grad log rho_i^infty`` for the KL force in Eq. (23)."""
    weights, means, variances = _validated_mixture_parameters(
        weights, means, variances
    )
    x = np.asarray(x, dtype=float)
    flat_x = x.reshape(-1)

    log_components = (
        np.log(weights)[:, None]
        - 0.5 * np.log(2.0 * np.pi * variances)[:, None]
        - 0.5 * (flat_x[None, :] - means[:, None]) ** 2 / variances[:, None]
    )
    responsibilities = np.exp(
        log_components - logsumexp(log_components, axis=0)[None, :]
    )
    gradients = (flat_x[None, :] - means[:, None]) / variances[:, None]
    return np.sum(responsibilities * gradients, axis=0).reshape(x.shape)


def gaussian_interaction_score(
    particles: np.ndarray,
    variance_floor: float = 1e-8,
) -> np.ndarray:
    """Gaussian closure for ``grad log rho_i`` in Eq. (23)."""
    x = np.asarray(particles, dtype=float)
    if x.ndim != 1:
        raise ValueError("particles must be one-dimensional")
    variance = float(np.var(x, ddof=1)) if x.size > 1 else variance_floor
    return -(x - np.mean(x)) / max(variance, variance_floor)


def diffusion_map_interaction_score(
    particles: np.ndarray,
    epsilon: float,
    denominator_floor: float = 1e-14,
) -> np.ndarray:
    """Diffusion-map approximation of ``grad log rho_i`` used in Eq. (23). See ref [11]"""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    x = np.asarray(particles, dtype=float)
    if x.ndim != 1:
        raise ValueError("particles must be one-dimensional")

    differences = x[None, :] - x[:, None]
    kernel = np.exp(-(differences * differences) / (4.0 * epsilon))
    q = np.maximum(np.sum(kernel, axis=1), denominator_floor)
    normalized_kernel = kernel / np.sqrt(q)[None, :]
    denominator = np.maximum(
        np.sum(normalized_kernel, axis=1), denominator_floor
    )
    return (
        np.sum(normalized_kernel * differences, axis=1)
        / (epsilon * denominator)
    )
