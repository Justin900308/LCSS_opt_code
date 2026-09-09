"""Two-population constrained distribution flow from draft Eqs. (21), (23), and (32)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from grad import (
    INTERACTION_METHODS,
    diffusion_map_interaction_score,
    gaussian_interaction_score,
    gaussian_mixture_density,
    gaussian_mixture_target_gradient,
)

if TYPE_CHECKING:
    from integrator import Trajectory


@dataclass(frozen=True)
class FlowParameters:
    """Parameters for Eq. (23) and the polynomial clock in Sec. VI."""

    p: float = 2.0
    penalty_mu: float = 1.0
    damping_zeta: float = 0.8
    epsilon: float = 0.03
    variance_floor: float = 1e-8
    metric_bins: int = 80

    def __post_init__(self) -> None:
        if self.p <= 0.0 or self.penalty_mu <= 0.0:
            raise ValueError("Require p>0 and penalty_mu>0.")
        if self.damping_zeta < 0.0:
            raise ValueError("Require damping_zeta>=0.")
        if self.epsilon <= 0.0 or self.variance_floor <= 0.0:
            raise ValueError("epsilon and variance_floor must be positive.")
        if self.metric_bins < 32:
            raise ValueError("metric_bins must be at least 32.")


@dataclass(frozen=True)
class PopulationConfig:
    """Target law and affine feature used in Sec. VI."""

    name: str
    feature_C: float
    initial_mean: float = 0.0
    initial_variance: float = 1.0
    target_weights: tuple[float, ...] = (1.0,)
    target_means: tuple[float, ...] = (0.0,)
    target_variances: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        weights = np.asarray(self.target_weights, dtype=float)
        means = np.asarray(self.target_means, dtype=float)
        variances = np.asarray(self.target_variances, dtype=float)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("target_weights must be nonempty and one-dimensional")
        if means.shape != weights.shape or variances.shape != weights.shape:
            raise ValueError("target mixture arrays must have equal lengths")
        if np.any(weights <= 0.0) or np.any(variances <= 0.0):
            raise ValueError("target weights and variances must be positive")
        if self.initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        object.__setattr__(self, "target_weights", tuple((weights / weights.sum()).tolist()))
        object.__setattr__(self, "target_means", tuple(means.tolist()))
        object.__setattr__(self, "target_variances", tuple(variances.tolist()))

    @classmethod
    def gaussian(
        cls,
        name: str,
        mean: float,
        variance: float,
        feature_C: float,
        initial_mean: float = 0.0,
        initial_variance: float = 1.0,
    ) -> "PopulationConfig":
        """Create a one-component Gaussian target."""
        return cls(
            name=name,
            feature_C=feature_C,
            initial_mean=initial_mean,
            initial_variance=initial_variance,
            target_weights=(1.0,),
            target_means=(mean,),
            target_variances=(variance,),
        )

    @classmethod
    def gaussian_mixture(
        cls,
        name: str,
        weights: tuple[float, ...],
        means: tuple[float, ...],
        variances: tuple[float, ...],
        feature_C: float,
        initial_mean: float = 0.0,
        initial_variance: float = 1.0,
    ) -> "PopulationConfig":
        """Create the Gaussian-mixture target used for rho_1."""
        return cls(
            name=name,
            feature_C=feature_C,
            initial_mean=initial_mean,
            initial_variance=initial_variance,
            target_weights=weights,
            target_means=means,
            target_variances=variances,
        )

    def target_density(self, x: np.ndarray) -> np.ndarray:
        """Evaluate rho_i^infinity for the Sec. VI objective."""
        return gaussian_mixture_density(
            x, self.target_weights, self.target_means, self.target_variances
        )

    def target_force(self, x: np.ndarray) -> np.ndarray:
        """Evaluate ``-grad log rho_i^infinity`` in Eq. (23)."""
        return gaussian_mixture_target_gradient(
            x, self.target_weights, self.target_means, self.target_variances
        )

    def feature(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the affine moment feature defining r(rho)."""
        return self.feature_C * np.asarray(x, dtype=float)

    def log_partition(self, multiplier: float) -> float:
        """Log partition of the exact KL exponential tilt."""
        weights = np.asarray(self.target_weights)
        means = np.asarray(self.target_means)
        variances = np.asarray(self.target_variances)
        linear = multiplier * self.feature_C
        terms = np.log(weights) - linear * means + 0.5 * linear**2 * variances
        return float(logsumexp(terms))

    def tilted_parameters(
        self, multiplier: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parameters of rho_i^* from the KL KKT exponential tilt."""
        weights = np.asarray(self.target_weights)
        means = np.asarray(self.target_means)
        variances = np.asarray(self.target_variances)
        linear = multiplier * self.feature_C
        log_weights = np.log(weights) - linear * means + 0.5 * linear**2 * variances
        tilted_weights = np.exp(log_weights - logsumexp(log_weights))
        tilted_means = means - linear * variances
        return tilted_weights, tilted_means, variances.copy()

    def tilted_moments(self, multiplier: float) -> tuple[float, float]:
        """Mean and variance of the exact constrained target."""
        weights, means, variances = self.tilted_parameters(multiplier)
        mean = float(np.sum(weights * means))
        variance = float(np.sum(weights * (variances + (means - mean) ** 2)))
        return mean, variance

    def tilted_density(self, x: np.ndarray, multiplier: float) -> np.ndarray:
        """Evaluate rho_i^* used in the unchanged density plots."""
        weights, means, variances = self.tilted_parameters(multiplier)
        return gaussian_mixture_density(x, weights, means, variances)


@dataclass(frozen=True)
class AnalyticalSolution:
    """Exact constrained KL solution used as a benchmark."""

    multiplier: float
    mean1: float
    mean2: float
    variance1: float
    variance2: float
    objective: float
    weights1: tuple[float, ...]
    component_means1: tuple[float, ...]
    component_variances1: tuple[float, ...]
    weights2: tuple[float, ...]
    component_means2: tuple[float, ...]
    component_variances2: tuple[float, ...]


def analytical_constrained_solution(
    pop1: PopulationConfig,
    pop2: PopulationConfig,
    resource_target: float,
) -> AnalyticalSolution:
    """Solve the scalar KKT feasibility condition for rho_i^*."""
    def residual(multiplier: float) -> float:
        mean1, _ = pop1.tilted_moments(multiplier)
        mean2, _ = pop2.tilted_moments(multiplier)
        return float(
            pop1.feature_C * mean1
            + pop2.feature_C * mean2
            - resource_target
        )

    if abs(residual(0.0)) <= 1e-14:
        multiplier = 0.0
    else:
        radius = 1.0
        for _ in range(80):
            left, right = -radius, radius
            f_left, f_right = residual(left), residual(right)
            if f_left >= 0.0 >= f_right:
                break
                break
            radius *= 2.0
        else:
            raise RuntimeError("Could not bracket the analytical multiplier.")
        multiplier = float(brentq(residual, left, right, xtol=1e-13, rtol=1e-13))

    mean1, variance1 = pop1.tilted_moments(multiplier)
    mean2, variance2 = pop2.tilted_moments(multiplier)
    weights1, means1, variances1 = pop1.tilted_parameters(multiplier)
    weights2, means2, variances2 = pop2.tilted_parameters(multiplier)
    objective = (
        -multiplier * resource_target
        - pop1.log_partition(multiplier)
        - pop2.log_partition(multiplier)
    )

    return AnalyticalSolution(
        multiplier=multiplier,
        mean1=mean1,
        mean2=mean2,
        variance1=variance1,
        variance2=variance2,
        objective=float(objective),
        weights1=tuple(weights1.tolist()),
        component_means1=tuple(means1.tolist()),
        component_variances1=tuple(variances1.tolist()),
        weights2=tuple(weights2.tolist()),
        component_means2=tuple(means2.tolist()),
        component_variances2=tuple(variances2.tolist()),
    )


@dataclass
class SixFlowSystem:
    """Finite-particle implementation of the six ODE blocks in Eq. (23)."""

    n_particles: int
    pop1: PopulationConfig
    pop2: PopulationConfig
    resource_target: float
    params: FlowParameters
    interaction_method: str = "diffusion_map"

    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError("Use at least two particles.")
        if self.interaction_method not in INTERACTION_METHODS:
            raise ValueError(f"interaction_method must be one of {INTERACTION_METHODS}")

    def unpack_state(
        self, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Split the flat adaptive-solver state into Eq. (23) blocks."""
        y = np.asarray(y, dtype=float)
        n = self.n_particles
        if y.size != 4 * n + 2:
            raise ValueError(f"Expected state size {4 * n + 2}, got {y.size}.")
        return (
            y[0:n],
            y[n : 2 * n],
            y[2 * n : 3 * n],
            y[3 * n : 4 * n],
            float(y[-2]),
            float(y[-1]),
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        """Approximate ``grad log rho_i`` in Eq. (23)."""
        if self.interaction_method == "gaussian":
            return gaussian_interaction_score(x, self.params.variance_floor)
        return diffusion_map_interaction_score(x, self.params.epsilon)

    def residuals(
        self,
        x1: np.ndarray,
        u1: np.ndarray,
        x2: np.ndarray,
        u2: np.ndarray,
    ) -> tuple[float, float]:
        """Return r(rho) and r(Z_rho) from Eq. (23)."""
        residual = float(
            np.mean(self.pop1.feature(x1))
            + np.mean(self.pop2.feature(x2))
            - self.resource_target
        )
        lookahead_residual = float(
            np.mean(self.pop1.feature(u1))
            + np.mean(self.pop2.feature(u2))
            - self.resource_target
        )
        return residual, lookahead_residual

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Evaluate the six finite-particle equations in draft Eq. (23)."""
        x1, u1, x2, u2, multiplier, multiplier_mirror = self.unpack_state(y)
        exp_a, exp_a_plus_b = time_coefficients(t, self.params)
        residual, lookahead_residual = self.residuals(x1, u1, x2, u2)

        dual_term = multiplier_mirror + self.params.penalty_mu * residual
        force1 = self.pop1.target_force(x1) + self.score(x1) + self.pop1.feature_C * dual_term
        force2 = self.pop2.target_force(x2) + self.score(x2) + self.pop2.feature_C * dual_term

        x1_dot = exp_a * (u1 - x1)
        x2_dot = exp_a * (u2 - x2)
        multiplier_dot = exp_a * (multiplier_mirror - multiplier)
        output = np.concatenate(
            (
                x1_dot,
                -exp_a_plus_b * force1,
                x2_dot,
                -exp_a_plus_b * force2,
                [
                    multiplier_dot,
                    exp_a_plus_b * lookahead_residual
                    - self.params.damping_zeta * multiplier_dot,
                ],
            )
        )

        return output


def time_coefficients(t: float, params: FlowParameters) -> tuple[float, float]:
    """Return exp(a) and exp(a+b) for the Sec. VI ideal scaling."""
    if t <= -1.0:
        raise ValueError("Require t>-1.")
    one_plus_t = 1.0 + t
    return (
        params.p / one_plus_t,
        params.p * one_plus_t ** (params.p - 1.0),
    )


def exp_b_coefficient(t: np.ndarray | float, params: FlowParameters) -> np.ndarray:
    """Return exp(b)=(1+t)^p used in draft Eq. (32)."""
    t = np.asarray(t, dtype=float)
    if np.any(t <= -1.0):
        raise ValueError("Require t>-1.")
    return (1.0 + t) ** params.p


def initial_state(
    system: SixFlowSystem,
    t0: float,
    seed: int = 4,
    initial_velocity: float = 0.0,
    multiplier0: float = 0.0,
    multiplier_velocity0: float = 0.0,
) -> np.ndarray:
    """Sample particles and enforce U=q+exp(-a)qdot from Eq. (20)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(
        system.pop1.initial_mean,
        np.sqrt(system.pop1.initial_variance),
        system.n_particles,
    )
    x2 = rng.normal(
        system.pop2.initial_mean,
        np.sqrt(system.pop2.initial_variance),
        system.n_particles,
    )
    exp_minus_a = (1.0 + t0) / system.params.p
    u1 = x1 + exp_minus_a * initial_velocity
    u2 = x2 + exp_minus_a * initial_velocity
    u_lambda = multiplier0 + exp_minus_a * multiplier_velocity0
    return np.concatenate((x1, u1, x2, u2, [multiplier0, u_lambda]))


def _smoothed_histogram_kl_against_density_series(
    particle_history: np.ndarray,
    reference_density: Callable[[np.ndarray], np.ndarray],
    support_points: np.ndarray,
    bins: int,
) -> np.ndarray:
    """Estimate the KL objective used in the convergence plot."""
    history = np.asarray(particle_history, dtype=float)
    support = np.asarray(support_points, dtype=float)
    lower = min(float(np.min(history)), float(np.min(support)))
    upper = max(float(np.max(history)), float(np.max(support)))
    margin = 0.03 * max(upper - lower, 1.0)
    edges = np.linspace(lower - margin, upper + margin, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    reference_mass = np.maximum(reference_density(centers) * widths, 1e-300)
    reference_mass /= np.sum(reference_mass)
    offsets = np.arange(-5, 6, dtype=float)
    kernel = np.exp(-0.5 * (offsets / 1.5) ** 2)
    kernel /= np.sum(kernel)

    def estimate(particles: np.ndarray) -> float:
        counts, _ = np.histogram(particles, bins=edges)
        empirical_mass = np.convolve(counts.astype(float), kernel, mode="same")
        empirical_mass = np.maximum(empirical_mass, 0.0)
        empirical_mass /= np.sum(empirical_mass)
        positive = empirical_mass > 0.0
        return float(
            np.sum(
                empirical_mass[positive]
                * np.log(empirical_mass[positive] / reference_mass[positive])
            )
        )

    return np.asarray([estimate(particles) for particles in history])


def _population_support_points(
    population: PopulationConfig,
    multiplier: float,
    standard_deviations: float = 8.0,
) -> np.ndarray:
    """Support for the histogram objective diagnostic."""
    target_means = np.asarray(population.target_means)
    target_std = np.sqrt(np.asarray(population.target_variances))
    _, optimal_means, optimal_variances = population.tilted_parameters(multiplier)
    optimal_std = np.sqrt(optimal_variances)
    return np.concatenate(
        (
            target_means - standard_deviations * target_std,
            target_means + standard_deviations * target_std,
            optimal_means - standard_deviations * optimal_std,
            optimal_means + standard_deviations * optimal_std,
        )
    )


def trajectory_metrics(
    trajectory: "Trajectory",
    system: SixFlowSystem,
    optimum: AnalyticalSolution,
) -> dict[str, np.ndarray]:
    """Compute the quantities plotted for draft Eqs. (21) and (32)."""
    n = system.n_particles
    y = trajectory.y
    x1, u1 = y[0:n].T, y[n : 2 * n].T
    x2, u2 = y[2 * n : 3 * n].T, y[3 * n : 4 * n].T
    multiplier, multiplier_mirror = y[-2], y[-1]

    mean1, mean2 = np.mean(x1, axis=1), np.mean(x2, axis=1)
    var1 = np.var(x1, axis=1, ddof=1)
    var2 = np.var(x2, axis=1, ddof=1)
    residual = (
        system.pop1.feature_C * mean1
        + system.pop2.feature_C * mean2
        - system.resource_target
    )
    lookahead_residual = (
        system.pop1.feature_C * np.mean(u1, axis=1)
        + system.pop2.feature_C * np.mean(u2, axis=1)
        - system.resource_target
    )

    support1 = _population_support_points(system.pop1, optimum.multiplier)
    support2 = _population_support_points(system.pop2, optimum.multiplier)
    kl1 = _smoothed_histogram_kl_against_density_series(
        x1, system.pop1.target_density, support1, system.params.metric_bins
    )
    kl2 = _smoothed_histogram_kl_against_density_series(
        x2, system.pop2.target_density, support2, system.params.metric_bins
    )
    objective = kl1 + kl2
    objective_gap_signed = objective - optimum.objective

    # Eq. (21): G_mu = F-F* + Lambda* r + mu |r|^2 / 2.
    augmented_saddle_gap = (
        objective_gap_signed
        + optimum.multiplier * residual
        + 0.5 * system.params.penalty_mu * residual**2
    )

    # Eq. (32): exp(b) r - U_Lambda - zeta Lambda is constant.
    exp_b = exp_b_coefficient(trajectory.t, system.params)
    dual_combination = multiplier_mirror + system.params.damping_zeta * multiplier
    dual_invariant = exp_b * residual - dual_combination
    dual_invariant_error = np.abs(dual_invariant - dual_invariant[0])

    mean_error = np.sqrt(
        (mean1 - optimum.mean1) ** 2 + (mean2 - optimum.mean2) ** 2
    )
    variance_error = np.sqrt(
        (var1 - optimum.variance1) ** 2 + (var2 - optimum.variance2) ** 2
    )

    return {
        "x1": x1,
        "u1": u1,
        "x2": x2,
        "u2": u2,
        "mean1": mean1,
        "mean2": mean2,
        "var1": var1,
        "var2": var2,
        "residual": residual,
        "extrapolated_residual": lookahead_residual,
        "constraint_violation": np.abs(residual),
        "lambda": multiplier,
        "u_lambda": multiplier_mirror,
        "kl1": kl1,
        "kl2": kl2,
        "objective": objective,
        "objective_gap": np.abs(objective_gap_signed),
        "objective_gap_signed": objective_gap_signed,
        "augmented_saddle_gap": augmented_saddle_gap,
        "indirect_augmented_gap": augmented_saddle_gap,
        "dual_combination": dual_combination,
        "dual_invariant": dual_invariant,
        "dual_invariant_error": dual_invariant_error,
        "mean_error": mean_error,
        "variance_error": variance_error,
        "multiplier_error": np.abs(multiplier - optimum.multiplier),
        "multiplier_mirror_error": np.abs(multiplier_mirror - optimum.multiplier),
    }


def __getattr__(name: str):
    # Keeps the untouched plotting modules import-compatible.
    if name == "Option" + "9SixFlowSystem":
        return SixFlowSystem
    raise AttributeError(name)
