#!/usr/bin/env python3
"""Small consistency checks for draft Eqs. (20), (23), and (32)."""
from __future__ import annotations

import numpy as np

from functions import (
    FlowParameters,
    PopulationConfig,
    SixFlowSystem,
    analytical_constrained_solution,
    exp_b_coefficient,
    initial_state,
    time_coefficients,
)
from grad import gaussian_mixture_density, gaussian_mixture_target_gradient


def build_system(n_particles: int = 20) -> SixFlowSystem:
    """Construct the Sec. VI test system."""
    pop1 = PopulationConfig.gaussian_mixture(
        name="rho1",
        weights=(0.55, 0.45),
        means=(-2.0, 1.0),
        variances=(0.95, 0.95),
        feature_C=1.0,
    )
    pop2 = PopulationConfig.gaussian(
        name="rho2", mean=3.0, variance=1.1, feature_C=0.8
    )
    return SixFlowSystem(
        n_particles=n_particles,
        pop1=pop1,
        pop2=pop2,
        resource_target=0.5,
        params=FlowParameters(p=2.0, penalty_mu=3.0, damping_zeta=0.8),
        interaction_method="diffusion_map",
    )


def check_target_gradient() -> float:
    """Check the target force used in Eq. (23)."""
    x = np.linspace(-4.0, 4.0, 17)
    weights, means, variances = (0.55, 0.45), (-2.0, 1.0), (0.95, 0.95)
    h = 1e-6
    plus = gaussian_mixture_density(x + h, weights, means, variances)
    minus = gaussian_mixture_density(x - h, weights, means, variances)
    finite_difference = (-np.log(plus) + np.log(minus)) / (2.0 * h)
    analytic = gaussian_mixture_target_gradient(x, weights, means, variances)
    return float(np.max(np.abs(finite_difference - analytic)))


def check_feasibility(system: SixFlowSystem) -> float:
    """Check feasibility of the exact tilted optimum."""
    optimum = analytical_constrained_solution(
        system.pop1, system.pop2, system.resource_target
    )
    return float(
        system.pop1.feature_C * optimum.mean1
        + system.pop2.feature_C * optimum.mean2
        - system.resource_target
    )


def check_lookahead(system: SixFlowSystem) -> float:
    """Check U=q+exp(-a)qdot from Eq. (20)."""
    t0 = 0.1
    velocity = 0.7
    multiplier_velocity = -0.3
    y = initial_state(
        system,
        t0,
        seed=1,
        initial_velocity=velocity,
        multiplier0=0.2,
        multiplier_velocity0=multiplier_velocity,
    )
    x1, u1, _, _, multiplier, u_lambda = system.unpack_state(y)
    exp_minus_a = (1.0 + t0) / system.params.p
    return float(
        max(
            np.max(np.abs((u1 - x1) - exp_minus_a * velocity)),
            abs((u_lambda - multiplier) - exp_minus_a * multiplier_velocity),
        )
    )


def check_dual_invariant(system: SixFlowSystem) -> float:
    """Check the derivative of the invariant in Eq. (32)."""
    y = initial_state(system, 0.1, seed=4)
    n = system.n_particles
    rng = np.random.default_rng(5)
    y[n : 2 * n] += rng.normal(0.0, 0.2, n)
    y[3 * n : 4 * n] += rng.normal(0.0, 0.2, n)
    y[-2], y[-1] = 0.4, -0.2

    t = 0.73
    dy = system.rhs(t, y)
    x1, u1, x2, u2, _, _ = system.unpack_state(y)
    residual, _ = system.residuals(x1, u1, x2, u2)
    residual_dot = (
        system.pop1.feature_C * np.mean(dy[0:n])
        + system.pop2.feature_C * np.mean(dy[2 * n : 3 * n])
    )
    exp_a, _ = time_coefficients(t, system.params)
    exp_b = float(exp_b_coefficient(t, system.params))
    invariant_dot = (
        exp_b * exp_a * residual
        + exp_b * residual_dot
        - dy[-1]
        - system.params.damping_zeta * dy[-2]
    )
    return float(abs(invariant_dot))


def main() -> None:
    """Run the implementation checks."""
    system = build_system()
    errors = {
        "target-gradient": check_target_gradient(),
        "analytical feasibility": abs(check_feasibility(system)),
        "lookahead initialization": check_lookahead(system),
        "dual invariant derivative": check_dual_invariant(system),
    }
    for name, value in errors.items():
        print(f"{name:28s}: {value:.3e}")

    assert errors["target-gradient"] < 1e-7
    assert errors["analytical feasibility"] < 1e-11
    assert errors["lookahead initialization"] < 1e-12
    assert errors["dual invariant derivative"] < 1e-12
    print("All draft-alignment checks passed.")


if __name__ == "__main__":
    main()
