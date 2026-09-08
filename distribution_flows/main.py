
import argparse
from pathlib import Path

import numpy as np

from functions import (
    FlowParameters,
    PopulationConfig,
    SixFlowSystem,
    analytical_constrained_solution,
    initial_state,
    trajectory_metrics,
)
from grad import INTERACTION_METHODS
from integrator import ALL_METHODS, integrate
from plotting_separated import save_plots, show_all_plots


def build_parser() -> argparse.ArgumentParser:
    """Command-line settings for the Sec. VI example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=200)
    parser.add_argument(
        "--interaction",
        choices=INTERACTION_METHODS,
        default="diffusion_map",
        help="Score approximation for grad log rho_i in Eq. (23), see ref[11].",
    )
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--method", choices=ALL_METHODS, default="RK45")
    parser.add_argument("--t0", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=20.0)
    parser.add_argument("--samples", type=int, default=801)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--penalty-mu", type=float, default=3.0)
    parser.add_argument("--damping-zeta", type=float, default=0.8)
    parser.add_argument("--resource-target", type=float, default=0.5)
    parser.add_argument("--metric-bins", type=int, default=80)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--max-step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output-dir", default="distribution_six_flow_output")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    return parser


def main() -> None:
    """Build Sec. VI, integrate Eq. (23), and save the figures."""
    args = build_parser().parse_args()

    pop1 = PopulationConfig.gaussian_mixture(
        name="fleet 1 (bimodal)",
        weights=(0.55, 0.45),
        means=(-2.0, 1.0),
        variances=(0.95, 0.95),
        feature_C=1.0,
        initial_mean=3.5,
        initial_variance=5.4,
    )

    pop2 = PopulationConfig.gaussian(
        name="fleet 2 (Gaussian)",
        mean=3.0,
        variance=1.1,
        feature_C=0.8,
        initial_mean=-1.5,
        initial_variance=1.4,
    )

    params = FlowParameters(
        p=args.p,
        penalty_mu=args.penalty_mu,
        damping_zeta=args.damping_zeta,
        epsilon=args.epsilon,
        metric_bins=args.metric_bins,
    )

    # Eq. (23) finite-particle dynamics.
    system = SixFlowSystem(
        n_particles=args.particles,
        pop1=pop1,
        pop2=pop2,
        resource_target=args.resource_target,
        params=params,
        interaction_method=args.interaction,
    )

    optimum = analytical_constrained_solution(pop1, pop2, args.resource_target)
    y0 = initial_state(system, args.t0, args.seed)

    trajectory = integrate(
        system=system,
        t_span=(args.t0, args.t0 + args.horizon),
        y0=y0,
        method=args.method,
        output_samples=args.samples,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
    )
    if not trajectory.success:
        raise RuntimeError(trajectory.message)

    # Eqs. (21) and (32) diagnostics used by the unchanged plots.
    metrics = trajectory_metrics(trajectory, system, optimum)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plots(trajectory, metrics, system, optimum, output_dir)

    np.savez_compressed(
        output_dir / "trajectory_distribution_six_flow.npz",
        t=trajectory.t,
        y=trajectory.y,
        **metrics,
    )

    summary = (
        "Constrained distributional six-flow example\n"
        f"interaction={args.interaction}, particles/population={args.particles}\n"
        f"integrator={args.method}, mu={args.penalty_mu:g}, "
        f"zeta={args.damping_zeta:g}, horizon={args.horizon:g}\n"
        f"analytical lambda*={optimum.multiplier:.8f}\n"
        f"optimal means=({optimum.mean1:.8f}, {optimum.mean2:.8f})\n"
        f"final means=({metrics['mean1'][-1]:.8f}, {metrics['mean2'][-1]:.8f})\n"
        f"final constraint violation={abs(metrics['residual'][-1]):.8e}\n"
        f"final lambda={metrics['lambda'][-1]:.8f}\n"
        f"final Eq.(21) saddle gap={metrics['augmented_saddle_gap'][-1]:.8e}\n"
        f"max dual-invariant drift={np.max(metrics['dual_invariant_error']):.8e}\n"
        f"runtime={trajectory.runtime_seconds:.4f} s\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Outputs: {output_dir.resolve()}")

    if args.show:
        show_all_plots()


if __name__ == "__main__":
    main()
