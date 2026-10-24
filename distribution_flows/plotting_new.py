from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from functions import AnalyticalSolution, SixFlowSystem
from integrator import Trajectory

# Use one consistent 16 pt font everywhere, including tick labels and legends.
FONT_SIZE = 18
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": 16,
        "ytick.labelsize": 15,
        "legend.fontsize": 16,
        "figure.titlesize": FONT_SIZE,
    }
)


def _save_figure(fig: plt.Figure, path: Path, paths: list[Path]) -> None:
    """Tighten and save one figure as PDF."""
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    paths.append(path)


def _power_reference(
    t: np.ndarray,
    values: np.ndarray,
    exponent: float = 2.0,
    floor: float = 1e-12,
) -> np.ndarray:
    """Return a visual O(t^{-exponent}) reference scaled to the first sample."""
    t_array = np.asarray(t, dtype=float)
    value_array = np.asarray(values, dtype=float)
    amplitude = max(float(np.abs(value_array[0])), floor)
    return amplitude * ((1.0 + t_array[0]) / (1.0 + t_array)) ** exponent


def _particle_kde(
    grid: np.ndarray,
    particles: np.ndarray,
    minimum_bandwidth: float = 0.08,
) -> np.ndarray:
    """Gaussian KDE used only for the density-snapshot visualization."""
    particles = np.asarray(particles, dtype=float)
    if particles.size > 1:
        sample_std = float(np.std(particles, ddof=1))
    else:
        sample_std = 0.0
    bandwidth = max(
        1.06 * sample_std * max(particles.size, 1) ** (-0.2),
        minimum_bandwidth,
    )
    density = np.mean(
        np.exp(-0.5 * ((grid[:, None] - particles[None, :]) / bandwidth) ** 2),
        axis=1,
    ) / (np.sqrt(2.0 * np.pi) * bandwidth)
    return density


def save_plots(
    trajectory: Trajectory,
    metrics: dict[str, np.ndarray],
    system: SixFlowSystem,
    optimum: AnalyticalSolution,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.asarray(trajectory.t, dtype=float)
    paths: list[Path] = []

    # A practical log-plot floor avoids isolated zero crossings from expanding
    # the y-axis to numerical underflow and hiding the meaningful trajectory.
    floor = 1e-12

    # ------------------------------------------------------------------
    # Particle trajectories and empirical means.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for particle_index in range(metrics["x1"].shape[1]):
        ax.plot(t, metrics["x1"][:, particle_index], alpha=0.14, linewidth=0.65)
    for particle_index in range(metrics["x2"].shape[1]):
        ax.plot(t, metrics["x2"][:, particle_index], alpha=0.14, linewidth=0.65)
    ax.plot(t, metrics["mean1"], linewidth=2.3, label=r"$\rho_1$ mean")
    ax.plot(t, metrics["mean2"], linewidth=2.3, label=r"$\rho_2$ mean")
    ax.axhline(optimum.mean1, linestyle="--", label=r"optimal $\rho_1$ mean")
    ax.axhline(optimum.mean2, linestyle=":", label=r"optimal $\rho_2$ mean")
    ax.set_xlabel("time")
    ax.set_ylabel("particle position")
    ax.set_title("Six-flow particle trajectories and empirical means")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, output_dir / "particle_trajectories_and_means.pdf", paths)

    # ------------------------------------------------------------------
    # Data needed for the combined 2 x 3 figure.
    # Top row: same three convergence panels as before.
    # Bottom row:
    #   (2,1) W2^2 curves for rho_1 and rho_2
    #   (2,2) compressed rho_1 snapshots
    #   (2,3) compressed rho_2 snapshots
    # ------------------------------------------------------------------
    objective_gap = np.maximum(np.abs(metrics["objective_gap_signed"]), floor)
    gap_raw = metrics.get("augmented_saddle_gap", metrics["indirect_augmented_gap"])
    gap = np.maximum(np.abs(gap_raw), floor)
    residual_squared = np.maximum(np.abs(metrics["residual"]) ** 2, floor)

    objective_ref = np.maximum(
        _power_reference(
            t, objective_gap, exponent=system.params.p, floor=floor
        ),
        floor,
    )
    ref = np.maximum(
        _power_reference(t, gap, exponent=system.params.p, floor=floor), floor
    )
    constraint_ref_1 = np.maximum(
        _power_reference(
            t,
            residual_squared,
            exponent=2.0 * system.params.p,
            floor=floor,
        ),
        floor,
    )
    constraint_ref_2 = np.maximum(
        _power_reference(
            t,
            residual_squared,
            exponent=1.0 * system.params.p,
            floor=floor,
        ),
        floor,
    )

    # W2^2 metrics from functions.py
    w2_squared1 = np.maximum(metrics["w2_squared1"], floor)
    w2_squared2 = np.maximum(metrics["w2_squared2"], floor)

    # Snapshot support/grid
    indices = [0, len(t) // 2, len(t) - 1]
    target_support: list[float] = []
    for population in (system.pop1, system.pop2):
        target_means = np.asarray(population.target_means)
        target_std = np.sqrt(np.asarray(population.target_variances))
        _, optimal_means, optimal_variances = population.tilted_parameters(
            optimum.multiplier
        )
        optimal_std = np.sqrt(optimal_variances)
        target_support.extend((target_means - 6.0 * target_std).tolist())
        target_support.extend((target_means + 6.0 * target_std).tolist())
        target_support.extend((optimal_means - 6.0 * optimal_std).tolist())
        target_support.extend((optimal_means + 6.0 * optimal_std).tolist())

    grid = np.linspace(
        min(float(metrics["x1"].min()), float(metrics["x2"].min()), min(target_support))
        - 0.5,
        max(float(metrics["x1"].max()), float(metrics["x2"].max()), max(target_support))
        + 0.5,
        900,
    )

    histories = (metrics["x1"], metrics["x2"])
    populations = (system.pop1, system.pop2)
    rho_labels = (r"\rho_1", r"\rho_2")

    # ------------------------------------------------------------------
    # Combined figure: 2 rows x 3 columns.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5))

    # ---------------- Top row: same convergence three columns ----------------
    axes[0, 0].set_xlim([0.1, 20])
    axes[0, 0].loglog(
        t,
        gap,
        linewidth=2.1,
        label=r"$|G_\mu|$",
    )
    axes[0, 0].semilogy(
        t,
        ref * 5,
        linestyle="--",
        linewidth=2.0,
        label=rf"$O((1+t)^{{-{system.params.p:g}}})$",
    )
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("Dual / saddle gap")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(loc="best", frameon=True)
    axes[0, 1].set_xlim([0.1, 20])
    axes[0, 1].loglog(
        t,
        objective_gap,
        linewidth=2.1,
        label=r"$|F(\rho_1,\rho_2)-F^\star|$",
    )
    axes[0, 1].semilogy(
        t,
        objective_ref * 5,
        linestyle="--",
        linewidth=2.0,
        label=rf"$O((1+t)^{{-{system.params.p:g}}})$",
    )
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("Objective gap")
    axes[0, 1].grid(True, which="both", alpha=0.3)
    axes[0, 1].legend(loc="best", frameon=True)
    axes[0, 2].set_xlim([0.1, 20])
    axes[0, 2].loglog(
        t,
        residual_squared,
        linewidth=2.1,
        label=r"$|r|^2$",
    )
    axes[0, 2].semilogy(
        t,
        constraint_ref_1 * 5,
        linestyle="--",
        linewidth=2.0,
        label=rf"$O((1+t)^{{-{2.0 * system.params.p:g}}})$",
    )
    axes[0, 2].semilogy(
        t,
        constraint_ref_2 * 5,
        linestyle="-.",
        linewidth=2.0,
        label=rf"$O((1+t)^{{-{system.params.p:g}}})$",
    )
    axes[0, 2].set_xlabel("time")
    axes[0, 2].set_ylabel(r"Squared residue")
    axes[0, 2].grid(True, which="both", alpha=0.3)
    axes[0, 2].legend(loc="best", frameon=True)

    # ---------------- Bottom-left: W2^2 plot ----------------
    axes[1, 0].loglog(
        t,
        w2_squared1,
        linewidth=2.1,
        label=r"$W_2^2(\rho_1(t),\rho_1^\star)$",
    )
    axes[1, 0].semilogy(
        t,
        w2_squared2,
        linewidth=2.1,
        label=r"$W_2^2(\rho_2(t),\rho_2^\star)$",
    )
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(r"Squared $W_2$")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend(loc="best", frameon=True)

    # ---------------- Bottom-middle and bottom-right: compressed snapshots ----------------
    snapshot_time_labels = [
        rf"$t={t[indices[0]]:.2f}$",
        rf"$t={t[indices[1]]:.2f}$",
        rf"$t={t[indices[2]]:.2f}$",
    ]
    snapshot_styles = ["-", "--", "-."]
    snapshot_linewidths = [2.2, 2.2, 2.2]

    for col, (history, population, rho_label) in enumerate(
        zip(histories, populations, rho_labels), start=1
    ):
        ax = axes[1, col]
        optimal_density = population.tilted_density(grid, optimum.multiplier)

        for idx, style, lw, time_label in zip(
            indices, snapshot_styles, snapshot_linewidths, snapshot_time_labels
        ):
            density = _particle_kde(grid, history[idx])
            ax.plot(
                grid,
                density,
                linestyle=style,
                linewidth=lw,
                label=rf"${rho_label}$ at {time_label}",
            )

        ax.plot(
            grid,
            optimal_density,
            linewidth=2.5,
            label=rf"${rho_label}^\star$",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("Density snaps")
        ax.grid(alpha=0.25)
        ax.legend( loc="best", frameon=True)

    fig.subplots_adjust(wspace=0.32, hspace=0.28)
    _save_figure(fig, output_dir / "combined_convergence_w2_snapshots.pdf", paths)

    # ------------------------------------------------------------------
    # Multiplier flows.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.plot(t, metrics["lambda"], label=r"$\lambda$")
    ax.plot(t, metrics["u_lambda"], label=r"$U_\lambda$")
    ax.plot(
        t,
        metrics["dual_combination"],
        linewidth=1.8,
        label=rf"$U_\lambda+\zeta\lambda$ ($\zeta={system.params.damping_zeta:g}$)",
    )
    ax.axhline(optimum.multiplier, linestyle="--", label=r"$\lambda^\star$")
    ax.set_xlabel("time")
    ax.set_ylabel("multiplier")
    ax.set_title("Multiplier position-mirror pair")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, output_dir / "multiplier_flows.pdf", paths)

    # ------------------------------------------------------------------
    # Exact-ideal-scaling dual invariant diagnostic.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.semilogy(
        t,
        np.maximum(metrics["dual_invariant_error"], floor),
        linewidth=2.1,
        label=r"$|e^b r-U_\lambda-\zeta\lambda-K_0|$",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("invariant drift")
    ax.set_title("Dual invariant numerical check")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save_figure(fig, output_dir / "dual_invariant_check.pdf", paths)

    # ------------------------------------------------------------------
    # Direct errors against the analytical constrained solution.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.semilogy(
        t,
        np.maximum(metrics["mean_error"], floor),
        label="combined mean error",
    )
    ax.semilogy(
        t,
        np.maximum(metrics["variance_error"], floor),
        label="combined variance error",
    )
    ax.semilogy(
        t,
        np.maximum(metrics["multiplier_error"], floor),
        label=r"$|\lambda-\lambda^\star|$",
    )
    ax.semilogy(
        t,
        np.maximum(metrics["multiplier_mirror_error"], floor),
        label=r"$|U_\lambda-\lambda^\star|$",
        alpha=0.85,
    )
    ax.set_xlabel("time")
    ax.set_ylabel("error")
    ax.set_title("Direct errors against the analytical constrained solution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save_figure(fig, output_dir / "analytical_solution_errors.pdf", paths)

    # ------------------------------------------------------------------
    # Mean/variance tracking: separate 2-row figure.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 10.5), sharex=True)

    axes[0].plot(t, metrics["mean1"], linewidth=2.2, label=r"$\rho_1$ mean")
    axes[0].plot(t, metrics["mean2"], linewidth=2.2, label=r"$\rho_2$ mean")
    axes[0].axhline(
        optimum.mean1,
        linestyle="--",
        linewidth=2.0,
        label=r"$\rho_1^*$ mean",
    )
    axes[0].axhline(
        optimum.mean2,
        linestyle=":",
        linewidth=2.0,
        label=r"$\rho_2^*$ mean",
    )
    axes[0].set_ylabel("mean")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, metrics["var1"], linewidth=2.2, label=r"$\rho_1$ variance")
    axes[1].plot(t, metrics["var2"], linewidth=2.2, label=r"$\rho_2$ variance")
    axes[1].axhline(
        optimum.variance1,
        linestyle="--",
        linewidth=2.0,
        label=r"$\rho_1^*$ variance",
    )
    axes[1].axhline(
        optimum.variance2,
        linestyle=":",
        linewidth=2.0,
        label=r"$\rho_2^*$ variance",
    )
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("variance")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    _save_figure(fig, output_dir / "mean_variance_tracking.pdf", paths)

    # ------------------------------------------------------------------
    # Objective value (retained from the original diagnostics).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.semilogy(
        t,
        np.maximum(metrics["objective"], floor),
        label=r"estimated $F_1+F_2$",
    )
    ax.axhline(
        max(optimum.objective, floor),
        linestyle="--",
        label="exact constrained optimum",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("objective value")
    ax.set_title("KL objective and exact constrained benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save_figure(fig, output_dir / "objective_value.pdf", paths)

    return paths


def show_all_plots() -> None:
    plt.show()