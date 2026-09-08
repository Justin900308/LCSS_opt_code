"""Mini-batch score approximation for the distributional six-flow example.

The draft requires an approximation of ``grad log rho_i`` when the population
law is represented by finitely many particles.  :mod:`grad` provides
all-particle deterministic approximations.  This module contains the optional
Monte-Carlo version of the ordinary Gaussian-KDE score

    grad log rho_epsilon(x_i)
      = (1/(2 epsilon))
        [sum_j g_epsilon(x_i,x_j) (x_j-x_i)]
        / [sum_j g_epsilon(x_i,x_j)],

with

    g_epsilon(x,y) = exp(-(x-y)^2/(4 epsilon)).

Only a random subset of interaction partners is used for each query particle.
The estimator is therefore a computational approximation to the smoothed KDE
score; it is not an exact formula for the unsmoothed ``grad log rho``.

For fixed-step Runge--Kutta integration, the sampled partner indices are frozen
through all stages of one time step and refreshed before the next time step.
This common-random-number convention makes all RK stages evaluate the same
random vector field instead of injecting an unrelated random draw at each
stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SampledKDEInteraction:
    """Mini-batch Monte-Carlo approximation of a one-dimensional KDE score.

    Parameters
    ----------
    epsilon:
        Positive kernel parameter in
        ``exp(-(x-y)^2/(4 epsilon))``.
    batch_size:
        Number of sampled interaction partners per query particle.  The query
        particle itself is always included, so the denominator remains
        strictly positive before floating-point underflow.
    seed:
        Random seed used for reproducible batches.
    denominator_floor:
        Small positive floor applied to the KDE denominator as a final
        numerical safeguard.

    Notes
    -----
    Sampling with replacement makes the vectorized implementation inexpensive.
    The numerator and denominator sample averages are individually unbiased,
    but their ratio is generally a finite-batch biased estimator.  This module
    is intended as a numerical approximation, not as a change to Eq. (23) of
    the draft.
    """

    epsilon: float = 0.01
    batch_size: int = 32
    seed: int = 17
    denominator_floor: float = 1e-14

    # Internal random-state fields are deliberately excluded from repr so that
    # printing a system configuration shows only mathematical/user parameters.
    _rng: np.random.Generator = field(init=False, repr=False)
    _indices: np.ndarray | None = field(default=None, init=False, repr=False)

    stochastic: bool = field(default=True, init=False)
    name: str = field(default="sampled_kde", init=False)

    def __post_init__(self) -> None:
        """Validate numerical parameters and initialize the random generator."""
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        if self.denominator_floor <= 0.0:
            raise ValueError("denominator_floor must be positive.")

        self._rng = np.random.default_rng(self.seed)

    def prepare_step(self, n_particles: int) -> None:
        """Draw and freeze one partner-index batch for the next time step.

        The fixed-step integrator calls this exactly once before evaluating all
        Euler/Heun/RK4 stages.  Thus, although the particle positions change
        between stages, the *labels* of the sampled partners remain fixed.
        """
        if n_particles < 1:
            raise ValueError("n_particles must be positive.")

        batch = min(self.batch_size, n_particles)

        # The degenerate one-partner case uses only self-interaction.  The KDE
        # score then evaluates to zero, which is the expected result because
        # x_j-x_i = 0 for the self partner.
        if batch == 1:
            self._indices = np.arange(n_particles, dtype=int)[:, None]
            return

        # Always include i itself as the first partner for query particle i.
        self_indices = np.arange(n_particles, dtype=int)[:, None]

        # Draw the remaining partners with replacement.  The shape is
        # (n_particles, batch-1), which allows the score computation to stay
        # fully vectorized.
        random_partners = self._rng.integers(
            low=0,
            high=n_particles,
            size=(n_particles, batch - 1),
            endpoint=False,
        )
        self._indices = np.concatenate((self_indices, random_partners), axis=1)

    def score(self, particles: np.ndarray) -> np.ndarray:
        """Evaluate the sampled approximation of ``grad log rho_epsilon``.

        Parameters
        ----------
        particles:
            One-dimensional current particle positions.

        Returns
        -------
        numpy.ndarray
            One score value per particle, in the same order as ``particles``.
        """
        x = np.asarray(particles, dtype=float)
        if x.ndim != 1:
            raise ValueError(
                "The sampling estimator currently supports 1-D particles only."
            )

        # When the estimator is called outside the fixed-step integrator, make
        # it self-contained by generating a batch on demand.
        if self._indices is None or self._indices.shape[0] != x.size:
            self.prepare_step(x.size)

        assert self._indices is not None

        # references[i,k] is the current position of the k-th sampled partner
        # for query particle i.  The difference sign is x_j-x_i, matching the
        # analytical derivative of the Gaussian KDE.
        references = x[self._indices]
        differences = references - x[:, None]

        weights = np.exp(-(differences * differences) / (4.0 * self.epsilon))
        denominator = np.maximum(
            np.sum(weights, axis=1), self.denominator_floor
        )
        numerator = np.sum(weights * differences, axis=1)

        return numerator / (2.0 * self.epsilon * denominator)
