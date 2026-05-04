from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .physics import FockPhysicsModel


@dataclass(frozen=True)
class MeasurementResponse:
    """Monotone map from physical Fock probability to observed yes probability.

    The real selective-pi-pulse measurement is not exactly ``Pr(yes)=P_n``:
    state decay during the pulse, readout assignment errors, and small coherent
    imperfections create a response curve with a nonzero floor, contrast below
    one, and mild nonlinearity.
    """

    false_positive: float = 0.0
    true_positive: float = 1.0
    curvature: float = 0.0


def observed_probability_from_physical(
    physical_probability: jax.Array,
    response: MeasurementResponse | None = None,
) -> jax.Array:
    """Apply a smooth monotone measurement-response curve."""
    physical_probability = jnp.clip(jnp.asarray(physical_probability), 0.0, 1.0)
    if response is None:
        return physical_probability

    # Endpoints remain fixed while the middle bends slightly. For
    # |curvature| <~ 0.5 this remains monotone on [0, 1].
    curved = physical_probability + response.curvature * (
        4.0
        * physical_probability
        * (1.0 - physical_probability)
        * (physical_probability - 0.5)
    )
    curved = jnp.clip(curved, 0.0, 1.0)
    false_positive = jnp.asarray(response.false_positive)
    true_positive = jnp.asarray(response.true_positive)
    return false_positive + (true_positive - false_positive) * curved


def sample_binomial_measurements(
    true_model: FockPhysicsModel,
    controls: jax.Array,
    key: jax.Array,
    *,
    shots: int | jax.Array,
    response: MeasurementResponse | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Simulate an experiment: one evolution plus response and binomial shots.

    The third return value is the hidden physical Fock probability, not the
    observed yes probability used for sampling. This keeps diagnostics readable
    while the optimizer only sees successes and shot counts.
    """
    key, shot_key = jax.random.split(key)
    true_probability = true_model.population_probability(controls)
    observed_probability = observed_probability_from_physical(true_probability, response)
    if jnp.ndim(jnp.asarray(shots)) == 0:
        shot_counts = jnp.full((controls.shape[0],), shots, dtype=jnp.float32)
    else:
        shot_counts = jnp.asarray(shots, dtype=jnp.float32)
    successes = jax.random.binomial(
        shot_key,
        n=shot_counts,
        p=observed_probability,
        shape=true_probability.shape,
    )
    return successes.astype(jnp.float32), shot_counts, true_probability, key


def append_dataset(
    controls: jax.Array | None,
    successes: jax.Array | None,
    shots: jax.Array | None,
    physics_probability: jax.Array | None,
    new_controls: jax.Array,
    new_successes: jax.Array,
    new_shots: jax.Array,
    new_physics_probability: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if controls is None:
        return (
            jnp.asarray(new_controls, dtype=jnp.float32),
            jnp.asarray(new_successes, dtype=jnp.float32),
            jnp.asarray(new_shots, dtype=jnp.float32),
            jnp.asarray(new_physics_probability, dtype=jnp.float32),
        )
    return (
        jnp.concatenate([controls, jnp.asarray(new_controls, dtype=jnp.float32)], axis=0),
        jnp.concatenate([successes, jnp.asarray(new_successes, dtype=jnp.float32)], axis=0),
        jnp.concatenate([shots, jnp.asarray(new_shots, dtype=jnp.float32)], axis=0),
        jnp.concatenate(
            [physics_probability, jnp.asarray(new_physics_probability, dtype=jnp.float32)],
            axis=0,
        ),
    )


def make_local_experiment_batch(
    center_controls: jax.Array,
    key: jax.Array,
    *,
    batch_size: int,
    noise_std: float = 0.025,
    param_clip: float = 2.0,
    include_center: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Generate pulses to evaluate around the GRAPE optimum."""
    center_controls = jnp.asarray(center_controls).reshape((-1,))
    key, noise_key = jax.random.split(key)
    count = batch_size - 1 if include_center else batch_size
    noise = noise_std * jax.random.normal(noise_key, (count, center_controls.shape[0]))
    local = jnp.clip(center_controls[None, :] + noise, -param_clip, param_clip)
    if include_center:
        local = jnp.concatenate([center_controls[None, :], local], axis=0)
    return local, key
