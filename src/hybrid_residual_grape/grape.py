from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .physics import (
    FockPhysicsModel,
    bounded_controls_from_raw,
    raw_from_bounded_controls,
)
from .residual import (
    RBFResidualModel,
    hybrid_probability_from_physics,
    residual_logit,
    rbf_support,
)


@dataclass(frozen=True)
class HybridGrapeConfig:
    maxiter: int = 120
    memory_size: int = 10
    noise_samples: int = 8
    control_noise_std: float = 0.02
    residual_support_penalty: float = 0.10
    residual_size_penalty: float = 0.01
    amplitude_l2: float = 1e-4
    smoothness_l2: float = 1e-4
    param_clip: float = 2.0
    grad_clip_norm: float = 20.0


def pulse_regularization(
    controls: jax.Array,
    *,
    amplitude_l2: float,
    smoothness_l2: float,
) -> jax.Array:
    coeffs = controls.reshape((4, -1))
    return amplitude_l2 * jnp.mean(coeffs**2) + smoothness_l2 * jnp.mean(
        jnp.diff(coeffs, axis=1) ** 2
    )


def hybrid_prediction(
    physics_model: FockPhysicsModel,
    residual_model: RBFResidualModel,
    controls: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    physics_probability = physics_model.photon_probability(controls)
    hybrid_probability = hybrid_probability_from_physics(
        physics_probability,
        residual_model,
        controls,
    )[0]
    support = rbf_support(residual_model, controls)[0]
    residual = jnp.abs(residual_logit(residual_model, controls)[0])
    return hybrid_probability, physics_probability, support, residual


def noisy_hybrid_objective(
    raw_controls: jax.Array,
    physics_model: FockPhysicsModel,
    residual_model: RBFResidualModel,
    fixed_noise: jax.Array,
    config: HybridGrapeConfig,
) -> tuple[jax.Array, jax.Array]:
    center_controls = bounded_controls_from_raw(
        raw_controls,
        param_clip=config.param_clip,
    )
    noisy_controls = jnp.clip(
        center_controls[None, :] + fixed_noise,
        -config.param_clip,
        config.param_clip,
    )

    def one(controls):
        hybrid_p, physics_p, support, residual = hybrid_prediction(
            physics_model,
            residual_model,
            controls,
        )
        penalty = (
            config.residual_support_penalty * (1.0 - support)
            + config.residual_size_penalty * residual**2
            + pulse_regularization(
                controls,
                amplitude_l2=config.amplitude_l2,
                smoothness_l2=config.smoothness_l2,
            )
        )
        return hybrid_p - penalty, jnp.array([hybrid_p, physics_p, support, residual, penalty])

    utilities, stats = jax.vmap(one)(noisy_controls)
    objective = jnp.mean(utilities)
    center_stats = stats[0]
    summary = jnp.concatenate(
        [
            jnp.array([objective]),
            center_stats,
            jnp.array([jnp.mean(stats[:, 0]), jnp.std(stats[:, 0])]),
        ]
    )
    return objective, summary


def optimize_hybrid_grape(
    physics_model: FockPhysicsModel,
    residual_model: RBFResidualModel,
    initial_controls: jax.Array,
    key: jax.Array,
    config: HybridGrapeConfig = HybridGrapeConfig(),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Warm-started AD-GRAPE with an RBF residual correction and Optax L-BFGS."""
    key, noise_key = jax.random.split(key)
    noise = config.control_noise_std * jax.random.normal(
        noise_key,
        (config.noise_samples, initial_controls.shape[0]),
    )
    noise = noise.at[0].set(jnp.zeros_like(initial_controls))
    raw_controls = raw_from_bounded_controls(
        initial_controls,
        param_clip=config.param_clip,
    )

    optimizer = optax.lbfgs(memory_size=config.memory_size)
    opt_state = optimizer.init(raw_controls)

    def loss_fn(raw):
        objective, aux = noisy_hybrid_objective(
            raw,
            physics_model,
            residual_model,
            noise,
            config,
        )
        return -objective, aux

    value_and_grad = optax.value_and_grad_from_state(lambda raw: loss_fn(raw)[0])

    def step(carry, _):
        raw, opt_state = carry
        value, grad = value_and_grad(raw, state=opt_state)
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = jnp.linalg.norm(grad)
        grad = grad * jnp.minimum(1.0, config.grad_clip_norm / (grad_norm + 1e-12))
        updates, opt_state = optimizer.update(
            grad,
            opt_state,
            raw,
            value=value,
            grad=grad,
            value_fn=lambda z: loss_fn(z)[0],
        )
        raw = optax.apply_updates(raw, updates)
        raw = jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        objective, summary = noisy_hybrid_objective(
            raw,
            physics_model,
            residual_model,
            noise,
            config,
        )
        return (raw, opt_state), summary

    (raw_controls, opt_state), history = jax.lax.scan(
        step,
        (raw_controls, opt_state),
        xs=None,
        length=config.maxiter,
    )
    controls = bounded_controls_from_raw(raw_controls, param_clip=config.param_clip)
    objective, summary = noisy_hybrid_objective(
        raw_controls,
        physics_model,
        residual_model,
        noise,
        config,
    )
    return controls, history, summary, key
