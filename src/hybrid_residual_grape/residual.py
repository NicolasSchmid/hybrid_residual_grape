from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def safe_logit(probability: jax.Array, eps: float = 1e-5) -> jax.Array:
    probability = jnp.clip(probability, eps, 1.0 - eps)
    return jnp.log(probability) - jnp.log1p(-probability)


@dataclass(frozen=True)
class RBFResidualConfig:
    max_centers: int = 128
    length_scale: float = 0.18
    ridge: float = 5e-3
    residual_clip: float = 1.2
    measurement_floor: float = 1e-3
    center_jitter: float = 1e-6


@dataclass(frozen=True)
class RBFResidualModel:
    centers: jax.Array
    weights: jax.Array
    bias: jax.Array
    active: jax.Array
    length_scale: float
    residual_clip: float


def empty_rbf_model(
    parameter_size: int,
    config: RBFResidualConfig = RBFResidualConfig(),
) -> RBFResidualModel:
    return RBFResidualModel(
        centers=jnp.zeros((config.max_centers, parameter_size), dtype=jnp.float32),
        weights=jnp.zeros((config.max_centers,), dtype=jnp.float32),
        bias=jnp.array(0.0, dtype=jnp.float32),
        active=jnp.zeros((config.max_centers,), dtype=jnp.float32),
        length_scale=config.length_scale,
        residual_clip=config.residual_clip,
    )


def measured_probability(successes: jax.Array, shots: jax.Array, floor: float = 1e-3) -> jax.Array:
    successes = jnp.asarray(successes, dtype=jnp.float32)
    shots = jnp.asarray(shots, dtype=jnp.float32)
    return (successes + floor * shots) / (shots * (1.0 + 2.0 * floor))


def select_rbf_centers(
    controls: jax.Array,
    successes: jax.Array,
    shots: jax.Array,
    *,
    max_centers: int,
    measurement_floor: float = 1e-3,
) -> tuple[jax.Array, jax.Array]:
    controls = jnp.asarray(controls)
    p_hat = measured_probability(successes, shots, measurement_floor)
    confidence_score = p_hat + 0.02 * jnp.log1p(shots)
    order = jnp.argsort(confidence_score)[::-1]
    ordered_controls = controls[order]
    num_data = controls.shape[0]
    num_active = jnp.minimum(num_data, max_centers)
    padded = jnp.zeros((max_centers, controls.shape[1]), dtype=controls.dtype)
    take = jnp.minimum(max_centers, ordered_controls.shape[0])
    padded = padded.at[:take].set(ordered_controls[:take])
    active = (jnp.arange(max_centers) < num_active).astype(jnp.float32)
    return padded, active


def rbf_features(
    controls: jax.Array,
    centers: jax.Array,
    active: jax.Array,
    *,
    length_scale: float,
) -> jax.Array:
    controls = jnp.asarray(controls)
    if controls.ndim == 1:
        controls = controls[None, :]
    normalized_diff = (controls[:, None, :] - centers[None, :, :]) / length_scale
    sqdist = jnp.mean(normalized_diff**2, axis=-1)
    return jnp.exp(-0.5 * sqdist) * active[None, :]


def residual_raw(model: RBFResidualModel, controls: jax.Array) -> jax.Array:
    features = rbf_features(
        controls,
        model.centers,
        model.active,
        length_scale=model.length_scale,
    )
    return model.bias + features @ model.weights


def residual_logit(model: RBFResidualModel, controls: jax.Array) -> jax.Array:
    return jnp.clip(
        residual_raw(model, controls),
        -model.residual_clip,
        model.residual_clip,
    )


def rbf_support(model: RBFResidualModel, controls: jax.Array) -> jax.Array:
    features = rbf_features(
        controls,
        model.centers,
        model.active,
        length_scale=model.length_scale,
    )
    return jnp.max(features, axis=-1)


def hybrid_probability_from_physics(
    physics_probability: jax.Array,
    residual_model: RBFResidualModel,
    controls: jax.Array,
) -> jax.Array:
    logits = safe_logit(physics_probability) + residual_logit(residual_model, controls)
    return jax.nn.sigmoid(logits)


def fit_rbf_residual(
    controls: jax.Array,
    physics_probability: jax.Array,
    successes: jax.Array,
    shots: jax.Array,
    config: RBFResidualConfig = RBFResidualConfig(),
) -> RBFResidualModel:
    controls = jnp.asarray(controls, dtype=jnp.float32)
    physics_probability = jnp.asarray(physics_probability, dtype=jnp.float32)
    successes = jnp.asarray(successes, dtype=jnp.float32)
    shots = jnp.asarray(shots, dtype=jnp.float32)

    centers, active = select_rbf_centers(
        controls,
        successes,
        shots,
        max_centers=config.max_centers,
        measurement_floor=config.measurement_floor,
    )
    features = rbf_features(
        controls,
        centers,
        active,
        length_scale=config.length_scale,
    )
    design = jnp.concatenate([jnp.ones((controls.shape[0], 1)), features], axis=1)
    p_hat = measured_probability(successes, shots, config.measurement_floor)
    target_residual = safe_logit(p_hat) - safe_logit(physics_probability)

    # Binomial variance weighting, clipped to avoid one lucky outcome dominating
    # when p_hat is nearly 0 or 1.
    variance = jnp.maximum(p_hat * (1.0 - p_hat), config.measurement_floor)
    weights = shots / variance
    sqrt_w = jnp.sqrt(weights)
    weighted_design = design * sqrt_w[:, None]
    weighted_target = target_residual * sqrt_w

    reg = config.ridge * jnp.eye(config.max_centers + 1, dtype=controls.dtype)
    reg = reg.at[0, 0].set(config.ridge * 0.05)
    lhs = weighted_design.T @ weighted_design + reg
    rhs = weighted_design.T @ weighted_target
    coeffs = jnp.linalg.solve(lhs, rhs)

    return RBFResidualModel(
        centers=centers,
        weights=coeffs[1:] * active,
        bias=coeffs[0],
        active=active,
        length_scale=config.length_scale,
        residual_clip=config.residual_clip,
    )


def rbf_diagnostics(
    residual_model: RBFResidualModel,
    controls: jax.Array,
    physics_probability: jax.Array,
    successes: jax.Array,
    shots: jax.Array,
) -> dict[str, jax.Array]:
    measured = successes / shots
    predicted = hybrid_probability_from_physics(
        physics_probability,
        residual_model,
        controls,
    )
    residual = residual_logit(residual_model, controls)
    support = rbf_support(residual_model, controls)
    mse = jnp.mean((predicted - measured) ** 2)
    weighted_mse = jnp.sum(shots * (predicted - measured) ** 2) / jnp.sum(shots)
    return {
        "predicted": predicted,
        "measured": measured,
        "residual": residual,
        "support": support,
        "mse": mse,
        "weighted_mse": weighted_mse,
        "active_centers": jnp.sum(residual_model.active),
    }
