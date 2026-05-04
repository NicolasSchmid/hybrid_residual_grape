from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .grape import HybridGrapeConfig, pulse_regularization
from .physics import (
    FockPhysicsModel,
    PhysicsParams,
    bounded_controls_from_raw,
    raw_from_bounded_controls,
)
from .residual import (
    RBFResidualConfig,
    RBFResidualModel,
    empty_rbf_model,
    fit_rbf_residual,
    hybrid_probability_from_physics,
    residual_logit,
    rbf_support,
)
from toolbox.quantmech.operators import hconj
from toolbox.quantmech.unit_evol import evol_hdt_exp


FEATURE_NAMES = (
    "P_phys",
    "mean_qubit_excitation",
    "late_qubit_excitation",
    "final_qubit_excitation",
    "mean_photon_number",
    "final_photon_number",
    "mean_photon_variance",
    "late_photon_number",
    "target_photon_arrival_time",
    "qubit_drive_power",
    "cavity_drive_power",
    "pulse_smoothness",
    "max_coefficient",
)


@dataclass(frozen=True)
class FeatureRBFResidualModel:
    """RBF residual model acting on physics-informed trajectory features."""

    rbf_model: RBFResidualModel
    feature_mean: jax.Array
    feature_scale: jax.Array
    feature_names: tuple[str, ...] = FEATURE_NAMES


def _expectation(psi: jax.Array, operator: jax.Array) -> jax.Array:
    return jnp.squeeze((hconj(psi) @ (operator @ psi)).real)


def trajectory_features(
    physics_model: FockPhysicsModel,
    controls: jax.Array,
    physics_params: PhysicsParams | None = None,
) -> jax.Array:
    """Compute differentiable low-dimensional summaries of a simulated pulse.

    These features are intentionally not arbitrary pulse coefficients. They are
    summaries of what the nominal simulator thinks happened during the pulse.
    This gives the residual model a chance to learn statements such as
    "pulses that excite the qubit for a long time are over-optimistic" and
    transfer that correction between different-looking coefficient vectors.
    """
    p = physics_model.physics_params if physics_params is None else physics_params
    q = physics_model.sim_config
    controls = jnp.asarray(controls, dtype=jnp.float32).reshape((-1,))
    e_qub, e_cav = physics_model.control_fields(controls)
    qubit_phase = jnp.exp(1j * p.qubit_phase)
    e_qub = qubit_phase * e_qub
    cavity_phase = jnp.exp(1j * p.cavity_phase)
    if p.grape_cavity_iq:
        e_cav_for_hamiltonian = cavity_phase * 1j * jnp.conj(e_cav)
    else:
        e_cav_for_hamiltonian = cavity_phase * e_cav
    h_drift = physics_model.h_drift_with_params(p)
    n_phot_sq = physics_model.n_phot @ physics_model.n_phot

    def step(psi, x):
        eq, ec, dt = x
        hmat = (
            h_drift
            + p.mu_qub * (eq * physics_model.sigp + jnp.conj(eq) * physics_model.sigm)
            + p.mu_cav * (ec * physics_model.adag + jnp.conj(ec) * physics_model.a)
        )
        psi = evol_hdt_exp(hmat, dt) @ psi
        qubit_excited = _expectation(psi, physics_model.qubit_excited)
        photon_number = _expectation(psi, physics_model.n_phot)
        photon_number_sq = _expectation(psi, n_phot_sq)
        psi_by_qubit = psi.reshape(2, q.n_cav)
        target_prob = jnp.sum(jnp.abs(psi_by_qubit[:, q.target_n]) ** 2).real
        return psi, jnp.array(
            [qubit_excited, photon_number, photon_number_sq, target_prob],
            dtype=jnp.float32,
        )

    psi_final, trajectory = jax.lax.scan(
        step,
        physics_model.psi0,
        (e_qub, e_cav_for_hamiltonian, physics_model.dt),
    )
    qubit_excited = trajectory[:, 0]
    photon_number = trajectory[:, 1]
    photon_number_sq = trajectory[:, 2]
    target_prob_traj = trajectory[:, 3]

    p_phys = target_prob_traj[-1]
    final_qubit_excited = qubit_excited[-1]
    final_photon_number = photon_number[-1] / q.n_cav
    mean_qubit_excited = jnp.mean(qubit_excited)
    mean_photon_number = jnp.mean(photon_number) / q.n_cav
    photon_variance = jnp.maximum(photon_number_sq - photon_number**2, 0.0)
    mean_photon_variance = jnp.mean(photon_variance) / (q.n_cav**2)

    t = physics_model.t_mids / q.t_drive
    late_qubit_excited = jnp.sum(t * qubit_excited) / (jnp.sum(qubit_excited) + 1e-6)
    late_photon_number = jnp.sum(t * photon_number) / (jnp.sum(photon_number) + 1e-6)
    target_arrival_time = jnp.sum(t * target_prob_traj) / (jnp.sum(target_prob_traj) + 1e-6)

    coeffs = controls.reshape((4, q.num_coeffs))
    scale = q.param_clip**2
    qubit_drive_power = jnp.mean(jnp.abs(e_qub) ** 2) / scale
    cavity_drive_power = jnp.mean(jnp.abs(e_cav) ** 2) / scale
    pulse_smoothness = jnp.mean(jnp.diff(coeffs, axis=1) ** 2) / scale
    max_coefficient = jnp.max(jnp.abs(controls)) / q.param_clip

    return jnp.array(
        [
            p_phys,
            mean_qubit_excited,
            late_qubit_excited,
            final_qubit_excited,
            mean_photon_number,
            final_photon_number,
            mean_photon_variance,
            late_photon_number,
            target_arrival_time,
            qubit_drive_power,
            cavity_drive_power,
            pulse_smoothness,
            max_coefficient,
        ],
        dtype=jnp.float32,
    )


def trajectory_feature_matrix(
    physics_model: FockPhysicsModel,
    controls: jax.Array,
    physics_params: PhysicsParams | None = None,
) -> jax.Array:
    return jax.vmap(lambda one_controls: trajectory_features(physics_model, one_controls, physics_params))(
        controls
    )


def _standardize_features(
    features: jax.Array,
    mean: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    return (features - mean) / scale


def empty_feature_rbf_model(
    config: RBFResidualConfig = RBFResidualConfig(),
) -> FeatureRBFResidualModel:
    feature_size = len(FEATURE_NAMES)
    return FeatureRBFResidualModel(
        rbf_model=empty_rbf_model(feature_size, config),
        feature_mean=jnp.zeros((feature_size,), dtype=jnp.float32),
        feature_scale=jnp.ones((feature_size,), dtype=jnp.float32),
    )


def fit_feature_rbf_residual(
    physics_model: FockPhysicsModel,
    controls: jax.Array,
    successes: jax.Array,
    shots: jax.Array,
    config: RBFResidualConfig = RBFResidualConfig(),
) -> tuple[FeatureRBFResidualModel, jax.Array, jax.Array]:
    features = trajectory_feature_matrix(physics_model, controls)
    physics_probability = features[:, 0]
    feature_mean = jnp.mean(features, axis=0)
    feature_scale = jnp.maximum(jnp.std(features, axis=0), 0.05)
    standardized = _standardize_features(features, feature_mean, feature_scale)
    rbf_model = fit_rbf_residual(
        standardized,
        physics_probability,
        successes,
        shots,
        config,
    )
    return (
        FeatureRBFResidualModel(
            rbf_model=rbf_model,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        ),
        features,
        physics_probability,
    )


def feature_hybrid_prediction(
    physics_model: FockPhysicsModel,
    residual_model: FeatureRBFResidualModel,
    controls: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    features = trajectory_features(physics_model, controls)
    physics_probability = features[0]
    standardized = _standardize_features(
        features[None, :],
        residual_model.feature_mean,
        residual_model.feature_scale,
    )
    hybrid_probability = hybrid_probability_from_physics(
        physics_probability,
        residual_model.rbf_model,
        standardized,
    )[0]
    support = rbf_support(residual_model.rbf_model, standardized)[0]
    residual = residual_logit(residual_model.rbf_model, standardized)[0]
    return hybrid_probability, physics_probability, support, residual, features


def feature_hybrid_probability_batch(
    physics_model: FockPhysicsModel,
    residual_model: FeatureRBFResidualModel,
    controls: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    features = trajectory_feature_matrix(physics_model, controls)
    physics_probability = features[:, 0]
    standardized = _standardize_features(
        features,
        residual_model.feature_mean,
        residual_model.feature_scale,
    )
    hybrid_probability = hybrid_probability_from_physics(
        physics_probability,
        residual_model.rbf_model,
        standardized,
    )
    support = rbf_support(residual_model.rbf_model, standardized)
    residual = residual_logit(residual_model.rbf_model, standardized)
    return hybrid_probability, physics_probability, support, residual, features


def noisy_feature_hybrid_objective(
    raw_controls: jax.Array,
    physics_model: FockPhysicsModel,
    residual_model: FeatureRBFResidualModel,
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
        hybrid_p, physics_p, support, residual, _ = feature_hybrid_prediction(
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


def optimize_feature_hybrid_grape(
    physics_model: FockPhysicsModel,
    residual_model: FeatureRBFResidualModel,
    initial_controls: jax.Array,
    key: jax.Array,
    config: HybridGrapeConfig = HybridGrapeConfig(),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Warm-started AD-GRAPE using an RBF over trajectory features."""
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
        objective, aux = noisy_feature_hybrid_objective(
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
        _, summary = noisy_feature_hybrid_objective(
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
    _, summary = noisy_feature_hybrid_objective(
        raw_controls,
        physics_model,
        residual_model,
        noise,
        config,
    )
    return controls, history, summary, key
