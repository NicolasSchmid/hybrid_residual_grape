from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .experiment import MeasurementResponse, observed_probability_from_physical
from .physics import FockPhysicsModel, PhysicsParams


def khz_to_rad_per_us_jax(value_khz: jax.Array | float) -> jax.Array:
    return 2.0 * jnp.pi * jnp.asarray(value_khz) / 1000.0


@dataclass(frozen=True)
class PhysicalCalibrationConfig:
    """Bounded physical-parameter calibration from binary measurements."""

    max_chi_offset_khz: float = 8.0
    max_cavity_detuning_khz: float = 20.0
    max_qubit_detuning_khz: float = 20.0
    max_cavity_self_kerr_abs_khz: float = 2.0
    max_drive_scale_fraction: float = 0.06
    max_cavity_phase_rad: float = 0.10
    max_qubit_phase_rad: float = 0.10
    max_lifetime_log_scale: float = 0.70
    fit_decoherence: bool = True
    prior_strength: float = 2e-3
    learning_rate: float = 0.04
    fit_steps: int = 80
    probability_floor: float = 1e-5
    measurement_response: MeasurementResponse | None = None


def physical_parameter_names(
    config: PhysicalCalibrationConfig = PhysicalCalibrationConfig(),
) -> tuple[str, ...]:
    names = (
        "chi_offset",
        "cavity_detuning",
        "qubit_detuning",
        "cavity_self_kerr",
        "mu_qub_scale",
        "mu_cav_scale",
        "cavity_phase",
        "qubit_phase",
    )
    if config.fit_decoherence:
        names = names + (
            "qubit_T1_scale",
            "qubit_T2_scale",
            "cavity_T1_scale",
            "cavity_T2_scale",
        )
    return names


def physical_parameter_size(
    config: PhysicalCalibrationConfig = PhysicalCalibrationConfig(),
) -> int:
    return len(physical_parameter_names(config))


def params_from_calibration_raw(
    raw: jax.Array,
    nominal_params: PhysicsParams,
    reference_params: PhysicsParams,
    config: PhysicalCalibrationConfig = PhysicalCalibrationConfig(),
) -> PhysicsParams:
    """Map unconstrained calibration variables to realistic physics parameters."""
    bounded = jnp.tanh(jnp.asarray(raw))

    chi = nominal_params.chi + khz_to_rad_per_us_jax(
        config.max_chi_offset_khz * bounded[0]
    )
    cavity_detuning = khz_to_rad_per_us_jax(
        config.max_cavity_detuning_khz * bounded[1]
    )
    qubit_detuning = khz_to_rad_per_us_jax(
        config.max_qubit_detuning_khz * bounded[2]
    )
    cavity_self_kerr = khz_to_rad_per_us_jax(
        config.max_cavity_self_kerr_abs_khz * bounded[3]
    )
    mu_qub = nominal_params.mu_qub * (
        1.0 + config.max_drive_scale_fraction * bounded[4]
    )
    mu_cav = nominal_params.mu_cav * (
        1.0 + config.max_drive_scale_fraction * bounded[5]
    )
    cavity_phase = config.max_cavity_phase_rad * bounded[6]
    qubit_phase = config.max_qubit_phase_rad * bounded[7]

    qubit_t1_us = None
    qubit_t2_us = None
    cavity_t1_us = None
    cavity_t2_us = None
    if config.fit_decoherence:
        lifetime_scale = jnp.exp(config.max_lifetime_log_scale * bounded[8:12])
        qubit_t1_us = reference_params.qubit_t1_us * lifetime_scale[0]
        qubit_t2_us = reference_params.qubit_t2_us * lifetime_scale[1]
        cavity_t1_us = reference_params.cavity_t1_us * lifetime_scale[2]
        cavity_t2_us = reference_params.cavity_t2_us * lifetime_scale[3]

    return PhysicsParams(
        chi=chi,
        cavity_self_kerr=cavity_self_kerr,
        cavity_detuning=cavity_detuning,
        qubit_detuning=qubit_detuning,
        mu_qub=mu_qub,
        mu_cav=mu_cav,
        grape_dispersive_frame=nominal_params.grape_dispersive_frame,
        grape_cavity_iq=nominal_params.grape_cavity_iq,
        cavity_phase=cavity_phase,
        qubit_phase=qubit_phase,
        qubit_t1_us=qubit_t1_us,
        qubit_t2_us=qubit_t2_us,
        cavity_t1_us=cavity_t1_us,
        cavity_t2_us=cavity_t2_us,
    )


def binomial_negative_log_likelihood(
    probability: jax.Array,
    successes: jax.Array,
    shots: jax.Array,
    *,
    probability_floor: float = 1e-5,
) -> jax.Array:
    probability = jnp.clip(probability, probability_floor, 1.0 - probability_floor)
    successes = jnp.asarray(successes)
    shots = jnp.asarray(shots)
    failures = shots - successes
    nll = -(successes * jnp.log(probability) + failures * jnp.log1p(-probability))
    return jnp.sum(nll) / jnp.maximum(jnp.sum(shots), 1.0)


def calibration_loss(
    raw: jax.Array,
    model: FockPhysicsModel,
    nominal_params: PhysicsParams,
    reference_params: PhysicsParams,
    controls: jax.Array,
    successes: jax.Array,
    shots: jax.Array,
    config: PhysicalCalibrationConfig = PhysicalCalibrationConfig(),
) -> tuple[jax.Array, jax.Array]:
    params = params_from_calibration_raw(raw, nominal_params, reference_params, config)
    probability = model.population_probability_with_params(controls, params)
    observed_probability = observed_probability_from_physical(
        probability,
        config.measurement_response,
    )
    nll = binomial_negative_log_likelihood(
        observed_probability,
        successes,
        shots,
        probability_floor=config.probability_floor,
    )
    prior = config.prior_strength * jnp.mean(jnp.asarray(raw) ** 2)
    loss = nll + prior
    aux = jnp.array(
        [
            loss,
            nll,
            prior,
            jnp.mean(probability),
            jnp.max(probability),
        ]
    )
    return loss, aux


def fit_physical_parameters(
    model: FockPhysicsModel,
    nominal_params: PhysicsParams,
    reference_params: PhysicsParams,
    controls: jax.Array,
    successes: jax.Array,
    shots: jax.Array,
    initial_raw: jax.Array | None = None,
    config: PhysicalCalibrationConfig = PhysicalCalibrationConfig(),
) -> tuple[jax.Array, PhysicsParams, jax.Array]:
    """Fit nuisance parameters with Adam on binomial negative log-likelihood."""
    if initial_raw is None:
        initial_raw = jnp.zeros((physical_parameter_size(config),), dtype=jnp.float64)
    else:
        initial_raw = jnp.asarray(initial_raw, dtype=jnp.float64)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(initial_raw)

    def objective(raw):
        return calibration_loss(
            raw,
            model,
            nominal_params,
            reference_params,
            controls,
            successes,
            shots,
            config,
        )

    value_and_grad = jax.value_and_grad(objective, has_aux=True)

    def step(carry, _):
        raw, opt_state = carry
        (loss, aux), grad = value_and_grad(raw)
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        updates, opt_state = optimizer.update(grad, opt_state, raw)
        raw = optax.apply_updates(raw, updates)
        raw = jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        row = jnp.concatenate([aux, jnp.array([jnp.linalg.norm(grad)])])
        return (raw, opt_state), row

    (raw, _), history = jax.lax.scan(
        step,
        (initial_raw, opt_state),
        xs=None,
        length=config.fit_steps,
    )
    params = params_from_calibration_raw(raw, nominal_params, reference_params, config)
    return raw, params, history


def adaptive_shot_counts(
    predicted_probability: jax.Array,
    *,
    low_shots: int = 1000,
    medium_shots: int = 4000,
    high_shots: int = 12000,
    medium_threshold: float = 0.85,
    high_threshold: float = 0.94,
) -> jax.Array:
    predicted_probability = jnp.asarray(predicted_probability)
    shots = jnp.full(predicted_probability.shape, low_shots, dtype=jnp.float32)
    shots = jnp.where(predicted_probability >= medium_threshold, medium_shots, shots)
    shots = jnp.where(predicted_probability >= high_threshold, high_shots, shots)
    return shots


def beta_normal_lower_bound(
    successes: jax.Array,
    shots: jax.Array,
    *,
    z: float = 2.0,
) -> jax.Array:
    """Cheap Jeffreys-posterior lower confidence score for pulse selection."""
    successes = jnp.asarray(successes)
    shots = jnp.asarray(shots)
    mean = (successes + 0.5) / (shots + 1.0)
    std = jnp.sqrt(jnp.maximum(mean * (1.0 - mean) / (shots + 2.0), 0.0))
    return jnp.clip(mean - z * std, 0.0, 1.0)


def calibration_parameter_summary(
    raw: jax.Array,
    params: PhysicsParams,
    nominal_params: PhysicsParams,
    reference_params: PhysicsParams,
    config: PhysicalCalibrationConfig = PhysicalCalibrationConfig(),
) -> dict[str, float]:
    """Convert calibrated parameters into readable engineering units."""
    bounded = jnp.tanh(jnp.asarray(raw))
    out = {
        "chi_offset_khz": float(config.max_chi_offset_khz * bounded[0]),
        "cavity_detuning_khz": float(config.max_cavity_detuning_khz * bounded[1]),
        "qubit_detuning_khz": float(config.max_qubit_detuning_khz * bounded[2]),
        "cavity_self_kerr_khz": float(
            config.max_cavity_self_kerr_abs_khz * bounded[3]
        ),
        "mu_qub": float(params.mu_qub),
        "mu_cav": float(params.mu_cav),
        "cavity_phase_rad": float(params.cavity_phase),
        "qubit_phase_rad": float(params.qubit_phase),
    }
    if config.fit_decoherence:
        out.update(
            {
                "qubit_T1_us": float(params.qubit_t1_us),
                "qubit_T2_us": float(params.qubit_t2_us),
                "cavity_T1_us": float(params.cavity_t1_us),
                "cavity_T2_us": float(params.cavity_t2_us),
                "qubit_T1_scale": float(params.qubit_t1_us / reference_params.qubit_t1_us),
                "qubit_T2_scale": float(params.qubit_t2_us / reference_params.qubit_t2_us),
                "cavity_T1_scale": float(params.cavity_t1_us / reference_params.cavity_t1_us),
                "cavity_T2_scale": float(params.cavity_t2_us / reference_params.cavity_t2_us),
            }
        )
    return out
