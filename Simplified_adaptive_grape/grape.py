"""Small, plain JAX GRAPE implementation for the qubit-cavity Fock problem.

This file is intentionally simple:

- no classes,
- no imports from the larger ``hybrid_residual_grape`` package,
- controls are 4 x 20 B-spline coefficients:
    [qubit I, qubit Q, cavity I, cavity Q],
- the coefficients are bounded to ``[-param_clip, param_clip]`` with ``tanh``.

The Hamiltonian is the same rotating/dispersive-frame model used in the larger
project, written here as direct functions so it is easy to read and tweak.
"""

from __future__ import annotations

import json
from pathlib import Path

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from toolbox.quantmech.operators import destroy, hconj, identity, sigma, tensor
from toolbox.quantmech.states import basis
from toolbox.quantmech.unit_evol import evol_hdt_exp


def khz_to_rad_per_us(value_khz):
    """Convert kHz to angular frequency in rad / microsecond."""
    return 2.0 * jnp.pi * value_khz / 1000.0


def load_physics_from_json(config_path, mu_qub=20.0, mu_cav=20.0):
    """Read only the Hamiltonian numbers we need from configuration.json."""
    with Path(config_path).open() as f:
        data = json.load(f)

    return {
        "chi": khz_to_rad_per_us(float(data["chi_kHz"])),
        "self_kerr": khz_to_rad_per_us(float(data.get("self_Kerr_kHz", 0.0))),
        "cavity_detuning": 0.0,
        "qubit_detuning": 0.0,
        "mu_qub": float(mu_qub),
        "mu_cav": float(mu_cav),
        "use_cavity_iq_convention": True,
    }


def make_settings():
    """Collect the knobs in one dictionary.

    The point is to keep function calls short. In a notebook, do:

    ``settings = make_settings()``

    and then directly edit values like ``settings["n_iter"] = 1000``.
    """
    return {
        "n_cav": 25,
        "target_n": 2,
        "initial_qubit": 0,
        "initial_cavity": 0,
        "t_drive": 1.408,
        "n_time_steps": 80,
        "n_bspline": 20,
        "spline_degree": 2,
        "skip_left": 2,
        "skip_right": 2,
        "param_clip": 2.0,
        "n_iter": 300,
        "learning_rate": 0.03,
        "amplitude_l2": 4e-5,
        "smoothness_l2": 6e-5,
        "grad_clip": 30.0,
    }


def bspline_knots(t0, t1, n_total, degree):
    """Uniform open knot vector for a B-spline basis."""
    left = jnp.full((degree,), t0)
    middle = jnp.linspace(t0, t1, n_total - degree + 1)
    right = jnp.full((degree,), t1)
    return jnp.concatenate([left, middle, right])


def bspline_basis(time_grid, settings):
    """Build 20 quadratic B-splines after skipping two splines on each edge.

    We first build ``20 + 2 + 2`` splines, then remove the first two and last
    two. The remaining 20 splines are shifted copies of each other, vanish at
    the pulse endpoints, and at most ``degree + 1 = 3`` overlap at a time.
    """
    degree = settings["spline_degree"]
    skip_left = settings["skip_left"]
    skip_right = settings["skip_right"]
    n_total = settings["n_bspline"] + skip_left + skip_right
    knots = bspline_knots(0.0, settings["t_drive"], n_total, degree)
    time_grid = jnp.asarray(time_grid)

    basis_values = (
        (time_grid[None, :] >= knots[:-1, None])
        & (time_grid[None, :] < knots[1:, None])
    ).astype(time_grid.dtype)

    for current_degree in range(1, degree + 1):
        new_count = basis_values.shape[0] - 1
        i = jnp.arange(new_count)

        left_den = knots[i + current_degree] - knots[i]
        right_den = knots[i + current_degree + 1] - knots[i + 1]

        left = jnp.where(
            left_den[:, None] > 0.0,
            (time_grid[None, :] - knots[i, None]) / left_den[:, None] * basis_values[:-1],
            0.0,
        )
        right = jnp.where(
            right_den[:, None] > 0.0,
            (knots[i + current_degree + 1, None] - time_grid[None, :])
            / right_den[:, None]
            * basis_values[1:],
            0.0,
        )
        basis_values = left + right

    return basis_values[skip_left : n_total - skip_right]


def make_system(settings):
    """Precompute grids, B-splines, operators, and the initial state."""
    n_cav = settings["n_cav"]
    t_edges = jnp.linspace(0.0, settings["t_drive"], settings["n_time_steps"] + 1)
    t_mid = 0.5 * (t_edges[1:] + t_edges[:-1])
    dt = t_edges[1:] - t_edges[:-1]

    a = tensor(identity(2), destroy(n_cav))
    adag = hconj(a)
    n_op = adag @ a
    one = identity(2 * n_cav)
    sigz = tensor(sigma.z, identity(n_cav))
    sigp = tensor(sigma.p, identity(n_cav))
    sigm = hconj(sigp)

    psi0 = tensor(
        basis(2, settings["initial_qubit"]),
        basis(n_cav, settings["initial_cavity"]),
    )

    return {
        "t_edges": t_edges,
        "t_mid": t_mid,
        "dt": dt,
        "basis_mid": bspline_basis(t_mid, settings),
        "basis_edges": bspline_basis(t_edges, settings),
        "a": a,
        "adag": adag,
        "n_op": n_op,
        "n2_minus_n": n_op @ n_op - n_op,
        "one": one,
        "sigz": sigz,
        "sigp": sigp,
        "sigm": sigm,
        "qubit_excited": 0.5 * (one - sigz),
        "psi0": psi0,
    }


def coefficients_from_raw(raw, settings):
    """Map unconstrained optimizer variables to bounded hardware coefficients."""
    return settings["param_clip"] * jnp.tanh(raw)


def raw_from_coefficients(coefficients, settings):
    """Inverse of ``coefficients_from_raw`` for warm starts."""
    x = jnp.asarray(coefficients) / settings["param_clip"]
    x = jnp.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return jnp.arctanh(x)


def pulse_fields(coefficients, system, settings):
    """Convert 80 B-spline coefficients into qubit and cavity complex envelopes."""
    coeffs = jnp.asarray(coefficients).reshape(4, settings["n_bspline"])
    real_channels = coeffs @ system["basis_mid"]
    qubit = real_channels[0] + 1j * real_channels[1]
    cavity = real_channels[2] + 1j * real_channels[3]
    return qubit, cavity


def drift_hamiltonian(system, physics):
    """Dispersive-frame drift Hamiltonian."""
    n_op = system["n_op"]
    one = system["one"]
    sigz = system["sigz"]
    return (
        -0.5 * physics["chi"] * (n_op @ (one - sigz))
        + 0.5 * physics["self_kerr"] * system["n2_minus_n"]
        + physics["cavity_detuning"] * n_op
        + physics["qubit_detuning"] * system["qubit_excited"]
    )


def final_state(coefficients, system, physics, settings):
    """Evolve the initial state through the piecewise-constant pulse."""
    qubit_drive, cavity_drive = pulse_fields(coefficients, system, settings)
    if physics["use_cavity_iq_convention"]:
        cavity_drive = 1j * jnp.conj(cavity_drive)

    h0 = drift_hamiltonian(system, physics)

    def step(psi, controls_at_t):
        eq, ec, dt = controls_at_t
        h = (
            h0
            + physics["mu_qub"] * (eq * system["sigp"] + jnp.conj(eq) * system["sigm"])
            + physics["mu_cav"] * (ec * system["adag"] + jnp.conj(ec) * system["a"])
        )
        return evol_hdt_exp(h, dt) @ psi, None

    psi_final, _ = jax.lax.scan(step, system["psi0"], (qubit_drive, cavity_drive, system["dt"]))
    return psi_final


def fock_probability(coefficients, system, physics, settings):
    """Probability that the cavity ends in ``target_n``, summed over qubit state."""
    psi = final_state(coefficients, system, physics, settings)
    psi = psi.reshape(2, settings["n_cav"])
    return jnp.sum(jnp.abs(psi[:, settings["target_n"]]) ** 2).real


def pulse_regularization(coefficients, settings):
    """Small penalties that keep GRAPE from using unnecessarily rough pulses."""
    coeffs = coefficients.reshape(4, settings["n_bspline"])
    amp = settings["amplitude_l2"] * jnp.mean(coeffs**2)
    smooth = settings["smoothness_l2"] * jnp.mean(jnp.diff(coeffs, axis=1) ** 2)
    return amp + smooth


def grape_objective(raw, system, physics, settings):
    """Quantity we maximize: target Fock probability minus gentle pulse penalties."""
    coefficients = coefficients_from_raw(raw, settings)
    prob = fock_probability(coefficients, system, physics, settings)
    reg = pulse_regularization(coefficients, settings)
    return prob - reg, jnp.array([prob, reg, jnp.max(jnp.abs(coefficients))])


def random_coefficients(key, settings, scale=0.05):
    """Small random initial pulse in coefficient space."""
    shape = (4 * settings["n_bspline"],)
    return scale * jax.random.normal(key, shape)


def run_grape(system, physics, settings, key, initial_coefficients=None):
    """Run a compact Adam-GRAPE loop.

    Returns a dictionary with:

    - ``coefficients``: optimized 80-vector,
    - ``history``: columns [P_n, regularization, max_abs_coeff, grad_norm],
    - ``final_probability``: final target Fock probability.
    """
    if initial_coefficients is None:
        initial_coefficients = random_coefficients(key, settings)
    raw0 = raw_from_coefficients(initial_coefficients, settings)

    learning_rate = settings["learning_rate"]
    grad_clip = settings["grad_clip"]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    def loss(raw):
        objective, aux = grape_objective(raw, system, physics, settings)
        return -objective, aux

    value_and_grad = jax.value_and_grad(loss, has_aux=True)

    def adam_step(carry, step_index):
        raw, m, v = carry
        (loss_value, aux), grad = value_and_grad(raw)

        grad_norm = jnp.linalg.norm(grad)
        grad = grad * jnp.minimum(1.0, grad_clip / (grad_norm + 1e-12))

        step_number = step_index + 1
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad**2
        m_hat = m / (1.0 - beta1**step_number)
        v_hat = v / (1.0 - beta2**step_number)
        raw = raw - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        history_row = jnp.array([aux[0], aux[1], aux[2], grad_norm, loss_value])
        return (raw, m, v), history_row

    @jax.jit
    def optimize(raw):
        zeros = jnp.zeros_like(raw)
        (raw, _, _), history = jax.lax.scan(
            adam_step,
            (raw, zeros, zeros),
            jnp.arange(settings["n_iter"]),
        )
        return raw, history

    raw_final, history = optimize(raw0)
    coefficients = coefficients_from_raw(raw_final, settings)
    final_probability = fock_probability(coefficients, system, physics, settings)

    return {
        "coefficients": coefficients,
        "history": history,
        "final_probability": final_probability,
    }
