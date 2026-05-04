"""Tiny helper functions for the simplified GRAPE notebook.

The notebook keeps the Hamiltonian, B-splines, JSON values, and optimizer loop
explicitly visible. This file only contains tiny repeated operations.
"""

import jax.numpy as jnp


def controls_from_coefficients(coefficients, bsplines):
    """4 x 20 real coefficients -> complex qubit and cavity envelopes."""
    real_channels = coefficients @ bsplines
    qubit = real_channels[0] + 1j * real_channels[1]
    cavity = real_channels[2] + 1j * real_channels[3]
    return qubit, cavity


def fock_probability_from_state(psi, n_cav, target_n):
    """P(cavity has target_n photons), summed over the two qubit states."""
    psi = psi.reshape(2, n_cav)
    return jnp.sum(jnp.abs(psi[:, target_n]) ** 2).real


def fock_probability_from_density(rho, n_cav, target_n):
    """Same probability as above, but from a full qubit-cavity density matrix."""
    rho = rho.reshape(2, n_cav, 2, n_cav)
    return (rho[0, target_n, 0, target_n] + rho[1, target_n, 1, target_n]).real


def pulse_penalty(coefficients, amplitude_l2=4e-5, smoothness_l2=6e-5):
    """Small regularization to avoid very large or rough B-spline coefficients."""
    return amplitude_l2 * jnp.mean(coefficients**2) + smoothness_l2 * jnp.mean(
        jnp.diff(coefficients, axis=1) ** 2
    )


def clip_coefficients(coefficients, param_clip=2.0):
    """Hard coefficient clipping, directly in the units you will send to hardware."""
    return jnp.clip(coefficients, -param_clip, param_clip)
