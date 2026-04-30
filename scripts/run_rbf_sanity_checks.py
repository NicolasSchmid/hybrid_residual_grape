from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from hybrid_residual_grape import (
    FockPhysicsModel,
    HybridGrapeConfig,
    PhysicsParams,
    RBFResidualConfig,
    SimulationConfig,
    empty_rbf_model,
    fit_rbf_residual,
    optimize_hybrid_grape,
)
from hybrid_residual_grape.config import khz_to_rad_per_us, self_kerr_rad_per_us
from hybrid_residual_grape.residual import (
    hybrid_probability_from_physics,
    measured_probability,
    residual_logit,
    safe_logit,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"


def mae(x: jax.Array, y: jax.Array) -> float:
    return float(jnp.mean(jnp.abs(jnp.asarray(x) - jnp.asarray(y))))


def deterministic_binomial_counts(probability: jax.Array, shots: int) -> jax.Array:
    return jnp.rint(jnp.clip(probability, 0.0, 1.0) * shots)


def make_models() -> tuple[SimulationConfig, FockPhysicsModel, FockPhysicsModel]:
    """Small, fast models used only for sanity checks.

    The true model has a controlled misspecification relative to the nominal
    model.  We keep the Hilbert space smaller than the research notebooks so
    this script can be run quickly during documentation updates.
    """
    q = SimulationConfig(n_cav=16, target_n=2, ndt_drive=56)
    nominal_params = PhysicsParams(
        cavity_self_kerr=0.0,
        cavity_detuning=0.0,
        qubit_detuning=0.0,
    )
    true_params = PhysicsParams(
        chi=nominal_params.chi + khz_to_rad_per_us(1.5),
        cavity_self_kerr=self_kerr_rad_per_us() + khz_to_rad_per_us(-0.12),
        cavity_detuning=khz_to_rad_per_us(1.2),
        qubit_detuning=khz_to_rad_per_us(-1.0),
        mu_qub=20.0 * 1.025,
        mu_cav=20.0 * 0.975,
        cavity_phase=0.02,
    )
    return q, FockPhysicsModel(q, nominal_params), FockPhysicsModel(q, true_params)


def optimize_reference_pulse(
    physics_model: FockPhysicsModel,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    initial_controls = 0.12 * jax.random.normal(key, (physics_model.parameter_size,))
    initial_controls = jnp.clip(initial_controls, -0.4, 0.4)
    empty_residual = empty_rbf_model(
        physics_model.parameter_size,
        RBFResidualConfig(max_centers=1),
    )
    grape_config = HybridGrapeConfig(
        maxiter=90,
        memory_size=8,
        noise_samples=1,
        control_noise_std=0.0,
        residual_support_penalty=0.0,
        residual_size_penalty=0.0,
        amplitude_l2=3e-5,
        smoothness_l2=1e-4,
    )
    controls, history, _, _ = optimize_hybrid_grape(
        physics_model,
        empty_residual,
        initial_controls,
        key,
        grape_config,
    )
    return controls, history


def seen_control_check(
    physics_model: FockPhysicsModel,
    true_model: FockPhysicsModel,
    base_controls: jax.Array,
    key: jax.Array,
) -> dict[str, float]:
    n_train = 64
    shots = 3000
    perturb = 0.10 * jax.random.normal(key, (n_train, physics_model.parameter_size))
    controls = jnp.clip(base_controls[None, :] + perturb, -2.0, 2.0)
    physics_p = physics_model.population_probability(controls)
    true_p = true_model.population_probability(controls)
    successes = deterministic_binomial_counts(true_p, shots)
    shots_arr = jnp.full((n_train,), shots)

    config = RBFResidualConfig(
        max_centers=n_train,
        length_scale=0.11,
        ridge=1e-5,
        residual_clip=3.0,
        measurement_floor=1e-4,
    )
    rbf = fit_rbf_residual(controls, physics_p, successes, shots_arr, config)
    measured = measured_probability(successes, shots_arr, config.measurement_floor)
    hybrid_p = hybrid_probability_from_physics(physics_p, rbf, controls)

    order = np.argsort(np.asarray(true_p))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].scatter(measured, physics_p, s=30, alpha=0.75, label="nominal physics")
    axes[0].scatter(measured, hybrid_p, s=30, alpha=0.75, label="physics + RBF")
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[0].set_xlabel("measured probability on seen controls")
    axes[0].set_ylabel("predicted probability")
    axes[0].set_title("Seen-control interpolation")
    axes[0].legend()

    axes[1].plot(np.asarray(measured)[order], "o", label="measured")
    axes[1].plot(np.asarray(physics_p)[order], "o", label="nominal physics")
    axes[1].plot(np.asarray(hybrid_p)[order], "o", label="physics + RBF")
    axes[1].set_xlabel("seen control, sorted by true probability")
    axes[1].set_ylabel("P_n")
    axes[1].set_title("The RBF should fit its own data")
    axes[1].legend()
    fig.savefig(RESULTS / "rbf_seen_controls_sanity.png", dpi=180)
    plt.close(fig)

    return {
        "seen_physics_mae_to_measured": mae(physics_p, measured),
        "seen_hybrid_mae_to_measured": mae(hybrid_p, measured),
        "seen_physics_mae_to_true": mae(physics_p, true_p),
        "seen_hybrid_mae_to_true": mae(hybrid_p, true_p),
        "seen_mean_abs_logit_residual": float(jnp.mean(jnp.abs(residual_logit(rbf, controls)))),
    }


def low_dimensional_rbf_check(
    physics_model: FockPhysicsModel,
    true_model: FockPhysicsModel,
    base_controls: jax.Array,
    key: jax.Array,
) -> dict[str, float]:
    shots = 3000
    key_directions, key_train = jax.random.split(key)
    raw_dirs = jax.random.normal(key_directions, (2, physics_model.parameter_size))
    directions = raw_dirs / (jnp.linalg.norm(raw_dirs, axis=1, keepdims=True) + 1e-12)
    directions = 3.5 * directions

    train_side = 11
    train_axis = jnp.linspace(-0.50, 0.50, train_side)
    aa, bb = jnp.meshgrid(train_axis, train_axis, indexing="ij")
    z_train = jnp.stack([aa.ravel(), bb.ravel()], axis=1)
    jitter = 0.012 * jax.random.normal(key_train, z_train.shape)
    z_train = z_train + jitter

    test_side = 45
    test_axis = jnp.linspace(-0.55, 0.55, test_side)
    aa_test, bb_test = jnp.meshgrid(test_axis, test_axis, indexing="ij")
    z_test = jnp.stack([aa_test.ravel(), bb_test.ravel()], axis=1)

    def controls_from_z(z):
        return jnp.clip(base_controls + z[0] * directions[0] + z[1] * directions[1], -2.0, 2.0)

    train_controls = jax.vmap(controls_from_z)(z_train)
    test_controls = jax.vmap(controls_from_z)(z_test)

    train_physics = physics_model.population_probability(train_controls)
    train_true = true_model.population_probability(train_controls)
    successes = deterministic_binomial_counts(train_true, shots)
    shots_arr = jnp.full((z_train.shape[0],), shots)

    config = RBFResidualConfig(
        max_centers=z_train.shape[0],
        length_scale=0.24,
        ridge=2e-3,
        residual_clip=1.5,
        measurement_floor=1e-4,
    )
    rbf = fit_rbf_residual(z_train, train_physics, successes, shots_arr, config)

    test_physics = physics_model.population_probability(test_controls)
    test_true = true_model.population_probability(test_controls)
    test_hybrid = hybrid_probability_from_physics(test_physics, rbf, z_test)
    learned_logit_residual = residual_logit(rbf, z_test)
    true_logit_residual = safe_logit(test_true) - safe_logit(test_physics)

    extent = [
        float(test_axis[0]),
        float(test_axis[-1]),
        float(test_axis[0]),
        float(test_axis[-1]),
    ]
    true_grid = np.asarray(test_true).reshape(test_side, test_side)
    physics_grid = np.asarray(test_physics).reshape(test_side, test_side)
    hybrid_grid = np.asarray(test_hybrid).reshape(test_side, test_side)
    correction_grid = np.asarray(learned_logit_residual).reshape(test_side, test_side)
    true_correction_grid = np.asarray(true_logit_residual).reshape(test_side, test_side)
    error_gain_grid = np.abs(physics_grid - true_grid) - np.abs(hybrid_grid - true_grid)

    vmax = max(
        float(np.max(np.abs(true_correction_grid))),
        float(np.max(np.abs(correction_grid))),
        1e-3,
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.3), constrained_layout=True)
    im0 = axes[0, 0].imshow(
        true_correction_grid.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0, 0].set_title("true logit residual")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(
        correction_grid.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0, 1].set_title("RBF learned logit residual")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(
        true_grid.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.2, float(np.max(true_grid))),
    )
    axes[1, 0].set_title("hidden true P_n on two-control manifold")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(
        error_gain_grid.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="PiYG",
        vmin=-float(np.max(np.abs(error_gain_grid))),
        vmax=float(np.max(np.abs(error_gain_grid))),
    )
    axes[1, 1].set_title("|physics error| - |hybrid error|")
    fig.colorbar(im3, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.set_xlabel("control coordinate a")
        ax.set_ylabel("control coordinate b")
    fig.savefig(RESULTS / "rbf_low_dimensional_sanity.png", dpi=180)
    plt.close(fig)

    measured_train = measured_probability(successes, shots_arr, config.measurement_floor)
    hybrid_train = hybrid_probability_from_physics(train_physics, rbf, z_train)
    return {
        "low_dim_train_physics_mae_to_measured": mae(train_physics, measured_train),
        "low_dim_train_hybrid_mae_to_measured": mae(hybrid_train, measured_train),
        "low_dim_test_physics_mae_to_true": mae(test_physics, test_true),
        "low_dim_test_hybrid_mae_to_true": mae(test_hybrid, test_true),
        "low_dim_test_fraction_improved": float(
            jnp.mean((jnp.abs(test_hybrid - test_true) < jnp.abs(test_physics - test_true)).astype(jnp.float32))
        ),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(7)
    key_ref, key_seen, key_low_dim = jax.random.split(key, 3)
    _, physics_model, true_model = make_models()

    base_controls, grape_history = optimize_reference_pulse(physics_model, key_ref)
    base_physics = float(physics_model.photon_probability(base_controls))
    base_true = float(true_model.photon_probability(base_controls))
    print(f"reference pulse nominal P_n: {base_physics:.6f}")
    print(f"reference pulse true P_n:    {base_true:.6f}")

    seen = seen_control_check(physics_model, true_model, base_controls, key_seen)
    low_dim = low_dimensional_rbf_check(physics_model, true_model, base_controls, key_low_dim)

    print("\nSeen-control RBF interpolation check")
    for name, value in seen.items():
        print(f"{name}: {value:.6f}")

    print("\nLow-dimensional two-control RBF check")
    for name, value in low_dim.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
