# hybrid_residual_grape

Self-contained JAX project for closed-loop Fock-state preparation with:

- a fixed physics simulator `f_phys(u, p1)`
- a simple RBF/ridge residual model `f_bb(u, p2)`
- warm-started AD-GRAPE using `jax.grad` through `f_phys + f_bb`
- experiment-like binomial measurements from the target photon-number test

The prediction model is

```text
logit(P_pred(u)) = logit(P_phys(u, p1)) + f_bb(u, p2)
```

This keeps predictions in `[0, 1]` and makes the residual a local correction to
the physics model rather than an unconstrained fidelity predictor.

The project is self-contained: `toolbox/` and `configuration.json` are included
locally, so nothing imports from another project folder.

## Run

```zsh
cd ~/coding/hybrid_residual_grape
uv run jupyter lab hybrid_residual_grape.ipynb
```

The notebook runs the loop:

1. cache `P_phys(u_i)` for measured pulses
2. fit an RBF/ridge residual from `(u_i, successes_i, shots_i)`
3. run L-BFGS GRAPE through `P_pred(u)`
4. measure `u*` and local noisy variants with binomial shots
5. append data and repeat, warm-starting from the previous best pulse

The simulator defaults are copied from `configuration.json`:

- `chi_kHz`
- `self_Kerr_kHz`
- `mu_qub = mu_cav = 20.0`
- quadratic B-splines with 20 active coefficients per real channel
- first/last 2 splines skipped so pulses start/end at zero
