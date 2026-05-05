# Simplified Adaptive GRAPE

This repository now contains only the cleaned, readable adaptive-GRAPE project.
The old RBF residual models, SPSA notebooks, and larger exploratory package were
removed so the codebase is easier to inspect and share.

## What This Project Does

The goal is to simulate a closed-loop calibration workflow for preparing a Fock
state in a cavity coupled to a qubit.

The control pulse is represented by `4 x 20` real coefficients:

- qubit I
- qubit Q
- cavity I
- cavity Q

Those coefficients are expanded into smooth quadratic B-spline pulses using the
small local `toolbox` package. The first two and last two quadratic B-splines are
skipped, so the played pulses start and end at zero and at most three basis
functions overlap at once.

The simplified notebooks compare three levels of simulation:

- unitary GRAPE on the nominal Hamiltonian
- non-unitary GRAPE with fixed `T1/T2`
- a hidden "true experiment" model with small Hamiltonian mismatches,
  amplitude miscalibration, decoherence, finite-shot binomial measurements, and
  a simple affine measurement-response distortion

## Useful Files

- `Simplified_adaptive_grape/grape.py`
  Tiny helper functions only: B-spline coefficient expansion, Fock-state
  probability extraction, regularization, and coefficient clipping.

- `Simplified_adaptive_grape/test_grape.ipynb`
  Minimal GRAPE notebook. It is the best place to check the Hamiltonian,
  B-spline basis, unitary GRAPE, non-unitary GRAPE, and the independent dynamiqs
  double check.

- `Simplified_adaptive_grape/adaptive_grape_parameter_fit.ipynb`
  Single-pass experiment: optimize a pulse, probe nearby controls on the hidden
  true model, then fit Hamiltonian parameters and the measurement response.

- `Simplified_adaptive_grape/closed_loop_adaptive_grape.ipynb`
  Closed-loop experiment: alternate GRAPE optimization, true-model probing, and
  parameter fitting. It first uses unitary evolution, then switches to
  non-unitary evolution.

- `configuration.json`
  Experimental parameters copied from the NNcat configuration. The simplified
  notebooks read values directly from this file.

- `toolbox/`
  The minimal subset of the toolbox needed by the simplified notebooks:
  B-splines, basic quantum operators/states, Hamiltonian trees, and simple
  Schrodinger/Lindblad solvers.

## Setup

From the repository root:

```bash
uv sync
```

Then open one of the notebooks in `Simplified_adaptive_grape/`.

On a DGX/Grace Hopper machine, keep the CUDA-enabled JAX dependency from
`pyproject.toml`. On a local Mac without an NVIDIA GPU, you may need to adjust
the JAX dependency locally before syncing.

## Current Direction

The main notebook to use now is:

```text
Simplified_adaptive_grape/closed_loop_adaptive_grape.ipynb
```

The intended workflow is:

1. Run GRAPE with the current calibrated model.
2. Probe a cloud of nearby pulses on the hidden true model.
3. Fit Hamiltonian and measurement-response parameters from measured data.
4. Repeat, warm-starting from the previous pulse and previous fitted parameters.

This keeps the code close to the physical picture: the experiment only returns
finite-shot measured values, while the model tries to calibrate the Hamiltonian
used by GRAPE.
