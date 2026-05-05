# Simplified Adaptive GRAPE

## A Small Textbook On The Method

This repository is about one question:

> If GRAPE works very well on a physics model, but the real experiment is
> slightly different from that model, can we use measured data from the
> experiment to adapt the model and therefore choose better pulses?

The project is intentionally not framed as reinforcement learning. There is no
long-horizon policy here. A single pulse is played, the cavity ends in some
state, and we measure whether the target Fock component was prepared. The
learning problem is closer to **model-based experimental design**:

1. keep a differentiable physics model,
2. use GRAPE to optimize the pulse in that model,
3. measure real or simulated experimental outcomes near that pulse,
4. recalibrate the model from those measurements,
5. repeat.

![Simplified adaptive GRAPE overview](docs/adaptive_grape_overview.png)

The exact equations and implementation choices are written below, because the
image is only the blackboard summary.

---

## 1. The Learning Problem

We want to prepare a target Fock state `|n>` in a 3D cavity coupled to a qubit.
The object we care about is

```text
P_true,n(u) = probability that the cavity contains n photons after pulse u.
```

In real hardware, `P_true,n(u)` is not directly visible. What we can do is:

1. play a pulse `u`,
2. apply a long selective qubit pulse that tests the photon number,
3. read out the qubit,
4. obtain a noisy binary result.

So the experiment is a black-box measurement process:

```text
u  ->  experiment  ->  measured readout y.
```

But unlike pure black-box optimization, we also have a strong prior: a
differentiable Hamiltonian simulator. The whole method is built around the idea
that the simulator is good enough to guide GRAPE, but not perfect enough to be
trusted forever without experimental feedback.

---

## 2. The Pulse Parameters `u`

The pulse is represented by `80` real numbers:

```text
u in R^80 = 4 channels x 20 B-spline coefficients.
```

The four channels are:

```text
qubit I, qubit Q, cavity I, cavity Q.
```

Those coefficients are expanded into smooth time-dependent envelopes:

```text
u  ->  e_q(t), e_c(t).
```

The notebooks use quadratic B-splines from the local `toolbox`. The first two
and last two splines are skipped so that the physical pulse starts and ends at
zero. With quadratic splines, at most three basis functions overlap at once,
which keeps the pulse shape close to something that can be played cleanly on
hardware.

Every coefficient is clipped:

```text
-2 <= u_j <= 2.
```

In machine-learning language, `u` is the **decision variable** or **design
variable**. It is the thing GRAPE updates.

---

## 3. The Model Parameters `p`

The simulator has parameters `p`. These are the things we believe might be
slightly wrong in the nominal Hamiltonian.

In the current closed-loop notebook, the fitted parameter vector has eight
entries:

```text
p_raw in R^8
```

After bounded `tanh` transforms, these correspond to:

```text
chi shift
qubit frequency shift
cavity frequency shift
cavity self-Kerr
qubit drive amplitude factor
cavity drive amplitude factor
readout contrast A
readout offset B
```

The fitted simulator predicts a physical Fock probability:

```text
P_model,n(u; p).
```

The readout part then converts that predicted physical probability into a
predicted measured value:

```text
y_pred(u; p, A, B) = A * P_model,n(u; p) + B.
```

At the moment, the closed-loop notebook keeps `T1/T2` fixed during fitting. The
non-unitary phase uses open-system evolution, but it does not fit the lifetime
parameters themselves. That is a deliberate simplification: first learn whether
the Hamiltonian and readout calibration are enough before opening a larger
parameter space.

---

## 4. The Observation Model

The hidden true simulator plays the role of the real experiment. It has small
mismatches relative to the model: small frequency shifts, Kerr terms, amplitude
miscalibration, and decoherence.

For a pulse `u`, the true simulator produces a hidden probability:

```text
P_true,n(u).
```

Then the experiment-like measurement samples finite shots:

```text
k ~ Binomial(N_shots, P_true,n(u)).
```

The notebook currently uses `N_shots = 500` for local probing. The measured
readout is then distorted by a simple affine contrast model:

```text
y = A_true * k / N_shots + B_true.
```

The values are chosen so that roughly

```text
P_true,n = 0  ->  y about 0.01
P_true,n = 1  ->  y about 0.93
```

This imitates the fact that the long selective pi pulse and qubit readout are
not an ideal photon-number oracle. Even a perfect Fock state would not
necessarily produce measured value `1.0`, and even a bad state can produce a
small false-positive offset.

The important conceptual distinction is:

```text
P_true,n     hidden physical fidelity, diagnostic only
y            noisy measured value, available to the optimizer
P_model,n    model-predicted physical fidelity
y_pred       model-predicted measurement
```

The calibration step should compare `y_pred` to `y`, not `P_model,n` directly
to `P_true,n`, because the real experiment does not reveal `P_true,n`.

---

## 5. Two Nested Optimizations

The method alternates two different optimization problems. Confusing these two
is the easiest way to misunderstand the notebook.

### 5.1 Control Update: GRAPE

During a GRAPE step, the model parameters are fixed.

```text
fixed:      p, A, B
optimized: u
```

The objective is approximately

```text
loss_u(u) = 1 - P_model,n(u; p) + pulse_penalty(u).
```

Because the simulator is written in JAX, the code computes

```text
grad_u loss_u
```

by automatic differentiation through the pulse expansion and time evolution.
This is the same practical role as GRAPE: obtain a gradient of final fidelity
with respect to pulse amplitudes.

After each optimizer step:

```text
u <- clip(u, -2, 2).
```

The next GRAPE run is warm-started from the previous best pulse. This matters:
once the pulse is already good, we do not want to rediscover it from random
initialization every round. We want to refine it under the newly calibrated
model.

### 5.2 Parameter Update: Model Calibration

During a calibration step, the measured controls are fixed.

```text
fixed:      measured pulses u_i and measured outcomes y_i
optimized: p, A, B
```

For each measured pulse, the model predicts

```text
y_pred_i = A * P_model,n(u_i; p) + B.
```

The current notebook fits parameters with a mean-squared prediction loss:

```text
loss_p(p, A, B) = mean_i((y_pred_i - y_i)^2).
```

Again, JAX differentiates through the simulator, but now the derivative is with
respect to the Hamiltonian/readout parameters, not the pulse:

```text
grad_p loss_p.
```

So the same differentiable simulator supports two different gradients:

```text
GRAPE:        d loss / d u
calibration:  d loss / d p
```

---

## 6. Do We Use All Previous Measurements?

Yes. The closed-loop notebook accumulates the full dataset.

After each local probing round, the new measured pulses are appended:

```text
D_t = D_{t-1} union {(u_i, y_i)} from the new round.
```

Then the parameter fit samples mini-batches from the full accumulated dataset,
not only from the latest 500 points.

This is the right default for a stationary simulated experiment because older
measurements are still valid information about the same Hamiltonian. Using all
past data helps stabilize the fit and reduces the chance that the fitted
parameters chase one noisy local cloud of measurements.

There are two cases where we might not want to use all data equally:

1. **The experiment drifts in time.**
   Then old data may become misleading. A sliding window or exponential
   forgetting factor would be better.

2. **The pulse moves to a very different region of control space.**
   Then old points may be less relevant to the local optimum. A weighted loss
   that gives more importance to nearby controls could help.

For now, the code assumes the hidden true model is stationary, so all measured
points are useful.

---

## 7. The Closed-Loop Algorithm

The main notebook follows this structure.

```text
initialize pulse u_0
initialize model parameters p_0
initialize empty dataset D

for each closed-loop round:

    1. GRAPE step
       optimize u using P_model,n(u; p)
       warm-start from previous best pulse

    2. local probing step
       generate pulses near the optimized pulse
       measure each pulse with finite-shot true-model readout
       append all results to D

    3. calibration step
       fit p, A, B on all accumulated data D
       warm-start from previous fitted p

    4. repeat
```

The first phase uses unitary evolution for speed. The second phase switches to
non-unitary evolution, so the optimized pulses are judged by a model that can
include decay and dephasing.

---

## 8. Why Not Just Fit A Neural Network?

A neural network could learn a flexible map

```text
u -> y.
```

That is attractive if the mismatch between model and experiment is complicated.
But it has two drawbacks here:

1. The input is 80-dimensional, and high-fidelity regions may be very small.
   A generic neural network may need many measurements before it extrapolates
   reliably.

2. GRAPE already gives a strong differentiable structure. If we throw that
   away, we lose a lot of physics.

The adaptive-GRAPE approach keeps the physics model as the main object and uses
measurements to correct the parameters that matter near the pulses we are
actually trying. This is a conservative first step before adding more flexible
residual models.

---

## 9. How To Read The Notebooks

The notebooks are written as experiments, not as a large software framework.
Most of the physics is kept visible in notebook cells so that it is easy to
modify.

Use this order:

```text
1. Simplified_adaptive_grape/test_grape.ipynb
2. Simplified_adaptive_grape/adaptive_grape_parameter_fit.ipynb
3. Simplified_adaptive_grape/closed_loop_adaptive_grape.ipynb
```

### `test_grape.ipynb`

This is the base GRAPE check. It shows the pulse basis, the Hamiltonian, unitary
GRAPE, non-unitary GRAPE, and a dynamiqs double check.

### `adaptive_grape_parameter_fit.ipynb`

This notebook checks whether a dataset of nearby measured pulses is enough to
fit Hamiltonian and readout parameters once.

### `closed_loop_adaptive_grape.ipynb`

This is the closed-loop version. It alternates GRAPE, local probing, and model
calibration over many rounds.

---

## 10. What To Look At In The Plots

When judging whether the method works, do not look at a single curve in
isolation.

The useful comparisons are:

### Model-predicted measurement versus measured value

This checks whether the fitted model predicts what the experiment actually
returns:

```text
A * P_model,n(u; p) + B   versus   measured y.
```

If this fails, the fitted model is not explaining the data.

### Model-predicted probability versus hidden true probability

This is a simulation-only diagnostic:

```text
P_model,n(u; p)   versus   P_true,n(u).
```

In real hardware we cannot plot `P_true,n`, but in simulation it tells us
whether the calibration moved the physics model closer to the hidden truth.

### Log infidelity

Near high fidelity, normal probability plots become visually uninformative.
Plotting

```text
log(1 - P_n)
```

makes the difference between `0.99`, `0.999`, and `0.9999` visible.

---

## 11. Current Simplifications

The current code is deliberately simple.

- The local probing cloud is random around the current pulse; it is not yet a
  sophisticated optimal experimental design strategy.
- The fit currently uses MSE on the observed readout values. A binomial
  likelihood using raw successes and shot counts would be statistically cleaner.
- `T1/T2` are used in the non-unitary simulation path but are not fitted in the
  current closed-loop parameter vector.
- The readout response is affine in the fit, while the real selective-pulse
  response might be mildly nonlinear.

These are good places to improve the method once the simplified loop is easy to
understand and debug.

---

## 12. Running The Project

From the repository root:

```bash
uv sync
```

Then open the notebooks in `Simplified_adaptive_grape/`.

The `pyproject.toml` is set up for the DGX/Grace Hopper environment with
CUDA-enabled JAX. On a local Mac without an NVIDIA GPU, use a local CPU-only JAX
setup instead.

The local `toolbox/` directory contains the small subset needed by the
notebooks: B-splines, quantum operators/states, and simple Schrodinger/Lindblad
solvers.
