# Simplified Adaptive GRAPE

This folder is intentionally small.

The only local code is:

- `grape.py`: tiny repeated helper functions.
- `test_grape.ipynb`: the readable notebook where the Hamiltonian, B-splines,
  unitary GRAPE, decay GRAPE, and dynamiqs double check are written explicitly.

The notebook uses the repository-level `toolbox` package for only these pieces:

- `toolbox/bspln.py`: B-spline basis construction.
- `toolbox/quantmech/operators.py`: tensor products and operators.
- `toolbox/quantmech/states.py`: basis states.
- `toolbox/quantmech/se_solve.py`: unitary Schrodinger evolution.
- `toolbox/quantmech/me_solve.py`: simple Lindblad master-equation evolution.
- `toolbox/quantmech/unit_evol.py` and `ham_trees.py`: support functions used by
  the solvers.

The toolbox clone in the reference zip also contained caches, checkpoints,
neural-network helpers, sampling helpers, and Wigner helpers. None of those are
needed for this simplified GRAPE test.
