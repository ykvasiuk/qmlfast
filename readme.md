# qmlfast

A fast and memory-efficient implementation of the multi-field [QML](https://arxiv.org/abs/astro-ph/0012120) (Quadratic Maximum Likelihood) power spectrum estimator for correlated scalar fields on the sphere. See [example.ipynb](example.ipynb) for a full working example.

```bash
conda env create -f env.yml -n qmlenv
```

## Features

- Multi-field QML estimator for an arbitrary number of correlated fields
- Packed storage: basis vectors stored without zero-padding, ~2x faster and ~2-4x less memory than the padded approach
- Memory-budgeted batching for large numbers of fields via `budget` parameter
- Support for arbitrary rank-one decompositions (not just spherical harmonics) via `pack_basis`
- Band-power Fisher matrix computation via eigendecomposition of binned basis elements
- Mode deprojection for removing unwanted low-$\ell$ multipoles
- Parallelized with [numba](https://numba.pydata.org/), optimized contractions with [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)

## Basic Usage

### Prepare the input

```python
import numpy as np
import healpy as hp
from utilities import theta_phi, sph_harm_y_real_all
from qmlfast import *

mask = hp.read_map('mask.fits')
nside = 16
lmax = 3 * nside - 1
Nf = 2  # number of correlated fields
Np = int(np.sum(mask))

theta, phi = theta_phi(nside)
theta, phi = theta[mask == 1], phi[mask == 1]

Y_r_all = sph_harm_y_real_all(3 * nside, theta, phi)

F_idx = np.array([(i, j, l) for l in range(3 * nside)
                  for i in range(Nf) for j in range(i, Nf)])
```

The user must also construct the inverse pixel-space covariance matrix `Cinv` of shape `(Nf*Np, Nf*Np)` and the block map `C_map` of shape `(Nf, Nf)` indicating which field-pair blocks are nonzero.

### Fisher matrix

```python
C_map = np.ones((Nf, Nf))
F = getF(Y_r_all, Cinv, F_idx, Nf, Np, C_map)
```

For memory-constrained problems (large `Nf`), pass `budget` to batch the computation:

```python
F = getF(Y_r_all, Cinv, F_idx, Nf, Np, C_map, budget=50)
```

### QML estimates

```python
obs = np.stack([map_a[mask == 1], map_b[mask == 1]])
y = get_y(obs, Y_r_all, Cinv, F_idx, Nf, Np)
c_hat = np.linalg.solve(F, y)
```

For simulation loops, pre-pack the basis to avoid repacking each call:

```python
V_packed, offsets, ranks = pack_sph_harm(Y_r_all)
for sim in range(n_sims):
    y = get_y_packed(x, V_packed, offsets, ranks, Cinv, F_idx, Nf, Np)
```

### Arbitrary rank-one decompositions

The Fisher matrix computation generalizes beyond spherical harmonics. Any set of basis elements that can be decomposed as a sum of rank-one matrices works:

```python
# V_list[i] has shape (r_i, npix) — the rank-one vectors for basis element i
V_packed, offsets, ranks = pack_basis(V_list)
F = getF_packed(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map)
```

This is used for band-power Fisher matrices where the binned basis $P_b = \sum_\ell S_{\ell b} P_\ell$ is eigendecomposed into rank-one vectors.

## API Reference

| Function | Description |
|----------|-------------|
| `getF(Y_r_all, Cinv, F_idx, Nf, Np, C_map, ranks=None, budget=None)` | Compute Fisher matrix. Uses packed storage internally. Pass `ranks` for non-spherical-harmonic bases, `budget` for memory batching. |
| `get_y(x, Y_r_all, Cinv, F_idx, Nf, Np, ranks=None)` | Compute QML estimator vector. Uses packed storage internally. |
| `getF_packed(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map)` | Fisher matrix from pre-packed basis vectors. |
| `get_y_packed(x, V_packed, offsets, ranks, Cinv, F_idx, Nf, Np)` | QML estimator vector from pre-packed basis vectors. Use in simulation loops. |
| `getF_batched(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map, budget)` | Memory-budgeted Fisher matrix. Peak memory: `2 * budget` VCinvV matrices. |
| `pack_sph_harm(Y_r_all)` | Convert padded spherical harmonics to packed format. Returns `(V_packed, offsets, ranks)`. |
| `pack_basis(V_list)` | Pack a list of per-mode basis arrays into packed format. Returns `(V_packed, offsets, ranks)`. |
| `profile_access(F_idx, Nf, C_map)` | Dry-run the Fisher loop and count VCinvV accesses per field pair. |

## Examples

| Notebook | Description |
|----------|-------------|
| [example.ipynb](example.ipynb) | Full 3-field QML pipeline: Fisher matrix, MC simulations, power spectrum estimates |
| [bandpower_explore.ipynb](bandpower_explore.ipynb) | Band-power Fisher: eigendecomposition, truncation, direct trace vs packed comparison |
| [bandpower_2field.ipynb](bandpower_2field.ipynb) | Binned estimator with 2 fields and uneven bins: eq. 24 vs direct binned, $F_b = S^T F_\ell S$ |

## Credits

Based on [Tegmark (1997)](https://arxiv.org/abs/astro-ph/9611174) and implemented following [Kvasiuk et al. (2025)](https://arxiv.org/abs/2510.05215). Works using this code should cite [2510.05215](https://arxiv.org/abs/2510.05215).
