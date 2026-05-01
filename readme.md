# qmlfast


```bash
conda env create -f env.yml -n qmlenv
```

## Features

- Multi-field QML estimator for an arbitrary number of correlated fields
- Packed storage: basis vectors stored without zero-padding, ~2x faster and ~2-4x less memory than the padded approach
- Memory-budgeted batching for large numbers of fields via `budget` parameter
- Support for arbitrary rank-one decompositions (not just spherical harmonics) via `pack_basis`
- Band-power Fisher matrix computation via eigendecomposition of binned basis elements
- Mode deprojection for removing unwanted low-$\ell$ multipoles, with the QR/projector construction as the main path and a Woodbury-limit inverse-covariance alternative
- Explicit packed noise-bias routines for dense general noise covariance and per-field white pixel noise
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

Y_r_all = sph_harm_y_real_all(3 * nside, theta, phi)  # progress bar enabled by default

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

### Noise bias

For white pixel noise with one variance per field, use the white-noise shortcut:

```python
noise_var = np.array([N_a / omega_pix, N_b / omega_pix])
noise_bias = get_noise_bias_packed_white(
    V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, noise_var)
```

For a dense pixel-noise covariance `N_pix` of shape `(Nf*Np, Nf*Np)`, use:

```python
noise_bias = get_noise_bias_packed_general(
    V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, N_pix)
```

### Mode deprojection

The main deprojection path constructs orthonormal low-$\ell$ modes `Z` and the corresponding pixel-space projector `pi = I - Z Z.T`:

```python
Z, pi = construct_Z_and_pi(theta, phi, lmax=lmax, ell0=ell0)

for i in range(Nf):
    for j in range(Nf):
        cov_block = C[block_np(i, j, Np)]
        C[block_np(i, j, Np)] = pi @ cov_block @ pi.T
```

As an alternative, low-$\ell$ modes can be projected out directly from an inverse covariance using the Woodbury-limit helper:

```python
Z_raw = low_ell_mode_matrix(theta, phi, ell0)
Z_all = scipy.linalg.block_diag(*[Z_raw] * Nf)
M = deproject_inverse_woodbury(Cinv, Z_all)
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
| `get_noise_bias_packed_white(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, noise_var)` | Noise-bias vector for per-field white pixel noise. |
| `get_noise_bias_packed_general(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, noise)` | Noise-bias vector for a dense pixel-noise covariance. |
| `pack_sph_harm(Y_r_all)` | Convert padded spherical harmonics to packed format. Returns `(V_packed, offsets, ranks)`. |
| `pack_basis(V_list)` | Pack a list of per-mode basis arrays into packed format. Returns `(V_packed, offsets, ranks)`. |
| `profile_access(F_idx, Nf, C_map)` | Dry-run the Fisher loop and count VCinvV accesses per field pair. |

Utility helpers:

| Function | Description |
|----------|-------------|
| `sph_harm_y_real_all(lmax, theta, phi, progress=True, desc='Y_lm')` | Real spherical harmonics with optional progress bar. |
| `get_Pl_ij(theta, phi, nside, lmax=None)` | Pixel-space Legendre kernels. |
| `low_ell_mode_matrix(theta, phi, ell0)` | Raw non-orthogonal low-$\ell$ mode matrix. |
| `construct_Z_and_pi(theta, phi, lmax, ell0)` | Main QR/projector low-$\ell$ deprojection construction. |
| `deproject_inverse_woodbury(Cinv, Z)` | Alternative inverse-covariance deprojection using the Woodbury limit. |

## Examples

All example notebooks use noise-bias subtraction (paper Eq. 9): signal is drawn band-limited via `synalm(S)` and white pixel noise is added separately, so the fiducial covariance equals the true data covariance ($\mathrm{Cov}(\hat y) = F$ exactly).

| Notebook | Description |
|----------|-------------|
| [example.ipynb](example.ipynb) | Full 3-field QML pipeline with mode deprojection, noise-bias subtraction, MC validation against $F^{-1}$ |
| [bandpower_1field.ipynb](bandpower_1field.ipynb) | 1-field binned estimator: posthoc (Eq. 24) and direct binned routes, both consistent with $F_b^{-1}$ |
| [bandpower_2field.ipynb](bandpower_2field.ipynb) | 2-field binned estimator: posthoc + direct binned for auto and cross spectra |
| [noise_mismatch_demo.ipynb](noise_mismatch_demo.ipynb) | Per-$\ell$ illustration of the mismodeling effect (Appendix A): empirical $\mathrm{Cov}$ matches $\tilde F^{-1}G\tilde F^{-1}$, not $\tilde F^{-1}$, when the noise model is mismatched |
| [noise_mismatch_demo_binned.ipynb](noise_mismatch_demo_binned.ipynb) | Same as above but with posthoc binning, on a small mask where per-$\ell$ inversion fails |

## Credits

Based on [Tegmark (1997)](https://arxiv.org/abs/astro-ph/9611174) and implemented following [Kvasiuk et al. (2025)](https://arxiv.org/abs/2510.05215). Works using this code should cite [2510.05215](https://arxiv.org/abs/2510.05215).
