# qmlfast
A fast and efficient implementation of the multi-field [QML](https://arxiv.org/abs/astro-ph/0012120) estimator. A minimalistic example of the main code functionality is in [example.ipynb](example.ipynb). To configure the environment:

```bash
conda env create -f env.yml -n qmlenv
```

## Credits
The theoretical foundations of the code is based on [arxiv:astro-ph/9611174](https://arxiv.org/abs/astro-ph/9611174) and implemented according to [arxiv to our qml paper]. Python librabries [numba](https://numba.pydata.org/) and [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/) are used for optimization. This code is designed to interact with 2-dimensional field in [Healpix](https://healpix.sourceforge.io/) projection. Works that involve the usage of the code should cite [arxiv to our qml paper] for reference.

## Basic Usage
### Prepare the Input
`qmlfast` takes arguments related to the input binary mask:
```python
import healpy as hp

mask = hp.read_map('/directory/to/the/binary/mask.fits')
nside = 16                    # resolution of the mask
Np = np.sum(mask, dtype=int)  # number of unmasked pixels
```
and two basic parameters `lmax`, `Nf` defined by the user:
```python
lmax = 3*nside - 1            # the maxmium multipole to be analyzed, no greater than 3*nside-1
Nf = 2                        # the total number of correlated fields
```
User instructs the outputing order of the Fisher elements by consturcting an array of index `F_idx`
```python
F_idx = np.array([(i, j, l) for l in range(0, lmax) for i in range(Nf) for j in range(i, Nf)])     # an array of index in order of (first field, second field, mode) to instruct the output order of Fisher matrix entries
```
The module `utilities.py` includes functions that generate inputs for the Fisher matrix calculation:
```python
from utilities import theta_phi, sph_harm_y_real_all
# getting angular positions of unmasked pixels
theta, phi = theta_phi(nside); 
theta = theta[mask==1]      
phi = phi[mask==1]
# getting transfomation matrix between Ylm's and pixels                
Y_r_all = sph_harm_y_real_all(lmax,theta,phi)     
```
Users are also required to construct the inverse pixel-space covariance matrix `Cinv` and the block map `C_map` for their fiducial powers. 

### Fisher Matrix Calculation
To compute the Fisher matrix, run the following:
```python
from qmlfast import *

F = getF(Y_r_all,Cinv,F_idx,Nf,Np,C_map)  #getting the Fisher matrix, the output order follows the indexing in F_idx that is passed
```
depending on the number of fields and resolutions, this process can take from seconds to hours on a single node. The `getF` routine is a convenience wrapper that combines the calculation of the $YC^{-1}Y$ products and Fisher matrix from them. In case of more heavy calculations, one might need / want to separate this process. The most compute-heavy part is to get $YC^{-1}Y$-s. However, one might want to configure the progressbar in the `_getF` routine. This can easily be accomplished with [numba_progress](https://github.com/mortacious/numba-progress)

### Obtaining Estimates
To estimate the full-sky power spectrum, user passes an array of stacked `Healpix` maps to the function `qmlfast.get_y` and solve for the QML estimates.
```python
obs = np.stack([map1_unmasked_pixels_only,
                 map2_unmasked_pixels_only, ...])               # order of stacking should be that same as the pixel covariance matrix
y = get_y(obs, Y_r_all,Cinv,F_idx,Nf,Np)                          # 'y' aren't the estimates yet 
e = np.linalg.solve(F,y)                               # solved 'e''s are the power spectrum estimates, the order of 'e' follows that in F_idx
```
See [`example.ipynb`](example.ipynb) for more details.
