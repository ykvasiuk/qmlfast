import numpy as np
from numba import njit, prange, types
from numba.typed import Dict
import opt_einsum as oe
from tqdm import tqdm

@njit(inline='always')
def block(i, j, npix):
    """
    A helper function that allows block indexing for a square block matrix
    eg. A[block(i,j,npix)] will yield (i,j) block of size (npix,npix) of matrix A
    """
    return (slice(i * npix, (i + 1) * npix),
            slice(j * npix, (j + 1) * npix))


def block_np(i, j, npix):
    return (slice(i * npix, (i + 1) * npix),
            slice(j * npix, (j + 1) * npix))

@njit
def idx_(i1, i2):
    # Assert that i1 is greater or equal to i2 according to our convention
    assert i1 >= i2
    return (i1 + 1) * i1 // 2 + i2

@njit
def K_func(idx, i1, i2):
    """
    Represents the sparse tensor K in P_alpha = KP_ell dexomposition
    """
    if i1 >= i2:
        idx_true = idx_(i1, i2)
    else:
        idx_true = idx_(i2, i1)
    return 1 if idx == idx_true else 0

@njit
def lower_tri_indices(n):
    count = n * (n + 1) // 2
    idxs = np.empty((count, 2), dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1):
            idxs[k, 0] = i
            idxs[k, 1] = j
            k += 1
    return idxs

@njit
def _trace_prod(A, B, imax, jmax):
    """Faster function to compute traces of products of the spherical harmonics arrays."""
    out = 0.0
    for i in range(2*imax+1):
        for j in range(2*jmax+1):
            out += A[i, j] * B[j, i]
    return out


@njit(parallel=True)
def _getF(F_idx, Nf, YCinvY, C_map):
    """
    This is the main routine for the Fisher. It can calculate both rediced and full one, dependent on C_map
    """
    n = F_idx.shape[0]
    F = np.zeros((n, n), dtype=C_map.dtype)
    
    pairs = lower_tri_indices(n)

    for index in prange(pairs.shape[0]):
        nF_i, nF_j = pairs[index]
        
        f1b, f1a, lbin1 = F_idx[nF_i]
        f2b, f2a, lbin2 = F_idx[nF_j]
        
        res = 0.0
        if (C_map[f1b,f2b]*C_map[f1a,f2a] == 0) and (C_map[f1b,f2a]*C_map[f1a,f2b] == 0): continue
        for i in range(Nf):
            for j in range(Nf):
                idx_1 = idx_(f1a, f1b)  
                if K_func(idx_1, i, j) == 0: continue
                for k in range(Nf):
                    map_jk = C_map[j,k]
                    if map_jk  == 0: continue
                    for l in range(Nf):
                        if C_map[l,i] == 0: continue
                        idx_2 = idx_(f2a, f2b)
                        if K_func(idx_2, k, l) == 0: continue
                                                                
                        e1 = YCinvY[(j,k)][lbin1, :, lbin2, :] if j >= k else YCinvY[(k,j)][lbin2, :, lbin1, :].T                                                  
                        e2 = YCinvY[(l,i)][lbin2, :, lbin1, :] if l >= i else YCinvY[(i,l)][lbin1, :, lbin2, :].T 
                        res += _trace_prod(e1,e2,lbin1,lbin2)
        
        F[nF_i, nF_j] = 0.5 * res
        F[nF_j, nF_i] = F[nF_i, nF_j]
    
    return F

def get_YCinvY(Y_r_all, Cinv, Cmap, Np): 
    constants = [0, 1]
    ops = Y_r_all, Y_r_all, (Np,Np)
    expr = oe.contract_expression('abm,cdn,mn->abcd', *ops, constants=constants)
    
    YCinvY = Dict.empty(key_type=types.UniTuple(types.int64, 2),
                        value_type = types.Array(types.float64, 4, 'C'))
    
    pairs = list(zip(*np.where(np.tril(Cmap!=0))))
    for i, j in tqdm(pairs,desc='YCinvY'):
        YCinvY[(i, j)] = expr(Cinv[block_np(i, j, Np)])
    return YCinvY   

    
def getF(Y_r_all, Cinv, F_idx, Nf, Np, C_map):
    YCinvY = get_YCinvY(Y_r_all,Cinv,C_map,Np)
    return _getF(F_idx, Nf, YCinvY, C_map)          


@njit(parallel=True)
def _get_y(F_idx, Cinv_rs_xy):
    Ntot = F_idx.shape[0]
    y_vec = np.zeros(Ntot)
    for n in prange(Ntot):
        f1,f2,l = F_idx[n]
        factor = 1 if f1==f2 else 0
        y_vec[n] = (2-factor)*np.dot(Cinv_rs_xy[f1,l],Cinv_rs_xy[f2,l])
    return y_vec
    
def get_y(x, Y_r_all,Cinv,F_idx,Nf,Np):
    Cinv_rs = Cinv.reshape(Nf, Np, Nf, Np).transpose(0, 2, 1, 3)
    Cinv_rs_xy = oe.contract('am,abmn,xyn->bxy',x,Cinv_rs,Y_r_all,optimize='optimal')
    y_vec = _get_y(F_idx, Cinv_rs_xy)
    return y_vec / 2
       