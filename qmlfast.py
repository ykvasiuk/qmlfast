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

@njit
def _trace_prod_general(A, B, ri, rj):
    """Trace of product for arbitrary rank-one decompositions with explicit ranks."""
    out = 0.0
    for i in range(ri):
        for j in range(rj):
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

@njit(parallel=True)
def _getF_general(F_idx, Nf, YCinvY, C_map, ranks):
    """
    Generalized Fisher matrix routine for arbitrary rank-one decompositions.
    Instead of assuming 2*l+1 components per mode (spherical harmonics),
    uses an explicit ranks array where ranks[i] gives the number of
    rank-one components for the i-th basis element.
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
                        res += _trace_prod_general(e1, e2, ranks[lbin1], ranks[lbin2])

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

    
def getF(Y_r_all, Cinv, F_idx, Nf, Np, C_map, ranks=None, budget=None):
    """
    Compute the Fisher matrix.

    Internally uses packed storage for efficiency (no zero-padding).

    Parameters:
        Y_r_all: (n_modes, max_rank, npix) basis array (e.g. from sph_harm_y_real_all).
                 Zero-padded if modes have different ranks.
        Cinv:    (Nf*Np, Nf*Np) inverse covariance matrix.
        F_idx:   (N, 3) array of (field1, field2, mode) indices.
        Nf:      number of fields.
        Np:      number of pixels.
        C_map:   (Nf, Nf) block structure map.
        ranks:   optional int64 array of actual ranks per mode.
                 If None, assumes spherical harmonics (ranks[l] = 2*l+1).
        budget:  optional int, max field pairs per batch for memory-budgeted
                 computation. If None, precomputes all VCinvV at once.
    """
    if ranks is None:
        V_packed, offsets, ranks = pack_sph_harm(Y_r_all)
    else:
        n_modes = Y_r_all.shape[0]
        V_list = [Y_r_all[i, :ranks[i], :] for i in range(n_modes)]
        V_packed, offsets, ranks = pack_basis(V_list)

    if budget is not None:
        return getF_batched(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map, budget)
    else:
        return getF_packed(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map)


### Packed storage routines — memory-efficient, no zero-padding ###

def pack_basis(V_list):
    """
    Pack a list of per-mode basis vector arrays into a single 2D array.

    Parameters:
        V_list: list of arrays, V_list[i] has shape (ranks[i], npix)

    Returns:
        V_packed: shape (total_rank, npix), all vectors concatenated
        offsets:  int64 array of length n_modes, offsets[i] = sum(ranks[0:i])
        ranks:    int64 array of length n_modes
    """
    ranks = np.array([v.shape[0] for v in V_list], dtype=np.int64)
    offsets = np.zeros(len(V_list), dtype=np.int64)
    offsets[1:] = np.cumsum(ranks[:-1])
    V_packed = np.vstack(V_list).astype(np.float64)
    return V_packed, offsets, ranks


def pack_sph_harm(Y_r_all):
    """
    Convert a padded spherical harmonics array to packed format.

    Parameters:
        Y_r_all: shape (lmax, 2*lmax-1, npix) from sph_harm_y_real_all

    Returns:
        V_packed, offsets, ranks (see pack_basis)
    """
    lmax = Y_r_all.shape[0]
    V_list = [Y_r_all[l, :2*l+1, :] for l in range(lmax)]
    return pack_basis(V_list)


def get_VCinvV(V_packed, Cinv, Cmap, Np):
    """
    Memory-efficient VCinvV computation using packed basis vectors.

    V_packed @ Cinv_block @ V_packed.T for each field pair — two BLAS calls,
    no einsum, no zero-padding overhead.

    Parameters:
        V_packed: shape (total_rank, npix)
        Cinv:     full inverse covariance, shape (Nf*Np, Nf*Np)
        Cmap:     field block map
        Np:       number of pixels

    Returns:
        dict mapping (i, j) -> 2D array of shape (total_rank, total_rank)
    """
    VCinvV = Dict.empty(key_type=types.UniTuple(types.int64, 2),
                        value_type=types.Array(types.float64, 2, 'C'))

    pairs = list(zip(*np.where(np.tril(Cmap != 0))))
    for i, j in tqdm(pairs, desc='VCinvV'):
        Cinv_block = Cinv[block_np(i, j, Np)]
        temp = V_packed @ Cinv_block       # (total_rank, Np)
        VCinvV[(i, j)] = temp @ V_packed.T # (total_rank, total_rank)
    return VCinvV


@njit(parallel=True)
def _getF_packed(F_idx, Nf, VCinvV, C_map, offsets, ranks):
    """
    Fisher matrix using packed (offset-indexed) VCinvV storage.
    Equivalent to _getF but without zero-padding overhead.
    """
    n = F_idx.shape[0]
    F = np.zeros((n, n), dtype=C_map.dtype)

    pairs = lower_tri_indices(n)

    for index in prange(pairs.shape[0]):
        nF_i, nF_j = pairs[index]

        f1b, f1a, lbin1 = F_idx[nF_i]
        f2b, f2a, lbin2 = F_idx[nF_j]

        res = 0.0
        if (C_map[f1b,f2b]*C_map[f1a,f2a] == 0) and (C_map[f1b,f2a]*C_map[f1a,f2b] == 0):
            continue

        o1, r1 = offsets[lbin1], ranks[lbin1]
        o2, r2 = offsets[lbin2], ranks[lbin2]

        for i in range(Nf):
            for j in range(Nf):
                idx_1 = idx_(f1a, f1b)
                if K_func(idx_1, i, j) == 0: continue
                for k in range(Nf):
                    if C_map[j,k] == 0: continue
                    for l in range(Nf):
                        if C_map[l,i] == 0: continue
                        idx_2 = idx_(f2a, f2b)
                        if K_func(idx_2, k, l) == 0: continue

                        if j >= k:
                            e1 = VCinvV[(j,k)][o1:o1+r1, o2:o2+r2]
                        else:
                            e1 = VCinvV[(k,j)][o2:o2+r2, o1:o1+r1].T

                        if l >= i:
                            e2 = VCinvV[(l,i)][o2:o2+r2, o1:o1+r1]
                        else:
                            e2 = VCinvV[(i,l)][o1:o1+r1, o2:o2+r2].T

                        res += _trace_prod_general(e1, e2, r1, r2)

        F[nF_i, nF_j] = 0.5 * res
        F[nF_j, nF_i] = F[nF_i, nF_j]

    return F


def getF_packed(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map):
    """
    Fisher matrix using packed basis vectors — memory-efficient, no zero-padding.

    Usage with spherical harmonics:
        V_packed, offsets, ranks = pack_sph_harm(Y_r_all)
        F = getF_packed(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map)

    Usage with arbitrary rank-one decomposition:
        V_packed, offsets, ranks = pack_basis([V0, V1, V2, ...])
        F = getF_packed(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map)
    """
    VCinvV = get_VCinvV(V_packed, Cinv, C_map, Np)
    return _getF_packed(F_idx, Nf, VCinvV, C_map, offsets, ranks)


@njit(parallel=True)
def _getF_packed_batched(F_idx, Nf, VCinvV, C_map, offsets, ranks,
                         pair_batch, ba, bb):
    """
    Fisher matrix contributions from a single batch-pair (ba, bb).
    Only accumulates terms where the e1 field pair belongs to one batch
    and the e2 field pair belongs to the other (or same if ba == bb),
    with canonical ordering min/max to avoid double-counting.
    """
    n = F_idx.shape[0]
    F = np.zeros((n, n), dtype=C_map.dtype)

    pairs = lower_tri_indices(n)

    for index in prange(pairs.shape[0]):
        nF_i, nF_j = pairs[index]

        f1b, f1a, lbin1 = F_idx[nF_i]
        f2b, f2a, lbin2 = F_idx[nF_j]

        res = 0.0
        if (C_map[f1b,f2b]*C_map[f1a,f2a] == 0) and \
           (C_map[f1b,f2a]*C_map[f1a,f2b] == 0):
            continue

        o1, r1 = offsets[lbin1], ranks[lbin1]
        o2, r2 = offsets[lbin2], ranks[lbin2]

        for i in range(Nf):
            for j in range(Nf):
                idx_1 = idx_(f1a, f1b)
                if K_func(idx_1, i, j) == 0: continue
                for k in range(Nf):
                    if C_map[j,k] == 0: continue
                    for l in range(Nf):
                        if C_map[l,i] == 0: continue
                        idx_2 = idx_(f2a, f2b)
                        if K_func(idx_2, k, l) == 0: continue

                        # Canonical field pairs for e1 and e2
                        e1a = j if j >= k else k
                        e1b = k if j >= k else j
                        e2a = l if l >= i else i
                        e2b = i if l >= i else l

                        # Check batch assignment — only accumulate if
                        # (min_batch, max_batch) == (ba, bb)
                        b_e1 = pair_batch[e1a, e1b]
                        b_e2 = pair_batch[e2a, e2b]
                        lo = b_e1 if b_e1 <= b_e2 else b_e2
                        hi = b_e2 if b_e1 <= b_e2 else b_e1
                        if lo != ba or hi != bb:
                            continue

                        if j >= k:
                            e1 = VCinvV[(j,k)][o1:o1+r1, o2:o2+r2]
                        else:
                            e1 = VCinvV[(k,j)][o2:o2+r2, o1:o1+r1].T

                        if l >= i:
                            e2 = VCinvV[(l,i)][o2:o2+r2, o1:o1+r1]
                        else:
                            e2 = VCinvV[(i,l)][o1:o1+r1, o2:o2+r2].T

                        res += _trace_prod_general(e1, e2, r1, r2)

        F[nF_i, nF_j] = 0.5 * res
        F[nF_j, nF_i] = F[nF_i, nF_j]

    return F


def getF_batched(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map, budget):
    """
    Memory-efficient Fisher matrix with batched VCinvV computation.

    Splits active field pairs into batches of size `budget`. For each
    pair of batches, computes the needed VCinvV matrices, accumulates
    Fisher contributions, then frees them.

    Peak memory: at most 2 * budget VCinvV matrices simultaneously.

    Parameters:
        budget: max field pairs per batch. If >= total active pairs,
                falls back to getF_packed (precompute everything).
    """
    import math

    active_pairs = [(int(i), int(j))
                    for i, j in zip(*np.where(np.tril(C_map != 0)))]
    n_pairs = len(active_pairs)

    if budget >= n_pairs:
        return getF_packed(V_packed, offsets, ranks, Cinv, F_idx, Nf, Np, C_map)

    # Split into batches
    n_batches = math.ceil(n_pairs / budget)
    batches = [active_pairs[i * budget:(i + 1) * budget]
               for i in range(n_batches)]

    # Batch assignment lookup: pair_batch[i, j] = batch index (i >= j)
    pair_batch = np.full((Nf, Nf), -1, dtype=np.int64)
    for batch_idx, batch in enumerate(batches):
        for (i, j) in batch:
            pair_batch[i, j] = batch_idx

    n = F_idx.shape[0]
    F_total = np.zeros((n, n), dtype=C_map.dtype)

    total_iters = n_batches * (n_batches + 1) // 2
    with tqdm(total=total_iters, desc='batched F') as pbar:
        for ba in range(n_batches):
            for bb in range(ba, n_batches):
                # Compute VCinvV only for pairs in batches[ba] ∪ batches[bb]
                pairs_needed = list(set(batches[ba]) | set(batches[bb]))

                VCinvV = Dict.empty(
                    key_type=types.UniTuple(types.int64, 2),
                    value_type=types.Array(types.float64, 2, 'C'))

                for (pi, pj) in pairs_needed:
                    Cinv_block = Cinv[block_np(pi, pj, Np)]
                    temp = V_packed @ Cinv_block
                    VCinvV[(pi, pj)] = temp @ V_packed.T

                F_total += _getF_packed_batched(
                    F_idx, Nf, VCinvV, C_map, offsets, ranks,
                    pair_batch, ba, bb)

                pbar.update(1)

    return F_total


def profile_access(F_idx, Nf, C_map):
    """
    Dry-run the Fisher loop logic and count how many times each
    VCinvV field pair (i, j) is accessed (stored as i >= j).

    Returns:
        counts: (Nf, Nf) array where counts[i, j] (i >= j) is the
                number of times VCinvV[(i, j)] is looked up.
    """
    counts = np.zeros((Nf, Nf), dtype=np.int64)
    n = F_idx.shape[0]

    for nF_i in range(n):
        for nF_j in range(nF_i + 1):
            f1b, f1a, lbin1 = F_idx[nF_i]
            f2b, f2a, lbin2 = F_idx[nF_j]

            if (C_map[f1b,f2b]*C_map[f1a,f2a] == 0) and \
               (C_map[f1b,f2a]*C_map[f1a,f2b] == 0):
                continue

            for i in range(Nf):
                for j in range(Nf):
                    idx_1 = (f1a + 1) * f1a // 2 + f1b if f1a >= f1b else (f1b + 1) * f1b // 2 + f1a
                    ij_idx = (i + 1) * i // 2 + j if i >= j else (j + 1) * j // 2 + i
                    if ij_idx != idx_1:
                        continue
                    for k in range(Nf):
                        if C_map[j, k] == 0:
                            continue
                        for l in range(Nf):
                            if C_map[l, i] == 0:
                                continue
                            kl_idx = (f2a + 1) * f2a // 2 + f2b if f2a >= f2b else (f2b + 1) * f2b // 2 + f2a
                            kl_test = (k + 1) * k // 2 + l if k >= l else (l + 1) * l // 2 + k
                            if kl_test != kl_idx:
                                continue

                            # e1 access
                            a, b = (j, k) if j >= k else (k, j)
                            counts[a, b] += 1
                            # e2 access
                            a, b = (l, i) if l >= i else (i, l)
                            counts[a, b] += 1

    return counts


@njit(parallel=True)
def _get_y(F_idx, Cinv_rs_xy):
    Ntot = F_idx.shape[0]
    y_vec = np.zeros(Ntot)
    for n in prange(Ntot):
        f1,f2,l = F_idx[n]
        factor = 1 if f1==f2 else 0
        y_vec[n] = (2-factor)*np.dot(Cinv_rs_xy[f1,l],Cinv_rs_xy[f2,l])
    return y_vec

def get_y(x, Y_r_all, Cinv, F_idx, Nf, Np, ranks=None):
    """
    Compute the QML estimator vector.

    Internally uses packed storage for efficiency.

    For hot loops (many simulations), prefer get_y_packed directly
    with pre-packed basis to avoid repacking each call.

    Parameters:
        x:       (Nf, Np) data vector (pixel maps per field).
        Y_r_all: (n_modes, max_rank, npix) basis array.
        Cinv:    (Nf*Np, Nf*Np) inverse covariance matrix.
        F_idx:   (N, 3) array of (field1, field2, mode) indices.
        Nf:      number of fields.
        Np:      number of pixels.
        ranks:   optional int64 array of actual ranks per mode.
                 If None, assumes spherical harmonics (ranks[l] = 2*l+1).
    """
    if ranks is None:
        V_packed, offsets, ranks = pack_sph_harm(Y_r_all)
    else:
        n_modes = Y_r_all.shape[0]
        V_list = [Y_r_all[i, :ranks[i], :] for i in range(n_modes)]
        V_packed, offsets, ranks = pack_basis(V_list)

    return get_y_packed(x, V_packed, offsets, ranks, Cinv, F_idx, Nf, Np)


@njit(parallel=True)
def _get_y_packed(F_idx, w, offsets, ranks):
    Ntot = F_idx.shape[0]
    y_vec = np.zeros(Ntot)
    for n in prange(Ntot):
        f1, f2, mode = F_idx[n]
        o = offsets[mode]
        r = ranks[mode]
        s = 0.0
        for i in range(r):
            s += w[f1, o + i] * w[f2, o + i]
        factor = 1 if f1 == f2 else 0
        y_vec[n] = (2 - factor) * s
    return y_vec


def get_y_packed(x, V_packed, offsets, ranks, Cinv, F_idx, Nf, Np):
    """
    Packed version of get_y — no zero-padding overhead.

    Parameters:
        x:        (Nf, Np) data vector (pixel maps per field)
        V_packed: (total_rank, Np) packed basis vectors
        offsets:  int64 array, offsets[i] = cumulative sum of ranks[0:i]
        ranks:    int64 array, ranks[i] = number of components for mode i
        Cinv:     (Nf*Np, Nf*Np) inverse covariance
        F_idx:    (N, 3) array of (field1, field2, mode) indices
        Nf:       number of fields
        Np:       number of pixels
    """
    Cinv_rs = Cinv.reshape(Nf, Np, Nf, Np).transpose(0, 2, 1, 3)
    # u[b,n] = sum_{a,m} x[a,m] * Cinv[a,b,m,n]  — field contraction
    u = oe.contract('am,abmn->bn', x, Cinv_rs, optimize='optimal')
    # w[b,r] = sum_n u[b,n] * V[r,n]  — project onto packed basis
    w = u @ V_packed.T  # (Nf, total_rank) — single BLAS matmul
    y_vec = _get_y_packed(F_idx, w, offsets, ranks)
    return y_vec / 2
       