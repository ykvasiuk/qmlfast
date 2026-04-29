import numpy as np
import numpy.typing as npt
from numba import njit, prange, types
from numba.typed import Dict
import opt_einsum as oe
from tqdm import tqdm

ArrayLike = npt.ArrayLike
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

@njit(inline='always')
def block(i: int, j: int, npix: int) -> tuple[slice, slice]:
    """
    Return slice indices for one field block in a square block matrix.

    Parameters
    ----------
    i, j
        Field-block row and column indices.
    npix
        Number of pixels in each field block.

    Returns
    -------
    tuple[slice, slice]
        Slices selecting the ``(i, j)`` block with shape ``(npix, npix)``.
    """
    return (slice(i * npix, (i + 1) * npix),
            slice(j * npix, (j + 1) * npix))


def block_np(i: int, j: int, npix: int) -> tuple[slice, slice]:
    """
    NumPy-compatible version of :func:`block`.

    Parameters
    ----------
    i, j
        Field-block row and column indices.
    npix
        Number of pixels in each field block.

    Returns
    -------
    tuple[slice, slice]
        Slices selecting the ``(i, j)`` block with shape ``(npix, npix)``.
    """
    return (slice(i * npix, (i + 1) * npix),
            slice(j * npix, (j + 1) * npix))

@njit
def idx_(i1: int, i2: int) -> int:
    """
    Pack a lower-triangular field pair into a one-dimensional index.

    Parameters
    ----------
    i1, i2
        Field indices with ``i1 >= i2``.

    Returns
    -------
    int
        Packed lower-triangular index ``i1 * (i1 + 1) // 2 + i2``.
    """
    assert i1 >= i2
    return (i1 + 1) * i1 // 2 + i2

@njit
def K_func(idx: int, i1: int, i2: int) -> int:
    """
    Evaluate the sparse field-pair selector used in the QML basis.

    Parameters
    ----------
    idx
        Packed lower-triangular field-pair index to test.
    i1, i2
        Candidate field indices. Ordering is ignored.

    Returns
    -------
    int
        ``1`` if ``idx`` matches the unordered pair ``(i1, i2)``, else ``0``.
    """
    if i1 >= i2:
        idx_true = idx_(i1, i2)
    else:
        idx_true = idx_(i2, i1)
    return 1 if idx == idx_true else 0

@njit
def lower_tri_indices(n: int) -> IntArray:
    """
    Return all lower-triangular matrix indices for an ``n x n`` matrix.

    Parameters
    ----------
    n
        Matrix side length.

    Returns
    -------
    ndarray of int64, shape (n * (n + 1) // 2, 2)
        Rows ``(i, j)`` with ``0 <= j <= i < n``.
    """
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
def _trace_prod(A: FloatArray, B: FloatArray, imax: int, jmax: int) -> float:
    """
    Trace ``A @ B`` for spherical-harmonic blocks with implicit ranks.

    Parameters
    ----------
    A
        Left block with at least shape ``(2 * imax + 1, 2 * jmax + 1)``.
    B
        Right block with at least shape ``(2 * jmax + 1, 2 * imax + 1)``.
    imax, jmax
        Multipole indices that determine the active ranks.

    Returns
    -------
    float
        ``trace(A @ B)`` over the active ranks.
    """
    out = 0.0
    for i in range(2*imax+1):
        for j in range(2*jmax+1):
            out += A[i, j] * B[j, i]
    return out

@njit
def _trace_prod_general(A: FloatArray, B: FloatArray, ri: int, rj: int) -> float:
    """
    Trace ``A @ B`` for blocks with explicit ranks.

    Parameters
    ----------
    A
        Left block with at least shape ``(ri, rj)``.
    B
        Right block with at least shape ``(rj, ri)``.
    ri, rj
        Active row/column ranks.

    Returns
    -------
    float
        ``trace(A @ B)`` over the active ranks.
    """
    out = 0.0
    for i in range(ri):
        for j in range(rj):
            out += A[i, j] * B[j, i]
    return out


@njit(parallel=True)
def _getF(F_idx: IntArray, Nf: int, YCinvY: object, C_map: FloatArray) -> FloatArray:
    """
    Compute the Fisher matrix from padded ``Y Cinv Y`` blocks.

    Parameters
    ----------
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``. Each row is
        ``(field_1, field_2, mode)`` with ``field_2 >= field_1``.
    Nf
        Number of fields.
    YCinvY
        Numba typed dict mapping field-pair tuples ``(i, j)`` with ``i >= j``
        to padded arrays with shape ``(n_modes, max_rank, n_modes, max_rank)``.
    C_map
        Float array of shape ``(Nf, Nf)``. Nonzero entries mark active
        covariance blocks.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers, n_bandpowers)
        Fisher matrix.
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
def _getF_general(
    F_idx: IntArray,
    Nf: int,
    YCinvY: object,
    C_map: FloatArray,
    ranks: IntArray,
) -> FloatArray:
    """
    Compute a Fisher matrix from padded blocks and explicit mode ranks.

    Parameters
    ----------
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``. Each row is
        ``(field_1, field_2, mode)`` with ``field_2 >= field_1``.
    Nf
        Number of fields.
    YCinvY
        Numba typed dict mapping field-pair tuples ``(i, j)`` with ``i >= j``
        to padded arrays with shape ``(n_modes, max_rank, n_modes, max_rank)``.
    C_map
        Float array of shape ``(Nf, Nf)``. Nonzero entries mark active
        covariance blocks.
    ranks
        Integer array of shape ``(n_modes,)``. ``ranks[m]`` is the active
        number of rank-one basis vectors for mode ``m``.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers, n_bandpowers)
        Fisher matrix.
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

def get_YCinvY(
    Y_r_all: FloatArray,
    Cinv: FloatArray,
    Cmap: FloatArray,
    Np: int,
) -> object:
    """
    Precompute padded ``Y @ Cinv_block @ Y.T`` arrays for active field pairs.

    Parameters
    ----------
    Y_r_all
        Float array of shape ``(n_modes, max_rank, Np)`` containing padded
        per-mode basis vectors.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``. Full inverse covariance.
    Cmap
        Float array of shape ``(Nf, Nf)``. Nonzero lower-triangular entries
        select field blocks to precompute.
    Np
        Number of pixels per field.

    Returns
    -------
    numba.typed.Dict
        Mapping ``(i, j)`` field pairs with ``i >= j`` to arrays of shape
        ``(n_modes, max_rank, n_modes, max_rank)``.
    """
    constants = [0, 1]
    ops = Y_r_all, Y_r_all, (Np,Np)
    expr = oe.contract_expression('abm,cdn,mn->abcd', *ops, constants=constants)
    
    YCinvY = Dict.empty(key_type=types.UniTuple(types.int64, 2),
                        value_type = types.Array(types.float64, 4, 'C'))
    
    pairs = list(zip(*np.where(np.tril(Cmap!=0))))
    for i, j in tqdm(pairs,desc='YCinvY'):
        YCinvY[(i, j)] = expr(Cinv[block_np(i, j, Np)])
    return YCinvY   

    
def getF(
    Y_r_all: FloatArray,
    Cinv: FloatArray,
    F_idx: IntArray,
    Nf: int,
    Np: int,
    C_map: FloatArray,
    ranks: IntArray | None = None,
    budget: int | None = None,
) -> FloatArray:
    """
    Compute the QML Fisher matrix.

    This convenience wrapper packs a padded basis array and dispatches to
    :func:`getF_packed` or :func:`getF_batched`.

    Parameters
    ----------
    Y_r_all
        Float array of shape ``(n_modes, max_rank, Np)``. Padded basis vectors,
        e.g. from ``sph_harm_y_real_all``.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``. Full inverse covariance.
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``. Each row is
        ``(field_1, field_2, mode)``.
    Nf
        Number of fields.
    Np
        Number of pixels per field.
    C_map
        Float array of shape ``(Nf, Nf)``. Nonzero entries mark active
        covariance blocks.
    ranks
        Optional integer array of shape ``(n_modes,)``. Active rank per mode.
        If omitted, spherical-harmonic ranks ``2 * ell + 1`` are assumed.
    budget
        Optional maximum number of field pairs per batch. If omitted, all
        ``V @ Cinv @ V.T`` blocks are precomputed at once.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers, n_bandpowers)
        Fisher matrix.
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

def pack_basis(V_list: list[FloatArray]) -> tuple[FloatArray, IntArray, IntArray]:
    """
    Pack per-mode basis-vector arrays into one contiguous basis matrix.

    Parameters
    ----------
    V_list
        List of float arrays. ``V_list[m]`` has shape ``(rank_m, Np)`` and
        contains the rank-one vectors for mode ``m``.

    Returns
    -------
    V_packed
        Float array of shape ``(sum(ranks), Np)`` containing all vectors
        concatenated along the rank axis.
    offsets
        Integer array of shape ``(n_modes,)``. ``offsets[m]`` is the starting
        row of mode ``m`` in ``V_packed``.
    ranks
        Integer array of shape ``(n_modes,)``. Rank of each mode.
    """
    ranks = np.array([v.shape[0] for v in V_list], dtype=np.int64)
    offsets = np.zeros(len(V_list), dtype=np.int64)
    offsets[1:] = np.cumsum(ranks[:-1])
    V_packed = np.vstack(V_list).astype(np.float64)
    return V_packed, offsets, ranks


def pack_sph_harm(Y_r_all: FloatArray) -> tuple[FloatArray, IntArray, IntArray]:
    """
    Pack a padded real spherical-harmonic basis.

    Parameters
    ----------
    Y_r_all
        Float array of shape ``(n_ell, 2 * n_ell - 1, Np)`` as returned by
        ``sph_harm_y_real_all``.

    Returns
    -------
    V_packed
        Float array of shape ``(n_ell ** 2, Np)`` with only active ``m`` rows.
    offsets
        Integer array of shape ``(n_ell,)``.
    ranks
        Integer array of shape ``(n_ell,)`` with ``ranks[ell] = 2 * ell + 1``.
    """
    lmax = Y_r_all.shape[0]
    V_list = [Y_r_all[l, :2*l+1, :] for l in range(lmax)]
    return pack_basis(V_list)


def get_VCinvV(
    V_packed: FloatArray,
    Cinv: FloatArray,
    Cmap: FloatArray,
    Np: int,
) -> object:
    """
    Precompute ``V_packed @ Cinv_block @ V_packed.T`` for field pairs.

    Parameters
    ----------
    V_packed
        Float array of shape ``(total_rank, Np)``.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``.
    Cmap
        Float array of shape ``(Nf, Nf)``. Nonzero lower-triangular entries
        select blocks to compute.
    Np
        Number of pixels per field.

    Returns
    -------
    numba.typed.Dict
        Mapping ``(i, j)`` field pairs with ``i >= j`` to float arrays of shape
        ``(total_rank, total_rank)``.
    """
    VCinvV = Dict.empty(key_type=types.UniTuple(types.int64, 2),
                        value_type=types.Array(types.float64, 2, 'C'))

    pairs = list(zip(*np.where(np.tril(Cmap != 0))))
    for i, j in tqdm(pairs, desc='VCinvV'):
        Cinv_block = Cinv[block_np(i, j, Np)]
        temp = V_packed @ Cinv_block       # (total_rank, Np)
        VCinvV[(i, j)] = temp @ V_packed.T # (total_rank, total_rank)
    return VCinvV


def get_noise_bias_packed_general(
    V_packed: FloatArray,
    offsets: IntArray,
    ranks: IntArray,
    Cinv: FloatArray,
    F_idx: IntArray,
    Nf: int,
    Np: int,
    noise: FloatArray,
) -> FloatArray:
    """
    Compute QML noise bias for a general dense pixel-noise covariance.

    Parameters
    ----------
    V_packed
        Float array of shape ``(total_rank, Np)``.
    offsets
        Integer array of shape ``(n_modes,)``. Starting row of each mode in
        ``V_packed``.
    ranks
        Integer array of shape ``(n_modes,)``. Active rank of each mode.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``. Inverse covariance.
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``. Rows are
        ``(field_1, field_2, mode)``.
    Nf
        Number of fields.
    Np
        Number of pixels per field.
    noise
        Float array of shape ``(Nf * Np, Nf * Np)``. Dense pixel-noise
        covariance.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers,)
        Additive QML noise bias for each row of ``F_idx``.
    """
    noise = np.asarray(noise, dtype=np.float64)
    if noise.shape != (Nf * Np, Nf * Np):
        raise ValueError("noise must have shape (Nf*Np, Nf*Np)")
    noise_weighted = Cinv @ noise @ Cinv
    C_map = np.ones((Nf, Nf), dtype=np.float64)
    VBV = get_VCinvV(V_packed, noise_weighted, C_map, Np)
    return _get_noise_bias_from_VBV(F_idx, VBV, offsets, ranks)


def get_noise_bias_packed_white(
    V_packed: FloatArray,
    offsets: IntArray,
    ranks: IntArray,
    Cinv: FloatArray,
    F_idx: IntArray,
    Nf: int,
    Np: int,
    noise_var: FloatArray,
) -> FloatArray:
    """
    Compute QML noise bias for per-field white pixel noise.

    Parameters
    ----------
    V_packed
        Float array of shape ``(total_rank, Np)``.
    offsets
        Integer array of shape ``(n_modes,)``. Starting row of each mode in
        ``V_packed``.
    ranks
        Integer array of shape ``(n_modes,)``. Active rank of each mode.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``. Inverse covariance.
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``. Rows are
        ``(field_1, field_2, mode)``.
    Nf
        Number of fields.
    Np
        Number of pixels per field.
    noise_var
        Float array of shape ``(Nf,)``. ``noise_var[g]`` is the white
        pixel-noise variance for field ``g``.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers,)
        Additive QML noise bias for each row of ``F_idx``.
    """
    noise_var = np.asarray(noise_var, dtype=np.float64)
    if noise_var.shape != (Nf,):
        raise ValueError("noise_var must have shape (Nf,)")

    projected = [[None for _ in range(Nf)] for _ in range(Nf)]
    for f in tqdm(range(Nf), desc='V M blocks'):
        for g in range(Nf):
            projected[f][g] = V_packed @ Cinv[block_np(f, g, Np)]

    bias = np.zeros(len(F_idx), dtype=np.float64)
    for a, (f1, f2, mode) in enumerate(tqdm(F_idx, desc='noise bias')):
        o = offsets[mode]
        r = ranks[mode]
        res = 0.0
        for g in range(Nf):
            left = projected[f1][g][o:o+r]
            right = projected[f2][g][o:o+r]
            res += noise_var[g] * np.sum(left * right)

        prefactor = 0.5 if f1 == f2 else 1.0
        bias[a] = prefactor * res

    return bias


def _get_noise_bias_from_VBV(
    F_idx: IntArray,
    VBV: object,
    offsets: IntArray,
    ranks: IntArray,
) -> FloatArray:
    """
    Trace packed ``V @ B @ V.T`` blocks into a QML noise-bias vector.

    Parameters
    ----------
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    VBV
        Numba typed dict mapping field-pair tuples to arrays with shape
        ``(total_rank, total_rank)``.
    offsets
        Integer array of shape ``(n_modes,)``.
    ranks
        Integer array of shape ``(n_modes,)``.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers,)
        Noise-bias vector.
    """
    bias = np.zeros(len(F_idx), dtype=np.float64)
    for a, (f1, f2, mode) in enumerate(tqdm(F_idx, desc='noise bias')):
        o = offsets[mode]
        r = ranks[mode]
        if f1 >= f2:
            block = VBV[(f1, f2)][o:o+r, o:o+r]
        else:
            block = VBV[(f2, f1)][o:o+r, o:o+r].T

        prefactor = 0.5 if f1 == f2 else 1.0
        bias[a] = prefactor * np.trace(block)

    return bias


@njit(parallel=True)
def _getF_packed(
    F_idx: IntArray,
    Nf: int,
    VCinvV: object,
    C_map: FloatArray,
    offsets: IntArray,
    ranks: IntArray,
) -> FloatArray:
    """
    Compute a Fisher matrix from packed ``V @ Cinv @ V.T`` blocks.

    Parameters
    ----------
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    Nf
        Number of fields.
    VCinvV
        Numba typed dict mapping field-pair tuples to arrays with shape
        ``(total_rank, total_rank)``.
    C_map
        Float array of shape ``(Nf, Nf)``. Nonzero entries mark active blocks.
    offsets
        Integer array of shape ``(n_modes,)``.
    ranks
        Integer array of shape ``(n_modes,)``.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers, n_bandpowers)
        Fisher matrix.
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


def getF_packed(
    V_packed: FloatArray,
    offsets: IntArray,
    ranks: IntArray,
    Cinv: FloatArray,
    F_idx: IntArray,
    Nf: int,
    Np: int,
    C_map: FloatArray,
) -> FloatArray:
    """
    Compute the QML Fisher matrix from packed basis vectors.

    Parameters
    ----------
    V_packed
        Float array of shape ``(total_rank, Np)``.
    offsets
        Integer array of shape ``(n_modes,)``. Starting row of each mode.
    ranks
        Integer array of shape ``(n_modes,)``. Active rank per mode.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``. Inverse covariance.
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    Nf
        Number of fields.
    Np
        Number of pixels per field.
    C_map
        Float array of shape ``(Nf, Nf)``. Nonzero entries mark active
        covariance blocks.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers, n_bandpowers)
        Fisher matrix.
    """
    VCinvV = get_VCinvV(V_packed, Cinv, C_map, Np)
    return _getF_packed(F_idx, Nf, VCinvV, C_map, offsets, ranks)


@njit(parallel=True)
def _getF_packed_batched(
    F_idx: IntArray,
    Nf: int,
    VCinvV: object,
    C_map: FloatArray,
    offsets: IntArray,
    ranks: IntArray,
    pair_batch: IntArray,
    ba: int,
    bb: int,
) -> FloatArray:
    """
    Compute Fisher contributions for one pair of field-pair batches.

    Parameters
    ----------
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    Nf
        Number of fields.
    VCinvV
        Numba typed dict containing the field pairs needed by this batch pair.
    C_map
        Float array of shape ``(Nf, Nf)``.
    offsets, ranks
        Integer arrays of shape ``(n_modes,)``.
    pair_batch
        Integer array of shape ``(Nf, Nf)`` mapping canonical field pairs to
        their batch index.
    ba, bb
        Batch indices to accumulate, with ``ba <= bb``.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers, n_bandpowers)
        Partial Fisher matrix for this batch pair.
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


def getF_batched(
    V_packed: FloatArray,
    offsets: IntArray,
    ranks: IntArray,
    Cinv: FloatArray,
    F_idx: IntArray,
    Nf: int,
    Np: int,
    C_map: FloatArray,
    budget: int,
) -> FloatArray:
    """
    Compute the QML Fisher matrix with batched ``VCinvV`` precomputation.

    Parameters
    ----------
    V_packed
        Float array of shape ``(total_rank, Np)``.
    offsets
        Integer array of shape ``(n_modes,)``.
    ranks
        Integer array of shape ``(n_modes,)``.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``.
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    Nf
        Number of fields.
    Np
        Number of pixels per field.
    C_map
        Float array of shape ``(Nf, Nf)``.
    budget
        Maximum number of active field pairs per batch. If ``budget`` is at
        least the total number of active pairs, this dispatches to
        :func:`getF_packed`.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers, n_bandpowers)
        Fisher matrix.
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


def profile_access(F_idx: IntArray, Nf: int, C_map: FloatArray) -> IntArray:
    """
    Count how often each packed field-pair block is accessed by Fisher loops.

    Parameters
    ----------
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    Nf
        Number of fields.
    C_map
        Float array of shape ``(Nf, Nf)``. Nonzero entries mark active blocks.

    Returns
    -------
    ndarray of int64, shape (Nf, Nf)
        ``counts[i, j]`` is the number of accesses to canonical pair
        ``(i, j)`` with ``i >= j``. Upper-triangular entries are unused.
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
def _get_y(F_idx: IntArray, Cinv_rs_xy: FloatArray) -> FloatArray:
    """
    Compute unnormalized QML quadratic forms from padded projected maps.

    Parameters
    ----------
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    Cinv_rs_xy
        Float array of shape ``(Nf, n_modes, max_rank)`` containing projected
        inverse-covariance-weighted maps.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers,)
        Quadratic estimator vector before the final ``1/2`` normalization.
    """
    Ntot = F_idx.shape[0]
    y_vec = np.zeros(Ntot)
    for n in prange(Ntot):
        f1,f2,l = F_idx[n]
        factor = 1 if f1==f2 else 0
        y_vec[n] = (2-factor)*np.dot(Cinv_rs_xy[f1,l],Cinv_rs_xy[f2,l])
    return y_vec

def get_y(
    x: FloatArray,
    Y_r_all: FloatArray,
    Cinv: FloatArray,
    F_idx: IntArray,
    Nf: int,
    Np: int,
    ranks: IntArray | None = None,
) -> FloatArray:
    """
    Compute the unnormalized QML quadratic-estimator vector.

    This convenience wrapper packs ``Y_r_all`` before calling
    :func:`get_y_packed`. For hot simulation loops, pre-pack the basis and call
    :func:`get_y_packed` directly.

    Parameters
    ----------
    x
        Float array of shape ``(Nf, Np)``. Pixel maps, one row per field.
    Y_r_all
        Float array of shape ``(n_modes, max_rank, Np)``. Padded basis vectors.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``. Inverse covariance.
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    Nf
        Number of fields.
    Np
        Number of pixels per field.
    ranks
        Optional integer array of shape ``(n_modes,)``. Active rank per mode.
        If omitted, spherical-harmonic ranks ``2 * ell + 1`` are assumed.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers,)
        Quadratic-estimator vector with the conventional final factor of
        ``1/2`` applied.
    """
    if ranks is None:
        V_packed, offsets, ranks = pack_sph_harm(Y_r_all)
    else:
        n_modes = Y_r_all.shape[0]
        V_list = [Y_r_all[i, :ranks[i], :] for i in range(n_modes)]
        V_packed, offsets, ranks = pack_basis(V_list)

    return get_y_packed(x, V_packed, offsets, ranks, Cinv, F_idx, Nf, Np)


@njit(parallel=True)
def _get_y_packed(
    F_idx: IntArray,
    w: FloatArray,
    offsets: IntArray,
    ranks: IntArray,
) -> FloatArray:
    """
    Compute packed QML quadratic forms from projected maps.

    Parameters
    ----------
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    w
        Float array of shape ``(Nf, total_rank)``. Packed basis projections of
        inverse-covariance-weighted maps.
    offsets
        Integer array of shape ``(n_modes,)``.
    ranks
        Integer array of shape ``(n_modes,)``.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers,)
        Quadratic estimator vector before the final ``1/2`` normalization.
    """
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


def get_y_packed(
    x: FloatArray,
    V_packed: FloatArray,
    offsets: IntArray,
    ranks: IntArray,
    Cinv: FloatArray,
    F_idx: IntArray,
    Nf: int,
    Np: int,
) -> FloatArray:
    """
    Compute the unnormalized QML quadratic-estimator vector from packed basis.

    Parameters
    ----------
    x
        Float array of shape ``(Nf, Np)``. Pixel maps, one row per field.
    V_packed
        Float array of shape ``(total_rank, Np)``.
    offsets
        Integer array of shape ``(n_modes,)``.
    ranks
        Integer array of shape ``(n_modes,)``.
    Cinv
        Float array of shape ``(Nf * Np, Nf * Np)``. Inverse covariance.
    F_idx
        Integer array of shape ``(n_bandpowers, 3)``.
    Nf
        Number of fields.
    Np
        Number of pixels per field.

    Returns
    -------
    ndarray of float64, shape (n_bandpowers,)
        Quadratic-estimator vector with the conventional final factor of
        ``1/2`` applied.
    """
    Cinv_rs = Cinv.reshape(Nf, Np, Nf, Np).transpose(0, 2, 1, 3)
    # u[b,n] = sum_{a,m} x[a,m] * Cinv[a,b,m,n]  — field contraction
    u = oe.contract('am,abmn->bn', x, Cinv_rs, optimize='optimal')
    # w[b,r] = sum_n u[b,n] * V[r,n]  — project onto packed basis
    w = u @ V_packed.T  # (Nf, total_rank) — single BLAS matmul
    y_vec = _get_y_packed(F_idx, w, offsets, ranks)
    return y_vec / 2
       
