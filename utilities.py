import numpy as np
import numpy.typing as npt
import healpy as hp
from scipy.special import sph_harm_y
from tqdm import tqdm

FloatArray = npt.NDArray[np.float64]


def sph_harm_y_real_all(
    lmax: int,
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    progress: bool = True,
    desc: str = 'Y_lm',
) -> FloatArray:
    """
    Evaluate real spherical harmonics on a set of angular coordinates.

    Parameters
    ----------
    lmax
        Number of multipoles to compute. The output contains ``ell = 0`` through
        ``ell = lmax - 1``.
    theta
        Array-like polar angles in radians. Shape ``pixel_shape``.
    phi
        Array-like azimuthal angles in radians. Must be broadcast-compatible
        with ``theta`` and is typically the same shape.
    progress
        If ``True``, show a progress bar over multipoles.
    desc
        Progress-bar label.

    Returns
    -------
    ndarray of float64
        Real harmonic values with shape ``(lmax, 2 * lmax - 1, *pixel_shape)``.
        For multipole ``ell``, active ``m`` values are stored at indices
        ``ell + m`` for ``-ell <= m <= ell``. Inactive entries are zero.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    shape = theta.shape
    out = np.zeros((lmax, 2*lmax-1) + shape, dtype=np.float64)
    ### TODO: rewrite vectorized to make it faster
    for l in tqdm(range(lmax), desc=desc, disable=not progress):
        for m in range(0, l+1):
            ylm = sph_harm_y(l, m, theta, phi)

            if m == 0:
                out[l, l] = ylm.real
            else:
                out[l, l + m] = np.sqrt(2) * ylm.real  # cos(mφ)
                out[l, l - m] = np.sqrt(2) * ylm.imag  # sin(mφ)
    return out

def get_Pl_ij(
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    nside: int,
    lmax: int | None = None,
) -> FloatArray:
    """
    Compute pixel-space Legendre kernels for all multipoles up to ``lmax``.

    Parameters
    ----------
    theta
        Array-like polar angles in radians with shape ``(Np,)``.
    phi
        Array-like azimuthal angles in radians with shape ``(Np,)``.
    nside
        HEALPix ``nside``. Used only to choose the default ``lmax``.
    lmax
        Maximum multipole to include. If omitted, uses ``3 * nside - 1``.

    Returns
    -------
    ndarray of float64, shape (lmax + 1, Np, Np)
        Kernels ``(2 ell + 1) P_ell(cos gamma_ij) / (4 pi)``.
    """
    cos_gamma = cosine_angle_matrix(theta, phi)
    if lmax is None:
        lmax = 3 * nside - 1

    Pl_ij = np.empty((lmax + 1, cos_gamma.shape[0], cos_gamma.shape[1]),
                     dtype=np.float64)
    Plm2 = np.ones_like(cos_gamma)

    for ell in tqdm(range(lmax + 1), desc='Pl_ij'):
        if ell == 0:
            P_ell = Plm2
        elif ell == 1:
            P_ell = cos_gamma.copy()
            Plm1 = P_ell
        else:
            P_ell = ((2 * ell - 1) * cos_gamma * Plm1 - (ell - 1) * Plm2) / ell
            Plm2, Plm1 = Plm1, P_ell

        Pl_ij[ell] = (2 * ell + 1) * P_ell / (4 * np.pi)

    return Pl_ij
    
def cosine_angle_matrix(theta: npt.ArrayLike, phi: npt.ArrayLike) -> FloatArray:
    """
    Compute pairwise angular cosines for points on the unit sphere.

    Parameters
    ----------
    theta
        Array-like polar angles in radians with shape ``(Np,)``.
    phi
        Array-like azimuthal angles in radians with shape ``(Np,)``.

    Returns
    -------
    ndarray of float64, shape (Np, Np)
        Matrix whose ``(i, j)`` entry is ``cos(gamma_ij)``.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    theta = theta[:, None]  # (N,1)
    phi = phi[:, None]      # (N,1)

    sin_theta = np.sin(theta)  # (N,1)
    cos_theta = np.cos(theta)  # (N,1)

    dphi = phi - phi.T  # (N,N)

    cos_gamma = sin_theta @ sin_theta.T * np.cos(dphi) + cos_theta @ cos_theta.T

    return cos_gamma

def theta_phi(nside: int) -> tuple[FloatArray, FloatArray]:
    """
    Return HEALPix pixel-center angular coordinates.

    Parameters
    ----------
    nside
        HEALPix ``nside`` parameter.

    Returns
    -------
    theta
        Float array of shape ``(12 * nside**2,)`` with polar angles in radians.
    phi
        Float array of shape ``(12 * nside**2,)`` with azimuthal angles in
        radians.
    """
    pix_idxs = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pix_idxs)
    return theta, phi


def low_ell_mode_matrix(theta: npt.ArrayLike, phi: npt.ArrayLike, ell0: int) -> FloatArray:
    """
    Build raw low-ell real spherical-harmonic mode columns.

    Parameters
    ----------
    theta
        Array-like polar angles in radians with shape ``(Np,)``.
    phi
        Array-like azimuthal angles in radians with shape ``(Np,)``.
    ell0
        Deproject all modes with ``ell < ell0``.

    Returns
    -------
    ndarray of float64, shape (Np, ell0**2)
        Matrix with one non-orthogonal real spherical-harmonic column per
        ``(ell, m)`` mode for ``ell < ell0``.
    """
    Yall = sph_harm_y_real_all(ell0, theta, phi)
    npix = np.asarray(theta).size

    cols = []
    for ell in range(ell0):
        for m in range(-ell, ell + 1):
            cols.append(Yall[ell, ell + m].reshape(npix))
    return np.vstack(cols).T


def deproject_inverse_woodbury(Cinv: FloatArray, Z: FloatArray) -> FloatArray:
    """
    Project modes out of an inverse covariance using the Woodbury limit.

    Parameters
    ----------
    Cinv
        Float array of shape ``(N, N)``. Inverse covariance before
        deprojection.
    Z
        Float array of shape ``(N, n_modes)``. Columns span the modes to
        deproject. Columns need not be orthonormal.

    Returns
    -------
    ndarray of float64, shape (N, N)
        Symmetrized inverse covariance with the column space of ``Z`` projected
        out:
        ``Cinv - Cinv @ Z @ inv(Z.T @ Cinv @ Z) @ Z.T @ Cinv``.
    """
    CinvZ = Cinv @ Z
    gram = Z.T @ CinvZ
    projected = Cinv - CinvZ @ np.linalg.solve(gram, CinvZ.T)
    return 0.5 * (projected + projected.T)


def construct_Z_and_pi(
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    lmax: int,
    ell0: int,
) -> tuple[FloatArray, FloatArray]:
    """
    Construct orthonormal low-ell modes and their pixel-space projector.

    Parameters
    ----------
    theta
        Array-like polar angles in radians with shape ``(Np,)``.
    phi
        Array-like azimuthal angles in radians with shape ``(Np,)``.
    lmax
        Kept for backward-compatible call signatures. The current
        implementation only needs ``ell0`` because it constructs modes
        ``ell < ell0``.
    ell0
        Project out all real spherical-harmonic modes with ``ell < ell0``.

    Returns
    -------
    Z
        Float array of shape ``(Np, ell0**2)``. Orthonormal columns spanning
        the low-ell mode subspace.
    pi
        Float array of shape ``(Np, Np)``. Orthogonal projector
        ``I - Z @ Z.T``.

    Notes
    -----
    The orthonormalization uses the unweighted Euclidean pixel-space inner
    product.
    """
    theta = np.asarray(theta)
    npix = theta.size

    # 1) build raw real harmonics for ell=0,...,ell0-1 and m=-ell...+ell
    Y_small = low_ell_mode_matrix(theta, phi, ell0)

    # 2) orthonormalize columns of Y_small via reduced QR
    #    Q has shape (npix, ell0**2) with Q.T @ Q = I
    Q, R = np.linalg.qr(Y_small, mode='reduced')
    Z = Q

    # 3) build projector Pi = I - Z Z^T
    pi = np.eye(npix) - Z @ Z.T

    return Z, pi     
