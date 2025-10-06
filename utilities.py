import numpy as np
import healpy as hp
from scipy.special import sph_harm_y, legendre_p_all

def sph_harm_y_real_all(lmax, theta, phi):
    """
    Returns real-valued spherical harmonics Y_lm^real for all l < lmax and -l <= m <= l,
    evaluated at (theta, phi). Output shape is (lmax+1, 2lmax+1, *theta.shape)
    with m index shifted to [0, 2lmax] where m = -l,...,l.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    shape = theta.shape
    out = np.zeros((lmax, 2*lmax-1) + shape, dtype=np.float64)
    ### TODO: rewrite vectorized to make it faster
    for l in range(lmax):
        for m in range(0, l+1):
            ylm = sph_harm_y(l, m, theta, phi)

            if m == 0:
                out[l, l] = ylm.real
            else:
                out[l, l + m] = np.sqrt(2) * ylm.real  # cos(mφ)
                out[l, l - m] = np.sqrt(2) * ylm.imag  # sin(mφ)
    return out

def get_Pl_ij(theta,phi,nside,lmax=None):
    cos_gamma = cosine_angle_matrix(theta,phi)
    if lmax is None:
        lmax=3*nside-1
    Pl_ij = legendre_p_all(lmax, cos_gamma)[0]*(2*np.arange(lmax+1)+1)[:,None,None]/4/np.pi
    return Pl_ij
    
def cosine_angle_matrix(theta, phi):
    """
    Compute matrix of cos(gamma_ij) between points (theta_i, phi_i) and (theta_j, phi_j)
    on the unit sphere.

    Parameters:
        theta : ndarray, shape (N,)
            Polar angles (colatitude, from 0 to pi)
        phi : ndarray, shape (N,)
            Azimuthal angles (longitude, from 0 to 2pi)

    Returns:
        cos_gamma : ndarray, shape (N, N)
            Matrix of cosines of angles between all pairs.
    """
    theta = theta[:, None]  # (N,1)
    phi = phi[:, None]      # (N,1)

    sin_theta = np.sin(theta)  # (N,1)
    cos_theta = np.cos(theta)  # (N,1)

    dphi = phi - phi.T  # (N,N)

    cos_gamma = sin_theta @ sin_theta.T * np.cos(dphi) + cos_theta @ cos_theta.T

    return cos_gamma

def theta_phi(nside):
    pix_idxs = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pix_idxs)
    return theta, phi


def construct_Z_and_pi(theta, phi, lmax, ell0):
    """
    Given arrays theta, phi of length n_pix (or shape e.g. (Nx,Ny) flattened below),
    and a real-sph-harm function sph_harm_y_real_all(lmax,theta,phi) returning an array
    of shape (lmax+1, 2*lmax+1, *theta.shape), this returns:

      Z  : (n_pix, ell0**2) matrix with Z.T @ Z = I
      pi : (n_pix, n_pix) projector that kills all modes ell<ell0

    Assumes trivial weighting (i.e. pixel–space inner product is just sum over pixels).
    """
    # 1) evaluate all real harmonics up to lmax
    Yall = sph_harm_y_real_all(lmax, theta, phi)
    # flatten pixel dims
    theta = np.asarray(theta)
    npix = theta.size

    # 2) build Y_small for ell=0,...,ell0-1 and m=-ell...+ell
    cols = []
    for ell in range(ell0):
        for m in range(-ell, ell+1):
            # m-shifted index = ell + m
            Y_ell_m = Yall[ell, ell + m].reshape(npix)
            cols.append(Y_ell_m)
    # stack into shape (npix, ell0**2)
    Y_small = np.vstack(cols).T

    # 3) orthonormalize columns of Y_small via reduced QR
    #    Q has shape (npix, ell0**2) with Q.T @ Q = I
    Q, R = np.linalg.qr(Y_small, mode='reduced')
    Z = Q

    # 4) build projector Pi = I - Z Z^T
    pi = np.eye(npix) - Z @ Z.T

    return Z, pi     