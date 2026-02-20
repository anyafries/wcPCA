"""
Utility functions for comparison simulations.

Shared helper functions for FairPCA, minPCA, and StablePCA comparisons.
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import from simulations
_parent_dir = str(Path(__file__).resolve().parents[1])
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from simulations.utils import get_random_covs


def f_pca_np(v, cov_matrix, norm_cst):
    """
    Compute explained variance ratio for a single covariance.

    Parameters:
        v: projection matrix (p x k)
        cov_matrix: covariance matrix (p x p)
        norm_cst: normalization constant (typically trace of original cov)

    Returns:
        Explained variance ratio
    """
    p = cov_matrix.shape[0]
    v = v.reshape(p, -1)
    numerator = np.trace(v.T @ cov_matrix @ v)
    denominator = norm_cst
    return numerator / denominator


def f_minpca_np(v, covs, norm_csts):
    """
    Compute minimum explained variance across all covariances.

    Parameters:
        v: projection matrix (p x k)
        covs: list of covariance matrices
        norm_csts: list of normalization constants

    Returns:
        Minimum explained variance across all groups
    """
    p = covs[0].shape[0]
    v = v.reshape(p, -1)
    # check if v is orthogonal
    if not np.allclose(v.T @ v, np.eye(v.shape[1]), atol=1e-5):
        rank = v.shape[1]
        u, _, vh = np.linalg.svd(v)
        v = u[:, :rank] @ vh
    fs = [f_pca_np(v, covs[i], norm_csts[i]) for i in range(len(covs))]
    return min(fs)


def f_regret_np(v, covs, norm_csts):
    """
    Compute minimum regret across all covariances.

    Regret = explained variance - optimal explained variance for that group.

    Parameters:
        v: projection matrix (p x k)
        covs: list of covariance matrices
        norm_csts: list of normalization constants

    Returns:
        Minimum regret across all groups (negative = better than optimal is impossible)
    """
    p = covs[0].shape[0]
    v = v.reshape(p, -1)

    # check if v is orthogonal
    if not np.allclose(v.T @ v, np.eye(v.shape[1]), atol=1e-5):
        rank = v.shape[1]
        u, _, vh = np.linalg.svd(v)
        v = u[:, :rank] @ vh

    # get the best rank-k solution for each group
    # best solution is given by the top-k eigenvectors of the covariance matrix
    k = v.shape[1]
    best_val = [np.sum(np.sort(np.linalg.eigvalsh(covs[i]))[-k:]) for i in range(len(covs))]

    fs = [f_pca_np(v, covs[i], norm_csts[i]) - best_val[i] for i in range(len(covs))]
    return min(fs)


def get_V_from_X(X, rank=None):
    """
    Extract V such that X ≈ V @ V.T from SDP solution matrix.

    Parameters:
        X: symmetric positive semi-definite matrix (SDP solution)
        rank: desired rank of factorization

    Returns:
        V: matrix such that X ≈ V @ V.T
    """
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(X)
    # set tiny eigenvalues (order of e-6) to zero
    eigvals[eigvals < 1e-6] = 0

    if rank is not None and np.sum(eigvals != 0) != rank:
        print("WARNING: The rank of X is not equal to the specified rank.")

    # Sort eigenvalues in descending order
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if rank is not None:
        eigvals = eigvals[:rank]
        eigvecs = eigvecs[:, :rank]

    # Keep only positive eigenvalues
    pos_idx = eigvals > 0
    eigvals = eigvals[pos_idx]
    eigvecs = eigvecs[:, pos_idx]

    # Construct V
    V = eigvecs @ np.diag(np.sqrt(eigvals))
    return V


