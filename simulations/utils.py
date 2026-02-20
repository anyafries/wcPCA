"""Shared utility functions for simulations."""

import numpy as np


def get_random_covs(p, rank, nenvs, a1=0.1, b1=1.0, a2=0.1, b2=1.0,
                    env_eigs_vary=False):
    """
    Generate nenvs random covariance matrices of size pxp
    with rank `rank` shared component and rank `rank` env-specific component.
    Eigenvalues of shared and env-specific components are preserved exactly
    if `env_eigs_vary=False`.

    Parameters
    ----------
    p : int
        The size of the covariance matrix
    rank : int
        The rank of each component (shared and env-specific)
    nenvs : int
        Number of environments
    a1, b1 : float
        Bounds for eigenvalues of the shared component
    a2, b2 : float
        Bounds for eigenvalues of the env-specific component
    env_eigs_vary : bool
        If True, env-specific eigenvalues vary across envs

    Returns
    -------
    list of np.ndarray
        List of nenvs covariance matrices, each normalized to trace 1
    """
    covs = []

    # 1. Generate shared eigenvalues
    eigvals_shared = np.random.uniform(a1, b1, rank)

    # 2. Generate env-specific eigenvalues (once if not varying)
    if not env_eigs_vary:
        eigvals_env = np.random.uniform(a2, b2, rank)

    for _ in range(nenvs):
        # 2. If env_eigs_vary, generate new env-specific eigenvalues
        if env_eigs_vary:
            eigvals_env = np.random.uniform(a2, b2, rank)

        # 3. Shared random orthonormal basis
        Q0, _ = np.linalg.qr(np.random.randn(p, rank))

        # 4. Environment-specific orthonormal basis, orthogonal to Q0
        Qi = np.random.randn(p, rank)
        # Make orthogonal to Q0
        Qi -= Q0 @ (Q0.T @ Qi)
        # Orthonormalize
        Qi, _ = np.linalg.qr(Qi)

        # 5. Stack into single orthonormal basis
        Qtilde = np.concatenate([Q0, Qi], axis=1)  # p x 2*rank
        Lambda_tilde = np.diag(np.concatenate([eigvals_shared, eigvals_env]))

        # 6. Construct covariance
        C = Qtilde @ Lambda_tilde @ Qtilde.T
        covs.append(C)

    return [C / np.linalg.trace(C) for C in covs]


def sample_from_convex_hull(item_list, n_samples):
    """
    Sample n_samples points uniformly from the convex hull of items.

    For each sample, draws weights from Dirichlet(1,...,1) (uniform on simplex)
    and returns the convex combination of items.

    Parameters
    ----------
    item_list : list of np.ndarray
        Items to form convex hull from
    n_samples : int
        Number of samples to draw

    Returns
    -------
    list of np.ndarray
        Sampled convex combinations
    """
    item_list = np.asarray(item_list)
    n_items = item_list.shape[0]
    samples = []

    for _ in range(n_samples):
        weights = np.random.gamma(shape=1.0, scale=1.0, size=n_items)  # Exp(1)
        weights /= weights.sum()
        samples.append(np.sum([w * item for w, item in zip(weights, item_list)], axis=0))

    return samples