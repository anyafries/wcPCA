"""
Utility functions for FLUXNET minPCA analysis.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA

from minPCA.minpca import minPCA, get_errs_pca, get_vars_pca, generate_params


def loo_time_split(
        Xpool_df_in,
        environments,
        Xs_dict,
        train_envs,
        test_envs,
        n_restarts=10,
        n_components=3,
    ):
    """
    Leave-one-out time split evaluation.

    Fits multiple PCA variants on training environments and evaluates on all environments.

    Parameters
    ----------
    Xpool_df_in : pd.DataFrame
        Pooled data (poolscale version)
    environments : pd.Series
        Environment labels for each row
    Xs_dict : dict
        Dictionary mapping environment names to torch tensors (envmeanzero version)
    train_envs : list
        List of training environment names
    test_envs : list
        List of test environment names
    n_restarts : int
        Number of restarts for minPCA optimization
    n_components : int
        Maximum number of components to compute

    Returns
    -------
    out_df : pd.DataFrame
        Results DataFrame with columns: environment, method, n_components, err, var, err_unnorm, var_unnorm
    pca_dict : dict
        Dictionary of fitted components for each method
    """
    # Training data
    Xs_train = []
    for env in train_envs:
        Xs_train.append(Xs_dict[env])
    covs = [torch.cov(X.T, correction=0) for X in Xs_train]

    Xpool = Xpool_df_in[environments.isin(train_envs)]
    Xpool_test = Xpool_df_in[environments.isin(test_envs)]
    Xpool = Xpool.to_numpy().astype(np.float32)
    Xpool_test = Xpool_test.to_numpy().astype(np.float32)
    Xpool_scaled = torch.cat(Xs_train, dim=0).cpu().numpy().astype(np.float32)

    # Fit PCA on pooled data
    pca = PCA(n_components=n_components)
    pca.fit(Xpool)
    pca_components = pca.components_.T

    # Fit PCA on pooled data with scaling
    pca_scaled = PCA(n_components=n_components)
    pca_scaled.fit(Xpool_scaled)
    pca_components_scaled = pca_scaled.components_.T

    # Fit PCA on average covariance
    avg_cov = torch.stack(covs).mean(dim=0).cpu().numpy()
    _, _, Vt = np.linalg.svd(avg_cov, full_matrices=False)
    components_avg_cov = Vt.T

    # Fit minPCA
    minpca_components = []
    for n in range(n_components):
        minpca = minPCA(n_components=n+1, norm=False)
        minpca = minpca.fit(covs, n_restarts=n_restarts)
        minpca_components.append(minpca.v_.detach().numpy())

    # Fit norm-minPCA
    normminpca_components = []
    for n in range(n_components):
        minpca = minPCA(n_components=n+1, norm=True)
        minpca = minpca.fit(covs, n_restarts=n_restarts)
        normminpca_components.append(minpca.v_.detach().numpy())

    # maxRCS
    maxrcs_components = []
    for n in range(n_components):
        minpca = minPCA(n_components=n+1, norm=False, function='maxrcs')
        minpca = minpca.fit(covs, n_restarts=n_restarts)
        maxrcs_components.append(minpca.v_.detach().numpy())

    # norm-maxRCS
    normmaxrcs_components = []
    for n in range(n_components):
        minpca = minPCA(n_components=n+1, norm=True, function='maxrcs')
        minpca = minpca.fit(covs, n_restarts=n_restarts)
        normmaxrcs_components.append(minpca.v_.detach().numpy())

    # Fit regret-variance minPCA
    regret_components = []
    for n in range(n_components):
        minpca = minPCA(n_components=n+1, norm=False, function='maxregret')
        minpca = minpca.fit(covs, n_restarts=n_restarts)
        regret_components.append(minpca.v_.detach().numpy())

    normregret_components = []
    for n in range(n_components):
        minpca = minPCA(n_components=n+1, norm=True, function='maxregret')
        minpca = minpca.fit(covs, n_restarts=n_restarts)
        normregret_components.append(minpca.v_.detach().numpy())

    # Collect all solutions
    pca_dict = {
        'PCA': [pca_components[:, :j+1] for j in range(n_components)],
        'scaledPCA': [pca_components_scaled[:, :j+1] for j in range(n_components)],
        'avgcovPCA': [components_avg_cov[:, :j+1] for j in range(n_components)],
        'minPCA': minpca_components,
        'norm-minPCA': normminpca_components,
        'maxRCS': maxrcs_components,
        'norm-maxRCS': normmaxrcs_components,
        'regret': regret_components,
        'norm-regret': normregret_components,
    }

    # Evaluate on training and testing environments
    out = []
    for env in train_envs + test_envs:
        X = Xs_dict[env]
        params = generate_params([X], norm=True)
        params_unnorm = generate_params([X], norm=False)

        for i, (name, components) in enumerate(pca_dict.items()):
            method_tag = name if env in test_envs else f"{name}_train"

            for j in range(n_components):
                v = torch.tensor(components[j], dtype=torch.float32)
                err, _ = get_errs_pca(v, params, from_cov=True)
                var, _ = get_vars_pca(v, params)
                err_unnorm, _ = get_errs_pca(v, params_unnorm, from_cov=True)
                var_unnorm, _ = get_vars_pca(v, params_unnorm)

                out.append({
                    'environment': env,
                    'method': method_tag,
                    'n_components': j+1,
                    'err': err,
                    'var': var,
                    'err_unnorm': err_unnorm,
                    'var_unnorm': var_unnorm,
                })

    # Also get the in-sample pooled error
    params_pooled = generate_params([Xpool], norm=True)
    params_pooled_test = generate_params([Xpool_test], norm=True)
    params_pooled_unnorm = generate_params([Xpool], norm=False)
    params_pooled_test_unnorm = generate_params([Xpool_test], norm=False)

    for i, (name, components) in enumerate(pca_dict.items()):
        for j in range(n_components):
            v = torch.tensor(components[j], dtype=torch.float32)
            err_train, _ = get_errs_pca(v, params_pooled, from_cov=True)
            err_test, _ = get_errs_pca(v, params_pooled_test, from_cov=True)
            var_train, _ = get_vars_pca(v, params_pooled)
            var_test, _ = get_vars_pca(v, params_pooled_test)
            err_train_unnorm, _ = get_errs_pca(v, params_pooled_unnorm, from_cov=True)
            err_test_unnorm, _ = get_errs_pca(v, params_pooled_test_unnorm, from_cov=True)
            var_train_unnorm, _ = get_vars_pca(v, params_pooled_unnorm)
            var_test_unnorm, _ = get_vars_pca(v, params_pooled_test_unnorm)

            out.append({
                'environment': 'pooled (in-sample)',
                'method': name,
                'n_components': j+1,
                'err': err_train,
                'var': var_train,
                'err_unnorm': err_train_unnorm,
                'var_unnorm': var_train_unnorm,
            })
            out.append({
                'environment': 'pooled (out-of-sample)',
                'method': name,
                'n_components': j+1,
                'err': err_test,
                'var': var_test,
                'err_unnorm': err_test_unnorm,
                'var_unnorm': var_test_unnorm,
            })

    return pd.DataFrame(out), pca_dict
