"""
Simulation 3: Finite-sample convergence

Compares empirical maxRCS with population maxRCS and pooled PCA across
different sample sizes.

Output: figures/sim3_finite_sample.png
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA

from minPCA import minPCA, generate_params, get_vars_pca
from utils import get_random_covs


# === Constants ===
SEED = 2
P = 20  # dimension
N_COMPONENTS = 5  # rank
N_ENVS = 5  # environments
N_SIMS = 25  # simulations per setting

SAMPLE_SIZES = [100, 250, 500, 1000, 2000, 5000, 10000]
HETEROGENEITY_LEVELS = [(0, 0.5), (0.5, 1), (1, 2), (2, 5)]

RESULTS_FILE = 'results/sim3_finite_sample.csv'
FIGURE_FILE = 'figures/sim3_finite_sample.png'


def run_simulation_single(sample_size, training_covs, n_components, rng, verbose=True):
    """Run a single finite-sample simulation."""
    if verbose:
        print(f'Running for n={sample_size}')

    mu = np.zeros(P)

    # Generate training data from covariances
    Xs = [
        rng.multivariate_normal(mu, cov, sample_size)
        for cov in training_covs
    ]
    X_pool = np.vstack(Xs)

    # Compute sample covariances
    covs = [np.cov(X.T) for X in Xs]
    cov_pool = np.cov(X_pool.T)

    # Fit PCA on pooled sample covariance
    _, _, V_T = np.linalg.svd(cov_pool)
    vpca = torch.tensor(V_T.T[:, :n_components]).float()

    # Fit minPCA on sample covariances
    model = minPCA(n_components=n_components, norm=True)
    model.fit(covs, n_restarts=10, lr=0.5, n_iters=1000)
    vminpca = model.v_

    out = []
    methods = [r'$\mathtt{PCA}$', r'$\mathtt{norm-minPCA}$']
    all_params = {
        'in-dist': (
            generate_params(Xs), 
            generate_params([X_pool])
        ),
        'ood': (
            generate_params(training_covs, from_cov=True),
            generate_params([np.mean(training_covs, axis=0)], from_cov=True)
        ),
    }
    for v, method in zip([vpca, vminpca], methods):
        for distr in ['in-dist', 'ood']:
            params, params_pooled = all_params[distr]
            var_wc, var_pool = get_vars_pca(v, params, params_pooled)
            out.append({
                'n': sample_size,
                'var_wc': var_wc,
                'var_pool': var_pool,
                'Method': method,
                'distr': distr,
            })

    return out


def run_simulation():
    """Run the full finite-sample simulation."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    results = []
    for a, b in HETEROGENEITY_LEVELS:
        for i in range(N_SIMS):
            # Generate training covariances
            training_covs = get_random_covs(
                P, N_COMPONENTS, N_ENVS, rng,
                a1=0.1, b1=1.0, a2=a, b2=b
            )
            avg_cov = np.mean(training_covs, axis=0)

            # Population results (oracle)
            _, _, V_T = np.linalg.svd(avg_cov)
            vpca = torch.tensor(V_T.T[:, :N_COMPONENTS]).float()

            model = minPCA(n_components=N_COMPONENTS, norm=True)
            model.fit(training_covs, n_restarts=10, lr=0.1, n_iters=100)
            vminpca = model.v_

            params = generate_params(training_covs, from_cov=True)
            params_pooled = generate_params([avg_cov], from_cov=True)

            var_wc_pca_pop, var_pool_pca_pop = get_vars_pca(
                vpca, params, params_pooled
            )
            var_wc_minpca_pop, var_pool_minpca_pop = get_vars_pca(
                vminpca, params, params_pooled
            )

            # Finite-sample results for each sample size
            for n in SAMPLE_SIZES:
                res = run_simulation_single(
                    n, training_covs, N_COMPONENTS, rng, verbose=True
                )
                for r in res:
                    r['sim'], r['a'], r['b'] = i, a, b
                    if r['Method'] == r'$\mathtt{PCA}$':
                        r['var_wc_pop'] = var_wc_pca_pop
                        r['var_pool_pop'] = var_pool_pca_pop
                    else:
                        r['var_wc_pop'] = var_wc_minpca_pop
                        r['var_pool_pop'] = var_pool_minpca_pop
                results += res

    return pd.DataFrame(results)


def compute_difference_df(df):
    """Compute difference metrics for plotting."""
    # Pivot to wide format by Method
    wide = df.pivot_table(
        index=['sim', 'n', 'a', 'b', 'distr'],
        columns='Method',
        values=['var_wc', 'var_wc_pop', 'var_pool_pop']
    ).reset_index()

    # Flatten column names
    wide.columns = [
        '_'.join(col).strip() if col[1] else col[0]
        for col in wide.columns.values
    ]

    diff_df = wide.copy()

    # Difference vs population maxRCS
    diff_df['var_wc_diff_vs_pop'] = (
        diff_df[r'var_wc_$\mathtt{norm-minPCA}$'] -
        diff_df[r'var_wc_pop_$\mathtt{norm-minPCA}$']
    )
    diff_df['rcs_wc_diff_vs_pop'] = -diff_df['var_wc_diff_vs_pop']

    # Difference vs pooled PCA
    diff_df['var_wc_diff_vs_pool'] = (
        diff_df[r'var_wc_$\mathtt{norm-minPCA}$'] -
        diff_df[r'var_wc_$\mathtt{PCA}$']
    )
    diff_df['rcs_wc_diff_vs_pool'] = -diff_df['var_wc_diff_vs_pool']

    return diff_df


def make_figure(df):
    """Create and save the figure."""
    plt.style.use('jmlr.mplstyle')

    diff_df = compute_difference_df(df)
    diff_df = diff_df[diff_df['n'] <= 5000]

    fig, ax = plt.subplots(1, 2, figsize=(6, 2.2))

    boxplot_kwargs = {
        'data': diff_df[diff_df['distr'] == 'ood'],
        'x': 'n',
        'color': "#2f85c3", #'#7AB', #"#a1d4f8", # 
        'fliersize': 2,
        'width': 0.65,
        'linewidth': 0.8,
    }

    # Left panel: empirical maxRCS vs population maxRCS
    sns.boxplot(ax=ax[0], y='rcs_wc_diff_vs_pop', **boxplot_kwargs)
    ax[0].axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax[0].set_ylabel(r'$\Delta$ worst-case RCS error')
    ax[0].set_xlabel('Sample size')
    ax[0].set_title(r'Empirical maxRCS vs.\ ' + '\n' + r'population maxRCS')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Right panel: empirical maxRCS vs empirical poolPCA
    sns.boxplot(ax=ax[1], y='rcs_wc_diff_vs_pool', **boxplot_kwargs)
    ax[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax[1].set_ylabel('\n'+r'$\Delta$ worst-case RCS error')
    ax[1].set_xlabel('Sample size')
    ax[1].set_title(r'Empirical maxRCS vs.\ ' + '\n' + r'empirical poolPCA')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Annotations
    ax[1].text(
        1.07, 0, r'$\rightarrow$ worse',
        transform=ax[1].get_yaxis_transform(),
        rotation=90, va='bottom', ha='left', fontsize=8
    )
    ax[1].text(
        1.07, 0, r'maxRCS better $\leftarrow$',
        transform=ax[1].get_yaxis_transform(),
        rotation=90, va='top', ha='left', fontsize=8
    )

    ax[0].set_ylim(-0.02, 0.075)
    ax[1].set_ylim(-0.14, 0.055)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'Saved figure to {FIGURE_FILE}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true',
                        help='Force rerun even if cached results exist')
    args = parser.parse_args()

    # Ensure directories exist
    Path('figures').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    # Run or load simulation
    if args.rerun or not os.path.exists(RESULTS_FILE):
        print('Running simulation...')
        df = run_simulation()
        df.to_csv(RESULTS_FILE, index=False)
        print(f'Saved results to {RESULTS_FILE}')
    else:
        print(f'Loading cached results from {RESULTS_FILE}')
        df = pd.read_csv(RESULTS_FILE)

    # Make figure
    make_figure(df)


if __name__ == '__main__':
    main()
