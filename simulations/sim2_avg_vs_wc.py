"""
Simulation 2: Average vs worst-case comparison

Compares maxRCS and poolPCA on average and worst-case explained variance
across varying levels of heterogeneity.

Output: figures/sim2_avg_vs_wc.png
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch

from minPCA import minPCA, generate_params, get_vars_pca
from utils import get_random_covs


# === Constants ===
SEED = 2
P = 20  # dimension
N_COMPONENTS = 5  # rank
N_ENVS = 5  # training environments
N_REPETITIONS = 25  # repetitions per heterogeneity level

# Heterogeneity levels: (a2, b2) for env-specific eigenvalues
HETEROGENEITY_LEVELS = [(0, 0.5), (0.5, 1), (1, 2), (2, 5)]

RESULTS_FILE = 'results/sim2_avg_vs_wc.csv'
FIGURE_FILE = 'figures/sim2_avg_vs_wc'

def avg_vs_wc(n_components, avg_cov, covs):
    """
    Compare pooled PCA and maxRCS on average and worst-case explained variance.

    Parameters
    ----------
    n_components : int
        Number of components (rank)
    avg_cov : np.ndarray
        Average covariance matrix
    covs : list of np.ndarray
        List of covariance matrices

    Returns
    -------
    list of dict
        Results for average and worst-case metrics
    """
    # SVD of the average covariance (pooled PCA solution)
    _, _, V_T = np.linalg.svd(avg_cov)
    vpca = torch.tensor(V_T.T[:, :n_components]).float()

    # maxRCS solution
    model = minPCA(n_components=n_components, norm=True)
    model.fit(covs, n_restarts=5, lr=0.1, n_iters=1000)
    vminpca = model.v_

    params = generate_params(covs, from_cov=True)
    params_pooled = generate_params([avg_cov], from_cov=True)

    var_wc_pca, var_pool_pca = get_vars_pca(vpca, params, params_pooled)
    var_wc_minpca, var_pool_minpca = get_vars_pca(vminpca, params, params_pooled)

    out = []
    out.append({
        'metric': r'$\Delta$ average',
        'value': var_pool_minpca - var_pool_pca,
        'pca': var_pool_pca,
        'minpca': var_pool_minpca
    })
    out.append({
        'metric': r'$\Delta$ worst-case',
        'value': var_wc_minpca - var_wc_pca,
        'pca': var_wc_pca,
        'minpca': var_wc_minpca
    })
    return out


def run_simulation():
    """Run the average vs worst-case comparison simulation."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    results = []
    for a, b in HETEROGENEITY_LEVELS:
        for i in range(N_REPETITIONS):
            covs = get_random_covs(
                P, N_COMPONENTS, N_ENVS, rng,
                a1=0.1, b1=1.0, a2=a, b2=b
            )
            avg_cov = np.mean(covs, axis=0)
            res = avg_vs_wc(N_COMPONENTS, avg_cov, covs)
            for r in res:
                r['a'] = a
                r['b'] = b
                r['i'] = i
            results += res

    df = pd.DataFrame(results)
    return df


def compute_diff(df, relative=True):
    """Add relative difference column to dataframe."""
    df = df.copy()
    rel_diff = []
    for i in range(len(df) // 2):
        # Relative difference normalized by PCA's error (1 - var)
        nd1 = df.iloc[i * 2]['value']
        nd2 = df.iloc[i * 2 + 1]['value']
        if relative:
            nd1 = nd1 / (1 - df.iloc[i * 2]['pca']) * 100
            nd2 = nd2 / (1 - df.iloc[i * 2]['pca']) * 100
        rel_diff += [nd1, nd2]
    df['diff'] = rel_diff
    df['var'] = -np.array(rel_diff)
    return df


def make_figure(df, relative=True):
    """Create and save the figure."""
    plt.style.use('jmlr.mplstyle')

    df = compute_diff(df, relative=relative)
    column = 'diff'

    # Compute colormap range
    mmax = max(abs(df[column].max()), abs(df[column].min()))

    # Colormap for relative differences
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom',
        [(0, 'tab:green'),     # negative -> blue
         (0.5, '#939393'),  # zero -> gray
         (1, 'tab:red')]       # positive -> orange
    )
    norm = mcolors.Normalize(vmin=-mmax, vmax=mmax)

    fig, ax = plt.subplots(figsize=(2.8, 2))

    # Reference line at zero
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)

    # Lines connecting paired points
    cmap_lines = {a: '#939393' for a in df['a'].unique()}
    for i in range(len(df) // 2):
        delta_pool = df.iloc[i * 2][column]
        delta_wc = df.iloc[i * 2 + 1][column]
        ax.plot(
            [r'$\Delta$ average', r'$\Delta$ worst-case'],
            [-delta_pool, -delta_wc],
            ':', linewidth=0.8, zorder=1,
            c=cmap_lines[df.iloc[i * 2]['a']]
        )

    # Scatter plot with color-coded points
    ax.scatter(
        x=df['metric'], y=-df[column],
        c=-df[column], cmap=cmap, norm=norm, zorder=10, s=10
    )

    # Color y-tick labels
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    for tick_val, tick_label in zip(yticks, ax.get_yticklabels()):
        color = cmap(norm(tick_val))
        tick_label.set_color(color)

    ax.set_xlim(-0.2, 1.2)
    ylabel = 'Relative change in\nreconstruction error (\\%)' if relative else r'$\Delta$ reconstruction error'
    ax.set_ylabel(ylabel)

    # Add annotation
    ax.text(
        1.1, 0, 'maxRCS is\nbetter \u2190\u2192 worse',
        transform=ax.get_yaxis_transform(),
        rotation=90, va='center', ha='center', fontsize=8
    )

    plt.tight_layout()
    fig_file = FIGURE_FILE + ('_relative.png' if relative else '_absolute.png')
    plt.savefig(fig_file, dpi=400)
    plt.close()
    print(f'Saved figure to {fig_file}')


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
    make_figure(df, relative=True)
    make_figure(df, relative=False)


if __name__ == '__main__':
    main()
