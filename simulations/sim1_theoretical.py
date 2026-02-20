"""
Simulation 1: Theoretical Result Visualization

Shows that maxRCS achieves a uniform bound on reconstruction error across all
test environments in the convex hull of training covariances.

Output: figures/sim_maxrcs_bound.png
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from minPCA import minPCA, generate_params, get_vars_pca
from utils import get_random_covs, sample_from_convex_hull


# === Constants ===
SEED = 123
P = 20  # dimension
N_COMPONENTS = 5  # rank
N_ENVS = 5  # training environments
N_TEST_ENVS = 50  # test environments from convex hull

RESULTS_FILE = 'results/sim1_theoretical.csv'
FIGURE_FILE = 'figures/sim_maxrcs_bound.png'


def run_simulation():
    """Run the theoretical result simulation."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Generate training covariances
    training_covs = get_random_covs(
        P, N_COMPONENTS, N_ENVS,
        a1=0.1, b1=1.0, a2=0.1, b2=1.0
    )
    avg_cov = np.mean(training_covs, axis=0)

    # Generate test covariances from convex hull
    test_covs = training_covs + sample_from_convex_hull(training_covs, N_TEST_ENVS)

    # Pooled PCA solution (SVD of average covariance)
    _, _, V_T = np.linalg.svd(avg_cov)
    vpca = torch.tensor(V_T.T[:, :N_COMPONENTS]).float()

    # maxRCS solution
    model = minPCA(n_components=N_COMPONENTS, norm=True)
    model.fit(training_covs, n_restarts=5, lr=0.1, n_iters=1500)
    vminpca = model.v_

    # Compute reconstruction error on test environments
    results = []
    for i, test_cov in enumerate(test_covs):
        params = generate_params([test_cov], from_cov=True)
        results.append({
            'var_pool': get_vars_pca(vpca, params)[0],
            'var_minpca': get_vars_pca(vminpca, params)[0],
            'i': i,
        })

    df = pd.DataFrame(results)

    # Store the minpca bound for plotting
    df['minpca_bound'] = model.minvar()

    return df


def make_figure(df):
    """Create and save the figure."""
    plt.style.use('jmlr.mplstyle')

    minpca_bound = df['minpca_bound'].iloc[0]

    plt.figure(figsize=(2.4, 2))
    for row in df.itertuples():
        plt.scatter(
            ['poolPCA', 'maxRCS'],
            [1 - row.var_pool, 1 - row.var_minpca],
            s=10, marker='o', c='#939393'
        )
        plt.plot(
            ['poolPCA', 'maxRCS'],
            [1 - row.var_pool, 1 - row.var_minpca],
            ':', linewidth=0.8, c='#939393'
        )

    plt.hlines(
        1 - minpca_bound, xmin=-0.1, xmax=1.1,
        linestyle='-', linewidth=2, color='blue',
        label=r'$m^*_\text{maxRCS}$'
    )

    plt.ylabel('Reconstruction error')
    plt.legend(loc='upper right')
    plt.xlim(-0.2, 1.2)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=400)
    plt.close()
    print(f'Saved figure to {FIGURE_FILE}')


def main():
    parser = argparse.ArgumentParser(description='Simulation 1: Theoretical result')
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
