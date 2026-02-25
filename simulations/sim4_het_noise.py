"""
Simulation 4: Heterogeneous Noise

Compares maxRCS and maxRegret objectives when training data has
heterogeneous measurement noise that is not present in test data.

Output: figures/sim_heterogeneous_noise_empirical.png
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from minPCA import minPCA, generate_params, get_vars_pca, get_errs_pca
from utils import get_random_covs


# === Constants ===
SEED = 10
P = 20  # dimension
N_COMPONENTS = 5  # rank
N_ENVS = 5  # environments
N_SAMPLE = 2000  # sample size per environment
N_SIMS = 25  # simulations

LB_NOISE = 0.0
UB_NOISE = 0.1  # upper bound for noise std

EMPIRICAL = True  # whether to use empirical data

SUFFIX = f'{'emp' if EMPIRICAL else 'pop'}_ub{int(UB_NOISE*100)}'
RESULTS_FILE = f'results/sim4_het_noise_{SUFFIX}.csv'
FIGURE_FILE = f'figures/sim_heterogeneous_noise_{SUFFIX}.pdf'

# Plot palette
PALETTE = {
    'maxRCS\n(noiseless)': "#2f85c3",
    'maxRegret\n(noiseless)': "#972597",
    'maxRCS\n(het. noise)': "#a1d4f8",
    'maxRegret\n(het. noise)': "#BB80BB",
}


def run_simulation():
    """Run the heterogeneous noise simulation."""
    results = []
    for sim in range(N_SIMS):
        rng = np.random.default_rng(SEED + sim)
        torch.manual_seed(SEED + sim)

        if sim % 5 == 0:
            print(f'Simulation {sim+1}/{N_SIMS}')

        # Generate covariances with varying env-specific eigenvalues
        covariances = get_random_covs(
            P, N_COMPONENTS, N_ENVS, rng,
            a1=0.1, b1=1.0, a2=0.1, b2=1.0,
            env_eigs_vary=True
        )

        # Generate heterogeneous noise levels
        stddev_noises = rng.uniform(LB_NOISE, UB_NOISE, size=N_ENVS)

        if EMPIRICAL:
            # Generate empirical data
            Xtrain = [rng.multivariate_normal(np.zeros(P), cov, N_SAMPLE)
                      for cov in covariances]
            Xnoise = [rng.normal(0, noise, size=(N_SAMPLE, P))
                      for noise in stddev_noises]
            Ztrain = [X + Xn for X, Xn in zip(Xtrain, Xnoise)]
            Xtest = [rng.multivariate_normal(np.zeros(P), cov, N_SAMPLE)
                     for cov in covariances]

            # Compute sample covariances
            covtrain = [np.cov(X.T, bias=True) for X in Xtrain]
            covtrain_noise = [np.cov(Z.T, bias=True) for Z in Ztrain]
            covtest = [np.cov(X.T, bias=True) for X in Xtest]
        else:
            covtrain = covariances
            covtrain_noise = [
                cov + noise**2 * np.eye(P) 
                for cov, noise in zip(covariances, stddev_noises)
            ]
            covtest = covariances

        params_test = generate_params(covtest, from_cov=True, norm=False)
        for rank in [N_COMPONENTS, 2 * N_COMPONENTS]:
            v_init = torch.randn(P, rank)
            for objective in ['maxrcs', 'maxregret']:
                for data_type in ['clean', 'noisy']:
                    model = minPCA(n_components=rank, norm=False, function=objective)
                    model.fit(
                        covtrain if data_type == 'clean' else covtrain_noise,
                        n_restarts=1, 
                        lr=0.1 if EMPIRICAL else 0.01, 
                        n_iters=1000,
                        v0=v_init.clone()
                    )

                    # Compute errors and variance on test data
                    v = model.v_
                    maxerr, _ = get_errs_pca(v, params_test, from_cov=True)
                    minvar, _ = get_vars_pca(v, params_test)

                    # Create label
                    obj_name = 'maxRCS' if objective == 'maxrcs' else 'maxRegret'
                    data_name = '(noiseless)' if data_type == 'clean' else '(het. noise)'
                    x_label = f'{obj_name}\n{data_name}'

                    results.append({
                        'sim': sim,
                        'objective': objective,
                        'data_type': data_type,
                        'maxerr': maxerr,
                        'minvar': minvar,
                        'rank': rank,
                        'x_label': x_label,
                    })

    return pd.DataFrame(results)


def plot_sim4_results(df, ax, add_lines=False):
    """Plot boxplot for sim4 results."""
    x_order = ['maxRCS\n(noiseless)', 'maxRegret\n(noiseless)',
               'maxRCS\n(het. noise)', 'maxRegret\n(het. noise)']

    sns.boxplot(
        x='x_label', y='maxerr', data=df,
        color='lightgray', width=0.5, showfliers=False,
        order=x_order, ax=ax,
        palette=PALETTE, hue='x_label', legend=False
    )

    ax.set_xlabel('minPCA variant\n(noise of training data)')
    ax.set_ylabel('Worst-case test RCS error')
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def make_figure(df):
    """Create and save the figure."""
    plt.style.use('jmlr.mplstyle')

    fig, ax = plt.subplots(1, 2, figsize=(6, 2.2))

    # Left: correct rank (rank = 10)
    plot_sim4_results(df[df['rank'] == 10], ax=ax[0], add_lines=False)

    # Right: lower rank (rank = 5)
    plot_sim4_results(df[df['rank'] == 5], ax=ax[1], add_lines=False)

    ax[0].set_title(r'rank-10 covariances, $k=10$')
    ax[1].set_title(r'rank-10 covariances, $k=5$')
    ax[1].set_ylabel('\nWorst-case test RCS error')
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    ax[0].tick_params(axis='x', labelsize=7)
    ax[1].tick_params(axis='x', labelsize=7)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, bbox_inches='tight')
    plt.close()
    print(f'Saved figure to {FIGURE_FILE}')


def main():
    parser = argparse.ArgumentParser(description='Simulation 4: Heterogeneous noise')
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
