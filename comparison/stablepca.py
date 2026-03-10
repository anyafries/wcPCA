"""
StablePCA Comparison Script

Runs StablePCA algorithm for worst-case PCA comparison.

Output:
    results/stablepca_{objective}_p{p}_ncomp{n_components}_ne{n_envs}.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from StablePCA.PCAalg import PCA_MP
from utils import get_random_covs, f_minpca_np, f_regret_np

# === Constants ===
SEED = 2
N_COMPONENTS = 5
RESULTS_DIR = Path(__file__).parent / 'results'


def run_stablepca(covs_norm, p, method, seed=SEED):
    """
    Run StablePCA for all ranks.

    Returns DataFrame with columns: rank, minvar, time
    """
    np.random.seed(seed)

    n = covs_norm[0].shape[0]
    k = len(covs_norm)

    minvars = []
    times = []

    # sqrt of sigma for X in PCA_MP
    jitter = 1e-8
    p_dim = covs_norm[0].shape[0]

    X_list = []
    for cov in covs_norm:
        # Add jitter to the diagonal
        cov_pd = cov + np.eye(p_dim) * jitter
        X_list.append(np.linalg.cholesky(cov_pd))

    for rank in range(1, n):
        print(f"    Rank: {rank}/{n}")
        model = PCA_MP(n_components=rank, method=method)

        t0 = time.time()
        model.fit(X_list, Sigma_list=covs_norm, verbose=False)
        t1 = time.time()

        v_stablepca = model.components_.T
        if method == 'stable':
            minvars.append(f_minpca_np(v_stablepca, covs_norm, [1.0] * k))
        elif method == 'fair':
            obj = f_regret_np(v_stablepca, covs_norm, 
                              [1.0] * k)
            minvars.append(obj)
        else:
            raise ValueError(f"Unsupported method: {method}")
        times.append(t1 - t0)

    return pd.DataFrame({
        'rank': list(range(1, n)),
        'minvar': minvars,
        'time': times
    })


def run_simulation(p, n_envs, method, seed=SEED):
    """Run StablePCA for given parameters."""
    print(f"Running StablePCA: p={p}, n_envs={n_envs}, objective=MM_Var")

    # Generate covariances
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    covs_norm = get_random_covs(p, N_COMPONENTS, n_envs, rng)

    # Run StablePCA
    print("  Running StablePCA optimizer...")
    df = run_stablepca(covs_norm, p, method=method, seed=seed)

    return df


def main():
    parser = argparse.ArgumentParser(description='StablePCA comparison simulation')
    parser.add_argument('--rerun', action='store_true',
                        help='Force rerun even if cached results exist')
    parser.add_argument('--p', type=int, default=None,
                        help='Dimension p')
    parser.add_argument('--n_envs', type=int, default=None,
                        help='Number of environments')
    parser.add_argument('--start_seed', type=int, default=SEED,
                        help='First seed (inclusive)')
    parser.add_argument('--end_seed', type=int, default=SEED,
                        help='Last seed (inclusive)')
    parser.add_argument('--objective', type=str, default='MM_Var',
                        help='Objective to optimize (default: MM_Var)')
    args = parser.parse_args()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # Parameter grid
    if args.p is not None and args.n_envs is not None:
        param_configs = [(args.p, args.n_envs)]
    else:
        param_configs = [(10, 5), (10, 50), (50, 5)]

    objective = args.objective
    method = 'stable' if objective == 'MM_Var' else 'fair'

    for seed in range(args.start_seed, args.end_seed + 1):
        for p, n_envs in param_configs:
            suffix = f"_{objective}_p{p}_ncomp{N_COMPONENTS}_ne{n_envs}_seed{seed}.csv"
            results_file = RESULTS_DIR / f"stablepca_new{suffix}"

            # Check cache
            if not args.rerun and results_file.exists():
                print(f"Cached results exist for p={p}, n_envs={n_envs}, seed={seed}, skipping...")
                continue

            # Run simulation
            df = run_simulation(p, n_envs, seed=seed, method=method)

            # Save results
            df.to_csv(results_file, index=False)
            print(f"  Saved: {results_file.name}")


if __name__ == '__main__':
    main()
