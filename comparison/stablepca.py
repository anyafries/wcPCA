"""
StablePCA Comparison Script

Runs StablePCA algorithm for worst-case PCA comparison.

Note: StablePCA only supports the MM_Var objective (minimum variance maximization).
      It does NOT support MM_Loss (regret minimization).

Output:
    results/stablepca_MM_Var_p{p}_ncomp{n_components}_ne{n_envs}.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stablepca_algo import StablePCA
from utils import get_random_covs, f_minpca_np

# === Constants ===
SEED = 2
N_COMPONENTS = 5
RESULTS_DIR = Path(__file__).parent / 'results'


def run_stablepca(covs_norm, p, seed=SEED):
    """
    Run StablePCA for all ranks.

    Returns DataFrame with columns: rank, minvar, time
    """
    np.random.seed(seed)

    n = covs_norm[0].shape[0]
    k = len(covs_norm)

    minvars = []
    times = []

    for rank in range(1, n + 1):
        print(f"    Rank: {rank}/{n}")
        model = StablePCA(n_components=rank)

        t0 = time.time()
        model.fit(covs_norm, verbose=False)
        t1 = time.time()

        v_stablepca = model.components_
        minvars.append(f_minpca_np(v_stablepca, covs_norm, [1.0] * k))
        times.append(t1 - t0)

    return pd.DataFrame({
        'rank': list(range(1, n + 1)),
        'minvar': minvars,
        'time': times
    })


def run_simulation(p, n_envs, seed=SEED):
    """Run StablePCA for given parameters."""
    print(f"Running StablePCA: p={p}, n_envs={n_envs}, objective=MM_Var")

    # Generate covariances
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    covs_norm = get_random_covs(p, N_COMPONENTS, n_envs, rng)

    # Run StablePCA
    print("  Running StablePCA optimizer...")
    df = run_stablepca(covs_norm, p, seed=seed)

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
    args = parser.parse_args()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # Parameter grid
    if args.p is not None and args.n_envs is not None:
        param_configs = [(args.p, args.n_envs)]
    else:
        param_configs = [(10, 5), (10, 50), (50, 5)]

    # StablePCA only supports MM_Var
    objective = 'MM_Var'

    for seed in range(args.start_seed, args.end_seed + 1):
        for p, n_envs in param_configs:
            suffix = f"_{objective}_p{p}_ncomp{N_COMPONENTS}_ne{n_envs}_seed{seed}.csv"
            results_file = RESULTS_DIR / f"stablepca{suffix}"

            # Check cache
            if not args.rerun and results_file.exists():
                print(f"Cached results exist for p={p}, n_envs={n_envs}, seed={seed}, skipping...")
                continue

            # Run simulation
            df = run_simulation(p, n_envs, seed=seed)

            # Save results
            df.to_csv(results_file, index=False)
            print(f"  Saved: {results_file.name}")


if __name__ == '__main__':
    main()
