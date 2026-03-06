"""
minPCA Comparison Script

Runs minPCA with PGD (Projected Gradient Descent) optimizer for worst-case PCA comparison.

Supports two objectives:
- MM_Var: Minimize maximum variance loss (maximize minimum explained variance)
- MM_Loss: Minimize maximum regret

Output:
    results/minPCA_{objective}_p{p}_ncomp{n_components}_ne{n_envs}.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import get_random_covs, f_regret_np
from minPCA.minpca import minPCA

# === Constants ===
SEED = 2
N_COMPONENTS = 5
RESULTS_DIR = Path(__file__).parent / 'results'

# Training hyperparameters
N_RESTARTS = 5
LR = 0.1
N_ITERS = 100


def run_minpca(covs_norm, objective, p, seed=SEED):
    """
    Run minPCA (PGD optimizer) for all ranks.

    Returns DataFrame with columns: rank, minvar, time
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n = covs_norm[0].shape[0]

    minvars = []
    times = []

    for rank in range(1, n + 1):
        # Choose function based on objective
        if objective == "MM_Loss":
            function = "maxregret"
        else:  # MM_Var
            function = "minpca"

        model = minPCA(n_components=rank, norm=False, function=function)

        t0 = time.time()
        model.fit(covs_norm, n_restarts=N_RESTARTS,
                  lr=LR, n_iters=N_ITERS)
        t1 = time.time()

        v_minpca = model.v_.detach().numpy()

        if objective == "MM_Loss":
            # Use f_regret_np to compute regret
            obj = f_regret_np(v_minpca, covs_norm, 
                              [1 for _ in range(len(covs_norm))])
            minvars.append(obj)
        else:  # MM_Var
            minvars.append(model.minvar())

        times.append(t1 - t0)

    return pd.DataFrame({
        'rank': list(range(1, n + 1)),
        'minvar': minvars,
        'time': times
    })


def run_simulation(p, n_envs, objective, seed=SEED):
    """Run minPCA for given parameters."""
    print(f"Running minPCA: p={p}, n_envs={n_envs}, objective={objective}")

    # Generate covariances
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    covs_norm = get_random_covs(p, N_COMPONENTS, n_envs, rng)

    # Run minPCA
    print("  Running PGD optimizer...")
    df = run_minpca(covs_norm, objective, p, seed=seed)

    return df


def main():
    parser = argparse.ArgumentParser(description='minPCA comparison simulation')
    parser.add_argument('--rerun', action='store_true',
                        help='Force rerun even if cached results exist')
    parser.add_argument('--objective', type=str, default=None,
                        choices=['MM_Var', 'MM_Loss'],
                        help='Run specific objective only')
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

    objectives = [args.objective] if args.objective else ['MM_Var', 'MM_Loss']

    for seed in range(args.start_seed, args.end_seed + 1):
        for p, n_envs in param_configs:
            for objective in objectives:
                suffix = f"_{objective}_p{p}_ncomp{N_COMPONENTS}_ne{n_envs}_seed{seed}.csv"
                results_file = RESULTS_DIR / f"minPCA{suffix}"

                # Check cache
                if not args.rerun and results_file.exists():
                    print(f"Cached results exist for p={p}, n_envs={n_envs}, {objective}, seed={seed}, skipping...")
                    continue

                # Run simulation
                df = run_simulation(p, n_envs, objective, seed=seed)

                # Save results
                df.to_csv(results_file, index=False)
                print(f"  Saved: {results_file.name}")


if __name__ == '__main__':
    main()
