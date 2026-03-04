"""
FairPCA comparison script

Runs FairPCA with SDP and MW optimizers for worst-case PCA comparison.
Requires the multiCriteriaDimReduction package from
https://github.com/SDPforAll/multiCriteriaDimReduction

Supports two objectives:
- MM_Var: Minimize maximum variance loss (maximize minimum explained variance)
- MM_Loss: Minimize maximum regret

Output:
    results/SDP_{objective}_p{p}_ncomp{n_components}_ne{n_envs}.csv
    results/MW_{objective}_p{p}_ncomp{n_components}_ne{n_envs}.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_random_covs, get_V_from_X, f_minpca_np, f_regret_np

from multi_criteria_dimensionality_reduction.SDP import fairDimReductionFractional
from multi_criteria_dimensionality_reduction.MW import fairDimReduction_MW

# === Constants ===
SEED = 2
N_COMPONENTS = 5
RESULTS_DIR = Path(__file__).parent / 'results'


def run_sdp(covs_norm, objective, p):
    """
    Run FairPCA with SDP optimizer.

    Returns DataFrame with columns: d, rank, time, obj, status, obj_trunc
    """
    n = covs_norm[0].shape[0]
    k = len(covs_norm)

    runstats, Xfrac = fairDimReductionFractional(
        n=n,
        k=k,
        B=covs_norm,
        list_d=range(1, n + 1),
        Obj=objective,
        return_option="frac_sol",
        savedPath=None,
    )

    # Compute truncated objective values (enforce exact rank)
    norm_csts = [1 for _ in covs_norm]
    obj_trunc = []
    for i in range(len(Xfrac)):
        d = Xfrac[i][1]
        X = np.array(Xfrac[i][2])
        V = get_V_from_X(X, rank=d)
        if objective == "MM_Var":
            obj = f_minpca_np(V, covs_norm, norm_csts)
        else:  # MM_Loss
            obj = f_regret_np(V, covs_norm, norm_csts)
        obj_trunc.append(obj)

    runstats['obj_trunc'] = obj_trunc
    runstats['obj'] = runstats['obj'].round(decimals=3)

    return runstats


def run_mw(covs_norm, objective, p):
    """
    Run FairPCA with MW (Multiplicative Weights) optimizer.

    Returns DataFrame with columns: d, obj, obj_last, time
    """
    n = covs_norm[0].shape[0]
    k = len(covs_norm)
    norm_csts = [1 for _ in covs_norm]

    results = []
    for d in range(1, n + 1):
        X_last, X_avg, runstats, total_time = fairDimReduction_MW(
            n=n,
            k=k,
            d=d,
            B=covs_norm,
            Obj=objective,
            verbose=False,
            T=100,
        )
        V_last = get_V_from_X(X_last, rank=d)
        if objective == "MM_Var":
            obj = f_minpca_np(V_last, covs_norm, norm_csts)
        else:  # MM_Loss
            obj = f_regret_np(V_last, covs_norm, norm_csts)
        results.append({
            "d": d,
            "obj": obj,
            "obj_last": runstats["minimum of m objective, that iterate"].values[-1],
            "time": total_time,
        })

    return pd.DataFrame(results)


def run_fairpca(p, n_envs, objective, seed=SEED):
    """Run both SDP and MW for given parameters."""
    print(f"Running FairPCA: p={p}, n_envs={n_envs}, objective={objective}")

    # Generate covariances
    np.random.seed(seed)
    covs_norm = get_random_covs(p, N_COMPONENTS, n_envs)

    # Run SDP
    print("  Running SDP optimizer...")
    df_sdp = run_sdp(covs_norm, objective, p)

    # Run MW
    print("  Running MW optimizer...")
    df_mw = run_mw(covs_norm, objective, p)

    return df_sdp, df_mw


def main():
    parser = argparse.ArgumentParser(description='FairPCA comparison simulation')
    parser.add_argument('--rerun', action='store_true',
                        help='Force rerun even if cached results exist')
    parser.add_argument('--objective', type=str, default=None,
                        choices=['MM_Var', 'MM_Loss'],
                        help='Run specific objective only')
    parser.add_argument('--p', type=int, default=None,
                        help='Dimension p')
    parser.add_argument('--n_envs', type=int, default=None,
                        help='Number of environments')
    args = parser.parse_args()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # Parameter grid
    if args.p is not None and args.n_envs is not None:
        param_configs = [(args.p, args.n_envs)]
    else:
        param_configs = [(10, 5), (10, 50), (50, 5)]

    objectives = [args.objective] if args.objective else ['MM_Var', 'MM_Loss']

    for i, (p, n_envs) in enumerate(param_configs):
        for objective in objectives:
            suffix = f"_{objective}_p{p}_ncomp{N_COMPONENTS}_ne{n_envs}.csv"
            sdp_file = RESULTS_DIR / f"SDP{suffix}"
            mw_file = RESULTS_DIR / f"MW{suffix}"

            # Check cache
            if not args.rerun and sdp_file.exists() and mw_file.exists():
                print(f"Cached results exist for p={p}, n_envs={n_envs}, {objective}, skipping...")
                continue

            # Run simulation
            df_sdp, df_mw = run_fairpca(p, n_envs, objective, seed=SEED + i)

            # Save results
            df_sdp.to_csv(sdp_file, index=False)
            df_mw.to_csv(mw_file, index=False)
            print(f"  Saved: {sdp_file.name}, {mw_file.name}")


if __name__ == '__main__':
    main()
