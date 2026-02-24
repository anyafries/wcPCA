"""
Simulation 5: Comparison of PCA, minPCA, poolMC, and maxMC

Compares four methods across heterogeneous multi-environment settings
with partially observed data (matrix completion).  Reproduces the key
figures from comparison_notebook_2026_0220.ipynb using the same
covariance generation as the other simulations in this repo.

Outputs:
  figures/sim5_comparison.png  — 2-panel scatter figure
  figures/sim5_all.png         — 3-panel figure (2 scatter + 1 boxplot)

Usage:
  python sim5_maxmc.py            # plot from cached results (or run if missing)
  python sim5_maxmc.py --rerun    # force recomputation
"""

import os
import sys

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from fancyimpute import MatrixFactorization, BiScaler
from minPCA.minpca import minPCA 

from solve_mc import altMinSense, get_Mhat_from_right_factor, pca
from utils import get_random_covs, sample_from_convex_hull

rng = np.random.default_rng()

# ================================================================ #
#                          Constants                               #
# ================================================================ #
FANCYIMPUTE = True


SEEDS               = list(range(1, 4))
P                   = 50       # dimension
N_ROW               = 500      # training rows per environment
N_ROW_TEST          = 100        # test rows per environment
RANK                = 10        # rank (shared + env-specific each rank//2)
N_ENVS              = 5         # training environments
N_TEST_ENVS         = 0        # extra test envs from convex hull
NORM_CST            = 100       # matrices scaled so Tr(Sigma) = NORM_CST^2

QS                  = [0.05, 0.1, 0.2, 0.5, 0.8]   # target qs for panel 3
HETEROGENEITY_LEVELS = [(0.05, 0.1)] #, (0.1, 0.5), (0.5, 1.0), (1.0, 5.0)]

FIGURE_COMP  = Path('figures/sim5_comparison.png')
FIGURE_ALL   = Path('figures/sim5_all.png')

# ================================================================ #
#                    Get and save solutions                        #
# ================================================================ #

def load_solution(file_prefix, args, envs, type='pool', override=False,
                  UV=False):
    if FANCYIMPUTE and type == 'pool':
        file = f'{file_prefix}_pool_fancyimpute.npz'
        
    else:
        file = f"{file_prefix}_{type}.npz"

    if os.path.exists(file) and not override:
        print(f"\t(Loading results from {file})")
        data = np.load(file)
        Us = [data[f'Us_{i}'] for i in range(envs)]
        V = data['V']
    else:
        print(f"\t(Computing {type}MC solution)")
        if type == 'pool':
            if not FANCYIMPUTE:
                Us, V = altMinSense(type='pool', **args)
            else: 
                # make args[nrow] x args[ncol] matrix with NaNs
                X_missing = [np.full((args['nrow'], args['ncol']), np.nan) for _ in range(envs)]
                for e in range(envs):
                    X_missing[e][args['omega_indices'][e]] = args['observed_entries'][e]
                X_missing = np.concatenate(X_missing, axis=0)
                print("\t(Fitting BiScaler)")
                biscaler = BiScaler(verbose=args['verbose'])
                X_incomplete_normalized = biscaler.fit_transform(X_missing)
                print("\t(Fitting MatrixFactorization)")
                X_filled_normalized = MatrixFactorization(
                    rank=args['rank'], 
                    max_iters=args['max_iters'], 
                    verbose=True, #args['verbose'],
                    learning_rate=0.001,
                    shrinkage_value=0,
                ).fit_transform(X_incomplete_normalized)
                X_filled = biscaler.inverse_transform(X_filled_normalized)
                U, S, Vt = np.linalg.svd(X_filled_normalized, full_matrices=False)
                left_factor = U[:, :args['rank']] @ np.diag(S[:args['rank']])
                Us = [U[e*args['nrow']:(e+1)*args['nrow'], :args['rank']] 
                      for e in range(envs)]
                V = Vt[:args['rank'], :].T


        elif type == 'wc':
            Us, V = altMinSense(type='wc', outerr='wc', **args)
        print(f"\t(Saving results to {file})")
        np.savez(file, V=V, **{f'Us_{i}': Us[i] for i in range(envs)})
    U = np.concat(Us)
    Mhat = U @ V.T
    Mhats = [Ui @ V.T for Ui in Us]
    
    qrV = np.linalg.qr(V)
    right_factor = qrV[0]
    if UV:
        return right_factor, Mhats, Mhat, Us, V
    else:
        return right_factor, Mhats, Mhat #Us, V, M_opt


def load_minpca_solution(data_prefix, full_prefix, rtest, covs, 
                         envs, omega_indices, Ms, override=False):
    file_r = f"{data_prefix}_minpca_r.npz"
    if os.path.exists(file_r) and not override:
        print(f"\t(Loading R minpca from {file_r}")
        data_r = np.load(file_r)
        r = data_r['r']
    else:
        print(f"\t(Computing minPCA right factor)")
        minpca = minPCA(n_components=rtest, function='minpca', norm=False)
        minpca.fit(covs, n_restarts=10, n_iters=1000)
        r = minpca.components() #.reshape(-1, rtest)
        r = r / np.linalg.norm(r, axis=0)
        np.savez(file_r, r=r)

    file_m = f"{full_prefix}_minpca_m.npz"
    if os.path.exists(file_m) and not override:
        print(f"\t(Loading Mhats minpca from {file_m})")
        data_m = np.load(file_m)
        Mhats = [data_m[f'Mhat_{i}'] for i in range(envs)]
    else:
        print(f"\t(Computing minPCA Mhats)")
        Mhats = get_Mhat_from_right_factor(r, omega_indices, Ms)
        print(f"\t(Saving results to {file_m})")
        np.savez(file_m, **{f'Mhat_{i}': Mhats[i] for i in range(envs)})
    Mhat = np.concat(Mhats)
    # qrR = np.linalg.qr(r)
    # right_factor = qrR[0]
    return r, Mhats, Mhat


# ================================================================ #
#                   Covariance / Data Helpers                      #
# ================================================================ #

def make_covs(seed, a, b):
    """Generate training and test covariances for one (seed, het) config."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw = get_random_covs(P, RANK // 2, N_ENVS, a1=0.1, b1=1.0, a2=a, b2=b)
    covs = [C * NORM_CST ** 2 for C in raw]   # scale to Tr(Sigma) = NORM_CST^2
    covs_test = covs + sample_from_convex_hull(covs, N_TEST_ENVS)
    return covs, covs_test


def generate_missing_entries(Ms, nrow, ncol, num_e, proba=0.7):
    """
    Generates missing entries for a matrix completion problem.
    
    Parameters:
    ----------
   nrow : int or list of int
        Number of rows in each environment of the output matrices.
    num_e : int, optional
        Number of environments. 
    ncol : int, optional
        Number of columns in each environment of the output matrices. 
        If None, defaults to `nrow`.
    proba : float, optional
        Probability of observing each entry in the matrices. Default is 0.7.
        
    Returns:
    -------
    omega_indices : list of tuple
        Indices of observed entries for each matrix in each environment, 
        represented as tuples of (rows, cols).
    omega_masks : list of numpy.ndarray
        Boolean masks indicating observed entries for each matrix 
        (each of shape `nrow` x `ncol`).
    observed_entries : list of numpy.ndarray
        Observed entries of each matrix in each environment, 
        extracted using `omega_indices`.
    """
    omega_masks = []
    omega_indices = []
    observed_entries = []
    for i in range(num_e):
        k = int(proba * ncol)    # number of ones per row
        mask = np.zeros((nrow, ncol), dtype=int)
        for j in range(nrow):
            cols = np.random.choice(ncol, size=k, replace=False)
            mask[j, cols] = 1
        indices = np.where(mask == 1)

        omega_masks.append(mask.astype(bool))
        omega_indices.append(indices)
        observed_entries.append(Ms[i][indices])
    return omega_masks, omega_indices, observed_entries


def generate_data(covs, proba, nrow, nenvs):
    """Generate training matrices and missingness at proba."""
    # _, _, Ms = generate_data(
    #     rank=RANK, nrow=N_ROW, num_e=N_ENVS, ncol=P,
    #     method='from-cov', population=False,
    #     covs=covs, norm_cst=NORM_CST,
    # )
    Ms = []
    for C in covs:
        try:
            # eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Z = np.random.standard_normal((N_ROW, P))
            # A = eigenvectors * np.sqrt(eigenvalues)
            # samples = Z.dot(A.T)
            # Ms.append(samples)
            rng.multivariate_normal(mean=np.zeros(P), cov=C, size=N_ROW,
                                    method='eigh')
        except np.linalg.LinAlgError:   
            print("PROBLEM")
    Ms = [rng.multivariate_normal(mean=np.zeros(P), cov=C, size=N_ROW, method='eigh') 
          for C in covs]
    _, omega_indices, observed_entries = generate_missing_entries(
        Ms, nrow, P, nenvs, proba=proba,
    )
    
    return Ms, omega_indices, observed_entries


def eval_on_test_cov(right_factors, cov, proba):
    """
    Generate fresh test data from cov at observation prob q, impute with
    each right factor, and return per-method MSE.
    """
    # _, _, Ms_t = generate_data(
    #     rank=RANK, nrow=N_ROW_TEST, num_e=1, ncol=P,
    #     method='from-cov', population=False,
    #     covs=[cov], norm_cst=NORM_CST,
    # )
    Ms, omega_indices, _ = generate_data([cov], proba, nrow=N_ROW_TEST, nenvs=1)
    errs = {}
    for method, R in right_factors.items():
        Mhats = get_Mhat_from_right_factor(R, omega_indices, Ms)
        errs[method] = np.mean((Ms[0] - Mhats[0]) ** 2)
    return errs


# ================================================================ #
#                         Solver                                   #
# ================================================================ #

def solve_factors(covs, Ms, omega_indices, observed_entries,
                  data_prefix, full_prefix, opt_tol, max_iters, override):
    """
    Compute (or load cached) right factors for PCA, minPCA, poolMC, maxMC.

    data_prefix: used for minPCA right factor (no missingness dependency)
    full_prefix:  used for poolMC, maxMC, and minPCA Mhats
    """
    emp_covs = [M.T @ M / N_ROW for M in Ms]

    # PCA — empirical pooled covariance
    cov_pooled_emp = np.mean(emp_covs, axis=0)
    R_pca = pca(cov_pooled_emp)[:, :RANK]

    # minPCA
    R_minpca, _, _ = load_minpca_solution(
        data_prefix, full_prefix,
        RANK, emp_covs, N_ENVS, omega_indices, Ms,
        override=override,
    )

    args = {
        'observed_entries': observed_entries,
        'omega_indices':    omega_indices,
        'nrow':             N_ROW,
        'ncol':             P,
        'rank':             RANK,
        'verbose':          False,
        'optTol':           opt_tol,
        'reruns':           1,
        'max_iters':        max_iters,
    }

    # poolMC
    R_pool, _, _ = load_solution(
        full_prefix, args, N_ENVS, type='pool', override=override,
    )

    # maxMC
    R_wc, _, _ = load_solution(
        full_prefix, args, N_ENVS, type='wc', override=override,
    )


    return {'PCA': R_pca, 'minPCA': R_minpca, 'poolMC': R_pool, 'maxMC': R_wc}


# ================================================================ #
#                       Main Simulation Loop                       #
# ================================================================ #

def run_simulation(prob_source, opt_tol, max_iters, results_miss,
                   override=False):
    """
    For each (seed, het_level): solve right factors, evaluate on test envs
    at prob_source and at each q in QS.  Saves two CSVs.
    """
    miss_rows = []

    for seed in SEEDS:
        for het_idx, (a, b) in enumerate(HETEROGENEITY_LEVELS):
            print(f"\nSeed {seed}, heterogeneity a={a}, b={b}")
            data_prefix = f'results/sim5/s{seed}_h{het_idx}'
            full_prefix = f'results/sim5/s{seed}_h{het_idx}_src{int(prob_source * 100)}_opt{int(opt_tol * 1e6)}_mi{max_iters}'

            covs, covs_test = make_covs(seed, a, b)
            Ms, omega_indices, observed_entries = generate_data(
                covs, prob_source, N_ROW, N_ENVS)

            right_factors = solve_factors(
                covs, Ms, omega_indices, observed_entries,
                data_prefix, full_prefix, opt_tol, max_iters, override,
            )

            current_qs = QS.copy()
            if prob_source not in current_qs:
                current_qs.append(prob_source)

            # --- Missingness sweep (for Figure 2 panel 2) ---
            for q in current_qs:
                env_errs_q = {m: [] for m in right_factors}
                for cov_t in covs_test:
                    e = eval_on_test_cov(right_factors, cov_t, q)
                    for m in right_factors:
                        env_errs_q[m].append(e[m])
                for m in right_factors:
                    miss_rows.append({
                        'seed':       seed,
                        'a':          a,
                        'b':          b,
                        'q':          q,
                        'method':     m,
                        'mean':       np.mean(env_errs_q[m]),
                        'worst_case': np.max(env_errs_q[m]),
                    })

    df_miss = pd.DataFrame(miss_rows)
    df_miss.to_csv(results_miss, index=False)
    print(f"\nSaved results to {results_miss}")
    return df_miss


# ================================================================ #
#                       Plotting Helpers                           #
# ================================================================ #

def comparison_plot_green_red(df, title='', maxmc=False, ax=None, s=10,
                               ylabel=r'$\Delta$ reconstruction error'):
    """
    Scatter plot of Δavg and Δwc for each (seed, het) pair.
    Points are colored green (better) → grey (neutral) → red (worse).
    Pairs connected by a dashed grey line.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(3, 2.5))

    method1, method2 = ('poolMC', 'maxMC') if maxmc else ('PCA', 'minPCA')
    het_pairs = df[['a', 'b']].drop_duplicates().values

    # Collect all deltas for symmetric colormap normalization
    all_deltas = []
    for seed in df['seed'].unique():
        for (a, b) in het_pairs:
            sub = df[(df['seed'] == seed) & (df['a'] == a) & (df['b'] == b)]
            m1 = sub[sub['method'] == method1]
            m2 = sub[sub['method'] == method2]
            if m1.empty or m2.empty:
                continue
            all_deltas.extend([
                m2['mean'].values[0]       - m1['mean'].values[0],
                m2['worst_case'].values[0] - m1['worst_case'].values[0],
            ])

    if not all_deltas:
        return

    mmax = max(abs(min(all_deltas)), abs(max(all_deltas)), 1e-8)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'gr', ['tab:green', '#939393', 'tab:red']
    )
    norm = mcolors.TwoSlopeNorm(vmin=-mmax, vcenter=0, vmax=mmax)

    x_labels = [r'$\Delta$ average', r'$\Delta$ worst-case']
    for seed in df['seed'].unique():
        for (a, b) in het_pairs:
            sub = df[(df['seed'] == seed) & (df['a'] == a) & (df['b'] == b)]
            m1 = sub[sub['method'] == method1]
            m2 = sub[sub['method'] == method2]
            if m1.empty or m2.empty:
                continue
            delta_avg = m2['mean'].values[0]       - m1['mean'].values[0]
            delta_wc  = m2['worst_case'].values[0] - m1['worst_case'].values[0]
            ax.plot(x_labels, [delta_avg, delta_wc],
                    ':', linewidth=0.8, color='#939393', zorder=1)
            ax.scatter(x_labels, [delta_avg, delta_wc],
                       s=s, marker='o', zorder=10,
                       c=[cmap(norm(delta_avg)), cmap(norm(delta_wc))])

    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.2, 1.2)
    ax.set_title(title)

    if created_fig:
        # Color y-tick labels by value when standalone
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        for tick_val, tick_label in zip(yticks, ax.get_yticklabels()):
            tick_label.set_color(cmap(norm(tick_val)))
        plt.tight_layout()
        plt.close()


# ================================================================ #
#                           Figures                                #
# ================================================================ #

def make_figure1(df_agg, prob_source):
    """2-panel scatter figure comparing minPCA vs PCA and maxMC vs poolMC."""
    fig, ax = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    comparison_plot_green_red(
        df_agg,
        title=rf'$q_\textrm{{source}}=1,\ q_\textrm{{target}}={prob_source}$',
        ax=ax[0], s=10,
    )
    comparison_plot_green_red(
        df_agg,
        title=rf'$q_\textrm{{source}}={prob_source},\ q_\textrm{{target}}={prob_source}$',
        maxmc=True, ax=ax[1], s=10, ylabel='',
    )
    plt.tight_layout()
    plt.savefig(FIGURE_COMP, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURE_COMP}")


def make_figure2(df_agg, df_miss, prob_source):
    """
    3-panel figure:
      Panel 0 — scatter: minPCA vs PCA (same data as figure 1)
      Panel 1 — scatter: maxMC vs poolMC (same data as figure 1)
      Panel 2 — sim3-style boxplot: Δ worst-case maxMC vs q
    """
    # Build wide-format DataFrame for panel 2
    df_wide = df_miss.pivot_table(
        index=['seed', 'a', 'b', 'q'],
        columns='method',
        values=['mean', 'worst_case'],
    ).reset_index()
    df_wide.columns = [
        '_'.join(col).strip() if col[1] else col[0]
        for col in df_wide.columns.values
    ]
    df_wide['diff_wc_maxMC'] = (
        df_wide['worst_case_maxMC'] - df_wide['worst_case_poolMC']
    )
    df_wide['q'] = df_wide['q'].round(2)

    fig, ax = plt.subplots(
        1, 3, figsize=(6.5, 1.6), sharey=True,
        gridspec_kw={'width_ratios': [1, 1, 1.4], 'wspace': 0.4},
    )

    # Panel 0: minPCA vs PCA
    comparison_plot_green_red(
        df_agg,
        title=f'Fully observed source,\n'
              rf'$q_\textrm{{source}}=1,\ q_\textrm{{target}}={prob_source}$',
        ax=ax[0], s=5, ylabel=r'$\Delta$ RCS error',
    )

    # Panel 1: maxMC vs poolMC
    comparison_plot_green_red(
        df_agg,
        title=f'Partially observed source,\n'
              rf'$q_\textrm{{source}}={prob_source},\ q_\textrm{{target}}={prob_source}$',
        maxmc=True, ax=ax[1], s=5, ylabel='',
    )

    # Panel 2: boxplot Δ wc maxMC vs q (sim3 style)
    sns.boxplot(
        data=df_wide, x='q', y='diff_wc_maxMC', ax=ax[2],
        color='#2f85c3', fliersize=2, width=0.65, linewidth=0.8,
    )
    ax[2].axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax[2].set_xlabel(r'$q_\textrm{target}$')
    ax[2].set_title(
        f'Partially observed source,\n'
        rf'$q_\textrm{{source}}={prob_source}$, varying $q_\textrm{{target}}$'
    )
    ax[2].set_ylabel(r'$\Delta$ worst-case RCS error')
    ax[2].yaxis.get_label().set_visible(True)
    plt.setp(ax[2].get_xticklabels(), rotation=90, ha='center')
    ax[2].text(
        1.07, 0, '→ worse',
        transform=ax[2].get_yaxis_transform(),
        rotation=90, va='bottom', ha='left', fontsize=8,
    )
    ax[2].text(
        1.07, 0, 'maxMC better ←',
        transform=ax[2].get_yaxis_transform(),
        rotation=90, va='top', ha='left', fontsize=8,
    )

    plt.savefig(FIGURE_ALL, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURE_ALL}")


# ================================================================ #
#                              Main                                #
# ================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description='Simulation 5: maxMC comparison across heterogeneous environments'
    )
    parser.add_argument(
        '--rerun', action='store_true',
        help='Rerun full simulation even if cached results exist',
    )
    parser.add_argument(
        "--prob_source", type=float, default=0.1,
        help="Source observation probability (default: 0.1)",
    )
    parser.add_argument(
        "--opt_tol", type=float, default=1e-4,
        help="Optimization tolerance for poolMC and maxMC (default: 1e-4)",
    )
    parser.add_argument(
        "--max_iters", type=int, default=50,
        help="Maximum iterations for poolMC and maxMC (default: 50)",
    )
    args = parser.parse_args()
    prob_source = args.prob_source
    opt_tol = args.opt_tol
    max_iters = args.max_iters

    Path('results').mkdir(exist_ok=True)
    Path('results/sim5').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)

    plt.style.use('jmlr.mplstyle')

    file_suffix = f'src{int(prob_source * 100)}_opt{int(opt_tol * 1e6)}_mi{max_iters}'
    file_suffix += f'_seed{SEEDS[0]}-{SEEDS[-1]}'
    results_miss = Path(f'results/sim5/miss_{file_suffix}.csv')
    print(f"Results file: {results_miss}")

    if args.rerun or not results_miss.exists():
        df_miss = run_simulation(
            prob_source, opt_tol, max_iters, results_miss,
            override=args.rerun,
        )
    else:
        print(f'Loading cached results from {results_miss}')
        df_miss = pd.read_csv(results_miss)

    df_agg = df_miss[df_miss['q'] == prob_source].copy()
    make_figure1(df_agg, prob_source)
    make_figure2(df_agg, df_miss, prob_source)


if __name__ == '__main__':
    main()
