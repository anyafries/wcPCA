"""
Simulation 5: Comparison of PCA, minPCA, poolMC, and maxMC
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch

from matplotlib.lines import Line2D
from minPCA.minpca import minPCA 
from pathlib import Path

from solve_mc import *
from utils import get_random_covs, sample_from_convex_hull

# ================================================================ #
#                          Constants                               #
# ================================================================ #
P                   = 500       # dimension
N_ROW               = 1000      # training rows per environment
N_ROW_TEST          = 1000        # test rows per environment
N_COMPONENTS        = 5        # rank 
N_ENVS              = 5         # training environments
N_TEST_ENVS         = 0        # extra test envs from convex hull
NORM_CST            = 1       # matrices scaled so Tr(Sigma) = NORM_CST^2

SUBDIR = f'mcam_p{P}_nrow{N_ROW}_norm{NORM_CST}'
SUBDIR += f'_ncomp{N_COMPONENTS}_nenvs{N_ENVS}'
FIGSUBDIR = SUBDIR + f'_nrowtest{N_ROW_TEST}_ntestenvs{N_TEST_ENVS}'

QS                  = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]   # target qs for panel 3
HETEROGENEITY_LEVELS = [(2, 5), (1, 2), (0.5, 1), (0, 0.5)]

FIGURE_ABS   = Path(f'figures/{FIGSUBDIR}/sim5_absolute.png')

METHOD_COLORS = {
    'PCA':    "#b41f1f",
    'minPCA': '#2f85c3',
    'poolMC': "#f75959",
    'maxMC':  "#57b6fa",
}

# ================================================================ #
#                    Get and save solutions                        #
# ================================================================ #

def load_solution(file_prefix, args, envs, type='pool', override=False,
                  UV=False):
    file = f"{file_prefix}_{type}.npz"

    if os.path.exists(file) and not override:
        print(f"\t(Loading results from {file})")
        data = np.load(file)
        Us = [data[f'Us_{i}'] for i in range(envs)]
        V = data['V']
    else:
        print(f"\t(Computing {type}MC solution)")
        if type == 'pool':
            Us, V = solve_mcam_multienv(**args, type='pool')

        elif type == 'wc':
            Us, V = solve_mcam_multienv(type='wc', **args)
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
        return right_factor, Mhats, Mhat


def load_minpca_solution(data_prefix, full_prefix, rtest, covs, 
                         envs, observed_indices, Ms, v0=None, override=False):
    file_r = f"{data_prefix}_minpca_r.npz"
    if os.path.exists(file_r) and not override:
        print(f"\t(Loading R minpca from {file_r}")
        data_r = np.load(file_r)
        r = data_r['r']
    else:
        print(f"\t(Computing minPCA right factor)")
        minpca = minPCA(n_components=rtest, function='maxrcs', norm=False)
        covs_scaled = [C * 1e3 for C in covs]  # scale up to avoid numerical issues
        minpca.fit(covs_scaled, n_restarts=10, n_iters=1000, lr=0.1, verbose=True)
        r = minpca.components() 
        r = r / np.linalg.norm(r, axis=0)
        np.savez(file_r, r=r)

    file_m = f"{full_prefix}_minpca_m.npz"
    if os.path.exists(file_m) and not override:
        print(f"\t(Loading Mhats minpca from {file_m})")
        data_m = np.load(file_m)
        Mhats = [data_m[f'Mhat_{i}'] for i in range(envs)]
    else:
        print(f"\t(Computing minPCA Mhats)")
        Mhats = get_Mhat_from_right_factor(r, observed_indices, Ms)
        print(f"\t(Saving results to {file_m})")
        np.savez(file_m, **{f'Mhat_{i}': Mhats[i] for i in range(envs)})

    Mhat = np.concat(Mhats)
    return r, Mhats, Mhat


# ================================================================ #
#                   Covariance / Data Helpers                      #
# ================================================================ #

def make_covs(seed, a, b, rng):
    """Generate training and test covariances for one (seed, het) config."""
    raw = get_random_covs(P, N_COMPONENTS, N_ENVS, rng, a1=0.1, b1=1.0, a2=a, b2=b)
    covs = [C * NORM_CST ** 2 for C in raw]   # scale to Tr(Sigma) = NORM_CST^2
    covs_test = covs + sample_from_convex_hull(covs, N_TEST_ENVS, rng)
    return covs, covs_test


def generate_missing_entries(Ms, nrow, ncol, num_e, rng_data, proba=0.7):
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
    rng_data : numpy.random.Generator
        A local random number generator for reproducibility.
    proba : float, optional
        Probability of observing each entry in the matrices. Default is 0.7.
        
    Returns:
    -------
    observed_indices : list of tuple
        Indices of observed entries for each matrix in each environment, 
        represented as tuples of (rows, cols).
    omega_masks : list of numpy.ndarray
        Boolean masks indicating observed entries for each matrix 
        (each of shape `nrow` x `ncol`).
    observed_entries : list of numpy.ndarray
        Observed entries of each matrix in each environment, 
        extracted using `observed_indices`.
    """
    omega_masks = []
    observed_indices = []
    observed_entries = []
    for i in range(num_e):
        k = int(proba * ncol)    # number of ones per row
        mask = np.zeros((nrow, ncol), dtype=int)
        for j in range(nrow):
            cols = rng_data.choice(ncol, size=k, replace=False)
            mask[j, cols] = 1
        indices = np.where(mask == 1)

        omega_masks.append(mask.astype(bool))
        observed_indices.append(indices)
        observed_entries.append(Ms[i][indices])
    return omega_masks, observed_indices, observed_entries


def generate_data(covs, proba, nrow, nenvs, rng_data):
    """Generate training matrices and missingness at proba."""
    Ms = [
        rng_data.multivariate_normal(mean=np.zeros(P), cov=C, size=nrow, method='eigh') 
        for C in covs
    ]
    _, observed_indices, observed_entries = generate_missing_entries(
        Ms, nrow, P, nenvs, proba=proba, rng_data=rng_data
    )
    
    return Ms, observed_indices, observed_entries


def eval_on_test_cov(right_factors, cov, proba, rng_data):
    """
    Generate fresh test data from cov at observation prob q, impute with
    each right factor, and return per-method MSE.
    """
    Ms, observed_indices, _ = generate_data([cov], proba, nrow=N_ROW_TEST, nenvs=1,
                                         rng_data=rng_data)
    errs = {}
    for method, R in right_factors.items():
        Mhats = get_Mhat_from_right_factor(R, observed_indices, Ms)
        errs[method] = np.mean((Ms[0] - Mhats[0]) ** 2)
    return errs


# ================================================================ #
#                         Solver                                   #
# ================================================================ #

def solve_factors(covs, Ms, observed_indices, observed_entries,
                  data_prefix, full_prefix, opt_tol, max_iters, 
                  rng, override):
    """
    Compute (or load cached) right factors for PCA, minPCA, poolMC, maxMC.

    data_prefix: used for minPCA right factor (no missingness dependency)
    full_prefix:  used for poolMC, maxMC, and minPCA Mhats
    """
    emp_covs = [M.T @ M / N_ROW for M in Ms]

    # PCA — empirical pooled covariance
    cov_pooled_emp = np.mean(emp_covs, axis=0)
    R_pca = pca(cov_pooled_emp)[:, :N_COMPONENTS]

    # minPCA
    R_minpca, _, _ = load_minpca_solution(
        data_prefix, full_prefix,
        N_COMPONENTS, emp_covs, N_ENVS, observed_indices, Ms,
        v0=R_pca, override=override,
    )

    args = {
        'observed_entries': observed_entries,
        'observed_indices':    observed_indices,
        'nrow':             N_ROW,
        'ncol':             P,
        'rank':             N_COMPONENTS,
        'optTol':           opt_tol,
        'max_iters':        max_iters,
    }

    # poolMC
    R_pool, _, _ = load_solution(
        full_prefix, args, N_ENVS, type='pool', override=False, #override,
    )

    # maxMC
    R_wc, _, _ = load_solution(
        full_prefix, args, N_ENVS, type='wc', override=False,#override,
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
    cache_dir = Path(f'results/sim5/{SUBDIR}/cache')
    cache_dir.mkdir(exist_ok=True)

    for seed in SEEDS:
        for het_idx, (a, b) in enumerate(HETEROGENEITY_LEVELS):
            print(f"\nSeed {seed}, heterogeneity a={a}, b={b}")
            data_prefix = f'results/sim5/{SUBDIR}/s{seed}_h{het_idx}'
            full_prefix = f'results/sim5/{SUBDIR}/s{seed}_h{het_idx}_src{int(prob_source * 100)}_opt{int(opt_tol * 1e6)}_mi{max_iters}'
            cache_file = f's{seed}_h{het_idx}_src{int(prob_source * 100)}_opt{int(opt_tol * 1e6)}_mi{max_iters}_nrowtest{N_ROW_TEST}_ntestenvs{N_TEST_ENVS}'
            cache_file += '_eval.csv'
            eval_cache = cache_dir / cache_file

            if eval_cache.exists() and not override:
                print(f"\t(Loading eval results from {eval_cache})")
                miss_rows.extend(pd.read_csv(eval_cache).to_dict('records'))
                continue

            train_seed = seed * 1000 + het_idx
            rng_train = np.random.default_rng(train_seed)
            np.random.seed(train_seed)
            torch.manual_seed(train_seed)

            covs, covs_test = make_covs(seed, a, b, rng_train)
            Ms, observed_indices, observed_entries = generate_data(
                covs, prob_source, N_ROW, N_ENVS, rng_train)

            right_factors = solve_factors(
                covs, Ms, observed_indices, observed_entries, data_prefix, 
                full_prefix, opt_tol, max_iters, rng_train, override,
            )

            current_qs = QS.copy()
            if prob_source not in current_qs:
                current_qs.append(prob_source)

            loop_rows = []
            for q in current_qs:
                env_errs_q = {m: [] for m in right_factors}
                for cov_idx, cov_t in enumerate(covs_test):
                    test_seed = train_seed * 10000 + int(q * 100) * 100 + cov_idx
                    rng_test = np.random.default_rng(test_seed)
                    e = eval_on_test_cov(right_factors, cov_t, q, rng_test)
                    for m in right_factors:
                        env_errs_q[m].append(e[m])
                for m in right_factors:
                    loop_rows.append({
                        'seed':       seed,
                        'a':          a,
                        'b':          b,
                        'q':          q,
                        'method':     m,
                        'mean':       np.mean(env_errs_q[m]),
                        'worst_case': np.max(env_errs_q[m]),
                    })

            pd.DataFrame(loop_rows).to_csv(eval_cache, index=False)
            print(f"\t(Saved eval results to {eval_cache})")
            miss_rows.extend(loop_rows)

    df_miss = pd.DataFrame(miss_rows)
    df_miss.to_csv(results_miss, index=False)
    print(f"\nSaved results to {results_miss}")
    return df_miss


# ================================================================ #
#                       Plotting Helpers                           #
# ================================================================ #

def comparison_plot_green_red(df, title='', maxmc=False, ax=None, s=10,
                               ylabel=r'$\Delta$ reconstruction error',
                               ymin=None, ymax=None):
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
            delta_avg = m2['mean'].values[0] - m1['mean'].values[0]
            delta_wc = m2['worst_case'].values[0] - m1['worst_case'].values[0]
            all_deltas.extend([delta_avg, delta_wc])

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
            delta_avg = m2['mean'].values[0] - m1['mean'].values[0]
            delta_wc  = m2['worst_case'].values[0] - m1['worst_case'].values[0]
            ax.plot(x_labels, [delta_avg, delta_wc],
                    ':', linewidth=0.8, color='#939393', zorder=1)
            ax.scatter(x_labels, [delta_avg, delta_wc],
                    s=s, marker='o', zorder=10,
                    c=[cmap(norm(delta_avg)), cmap(norm(delta_wc))])

    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)

    if created_fig:
        # Color y-tick labels by value when standalone
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        for tick_val, tick_label in zip(yticks, ax.get_yticklabels()):
            tick_label.set_color(cmap(norm(tick_val)))
        plt.tight_layout()
        plt.close()


def absolute_errors_plot(df, title='', ax=None, methods=None):
    """
    Line plot of absolute average and worst-case reconstruction errors.

    x-axis: ['average', 'worst-case'] (same structure as comparison_plot_green_red)
    Lines colored by method; markers styled by error type:
      'o' at 'average', 's' at 'worst-case'.
    One thin line per (method, seed, het); a thicker mean line per method.

    methods: optional list of method names to include (default: all in df)
    """

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(3, 2.5))

    x_labels = ['average', 'worst-case']
    methods   = list(methods) if methods is not None else list(df['method'].unique())
    het_pairs = df[['a', 'b']].drop_duplicates().values

    for method in methods:
        color = METHOD_COLORS.get(method, None)
        for seed in df['seed'].unique():
            for (a, b) in het_pairs:
                sub = df[
                    (df['seed'] == seed) & (df['a'] == a) &
                    (df['b'] == b) & (df['method'] == method)
                ]
                if sub.empty:
                    continue
                avg_err = sub['mean'].values[0]
                wc_err  = sub['worst_case'].values[0]
                ax.plot(x_labels, [avg_err, wc_err],
                        '-', linewidth=0.6, alpha=0.4, color=color, zorder=1)
                # x positions 0/1 correspond to categorical 'average'/'worst-case'
                ax.text(0, avg_err, str(seed), ha='center', va='center',
                        color=color, fontsize=5, fontweight='bold', zorder=10)
                ax.text(1, wc_err,  str(seed), ha='center', va='center',
                        color=color, fontsize=5, fontweight='bold', zorder=10)

    # Legend: method colors only (x-position already encodes error type)
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS.get(m, 'grey'), linewidth=1.5, label=m)
        for m in methods
    ]
    ax.legend(handles=legend_handles, fontsize=5, frameon=False,
              loc='upper left')
    ax.set_ylabel('reconstruction error')
    ax.set_xlim(-0.3, 1.3)
    ax.set_title(title)

    if created_fig:
        plt.tight_layout()
        plt.close()


# ================================================================ #
#                           Figures                                #
# ================================================================ #


def plot_absolute_errors(df_agg, prob_source, prob_target):
    """2-panel figure: absolute errors — (PCA, minPCA) left, (poolMC, maxMC) right."""
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    absolute_errors_plot(
        df_agg,
        title=rf'$q_\textrm{{source}}=1,\ q_\textrm{{target}}={prob_target}$',
        ax=axes[0],
        methods=['PCA', 'minPCA'],
    )
    absolute_errors_plot(
        df_agg,
        title=rf'$q_\textrm{{source}}={prob_source},\ q_\textrm{{target}}={prob_target}$',
        ax=axes[1],
        methods=['poolMC', 'maxMC'],
    )
    axes[1].set_ylabel('')
    plt.tight_layout()
    plt.savefig(FIGURE_ABS, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURE_ABS}")


def plot_test_errors(df_agg, df_miss, prob_source, ymin=None, ymax=None):
    """
    """
    methods = ['PCA', 'minPCA'] if prob_source == 1 else ['poolMC', 'maxMC']
    metric = 'diff_wc_maxRCS' if prob_source == 1 else 'diff_wc_maxMC'

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
    df_wide['diff_wc_maxRCS'] = (
        df_wide['worst_case_minPCA'] - df_wide['worst_case_PCA']
    )
    print(df_wide[(df_wide['diff_wc_maxRCS'] > 0) & (df_wide['q'] == 1)][['a', 'b']].value_counts())
    df_wide['q'] = df_wide['q'].round(2)

    
    fig, ax = plt.subplots(1, 2, figsize=(6, 1.6), 
                             sharey=True,
                           gridspec_kw={'width_ratios': [1, 1.4], 'wspace': 0.6})
    comparison_plot_green_red(
        df_agg,
        maxmc=prob_source != 1, 
        ax=ax[0], s=5, ylabel=r'$\Delta$ test RCS error',
        ymin=ymin, ymax=ymax,
    )


    # Panel 2: boxplot Δ wc maxMC vs q (sim3 style)
    df_wide['perc_missing'] = np.round(1 - df_wide['q'], 2)
    spec = dict(
        data=df_wide, x='perc_missing', y=metric, ax=ax[1], fliersize=0.5, 
        width=0.8, orient='v', hue='a', 
        # palette=["#9be15d", "#49b583", "#3a7a8a", "#3e4989"],
        palette=["#de9542", "#b85555", "#872888", "#450f68"] #["#f9831b", "#c95d77", "#6549ca", "#2e127c"]
    )
    sns.boxplot(**spec, linewidth=0, showfliers=False, boxprops=dict(alpha=.5), legend=True)
    sns.boxplot(**spec, fill=False, linewidth=0.7, legend=False)
    # ax[j].axhline(0, color='black', linestyle='--', zorder=1, linewidth=1)
    ax[1].set_ylabel(r'$\Delta$ worst-case' + '\ntest RCS error')
    ax[1].text(
        1.07, 0, '→ worse',
        transform=ax[1].get_yaxis_transform(),
        rotation=90, va='bottom', ha='left', fontsize=8,
    )
    ax[1].text(
        1.07, 0, 'maxMC is better ←',
        transform=ax[1].get_yaxis_transform(),
        rotation=90, va='top', ha='left', fontsize=8,
    )
    plt.setp(ax[1].get_xticklabels(), rotation=90, ha='center')
    ax[1].set_xlabel('test \\% missing')
    ax[1].yaxis.get_label().set_visible(True)
    # Ensure the y-axis itself, ticks and tick labels are visible
    ax[1].yaxis.set_visible(True)
    ax[1].tick_params(axis='y', which='both', labelleft=True)
    for lbl in ax[1].get_yticklabels():
        lbl.set_visible(True)
    
    fig.subplots_adjust(right=0.7)
    handles, labels = ax[1].get_legend_handles_labels()
    new_labels = {'0.0': '(0, 0.5)', '0.5': '(0.5, 1)', '1.0': '(1, 2)', '2.0': '(2, 5)'}
    new_handles = [handles[i] for i in range(len(labels)) if labels[i] in new_labels]
    new_labels = [new_labels[labels[i]] for i in range(len(labels)) if labels[i] in new_labels]
    ax[1].legend(
        new_handles, new_labels, 
        loc='center left',
        bbox_to_anchor=(1.2, 0.5),
        borderaxespad=0,
        frameon=False,
        title=r'$(\alpha, \beta)$',
        # --- Sizing Adjustments ---
        handlelength=1.0,  # Default is 2.0; making it 1.0 makes it a square
        handleheight=0.6,  # Adjusts the vertical thickness of the box
        handletextpad=0.5, # Adjusts spacing between the box and the text
        labelspacing=0.5   # Adjusts vertical spacing between different legend items
    )
     
    for j in range(2):
        ax[j].grid(axis='y', linestyle='--', alpha=0.7)
        
    figpath = Path(f'figures/{FIGSUBDIR}/sim5_main_{'obs' if prob_source == 1.0 else 'miss'}.png')
    plt.savefig(figpath, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {figpath}")


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
        "--max_iters", type=int, default=100,
        help="Maximum iterations for poolMC and maxMC (default: 50)",
    )
    parser.add_argument(
        "--start_seed", type=int, default=0,
        help="Starting seed index (default: 0)",
    )
    parser.add_argument(
        "--end_seed", type=int, default=25,
        help="Ending seed index (default: 25)",
    )
    args = parser.parse_args()
    prob_source = args.prob_source
    opt_tol = args.opt_tol
    max_iters = args.max_iters
    global SEEDS
    SEEDS = list(range(args.start_seed, args.end_seed))

    Path('results').mkdir(exist_ok=True)
    Path('results/sim5').mkdir(exist_ok=True)
    Path(f'results/sim5/{SUBDIR}').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)
    Path(f'figures/{FIGSUBDIR}').mkdir(exist_ok=True)

    plt.style.use('jmlr.mplstyle')

    file_suffix = f'src{int(prob_source * 100)}_opt{int(opt_tol * 1e6)}_mi{max_iters}'
    file_suffix += f'_seed{SEEDS[0]}-{SEEDS[-1]}_nrowtest{N_ROW_TEST}_ntestenvs{N_TEST_ENVS}'
    results = Path(f'results/sim5/{SUBDIR}/{file_suffix}.csv')
    print(f"Results file: {results}")

    if args.rerun or not results.exists():
        df = run_simulation(
            prob_source, opt_tol, max_iters, results,
            override=args.rerun,
        )
    else:
        print(f'Loading cached results from {results}')
        df = pd.read_csv(results)

    df['mean'] = 1e4 * df['mean']  # scale up for better visibility in figures
    df['worst_case'] = 1e4 * df['worst_case']
    df_plot = df[df['q'].isin([0.05, 0.2, 0.5, 0.8])]
    prob_target = 0.2 # prob_source
    df_agg = df[df['q'] == prob_target].copy()
    
    plot_absolute_errors(df_agg, prob_source, prob_target)
    plot_test_errors(df_agg, df_plot, prob_source=1, ymin=-2.4, ymax=0.7)
    plot_test_errors(df_agg, df_plot, prob_source=0.1, ymin=-2.4, ymax=0.7)


if __name__ == '__main__':
    main()
