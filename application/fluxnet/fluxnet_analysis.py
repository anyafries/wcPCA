"""
FLUXNET minPCA Analysis Script

This computes minPCA variants on FLUXNET data grouped by TRANSCOM regions and generates
comparison plots.

Usage:
    python fluxnet_analysis.py [--rerun]

    --rerun: Force recomputation of results (otherwise loads from cache if available)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from utils import loo_time_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_preprocessing import create_poolscale, create_envzeromean, build_env_dicts

# =============================================================================
# Configuration
# =============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
RESULTS_DIR = SCRIPT_DIR / "results"
STYLE_PATH = SCRIPT_DIR.parent.parent / "jmlr.mplstyle"

# Analysis parameters
N_COMPONENTS = 6       # Maximum number of components to compute
NUM_SPLITS = 20        # Number of random train/test splits for many-splits analysis
N_RESTARTS = 10        # Number of restarts for minPCA optimization
SEED = 5               # Seed for the single-split analysis
N_PCS = 2              # Number of PCs for Plot 4 (environment comparison)

# Method names (as used in the results dataframes)
BASE_METHOD = 'PCA'          # baseline method
MAIN_METHOD = 1 # index of the main method
COMPARISON_METHODS = ['regret', 'norm-regret', 'norm-maxRCS', 'avgcovPCA']  # minPCA method

# Display labels for legends
COMPARISON_METHODS_LABEL = ['maxRegret', 'norm-maxRegret', 'norm-maxRCS', 'avgcovPCA']
BASE_METHOD_LABEL = 'poolPCA'

# Columns for minPCA analysis
MINPCA_COLS = [
    'GPP', 'Tair', 'vpd',
    'LST_TERRA_Day', 'LST_TERRA_Night', 'EVI', 'NIRv', 'NDWI_band7',
    'LAI', 'fPAR',
]

# X-axis ordering for Plot 4 (set to None for default order, or provide list of region names)
# These will be set once you see the data
# Source regions order (list of environment_tag strings)
x_order1 = [
    'Eurasia Boreal', 'Eurasia Temperate',
    'S. American Temperate', 'N. American Temperate', 'N. Africa'
]  
# Target regions order (list of environment_tag strings)
x_order2 = [
    'Australia', 'Europe', 'N. Ocean',  'Tropical Asia', 'S. Africa', 
    'S. American Tropical', 'N. American Boreal',  'N. Pacific Temperate'
]

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_preprocess_data():
    """Load and preprocess FLUXNET data."""
    # Load data
    df = pd.read_csv(DATA_DIR / "daily.csv", index_col=0)
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df = df[df['year'] > 1999]

    # Load region mapping
    regions = pd.read_csv(DATA_DIR / "fluxnet_site_region_dict.csv")
    df['region'] = df['site_id'].map(regions.set_index('site_id')['Region'])
    df = df[df['region'].notna()]

    environment_column = 'region'

    # Clean data
    minpca_df = df[MINPCA_COLS].copy()

    # Outlier filtering using IQR
    Q1 = minpca_df.quantile(0.25)
    Q3 = minpca_df.quantile(0.75)
    IQR = Q3 - Q1
    cutoff = 2
    filter_mask = ~(
        (minpca_df < (Q1 - cutoff * IQR)) | 
        (minpca_df > (Q3 + cutoff * IQR))
    ).any(axis=1)

    # Keep only complete rows
    filter_mask = filter_mask & ~minpca_df.isna().any(axis=1)

    # Keep only positive GPP values
    filter_mask = filter_mask & (minpca_df['GPP'] >= 0)

    # Check how many samples were removed per environment
    raw_n = []
    clean_n = []
    for env in df[environment_column].unique():
        idx = df[environment_column] == env
        n_samples_raw = df[idx].shape[0]
        n_samples_clean = df[idx & filter_mask].shape[0]
        raw_n.append(n_samples_raw)
        clean_n.append(n_samples_clean)
    raw_n = np.array(raw_n)
    clean_n = np.array(clean_n)

    # Drop environments where more than 50% of data was removed
    envs_50perc_removed = [df[environment_column].unique()[i]
                          for i in np.where((raw_n - clean_n)/raw_n * 100 > 50)[0]]
    print(f"Dropping environments with >50% data removed: {envs_50perc_removed}")

    filter_mask &= ~df[environment_column].isin(envs_50perc_removed)
    minpca_df = minpca_df[filter_mask]
    environments = df[environment_column][filter_mask]

    # Create poolscale version (standardized on pooled data)
    X_poolscale_df = create_poolscale(minpca_df, MINPCA_COLS, ddof=0)

    # Create envmeanzero version (centered by environment, scaled on pool)
    X_envzeromean_df = create_envzeromean(minpca_df, MINPCA_COLS, environments, ddof=0)

    # Build environment dictionaries
    env_dfs_poolscale, env_dfs_envmeanzero = build_env_dicts(
        X_poolscale_df, X_envzeromean_df, environments
    )

    print(f"Remaining regions: {environments.nunique()}")
    print(f"Total samples: {len(minpca_df)}")

    return X_envzeromean_df, environments, env_dfs_envmeanzero


# =============================================================================
# Analysis Functions
# =============================================================================

def run_many_splits(X_poolscale_df, environments, env_dfs_envmeanzero,
                    num_splits=NUM_SPLITS, n_restarts=N_RESTARTS,
                    n_components=N_COMPONENTS, rerun=False):
    """Run analysis over multiple random train/test splits.

    Results are cached to disk. Set rerun=True to force recomputation.
    """
    cache_suffix = f"_splits{num_splits}_restarts{n_restarts}_ncomp{n_components}"
    wc_in_path = RESULTS_DIR / f"wc_in_all_df{cache_suffix}.csv"
    wc_out_path = RESULTS_DIR / f"wc_out_all_df{cache_suffix}.csv"
    pool_in_path = RESULTS_DIR / f"pool_in_all_df{cache_suffix}.csv"
    pool_out_path = RESULTS_DIR / f"pool_out_all_df{cache_suffix}.csv"
    cache_suffix2 = f"_seed{SEED}_restarts{n_restarts * 2}_ncomp{n_components}"
    out_ts_path = RESULTS_DIR / f"out_ts{cache_suffix2}.csv"

    if not rerun and wc_in_path.exists() and wc_out_path.exists():
        print("Loading cached many-splits results...")
        wc_in_df = pd.read_csv(wc_in_path)
        wc_out_df = pd.read_csv(wc_out_path)
        pool_in_df = pd.read_csv(pool_in_path)
        pool_out_df = pd.read_csv(pool_out_path)
        out_single_run = pd.read_csv(out_ts_path)
        return wc_in_df, wc_out_df, pool_in_df, pool_out_df, out_single_run

    out_lists = {
        'wc_in': [],
        'wc_out': [],
        'pool_in': [],
        'pool_out': [],
    }
    out_single_run = None

    for seed in range(num_splits):
        print(f"Processing split {seed + 1}/{num_splits}...")
        np.random.seed(seed)
        unique_envs = np.sort(environments.unique())
        train_envs = list(np.random.choice(unique_envs, 5, replace=False))
        test_envs = [e for e in unique_envs if e not in train_envs]

        out_ts, _ = loo_time_split(
            Xpool_df_in=X_poolscale_df,
            environments=environments,
            Xs_dict=env_dfs_envmeanzero,
            train_envs=train_envs,
            test_envs=test_envs,
            n_restarts=n_restarts,
            n_components=n_components,
        )

        if seed == SEED:
            out_single_run = out_ts.copy()
            out_ts.to_csv(out_ts_path, index=False)

        out_ts2 = out_ts.copy() 
        for key in out_lists.keys():
            if key in ['wc_in', 'wc_out']:
                df = out_ts2[~out_ts2['environment'].isin(
                    ['pooled (in-sample)', 'pooled (out-of-sample)'])]
                agg_err, agg_var = 'max', 'min'
                if key == 'wc_in':
                    df = df[df['method'].str.contains('_train')]
                else: # 'wc_out'
                    df = df[~df['method'].str.contains('_train')]
            else:
                agg_err, agg_var = 'mean', 'mean'
                if key == 'pool_in':
                    df = out_ts2[out_ts2['environment'] == 'pooled (in-sample)']
                else: # 'pool_out'
                    df = out_ts2[out_ts2['environment'] == 'pooled (out-of-sample)']

            df = (
                df
                .groupby(['n_components', 'method'])
                .agg({
                    'err': agg_err, 'var': agg_var,
                    'err_unnorm': agg_err, 'var_unnorm': agg_var
                }).reset_index()
            )
            df['seed'] = seed
            out_lists[key].append(df)

    wc_in_df = pd.concat(out_lists['wc_in'], ignore_index=True)
    wc_out_df = pd.concat(out_lists['wc_out'], ignore_index=True)
    pool_in_df = pd.concat(out_lists['pool_in'], ignore_index=True)
    pool_out_df = pd.concat(out_lists['pool_out'], ignore_index=True)

    # Save to cache
    wc_in_df.to_csv(wc_in_path, index=False)
    wc_out_df.to_csv(wc_out_path, index=False)
    pool_in_df.to_csv(pool_in_path, index=False)
    pool_out_df.to_csv(pool_out_path, index=False)

    return wc_in_df, wc_out_df, pool_in_df, pool_out_df, out_single_run


def run_single_split(X_poolscale_df, environments, env_dfs_envmeanzero,
                     seed=SEED, n_restarts=N_RESTARTS, n_components=N_COMPONENTS,
                     rerun=False):
    """Run analysis for a single train/test split.

    Results are cached to disk. Set rerun=True to force recomputation.
    """
    np.random.seed(seed)
    unique_envs = np.sort(environments.unique())
    train_envs = list(np.random.choice(unique_envs, 5, replace=False))
    test_envs = [e for e in unique_envs if e not in train_envs]

    print(f"Training environments: {train_envs}")
    print(f"Test environments: {test_envs}")

    cache_suffix = f"_seed{seed}_restarts{n_restarts * 2}_ncomp{n_components}"
    out_ts_path = RESULTS_DIR / f"out_ts{cache_suffix}.csv"

    if not rerun and out_ts_path.exists():
        print("Loading cached single-split results...")
        out_ts = pd.read_csv(out_ts_path)
    else:
        out_ts, _ = loo_time_split(
            Xpool_df_in=X_poolscale_df,
            environments=environments,
            Xs_dict=env_dfs_envmeanzero,
            train_envs=train_envs,
            test_envs=test_envs,
            n_restarts=n_restarts * 2,  # More restarts for single split
            n_components=n_components,
        )
        out_ts.to_csv(out_ts_path, index=False)

    return out_ts


# =============================================================================
# Plotting Functions
# =============================================================================

def compute_diffs(df, y, method1, method2, variance=False, return_seeds=False):
    assert ((y == 'err' and not variance) or (y == 'var' and variance))
    diffs = []
    relative_diffs = []
    for seed in df['seed'].unique():
        df1 = df[(df['method'] == method1) & (df['seed'] == seed)]
        df2 = df[(df['method'] == method2) & (df['seed'] == seed)]
        diff = df2[y].values - df1[y].values
        assert len(diff) == 1, f"Expected one row per method/seed, got {len(diff)}"
        diffs.append(diff[0])
        relative_diffs.append(diff[0] / df1[y].values[0])
    
    if return_seeds:
        return diffs, relative_diffs, list(df['seed'].unique())
    else:
        return diffs, relative_diffs


def compare_errs_across_methods(df, ax, y, 
                                num_components=2, 
                                relative=True,
                                variance=True):
    df_sub = df[df['n_components'] == num_components]
    diffs = []
    relative_diffs = []
    names = []

    train = ('PCA_train' in df_sub['method'].unique())
    base_method = f'{BASE_METHOD}_train' if train else BASE_METHOD
    comparison_methods = [f'{m}_train' if train else m for m in COMPARISON_METHODS]

    for method2, method2_label in zip(comparison_methods, COMPARISON_METHODS_LABEL):
        diff, relative_diff = compute_diffs(df_sub, y, base_method, method2, 
                                            variance=variance)
        diffs += diff
        relative_diffs += relative_diff
        names += [method2_label for _ in diff]
    
    df_diffs = pd.DataFrame({
        'method': names,
        'diff': diffs,
        'relative_diff': relative_diffs,
    })
    # print(df_diffs[df_diffs['method'] == COMPARISON_METHODS_LABEL[MAIN_METHOD]])
    print(df_diffs.groupby('method')[['relative_diff', 'diff']].median().reset_index())
    sns.boxplot(data=df_diffs, x='method', 
                y='relative_diff' if relative else 'diff', ax=ax, width=0.7,
                linewidth=1, order=COMPARISON_METHODS_LABEL,
                boxprops=dict(facecolor="#A4D4E0"), fliersize=3)
    


def compare_errs_across_pcs(df, ax, y, method1, method2, relative=True, 
                            variance=True):
    df_sub = df[df['method'].isin([method1, method2])]
    
    diffs = []
    relative_diffs = []
    n_components = []

    for k in df_sub['n_components'].unique():
        diff, relative_diff = compute_diffs(df_sub[df_sub['n_components'] == k], 
                                            y, method1, method2, 
                                            variance=variance)
        diffs += diff
        relative_diffs += relative_diff
        n_components += [k for _ in diff]
    
    df_diffs = pd.DataFrame({
        'n_components': n_components,
        'diff': diffs,
        'relative_diff': relative_diffs,
    })

    sns.boxplot(data=df_diffs, x='n_components', 
                y='relative_diff' if relative else 'diff', 
                ax=ax, width=0.7, linewidth=1,
                boxprops=dict(facecolor="#A4D4E0"), fliersize=3)
    

def plot_boxplot_npcs(wc_out_df, num_components=2, y='err', average=False,
                      save_path=None, source=False, relative=True, variance=True):
    fig, ax = plt.subplots(figsize=(2.5, 2.5), sharey=False)
    compare_errs_across_methods(
        wc_out_df, ax, y,
        num_components=num_components,
        variance=variance, relative=relative
    )
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    metric = 'average' if average else 'worst-case'
    domain = 'source' if source else 'target'
    ylabel = r'Relative $\Delta$' if relative else 'Difference'
    ylabel += f' {metric}\n{domain} \\% expl. var.'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Method')
    ax.tick_params(axis='x', rotation=90)
    ax.text(1.07, 0, '→ better than poolPCA',
            transform=ax.get_yaxis_transform(),
            rotation=90, va='bottom', ha='left', fontsize=8)
    ax.text(1.07, 0, 'worse ←',
            transform=ax.get_yaxis_transform(),
            rotation=90, va='top', ha='left', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_boxplot_comparison(wc_in_df, wc_out_df, method, method_label, 
                            relative=True, variance=True,
                            highlight_seed=None, ax=None, save_path=None):
    """
    Boxplot comparing method vs BASE_METHOD across splits.

    Parameters
    ----------
    highlight_seed : int or None
        If provided, overlay scatter points for the specified seed.
    ax : array of 2 Axes or None
        If provided, plot on these axes. Otherwise create a new figure.
    """
    # Compute comparison DataFrames for the scatter overlay (if needed)
    comp_dfs = None
    if highlight_seed is not None:
        comp_dfs = []
        for i, df1 in enumerate([wc_in_df, wc_out_df]):
            diffs, rel_diffs, seeds = [], [], []
            n_components = []
            for k in df1['n_components'].unique():
                diff, rel_diff, seed_list = compute_diffs(
                    df1[df1['n_components'] == k], 'var', 
                    f'{BASE_METHOD}_train' if i == 0 else BASE_METHOD, 
                    f'{method}_train' if i == 0 else method, 
                    variance=variance, return_seeds=True
                )
                diffs += diff
                rel_diffs += rel_diff
                seeds += seed_list
                n_components += [k for _ in diff]   
            comp_df = pd.DataFrame({
                'n_components': n_components,  # Adjust for boxplot x-axis
                'diff': diffs,
                'relative_diff': rel_diffs,
                'seed': seeds
            })
            comp_dfs.append(comp_df)

    # Create figure if ax not provided
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(1, 2, figsize=(4, 1.8), sharey=True)

    compare_errs_across_pcs(wc_in_df, ax[0], 'var', 
                            f'{BASE_METHOD}_train',
                            f'{method}_train',
                            variance=variance, relative=relative)
    ax[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    ymin, ymax = (-0.85, 0.85) if relative else (-0.25, 0.4)
    ax[0].set_ylim(ymin, ymax)
    if comp_dfs is not None:
        sns.scatterplot(comp_dfs[0][comp_dfs[0].seed == highlight_seed],
                        x='n_components', 
                        y='relative_diff' if relative else 'diff', 
                        ax=ax[0], zorder=10,
                        color='tomato', s=25)

    compare_errs_across_pcs(wc_out_df, ax[1], 'var', BASE_METHOD, method,
                            variance=variance, relative=relative)
    ax[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    if comp_dfs is not None:
        sns.scatterplot(comp_dfs[1][comp_dfs[1].seed == highlight_seed],
                        x='n_components', 
                        y='relative_diff' if relative else 'diff',
                        ax=ax[1], zorder=10,
                        color='tomato', s=25)

    # Only set labels/titles if we own the figure
    if own_figure:
        ax[0].set_ylabel('Difference in worst-case\n\\% explained variance', fontsize=7)
        ax[0].set_title('Source regions', fontsize=8)
        ax[0].set_xlabel('Number of PCs', fontsize=7)
        ax[1].set_title('Target regions', fontsize=8)
        ax[1].set_xlabel('Number of PCs', fontsize=7)
        if method_label:
            plt.suptitle(f"{method_label} vs. {BASE_METHOD_LABEL}", x=0, y=0.92,
                         ha='left', va='top', fontsize=9, fontweight='bold')
        plt.tight_layout()
        if save_path:
            print(f"{save_path}_{method}.pdf")
            plt.savefig(f"{save_path}_{method}.pdf", bbox_inches='tight')
        plt.close()


def plot_boxplot_comparison_grid(wc_in_df, wc_out_df, methods, method_labels,
                                 relative=True, variance=True,
                                 highlight_seed=None, save_path=None):
    """
    Plot a grid of boxplot comparisons for multiple methods.

    Parameters
    ----------
    methods : list of str
        Method names to compare against BASE_METHOD.
    method_labels : list of str
        Display labels for each method.
    highlight_seed : int or None
        If provided, overlay scatter points for the specified seed.
    """
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 2, figsize=(4, 1.2 * n_methods),
                              sharex=True, sharey=True)

    for row, (method, method_label) in enumerate(zip(methods, method_labels)):
        ax_row = axes[row]
        plot_boxplot_comparison(
            wc_in_df, wc_out_df, method, method_label,
            highlight_seed=highlight_seed, ax=ax_row, 
            variance=variance, relative=relative
        )

        # Add bold method label rotated vertically to the left of ylabel
        ax_row[0].text(-0.45, 0.5, method_label, transform=ax_row[0].transAxes,
                       fontsize=8, fontweight='bold', rotation=90,
                       va='center', ha='center')

        # Only show x-axis labels on bottom row
        if row < n_methods - 1:
            ax_row[0].set_xlabel('')
            ax_row[1].set_xlabel('')
        else:
            ax_row[0].set_xlabel('Number of PCs', fontsize=7)
            ax_row[1].set_xlabel('Number of PCs', fontsize=7)

        ylab = 'Relative ' if relative else ''
        ylab += r'$\Delta$ worst-case'
        ylab += '\n' if relative else ''
        ylab += '\\% expl. var.'
        ax_row[0].set_ylabel(ylab, fontsize=6)

    # Set column titles only on top row
    axes[0, 0].set_title('Source regions', fontsize=8)
    axes[0, 1].set_title('Target regions', fontsize=8)

    plt.tight_layout()
    fig.subplots_adjust(left=0.22)  # Make room for rotated method labels

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_scree(environments, env_dfs, save_path=None):
    """
    Plot 3: Scree plot of eigenvalues per training region.
    """
    Xs_train = [env_dfs[env] for env in environments]
    covs = [torch.cov(X.T) for X in Xs_train]

    plt.figure(figsize=(2.5, 1.7))
    for C, name in zip(covs, environments):
        eigvals, _ = torch.linalg.eigh(C)
        eigvals = eigvals.flip(0).cpu().numpy()
        plt.plot(range(1, len(eigvals)+1), eigvals, label=name,
                 markersize=4, linewidth=1.5, marker='o')
    plt.xlabel('PC')
    plt.ylabel('Eigenvalue')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_environment_comparison(out_ts,
                                method=COMPARISON_METHODS[MAIN_METHOD], 
                                method_label=COMPARISON_METHODS_LABEL[MAIN_METHOD],
                                n_pcs=N_PCS, ylim=[None, None],
                                x_order1=None, x_order2=None, save_path=None):
    """
    Plot 4: Line plots showing % explained variance per environment.

    Parameters
    ----------
    out_ts : pd.DataFrame
        Results from loo_time_split
    n_pcs : int
        Number of PCs to show
    ylim : list of 2 floats
        Y-axis limits for the two subplots (source and target)
    x_order1 : list or None
        Order for source regions x-axis (use environment_tag values)
    x_order2 : list or None
        Order for target regions x-axis (use environment_tag values)
    save_path : str or None
        Path to save figure
    """
    out_ts2 = out_ts[out_ts['n_components'] == n_pcs].copy() 

    # Create environment tags (shortened names)
    out_ts2['environment_tag'] = (out_ts2['environment']
        .str.replace('Northern', 'N.')
        .str.replace('Southern', 'S.')
        .str.replace('North', 'N.')
        .str.replace('South', 'S.')
    )

    # Source regions (training)
    yy = out_ts2[out_ts2['method'].isin([f'{BASE_METHOD}_train', f'{method}_train'])].copy()

    # Target regions (test)
    zz = out_ts2[out_ts2['method'].isin([BASE_METHOD, method])].copy()

    # Apply custom ordering if provided
    if x_order1 is not None:
        yy['environment_tag'] = pd.Categorical(
            yy['environment_tag'],
            categories=x_order1,
            ordered=True
        )
        yy = yy.sort_values('environment_tag').reset_index(drop=True)

    if x_order2 is not None:
        zz['environment_tag'] = pd.Categorical(
            zz['environment_tag'],
            categories=x_order2,
            ordered=True
        )
        zz = zz.sort_values('environment_tag').reset_index(drop=True)

    fig, ax = plt.subplots(1, 2, figsize=(5.3, 2.5), sharey=False, 
                           width_ratios=[1, 1.3])

    for i, df in enumerate([yy, zz]):
        sns.lineplot(data=df, x='environment_tag', y='var', hue='method',
                     marker='o', palette=['tab:blue', 'tab:red'], ax=ax[i])
        ax[i].set_ylim(ylim[i])
        ax[i].tick_params(axis='x', rotation=30, labelsize=7)
        for label in ax[i].get_xticklabels():
            label.set_ha('right')
        domain = 'Source' if i == 0 else 'Target'
        ax[i].set_xlabel(f'{domain} regions')
        ax[i].set_ylabel('\n\\% explained variance')

        # Add annotation arrow for worst environment
        worst_env = df.loc[df['var'].idxmin()]['environment']
        method_suffix = '_train' if i == 0 else ''
        y_end = df[(df['environment'] == worst_env) &
                   (df['method'] == method + method_suffix)]['var'].values[0]
        y_start = df[(df['environment'] == worst_env) &
                     (df['method'] == BASE_METHOD + method_suffix)]['var'].values[0]
        x_pos = df[df['environment'] == worst_env].index[0] // 2
        ax[i].annotate('',
                       xy=(x_pos, y_end),
                       xytext=(x_pos, y_start),
                       arrowprops=dict(arrowstyle='->', color='tab:green', lw=2))
        diff = (y_end - y_start) / y_start * 100
        ax[i].text(x_pos + 0.25 * (i + 1), (y_start + y_end) / 2 - 0.02, 
                   f'{diff:.1f}\\%',
                   color='tab:green', fontsize=9, ha='left', va='center')

    # Remove legends from both axes
    ax[0].legend_.remove()
    if ax[1].get_legend() is not None:
        ax[1].legend_.remove()

    # Apply tight_layout first, then add legend outside
    plt.tight_layout()
    fig.subplots_adjust(right=0.82)

    # Create combined legend
    handles, _ = ax[0].get_legend_handles_labels()
    new_labels = [BASE_METHOD_LABEL, method_label]
    fig.legend(
        handles[:2],
        new_labels,
        title='Method',
        loc='center left',
        bbox_to_anchor=(0.83, 0.7),
        borderaxespad=0,
        frameon=False
    )
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main(rerun=False):
    """Main function to run the analysis."""
    # Set style
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    # Set seeds
    np.random.seed(2)
    torch.manual_seed(2)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_poolscale_df, environments, env_dfs_envmeanzero = load_and_preprocess_data()

    # -------------------------------------------------------------------------
    # Many splits analysis 
    # -------------------------------------------------------------------------
    print("Running many-splits analysis...")
    wc_in_df, wc_out_df, pool_in_df, pool_out_df, out_single = run_many_splits(
        X_poolscale_df, environments, env_dfs_envmeanzero, rerun=rerun
    )

    # Compute differences across splits
    comp_dfs = []
    for i, df1 in enumerate([wc_in_df, wc_out_df]):
        diffs = []
        ranks = []
        seeds = []
        for seed in wc_in_df['seed'].unique():
            m1, m2 = COMPARISON_METHODS[MAIN_METHOD], BASE_METHOD
            if i == 0:
                m1 = f'{m1}_train'
                m2 = f'{m2}_train'
            df1_m1_seed = df1[(df1['method'] == m1) & (df1['seed'] == seed)]
            df1_m2_seed = df1[(df1['method'] == m2) & (df1['seed'] == seed)]
            x = df1_m1_seed['n_components'].unique()
            y1_seed = df1_m1_seed['var'].values - df1_m2_seed['var'].values
            diffs.append(y1_seed)
            ranks.append(x)
            seeds += [seed for _ in range(len(x))]

        df2 = pd.DataFrame({
            'n_components': np.concatenate(ranks) - 1,
            'diff': np.concatenate(diffs),
            'seed': seeds
        })
        comp_dfs.append(df2)

    # Print summary: how many splits show improvement at N_PCS
    df_in = comp_dfs[0]
    df_out = comp_dfs[1]
    n_positive_in = np.sum(df_in[df_in['n_components'] == N_PCS - 1]['diff'] > 0)
    n_positive_out = np.sum(df_out[df_out['n_components'] == N_PCS - 1]['diff'] > 0)
    n_total = len(df_in[df_in['n_components'] == N_PCS - 1])
    print(f"Summary of improvements for {COMPARISON_METHODS_LABEL[MAIN_METHOD]}:")
    print(f"Splits with positive improvement at {N_PCS} PCs:")
    print(f"  In-sample (source): {n_positive_in}/{n_total}")
    print(f"  Out-of-sample (target): {n_positive_out}/{n_total}")

    # -------------------------------------------------------------------------
    # Pooled results comparison
    # -------------------------------------------------------------------------
    print("\nPooled results:")
    pool_outsample = out_single[out_single['environment'] == 'pooled (out-of-sample)']
    pool_insample = out_single[out_single['environment'] == 'pooled (in-sample)']

    aa = pool_outsample[pool_outsample['n_components'] == N_PCS]
    a1 = aa[aa['method'] == COMPARISON_METHODS[MAIN_METHOD]]['var'].values[0]
    a2 = aa[aa['method'] == BASE_METHOD]['var'].values[0]

    bb = pool_insample[pool_insample['n_components'] == N_PCS]
    b1 = bb[bb['method'] == COMPARISON_METHODS[MAIN_METHOD]]['var'].values[0]
    b2 = bb[bb['method'] == BASE_METHOD]['var'].values[0]

    label = COMPARISON_METHODS_LABEL[MAIN_METHOD]
    print(f"  Out-of-sample rel decrease for {label}: {(a1-a2)/a2 * 100:.3f}%")
    print(f"  In-sample rel decrease for {label}: {(b1-b2)/b2 * 100:.3f}%")

    # -------------------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------------------
    print("\nGenerating plots...")

    # Plot 1: Boxplot comparison grid (suppl.)
    print("  Plot 1: Boxplot comparison grid with/out highlight")
    plot_boxplot_comparison_grid(
        wc_in_df, wc_out_df, COMPARISON_METHODS, COMPARISON_METHODS_LABEL,
        save_path=FIGURES_DIR / "fluxnet_boxplot_comparison_grid.pdf"
    )
    plot_boxplot_comparison_grid(
        wc_in_df, wc_out_df, COMPARISON_METHODS, COMPARISON_METHODS_LABEL,
        highlight_seed=SEED,
        save_path=FIGURES_DIR / "fluxnet_boxplot_highlight_grid.pdf"
    )

    # Plot 2: Boxplot comparison (main, only 2 PCs)
    print(f"  Plot 2: Boxplot comparison for {N_PCS} PCs")
    for domain in ['source', 'target']:
        for metric in ['worst-case', 'average']:
            if metric == 'worst-case':
                df = wc_in_df if domain == 'source' else wc_out_df
            else:
                df = pool_in_df if domain == 'source' else pool_out_df
            print(f"---> Median {metric} {domain} difference")
            plot_boxplot_npcs(
                df,
                num_components=N_PCS, y='var',
                average=(metric == 'average'), source=(domain == 'source'),
                save_path=FIGURES_DIR / f"fluxnet_boxplot_npcs_{domain}_{metric}.pdf"
            )

    # Plot 3: Scree plot
    print("  Plot 3: Scree plot")
    plot_scree(
        environments.unique(), env_dfs_envmeanzero,
        save_path=FIGURES_DIR / "fluxnet_scree_plot.pdf"
    )

    # Plot 4: Environment comparison
    print("  Plot 4: Environment comparison")
    plot_environment_comparison(
        out_single, n_pcs=N_PCS,
        method='norm-regret', method_label='norm-maxRegret',
        x_order1=x_order1, x_order2=x_order2,
        save_path=FIGURES_DIR / "fluxnet_environment_comparison.png"
    )

    print("\nDone! Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUXNET minPCA Analysis")
    parser.add_argument("--rerun", action="store_true",
                        help="Force recomputation of results")
    args = parser.parse_args()

    main(rerun=args.rerun)
