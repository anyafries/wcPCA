"""
Ecosystem minPCA Analysis Script

In-sample analysis of ecosystem functional properties across 6 continents.
All environments are used for fitting and evaluation (no train/test splits).

Usage:
    python ecosystem_analysis.py [--rerun]

    --rerun: Force recomputation of results (otherwise loads from cache if available)
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA

from minPCA.minpca import minPCA, get_errs_pca, get_vars_pca, generate_params

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_preprocessing import create_poolscale, create_envzeromean, build_env_dicts

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
RESULTS_DIR = SCRIPT_DIR / "results"
STYLE_PATH = SCRIPT_DIR.parent.parent / "jmlr.mplstyle"

N_COMPONENTS = 5
N_RESTARTS = 10

MINPCA_COLS = [
    'uWUE', 'ETmax', 'GSmax', 'G1', 'EF', 'EFampl',
    'GPPsat', 'NEPmax', 'Rb', 'Rbmax', 'aCUE', 'WUEt',
]

ENVIRONMENT_COLUMN = 'continent'

# Methods to include in the main comparison line plot
PLOT_METHODS = ['PCA(zeromean)', 'norm-minPCA', 'norm-maxRegret']
PLOT_LABELS = ['poolPCA', 'norm-maxRCS', 'norm-maxRegret'] 

PLOT_METHODS_EXT = PLOT_METHODS + ['Average covariance', 'maxRegret']
PLOT_LABELS_EXT = PLOT_LABELS + ['avgcovPCA', 'maxRegret']

# Custom row ordering for component bar plots
COMPONENT_ROW_ORDER = np.flip(np.array([11, 0, 9, 8, 7, 2, 6, 3, 1, 5, 4, 10]))

palette = {
    'norm-maxRCS': '#2f85c3',
    'norm-maxRegret': "#972597",
    'poolPCA': "tab:red",
    'avgcovPCA': "#ec732c",
    'maxRegret': "#BB80BB",
}


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_preprocess_data():
    """Load ecosystem CSV and preprocess into analysis-ready formats.

    Returns
    -------
    env_dfs_poolscale : dict[str, torch.Tensor]
    env_dfs_envmeanzero : dict[str, torch.Tensor]
    X_poolscale_df : pd.DataFrame
    X_envzeromean_df : pd.DataFrame
    continents : pd.Series
    """
    df = pd.read_csv(DATA_DIR / "1-1-EFPN_with_pcs_and_continents.csv", index_col=0)
    continents = df[ENVIRONMENT_COLUMN]

    minpca_df = df[MINPCA_COLS].copy()
    minpca_df = minpca_df.fillna(minpca_df.mean())

    # Verify no zero-variance columns per continent
    for c in continents.unique():
        dfe = minpca_df[continents == c]
        assert dfe.std().min() > 1e-6, f"{c} has zero-variance columns"
        assert dfe.isna().sum().sum() == 0

    X_poolscale_df = create_poolscale(minpca_df, MINPCA_COLS, ddof=0)
    X_envzeromean_df = create_envzeromean(minpca_df, MINPCA_COLS, continents, ddof=0)

    env_dfs_poolscale, env_dfs_envmeanzero = build_env_dicts(
        X_poolscale_df, X_envzeromean_df, continents
    )

    print(f"Data: {len(minpca_df)} sites, {len(MINPCA_COLS)} features, "
          f"{continents.nunique()} continents")

    return (env_dfs_poolscale, env_dfs_envmeanzero,
            X_poolscale_df, X_envzeromean_df, continents)


# =============================================================================
# Analysis Functions
# =============================================================================

def fit_all_methods(covs_envmeanzero, X_pool_zeromean, X_poolscale,
                    n_components=N_COMPONENTS, n_restarts=N_RESTARTS):
    """Fit all 8 PCA methods.

    Returns
    -------
    pca_dict : dict[str, list[np.ndarray]]
        Method name -> list of component arrays (one per rank 1..n_components).
    fitted_objects : dict
        Fitted PCA/minPCA objects needed for component plots.
    """
    # 1. PCA on pooled envzeromean data
    pca_zeromean = PCA(n_components=n_components)
    pca_zeromean.fit(X_pool_zeromean)

    # 2. PCA on pooled poolscale data
    pca_poolscale = PCA(n_components=n_components)
    pca_poolscale.fit(X_poolscale)

    # 3. Average covariance
    avg_cov = torch.stack(covs_envmeanzero).mean(dim=0)
    _, _, V = torch.svd(avg_cov)
    components_avg_cov = V.numpy()

    # 4. norm-minPCA (joint, fit separately for each rank)
    joint_components = []
    for i in range(n_components):
        m = minPCA(n_components=i + 1, norm=True)
        m.fit(covs_envmeanzero, n_restarts=n_restarts)
        joint_components.append(m.components())

    # 5. minPCA (unnormalized)
    components_minpca_unnorm = []
    for i in range(n_components):
        m = minPCA(n_components=i + 1, norm=False)
        m.fit(covs_envmeanzero, n_restarts=n_restarts)
        components_minpca_unnorm.append(m.components())

    # 6. maxRegret
    components_regret = []
    for i in range(n_components):
        m = minPCA(n_components=i + 1, norm=False, function='maxregret')
        m.fit(covs_envmeanzero, n_restarts=n_restarts)
        components_regret.append(m.components())

    # 7. norm-maxRegret
    components_normregret = []
    for i in range(n_components):
        m = minPCA(n_components=i + 1, norm=True, function='maxregret')
        m.fit(covs_envmeanzero, n_restarts=n_restarts)
        components_normregret.append(m.components())

    pca_dict = {
        'PCA(zeromean)': [pca_zeromean.components_.T[:, :i]
                          for i in range(1, n_components + 1)],
        'PCA(poolscale)': [pca_poolscale.components_.T[:, :i]
                           for i in range(1, n_components + 1)],
        'Average covariance': [components_avg_cov[:, :i]
                               for i in range(1, n_components + 1)],
        'norm-minPCA': joint_components,
        'minPCA': components_minpca_unnorm,
        'maxRegret': components_regret,
        'norm-maxRegret': components_normregret,
    }

    fitted_objects = {
        'pca_zeromean': pca_zeromean,
        'pca_poolscale': pca_poolscale,
    }

    return pca_dict, fitted_objects


def evaluate_methods(pca_dict, Xs_envmeanzero, X_pool_zeromean,
                     n_components=N_COMPONENTS):
    """Compute in-sample worst-case and pooled error/variance for all methods.

    Returns
    -------
    errs_df : pd.DataFrame
        Columns: method, n_components, wc_err, pooled_err, wc_var, pooled_var,
                 wc_err_unnorm, pooled_err_unnorm, wc_var_unnorm, pooled_var_unnorm
    """
    params = generate_params(Xs_envmeanzero)
    params_pooled = generate_params([X_pool_zeromean])

    params2 = params.copy()
    params2['norm_csts'] = [1 / X.shape[0] for X in Xs_envmeanzero]
    params_pooled2 = params_pooled.copy()
    params_pooled2['norm_csts'] = [1 / X_pool_zeromean.shape[0]]

    errs = {
        'method': [], 'n_components': [],
        'wc_err': [], 'pooled_err': [],
        'wc_var': [], 'pooled_var': [],
        'wc_err_unnorm': [], 'pooled_err_unnorm': [],
        'wc_var_unnorm': [], 'pooled_var_unnorm': [],
    }

    for name, components in pca_dict.items():
        for j, vv in enumerate(components):
            v = torch.tensor(vv, dtype=torch.float32)
            wc_err, pooled_err = get_errs_pca(v, params, params_pooled, from_cov=True)
            wc_var, pooled_var = get_vars_pca(v, params, params_pooled)
            wc_err2, pooled_err2 = get_errs_pca(v, params2, params_pooled2, from_cov=True)
            wc_var2, pooled_var2 = get_vars_pca(v, params2, params_pooled2)

            errs['method'].append(name)
            errs['n_components'].append(j + 1)
            errs['wc_err'].append(wc_err)
            errs['pooled_err'].append(pooled_err)
            errs['wc_var'].append(wc_var)
            errs['pooled_var'].append(pooled_var)
            errs['wc_err_unnorm'].append(wc_err2)
            errs['pooled_err_unnorm'].append(pooled_err2)
            errs['wc_var_unnorm'].append(wc_var2)
            errs['pooled_var_unnorm'].append(pooled_var2)

    return pd.DataFrame(errs)


def fit_ordered_minpca(covs_envmeanzero, n_components=N_COMPONENTS,
                       n_restarts=N_RESTARTS):
    """Fit joint norm-minPCA and reorder components by explained variance.

    Returns
    -------
    minpca_ordered : minPCA
        Fitted object with ordered components and updated cumsum_minvar_.
    """
    m = minPCA(n_components=n_components, norm=True)
    m.fit(covs_envmeanzero, n_restarts=n_restarts)
    m.components(ordered=True, lr=0.1, n_iters=1000)
    return m


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_variance_heatmap(X_envzeromean_df, continents, row_order=None, save_path=None):
    """Heatmap of per-continent variance for each variable (env-zero-mean data)."""
    if row_order is None:
        row_order = COMPONENT_ROW_ORDER[::-1] 
    df = X_envzeromean_df.copy()
    df['continent'] = continents.values
    var_per_continent = df.groupby('continent').var()  # continents x variables
    print(var_per_continent.div(var_per_continent.sum(axis=1),axis=0 ).head())
    var_per_continent = var_per_continent.div(var_per_continent.sum(axis=1),axis=0)
    var_per_continent = var_per_continent.iloc[:, row_order]
    var_per_continent.loc['Minimum'] = var_per_continent.min(axis=0)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(var_per_continent.T, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Variance'})
    ax.set_xlabel('Continent')
    ax.set_ylabel('Variable')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_scree(covs, env_names, save_path=None):
    """Scree plot: eigenvalue decay per continent."""
    plt.figure(figsize=(3, 2))
    for C, name in zip(covs, env_names):
        eigvals, _ = torch.linalg.eigh(C)
        eigvals = eigvals.flip(0).cpu().numpy()
        plt.plot(range(1, len(eigvals) + 1), eigvals,
                 label=name, markersize=4, linewidth=1.5, marker='o')
    plt.xlabel('PC')
    plt.ylabel('Eigenvalue')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_comparison(errs_df, methods_to_plot, labels=None, n_components=N_COMPONENTS,
                    col_wc='wc_var', col_pool='pooled_var', save_path=None):
    """Line plot with 2 subplots: average and worst-case % variance explained."""
    df_plot = errs_df[errs_df['method'].isin(methods_to_plot)].copy()
    if labels is not None:
        label_map = dict(zip(methods_to_plot, labels))
        df_plot['method'] = df_plot['method'].map(label_map)

    fig, ax = plt.subplots(1, 2, figsize=(5.6, 1.8), layout="constrained")

    ymin = 0.2
    ymax = max(df_plot[col_wc].max(), df_plot[col_pool].max()) + 0.05

    kw = dict(marker='o', hue='method', x='n_components', data=df_plot,
              palette=palette, style='method', markers=True, dashes=False,
              markeredgecolor="none", markersize=4.5, linewidth=1)

    sns.lineplot(y=col_pool, ax=ax[0], legend=False, **kw)
    sns.lineplot(y=col_wc, ax=ax[1], **kw)

    ax[1].legend(title='Method', bbox_to_anchor=(1.05, 0.5), loc='center left',
                #  borderaxespad=0,
                frameon=False)
    ax[0].set_xlabel('Rank')
    ax[0].set_ylabel('Average in-sample\n\% explained variance')
    ax[0].set_xticks(range(1, n_components + 1))
    ax[1].set_ylabel('\nWorst-case in-sample\n\% explained variance')
    ax[1].set_xlabel('Rank')
    ax[1].set_xticks(range(1, n_components + 1))
    ax[0].set_ylim(ymin, ymax)
    ax[1].set_ylim(ymin, ymax)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_cumsum_minvar(minpca_ordered, minpca_seq, n_components=N_COMPONENTS,
                       save_path=None):
    """Plot cumulative worst-case variance: ordered joint vs sequential."""
    plt.figure(figsize=(3, 2))
    plt.plot(range(1, n_components + 1),
             minpca_ordered.cumsum_minvar_, label='minPCA (ordered)')
    plt.plot(range(1, n_components + 1),
             minpca_seq.cumsum_minvar_, label='seq-minPCA')
    plt.xlabel('Number of PCs')
    plt.ylabel('Cumulative worst-case\n\\% explained variance')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_components(pca_obj, minpca_obj, colnames, signs=None,
                    label_pca='PCA', label_minpca='minPCA',
                    row_order=None, save_path=None):
    """Bar chart comparing PCA vs minPCA loadings for first 3 PCs.

    Parameters
    ----------
    pca_obj : sklearn PCA
        Fitted PCA object.
    minpca_obj : minPCA
        Fitted minPCA with components.
    colnames : list[str]
        Feature names.
    signs : list[int]
        Sign flips for each PC (length 3).
    label : str
        Legend label for minPCA.
    row_order : np.ndarray
        Custom ordering of feature rows.
    """
    if signs is None:
        signs = [1, 1, 1]
    if row_order is None:
        row_order = COMPONENT_ROW_ORDER

    n_features = len(colnames)
    ordered_colnames = [colnames[i] for i in row_order]
    reordered_minpca = np.array([minpca_obj.components()[i] for i in row_order])
    reordered_pca = np.array([pca_obj.components_.T[i] for i in row_order])

    fig, axs = plt.subplots(1, 3, figsize=(5.8, 2.2), sharey=True)
    for i in range(3):
        axs[i].barh(range(1, n_features + 1), reordered_pca[:, i],
                     alpha=0.5, label=label_pca, color='tab:red')
        axs[i].barh(range(1, n_features + 1),
                     signs[i] * reordered_minpca[:, i],
                     alpha=0.4, label=label_minpca, color="#1e76b4")
        axs[i].set_title(f'PC {i + 1}')
        axs[i].set_yticks(range(1, n_features + 1))
        axs[i].set_yticklabels(ordered_colnames)
    
    axs[0].set_ylabel('Ecosystem functional property')
    # axs[1].legend(loc='lower left')
    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False,
                  title='Method')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_combined(errs_df, methods_to_plot, labels,
                  pca_obj, minpca_obj, colnames,
                  signs=None, label_pca='poolPCA', label_minpca='norm-maxRCS',
                  row_order=None, n_components=N_COMPONENTS,
                  col_wc='wc_var', col_pool='pooled_var', save_path=None):
    """Combined plot: LEFT/CENTRE = comparison line plots, RIGHT = component bars stacked vertically."""
    if signs is None:
        signs = [1, 1, 1]
    if row_order is None:
        row_order = COMPONENT_ROW_ORDER

    # --- Prepare comparison data ---
    df_plot = errs_df[errs_df['method'].isin(methods_to_plot)].copy()
    if labels is not None:
        label_map = dict(zip(methods_to_plot, labels))
        df_plot['method'] = df_plot['method'].map(label_map)

    # --- Prepare component data ---
    n_features = len(colnames)
    ordered_colnames = [colnames[i] for i in row_order]
    reordered_minpca = np.array([minpca_obj.components()[i] for i in row_order])
    reordered_pca = np.array([pca_obj.components_.T[i] for i in row_order])

    # --- Create figure: 2 columns for comparison + 1 column for components ---
    fig = plt.figure(figsize=(6, 2.7), layout="constrained")
    # Top-level: 3 columns. Left two share a single row, right gets 3 stacked rows.
    w1, w2 = 1, 1.3
    wt = 2 * w1 + w2
    gs = fig.add_gridspec(1, 3, width_ratios=[w1, w1, w2])

    # Left two plots: top-aligned, shorter than the right column
    gs_left = gs[0, 0].subgridspec(3, 1, height_ratios=[1, 4, 1])
    gs_centre = gs[0, 1].subgridspec(3, 1, height_ratios=[1, 4, 1])
    ax_pool = fig.add_subplot(gs_left[1, 0])
    ax_wc = fig.add_subplot(gs_centre[1, 0])

    # Right column: 3 stacked component axes with shared x
    gs_right = gs[0, 2].subgridspec(3, 1)
    ax_comp = [fig.add_subplot(gs_right[0, 0])]
    for i in range(1, 3):
        ax_comp.append(fig.add_subplot(gs_right[i, 0], sharex=ax_comp[0]))

    # --- LEFT & CENTRE: comparison line plots ---
    ymin = 0.2
    ymax = max(df_plot[col_wc].max(), df_plot[col_pool].max()) + 0.05

    kw = dict(marker='o', hue='method', x='n_components', data=df_plot,
              palette=palette, style='method', markers=True, dashes=False,
              markeredgecolor="none", markersize=4.5, linewidth=1)

    sns.lineplot(y=col_pool, ax=ax_pool, legend=False, **kw)
    sns.lineplot(y=col_wc, ax=ax_wc, legend=True, **kw)

    # Single shared legend with bar-color backgrounds for methods that appear in component plots
    handles, leg_labels = ax_wc.get_legend_handles_labels()
    ax_wc.get_legend().remove()
    bar_colors = {label_pca: ('tab:red', 0.3), label_minpca: ('#1e76b4', 0.25)}
    leg = fig.legend(handles, leg_labels,
                         loc='lower center', 
                         bbox_to_anchor=(0.5, -0.08),
                         title='Method', ncol=len(labels),
                         frameon=False)
    for text in leg.get_texts():
        label = text.get_text()
        if label in bar_colors:
            color, alpha = bar_colors[label]
            text.set_bbox(dict(facecolor=color, alpha=alpha, edgecolor='none',
                               boxstyle='round,pad=0.2'))

    ax_pool.set_xlabel('Rank')
    ax_pool.set_ylabel('Average in-sample\n\% explained variance')
    ax_pool.set_xticks(range(1, n_components + 1))
    ax_pool.set_ylim(ymin, ymax)

    ax_wc.set_xlabel('Rank')
    ax_wc.set_ylabel('Worst-case in-sample\n\% explained variance')
    ax_wc.set_xticks(range(1, n_components + 1))
    ax_wc.set_ylim(ymin, ymax)

    # --- RIGHT: component bar plots stacked vertically (vertical bars) ---
    for i in range(3):
        ax = ax_comp[i]
        ax.bar(range(1, n_features + 1), reordered_pca[:, i],
               alpha=0.5, label=label_pca, color='tab:red')
        ax.bar(range(1, n_features + 1),
               signs[i] * reordered_minpca[:, i],
               alpha=0.4, label=label_minpca, color="#1e76b4")
        ax.annotate(f'PC {i + 1}', xy=(0.02, 0.02), xycoords='axes fraction',
                    va='bottom', ha='left', fontweight='bold')
        ax.set_xticks(range(1, n_features + 1))
        if i == 2:
            ax.set_xticklabels(ordered_colnames, rotation=90)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        if i == 1: ax.set_ylabel('Loading')
        ymin = min(reordered_pca[:, i].min(), (signs[i] * reordered_minpca[:, i]).min()) - 0.1
        ymax = max(reordered_pca[:, i].max(), (signs[i] * reordered_minpca[:, i]).max()) + 0.05
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main(rerun=False):
    """Main function to run the ecosystem analysis."""
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(2)
    torch.manual_seed(2)

    # ---- Load data ----
    print("Loading and preprocessing data...")
    (env_dfs_poolscale, env_dfs_envmeanzero,
     X_poolscale_df, X_envzeromean_df, continents) = load_and_preprocess_data()

    Xs_envmeanzero = list(env_dfs_envmeanzero.values())
    covs_poolscale = [torch.cov(X.T, correction=0)
                      for X in env_dfs_poolscale.values()]
    covs_envmeanzero = [torch.cov(X.T, correction=0) for X in Xs_envmeanzero]

    X_pool_zeromean = torch.tensor(X_envzeromean_df.values, dtype=torch.float32)
    X_poolscale = torch.tensor(X_poolscale_df.values, dtype=torch.float32)

    # ---- Plot 1: Scree plot ----
    print("Generating scree plot...")
    plot_scree(
        covs_poolscale, list(env_dfs_poolscale.keys()),
        save_path=FIGURES_DIR / "ecosystem_scree_plot.pdf"
    )

    # ---- Plot: Variance heatmap per continent ----
    print("Generating variance heatmap...")
    plot_variance_heatmap(
        X_envzeromean_df, continents,
        save_path=FIGURES_DIR / "ecosystem_variance_heatmap.pdf"
    )

    # ---- Fit methods and evaluate ----
    fit_cache_path = RESULTS_DIR / f"fitted_ncomp{N_COMPONENTS}_restarts{N_RESTARTS}.pkl"
    if not rerun and fit_cache_path.exists():
        print("Loading cached fitted objects...")
        with open(fit_cache_path, 'rb') as f:
            cached = pickle.load(f)
        pca_dict, fitted_objects = cached['pca_dict'], cached['fitted_objects']
    else:
        print("Fitting all methods...")
        pca_dict, fitted_objects = fit_all_methods(
            covs_envmeanzero, X_pool_zeromean, X_poolscale
        )
        with open(fit_cache_path, 'wb') as f:
            pickle.dump({'pca_dict': pca_dict, 'fitted_objects': fitted_objects}, f)

    cache_path = RESULTS_DIR / f"errs_df_ncomp{N_COMPONENTS}_restarts{N_RESTARTS}.csv"
    if not rerun and cache_path.exists():
        print("Loading cached evaluation results...")
        errs_df = pd.read_csv(cache_path)
    else:
        print("Evaluating methods...")
        errs_df = evaluate_methods(pca_dict, Xs_envmeanzero, X_pool_zeromean)
        errs_df.to_csv(cache_path, index=False)

    # ---- Plot 2: Comparison plot ----
    print("Generating comparison plot...")
    plot_comparison(
        errs_df, PLOT_METHODS_EXT, labels=PLOT_LABELS_EXT,
        save_path=FIGURES_DIR / "ecosystem_comparison.pdf"
    )

    # ---- Plot 3: Component ordering ----
    ordered_cache_path = RESULTS_DIR / f"minpca_ordered_ncomp{N_COMPONENTS}_restarts{N_RESTARTS}.pkl"
    if not rerun and ordered_cache_path.exists():
        print("Loading cached ordered minPCA...")
        with open(ordered_cache_path, 'rb') as f:
            minpca_ordered = pickle.load(f)
    else:
        print("Fitting ordered minPCA...")
        min_explained = 0
        while min_explained < 0.36:
            minpca_ordered = fit_ordered_minpca(covs_envmeanzero)
            min_explained = minpca_ordered.cumsum_minvar_[0]
        with open(ordered_cache_path, 'wb') as f:
            pickle.dump(minpca_ordered, f)
    print(f"Cumulative worst-case var (ordered): {minpca_ordered.cumsum_minvar_}")

    # ---- Plot 4 & 5: Component loading comparisons ----
    print("Generating component plots...")
    plot_components(
        fitted_objects['pca_zeromean'], minpca_ordered,
        list(X_envzeromean_df.columns),
        signs=[-1, 1, -1],
        label_pca='poolPCA',
        label_minpca='norm-maxRCS',
        save_path=FIGURES_DIR / "ecosystem_components_zeromean.pdf"
    )
    plot_components(
        fitted_objects['pca_poolscale'], minpca_ordered,
        list(X_envzeromean_df.columns),
        signs=[-1, 1, -1],
        label_pca=r'poolPCA$^*$',
        label_minpca='norm-maxRCS',
        save_path=FIGURES_DIR / "ecosystem_components_poolscale.pdf"
    )

    # ---- Plot 6: Combined comparison + components ----
    print("Generating combined plot...")
    plot_combined(
        errs_df, PLOT_METHODS, labels=PLOT_LABELS,
        pca_obj=fitted_objects['pca_zeromean'],
        minpca_obj=minpca_ordered,
        colnames=list(X_envzeromean_df.columns),
        signs=[-1, 1, -1],
        save_path=FIGURES_DIR / "ecosystem_combined.pdf"
    )

    print(f"\nDone! Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ecosystem minPCA Analysis")
    parser.add_argument("--rerun", action="store_true",
                        help="Force recomputation of results")
    args = parser.parse_args()
    main(rerun=args.rerun)
