"""
Comparison script

Runs all comparison methods (FairPCA, minPCA, StablePCA) and generates
comparison figures.

Usage:
    python comparison.py --start_seed 2 --end_seed 11   # Run all methods and generate plots
    python comparison.py --start_seed 2 --end_seed 11 --rerun  # Force rerun all methods
    python comparison.py --start_seed 2 --end_seed 11 --plots_only  # Only generate plots

Output (individual, one per (p, n_envs, objective)):
    figures/comparison_relative_{objective}_p{p}_ncomp{n}_ne{e}.png

Output (combined, one per objective):
    figures/comparison_relative_{objective}.png
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# === Constants ===
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
FIGURES_DIR = SCRIPT_DIR / 'figures'
N_COMPONENTS = 5
SEED = 2

PARAM_CONFIGS = [(10, 5), (10, 50), (50, 5)]
OBJECTIVES = ['MM_Var', 'MM_Loss']

COLORS = {
    'PGD': 'tab:blue',
    'SDP': 'tab:orange',
    'MW': 'tab:red',
    'StablePCA': 'tab:green',
    'StablePCA_new': '#2ca02c',
}


def run_all_methods(rerun=False, start_seed=SEED, end_seed=SEED):
    """Run all comparison methods via subprocess."""
    rerun_flag = ['--rerun'] if rerun else []
    seed_args = ['--start_seed', str(start_seed), '--end_seed', str(end_seed)]

    # Run FairPCA (SDP + MW)
    print("\n=== Running FairPCA ===")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / 'fairpca.py')] + rerun_flag + seed_args,
        check=True
    )

    # Run minPCA (PGD)
    print("\n=== Running minPCA ===")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / 'minpca_sim.py')] + rerun_flag + seed_args,
        check=True
    )

    # Run StablePCA (MM_Var only)
    print("\n=== Running StablePCA ===")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / 'stablepca.py')] + rerun_flag + seed_args,
        check=True
    )

    # Run StablePCA (new version)
    print("\n=== Running StablePCA (new version) ===")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / 'stablepca_new.py')] + rerun_flag + seed_args,
        check=True
    )


def load_results(p, n_envs, objective, start_seed=SEED, end_seed=SEED):
    """Load multi-seed results from all methods for a given configuration.

    Seeds are filtered to those for which all required methods have results.
    A warning is printed for any dropped seeds.

    Returns a DataFrame with columns: rank, obj, time, Method, seed
    """
    all_dfs = []
    dropped_seeds = []

    for seed in range(start_seed, end_seed + 1):
        suffix = f"_{objective}_p{p}_ncomp{N_COMPONENTS}_ne{n_envs}_seed{seed}.csv"

        files = {
            'PGD': RESULTS_DIR / f"minPCA{suffix}",
            'SDP': RESULTS_DIR / f"SDP{suffix}",
            'MW': RESULTS_DIR / f"MW{suffix}",
        }
        if objective == 'MM_Var':
            files['StablePCA'] = RESULTS_DIR / f"stablepca{suffix}"
            files['StablePCA_new'] = RESULTS_DIR / f"stablepca_stable_{suffix}"

        missing = [name for name, f in files.items() if not f.exists()]
        if missing:
            dropped_seeds.append((seed, missing))
            continue

        # Load minPCA results
        df_minpca = pd.read_csv(files['PGD'])
        df_minpca['Method'] = 'PGD'
        df_minpca['obj'] = df_minpca['minvar']
        df_minpca['seed'] = seed

        # Load SDP results
        df_sdp = pd.read_csv(files['SDP'])
        df_sdp['Method'] = 'SDP'
        df_sdp['obj'] = df_sdp['obj_trunc']
        df_sdp['rank'] = df_sdp['d']
        df_sdp['seed'] = seed

        # Load MW results
        df_mw = pd.read_csv(files['MW'])
        df_mw['Method'] = 'MW'
        df_mw['rank'] = df_mw['d']
        df_mw['seed'] = seed

        seed_dfs = [df_minpca, df_sdp, df_mw]

        if objective == 'MM_Var':
            df_stablepca = pd.read_csv(files['StablePCA'])
            df_stablepca['Method'] = 'StablePCA'
            df_stablepca['obj'] = df_stablepca['minvar']
            df_stablepca['seed'] = seed
            seed_dfs.append(df_stablepca)

            df_stablepca_new = pd.read_csv(files['StablePCA_new'])
            df_stablepca_new['Method'] = 'StablePCA_new'
            df_stablepca_new['obj'] = df_stablepca_new['minvar']
            df_stablepca_new['seed'] = seed
            seed_dfs.append(df_stablepca_new)

        all_dfs.extend(seed_dfs)

    if dropped_seeds:
        for seed, missing in dropped_seeds:
            print(f"Warning: Dropping seed {seed} for p={p}, n_envs={n_envs}, "
                  f"{objective}: missing results for {missing}")

    if not all_dfs:
        raise FileNotFoundError(
            f"No complete results found for p={p}, n_envs={n_envs}, {objective} "
            f"in seed range [{start_seed}, {end_seed}]"
        )

    df = pd.concat(all_dfs, ignore_index=True)

    # Filter ranks for large p
    if p == 50:
        df = df[df['rank'] <= 30]

    if objective == 'MM_Loss':
        df['obj'] = -df['obj']

    return df


def _add_relative_perf(df):
    """Add 'rel' column: (PGD_obj - method_obj) / |PGD_obj| per (rank, seed)."""
    df_pgd = (df[df['Method'] == 'PGD'][['rank', 'seed', 'obj']]
              .rename(columns={'obj': 'pgd_obj'}))
    df = df.merge(df_pgd, on=['rank', 'seed'])
    df['rel'] = (df['pgd_obj'] - df['obj']) / df['pgd_obj'].abs()
    return df


def _plot_percentile_lines(ax, df, y_col, methods):
    """Plot median line + 25th/75th percentile band per method."""
    q_low, q_high = 0.25, 0.75
    for method in methods:
        df_m = df[df['Method'] == method]
        stats = df_m.groupby('rank')[y_col].quantile([q_low, 0.5, q_high]).unstack()
        color = COLORS[method]
        ax.plot(stats.index, stats[0.5], color=color, label=method, linewidth=1.0)
        ax.fill_between(stats.index, stats[q_low], stats[q_high],
                        color=color, alpha=0.2, linewidth=0)


def make_individual_plot(p, n_envs, objective, start_seed=SEED, end_seed=SEED,
                         ymin=None, ymax=None):
    """Create individual relative performance plot for one (p, n_envs, objective).

    Saved as: figures/comparison_relative_{objective}_p{p}_ncomp{n}_ne{e}.png
    Left subplot:  relative gap vs PGD (median + IQR band), non-PGD methods only
    Right subplot: runtime (median + IQR band), all methods
    """
    df = load_results(p, n_envs, objective, start_seed, end_seed)
    df = _add_relative_perf(df)

    if p == 50:
        df = df[df['rank'] % 2 == 1]
    if (objective == 'MM_Loss') and (p == 10):
            df = df[df['rank'] <= 9] 

    available_methods = df['Method'].unique()
    non_pgd = [m for m in ['SDP', 'MW', 'StablePCA'] if m in available_methods]
    all_methods = [m for m in ['PGD', 'SDP', 'MW', 'StablePCA'] if m in available_methods]

    fig, axes = plt.subplots(1, 2, figsize=(4.5, 1.8))

    # Relative performance subplot (left)
    _plot_percentile_lines(axes[0], df[df['Method'].isin(non_pgd)], 'rel', non_pgd)
    axes[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[0].set_xlabel('Rank of solution')
    ylab = 'explained variance' if objective == 'MM_Var' else 'regret'
    axes[0].set_ylabel(r'$\Delta$ ' + ylab)
    axes[0].set_ylim(ymin, ymax)

    # Runtime subplot (right)
    _plot_percentile_lines(axes[1], df, 'time', all_methods)
    axes[1].set_xlabel('Rank of solution')
    axes[1].set_ylabel('Time (s)')
    ylim = axes[1].get_ylim()
    if ylim[1] > 60:
        axes[1].set_ylim(0, 60)

    # Shared legend (combine entries from both subplots without duplicates)
    all_labels_seen = set()
    legend_handles, legend_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in all_labels_seen:
                legend_handles.append(handle)
                legend_labels.append(label)
                all_labels_seen.add(label)

    higher = 'better' if objective == 'MM_Var' else 'worse'
    lower = 'worse' if objective == 'MM_Var' else 'better'
    axes[0].text(1.07, 0, f'→ {higher}',
            transform=axes[0].get_yaxis_transform(),
            rotation=90, va='bottom', ha='left', fontsize=8)
    axes[0].text(1.07, 0, f'PGD {lower} ←',
            transform=axes[0].get_yaxis_transform(),
            rotation=90, va='top', ha='left', fontsize=8)

    fig.legend(legend_handles, legend_labels, loc='center right',
               bbox_to_anchor=(1.05, 0.55), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    suffix = f"_{objective}_p{p}_ncomp{N_COMPONENTS}_ne{n_envs}"
    fig_path = FIGURES_DIR / f"comparison_relative{suffix}.png"
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path.name}")


def make_combined_plot(objective, start_seed=SEED, end_seed=SEED,
                       ymin=None, ymax=None):
    """Create combined relative performance plot (3 configs × 2 metrics).

    Saved as: figures/comparison_relative_{objective}.png
    Top row:    relative gap vs PGD for each param config
    Bottom row: runtime for each param config
    """
    plt.style.use(str(SCRIPT_DIR.parent / 'jmlr.mplstyle'))

    fig, ax = plt.subplots(2, 3, figsize=(6, 3.3))

    legend_handles, legend_labels = [], []

    for i, (p, n_envs) in enumerate(PARAM_CONFIGS):
        df = load_results(p, n_envs, objective, start_seed, end_seed)
        df = _add_relative_perf(df)

        if p == 50:
            df = df[df['rank'] % 2 == 1]
        if (objective == 'MM_Loss') and (p == 10):
            df = df[df['rank'] <= 9] 

        available_methods = df['Method'].unique()
        non_pgd = [m for m in ['SDP', 'MW', 'StablePCA'] if m in available_methods]
        all_methods = [m for m in ['PGD', 'SDP', 'MW', 'StablePCA'] if m in available_methods]

        # Relative performance (top row)
        _plot_percentile_lines(ax[0, i], df[df['Method'].isin(non_pgd)], 'rel', non_pgd)
        ax[0, i].axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax[0, i].set_title(f"p={p}, E={n_envs}\n", fontsize=10)
        ax[0, i].set_xlabel('Rank of solution')
        if i == 0:
            ylab = 'explained variance' if objective == 'MM_Var' else 'regret'
            ax[0, i].set_ylabel(r'$\Delta$ ' + ylab)
        ax[0, i].set_ylim(ymin, ymax)

        # Runtime (bottom row)
        _plot_percentile_lines(ax[1, i], df, 'time', all_methods)
        ax[1, i].set_xlabel('Rank of solution')
        if i == 0:
            ax[1, i].set_ylabel('Time (s)')

        ylim = ax[1, i].get_ylim()
        if ylim[1] > 20:
            ax[1, i].set_ylim(-2, 20)

        # Collect legend entries from the first config (all subsequent are the same)
        if i == 0:
            seen = set()
            for method in non_pgd + ['PGD']:
                if method in available_methods and method not in seen:
                    legend_handles.append(
                        plt.Line2D([0], [0], color=COLORS[method], linewidth=1.0)
                    )
                    legend_labels.append(method)
                    seen.add(method)

    last = len(PARAM_CONFIGS) - 1
    higher = 'better' if objective == 'MM_Var' else 'worse'
    lower = 'worse' if objective == 'MM_Var' else 'better'
    ax[0, last].text(1.07, 0, f'→ {higher}',
            transform=ax[0, last].get_yaxis_transform(),
            rotation=90, va='bottom', ha='left', fontsize=8)
    ax[0, last].text(1.07, 0, f'PGD {lower} ←',
            transform=ax[0, last].get_yaxis_transform(),
            rotation=90, va='top', ha='left', fontsize=8)

    fig.legend(legend_handles, legend_labels, loc='center right',
               bbox_to_anchor=(0.99, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    fig_path = FIGURES_DIR / f"comparison_relative_{objective}.png"
    plt.savefig(fig_path, dpi=400)
    plt.close()
    print(f"Saved: {fig_path.name}")


def make_all_plots(start_seed=SEED, end_seed=SEED):
    """Generate all comparison plots."""
    print("\n=== Generating plots ===")

    FIGURES_DIR.mkdir(exist_ok=True)

    style_file = SCRIPT_DIR.parent / 'jmlr.mplstyle'
    if style_file.exists():
        plt.style.use(str(style_file))

    # Individual plots: one per (p, n_envs, objective)
    for p, n_envs in PARAM_CONFIGS:
        for objective in OBJECTIVES:
            try:
                ymin, ymax = (None, None) if objective == 'MM_Var' else (-0.5, None)
                make_individual_plot(p, n_envs, objective, start_seed, end_seed,
                                     ymin, ymax)
            except FileNotFoundError as e:
                print(f"Warning: {e}")

    # Combined plots: one per objective
    for objective in OBJECTIVES:
        try:
            ymin, ymax = (-0.025, 0.1) if objective == 'MM_Var' else (None, None)
            make_combined_plot(objective, start_seed, end_seed, ymin, ymax)
        except FileNotFoundError as e:
            print(f"Warning: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true',
                        help='Force rerun all methods')
    parser.add_argument('--plots_only', action='store_true',
                        help='Skip running methods, only generate plots')
    parser.add_argument('--start_seed', type=int, default=SEED,
                        help='First seed (inclusive)')
    parser.add_argument('--end_seed', type=int, default=SEED,
                        help='Last seed (inclusive)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    if not args.plots_only:
        run_all_methods(rerun=args.rerun,
                        start_seed=args.start_seed,
                        end_seed=args.end_seed)

    make_all_plots(start_seed=args.start_seed, end_seed=args.end_seed)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
