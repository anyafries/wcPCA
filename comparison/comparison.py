"""
Comparison script

Runs all comparison methods (FairPCA, minPCA, StablePCA) and generates
comparison figures.

Usage:
    python comparison.py              # Run all methods and generate plots
    python comparison.py --rerun      # Force rerun all methods
    python comparison.py --plots-only # Only generate plots from existing results

Output:
    figures/comparison_MM_Var.png
    figures/comparison_MM_Loss.png
    figures/comparison_{objective}_p{p}_ncomp{n}_ne{e}.png (individual plots)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# === Constants ===
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
FIGURES_DIR = SCRIPT_DIR / 'figures'
N_COMPONENTS = 5

PARAM_CONFIGS = [(10, 5), (10, 50), (50, 5)]
OBJECTIVES = ['MM_Var', 'MM_Loss']

COLORS = {
    'PGD': 'tab:blue',
    'SDP': 'tab:orange',
    'MW': 'tab:red',
    'StablePCA': 'tab:green',
}

PLOT_KWARGS = {
    'ms': 3,
    'linewidth': 0.5,
    'markeredgewidth': 0,
    'hue': 'Method',
    'style': 'Method',
    'markers': True,
}


def run_all_methods(rerun=False):
    """Run all comparison methods via subprocess."""
    rerun_flag = ['--rerun'] if rerun else []

    # Run FairPCA (SDP + MW)
    print("\n=== Running FairPCA ===")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / 'fairpca.py')] + rerun_flag,
        check=True
    )

    # Run minPCA (PGD)
    print("\n=== Running minPCA ===")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / 'minpca_sim.py')] + rerun_flag,
        check=True
    )

    # Run StablePCA (MM_Var only)
    print("\n=== Running StablePCA ===")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / 'stablepca.py')] + rerun_flag,
        check=True
    )


def load_results(p, n_envs, objective):
    """Load results from all methods for given configuration."""
    suffix = f"_{objective}_p{p}_ncomp{N_COMPONENTS}_ne{n_envs}.csv"

    # Load minPCA results
    df_minpca = pd.read_csv(RESULTS_DIR / f"minPCA{suffix}")
    df_minpca['Method'] = 'PGD'
    df_minpca['obj'] = df_minpca['minvar']

    # Load SDP results
    df_sdp = pd.read_csv(RESULTS_DIR / f"SDP{suffix}")
    df_sdp['Method'] = 'SDP'
    df_sdp['obj'] = df_sdp['obj_trunc']
    df_sdp['rank'] = df_sdp['d']

    # Load MW results
    df_mw = pd.read_csv(RESULTS_DIR / f"MW{suffix}")
    df_mw['Method'] = 'MW'
    df_mw['rank'] = df_mw['d']

    dfs = [df_minpca, df_sdp, df_mw]

    # Load StablePCA results (only for MM_Var)
    if objective == "MM_Var":
        stablepca_file = RESULTS_DIR / f"stablepca{suffix}"
        if stablepca_file.exists():
            df_stablepca = pd.read_csv(stablepca_file)
            df_stablepca['Method'] = 'StablePCA'
            df_stablepca['obj'] = df_stablepca['minvar']
            dfs.append(df_stablepca)

    df = pd.concat(dfs, ignore_index=True)

    # Filter ranks for large p
    if p == 50:
        df = df[df['rank'] <= 30]

    if objective == "MM_Loss":
        df['obj'] = - df['obj']

    return df


def make_individual_plot(p, n_envs, objective):
    """Create individual comparison plot for one configuration."""
    suffix = f"_{objective}_p{p}_ncomp{N_COMPONENTS}_ne{n_envs}"

    df = load_results(p, n_envs, objective)

    fig, ax = plt.subplots(1, 2, figsize=(4.5, 1.4))

    # Objective plot
    sns.lineplot(data=df, x='rank', y='obj', ax=ax[0], palette=COLORS, **PLOT_KWARGS)
    ax[0].set_xlabel('Rank of solution')
    if objective == "MM_Var":
        ax[0].set_ylabel('- minimum\n variance')
    else:
        ax[0].set_ylabel('Maximum \nregret')

    # Time plot
    sns.lineplot(data=df, x='rank', y='time', ax=ax[1], palette=COLORS, **PLOT_KWARGS)
    ax[1].set_xlabel('Rank of solution')
    ax[1].set_ylabel('\nTime (s)')

    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend_.remove()
    ax[1].legend_.remove()
    fig.legend(handles, labels, loc='center right',
               bbox_to_anchor=(1.1, 0.55), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    fig_path = FIGURES_DIR / f"comparison{suffix}.png"
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path.name}")


def make_combined_plot(objective):
    """Create combined comparison plot (3 configs x 2 metrics)."""
    plt.style.use(str(SCRIPT_DIR.parent / 'jmlr.mplstyle'))

    fig, ax = plt.subplots(2, 3, figsize=(6, 3.1))

    for i, (p, n_envs) in enumerate(PARAM_CONFIGS):
        df = load_results(p, n_envs, objective)

        # Filter for better visibility on large p
        if p == 50:
            df = df[df['rank'] % 2 == 1]  # Keep odd ranks only

        # Objective plot (top row)
        sns.lineplot(data=df, x='rank', y='obj', ax=ax[0, i], palette=COLORS, **PLOT_KWARGS)
        ax[0, i].set_title(f"p={p}, E={n_envs}\n", fontsize=10)
        ax[0, i].set_xlabel('Rank of solution')
        if i == 0:
            if objective == "MM_Var":
                ax[0, i].set_ylabel('Minimum\nexplained variance')
            else:
                ax[0, i].set_ylabel('Minimum\nregret')
        else:
            ax[0, i].set_ylabel('')

        # Time plot (bottom row)
        sns.lineplot(data=df, x='rank', y='time', ax=ax[1, i], palette=COLORS, **PLOT_KWARGS)
        ax[1, i].set_xlabel('Rank of solution')
        if i == 0:
            ax[1, i].set_ylabel('\nTime (s)')
        else:
            ax[1, i].set_ylabel('')

        # Remove individual legends
        handles, labels = ax[0, i].get_legend_handles_labels()
        ax[0, i].legend_.remove()
        ax[1, i].legend_.remove()

        # Add combined legend (only once)
        if i == 0:
            fig.legend(handles, labels, loc='center right',
                       bbox_to_anchor=(0.95, 0.5), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    fig_path = FIGURES_DIR / f"comparison_{objective}.png"
    plt.savefig(fig_path, dpi=400)
    plt.close()
    print(f"Saved: {fig_path.name}")


def make_all_plots():
    """Generate all comparison plots."""
    print("\n=== Generating plots ===")

    # Ensure figures directory exists
    FIGURES_DIR.mkdir(exist_ok=True)

    # Load matplotlib style
    style_file = SCRIPT_DIR.parent / 'jmlr.mplstyle'
    if style_file.exists():
        plt.style.use(str(style_file))

    # Individual plots
    for p, n_envs in PARAM_CONFIGS:
        for objective in OBJECTIVES:
            try:
                make_individual_plot(p, n_envs, objective)
            except FileNotFoundError as e:
                print(f"Warning: Missing results for p={p}, n_envs={n_envs}, {objective}: {e}")

    # Combined plots
    for objective in OBJECTIVES:
        try:
            make_combined_plot(objective)
        except FileNotFoundError as e:
            print(f"Warning: Missing results for combined {objective} plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Comparison orchestrator and plotting')
    parser.add_argument('--rerun', action='store_true',
                        help='Force rerun all methods')
    parser.add_argument('--plots-only', action='store_true',
                        help='Only generate plots from existing results')
    args = parser.parse_args()

    # Ensure directories exist
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # Run methods (unless plots-only)
    if not args.plots_only:
        run_all_methods(rerun=args.rerun)

    # Generate plots
    make_all_plots()

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
