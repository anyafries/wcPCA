"""
Illustrative example: 3D visualization of minPCA vs other methods 
(minPCA, poolPCA, sepPCA, and maxRegret) 

Output: figures/eg-minpca.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


# === Constants ===
SEED = 123
N_SAMPLES = 150

# Covariance matrices
SIGMA1 = np.array([[0.9, 0, 0], [0, 0.1, 0], [0, 0, 0]])
SIGMA2 = np.array([[0, 0, 0], [0, 0.4, 0], [0, 0, 0.6]])

# Arrow vectors (principal directions for different methods)
V_MINPCA = np.sqrt([0.4, 0, 0.6])
V_POOL = np.array([1.0, 0.0, 0.0])
V_SEP = np.array([0.0, 0.0, 1.0])
V_REGRET = np.sqrt([0.6, 0, 0.4])

FIGURE_FILE = 'figures/eg-minpca.png'


def compute_plot_data():
    """Generate samples and compute PDF grids."""
    np.random.seed(SEED)

    mean = np.zeros(3)

    # Sample from the distributions
    samples1 = np.random.multivariate_normal(mean, SIGMA1, size=N_SAMPLES)
    samples2 = np.random.multivariate_normal(mean, SIGMA2, size=N_SAMPLES)

    # Determine plot limits from data
    all_samples = np.vstack([samples1, samples2])
    lims = np.array([all_samples.min(axis=0) - 1, all_samples.max(axis=0) + 1]).T
    lims = lims / 1.5

    # Grid and PDF for distribution 1 (XY plane)
    x1, y1 = np.mgrid[lims[0][0]:lims[0][1]:.01, lims[1][0]:lims[1][1]:.01]
    Sigma1_small = SIGMA1[:2, :2]
    pos1 = np.dstack((x1, y1))
    rv1 = multivariate_normal([0, 0], Sigma1_small)
    z_pdf1 = rv1.pdf(pos1)

    # Grid and PDF for distribution 2 (YZ plane)
    y2, z2 = np.mgrid[lims[1][0]:lims[1][1]:.01, lims[2][0]:lims[2][1]:.01]
    Sigma2_small = SIGMA2[1:, 1:]
    pos2 = np.dstack((y2, z2))
    rv2 = multivariate_normal([0, 0], Sigma2_small)
    x_pdf2 = rv2.pdf(pos2)

    return {
        'lims': lims,
        'x1': x1, 'y1': y1, 'z_pdf1': z_pdf1,
        'y2': y2, 'z2': z2, 'x_pdf2': x_pdf2,
    }


def make_figure(data):
    """Create and save the 3D visualization figure."""
    plt.style.use('jmlr.mplstyle')

    lims = data['lims']
    x1, y1, z_pdf1 = data['x1'], data['y1'], data['z_pdf1']
    y2, z2, x_pdf2 = data['y2'], data['z2'], data['x_pdf2']

    # Contour parameters
    kw = {
        'vmin': 1.3*np.min([np.min(z_pdf1), np.min(x_pdf2)]),
        'vmax': 1.3*np.max([np.max(z_pdf1), np.max(x_pdf2)]),
        'alpha': 0.9
    }
    kw['levels'] = np.linspace(kw['vmin'], kw['vmax'], 10)

    # Font sizes
    font1 = 10
    font2 = 8
    font3 = 9

    # Colors
    color1 = 'yellow'
    color2 = 'limegreen'
    cmap1 = plt.get_cmap('Oranges')
    cmap2 = plt.get_cmap('Greens')
    arrowcolor1 = 'blue'
    arrowcolor2 = 'red'
    arrowcolor3 = arrowcolor2
    arrowcolor4 = 'purple'

    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Compute axis limits
    xmax = np.max(np.abs(lims[0]))
    ymax = np.max(np.abs(lims[1]))
    zmax = np.max(np.abs(lims[2]))
    x = np.linspace(-xmax, xmax, 10)
    y = np.linspace(-ymax, ymax, 10)
    z = np.linspace(-zmax, zmax, 10)

    # Add transparent XY plane at Z = 0
    plane_scale = 0.85
    X, Y = np.meshgrid(plane_scale*x, plane_scale*y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color=color2, label='E=1')

    # Add transparent YZ plane at X = 0
    Y, Z = np.meshgrid(plane_scale*y, plane_scale*z)
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, alpha=0.1, color=color1, label='E=2')

    # Add contours for the PDFs
    ax.contour(x1, y1, z_pdf1, cmap=cmap2, zdir='z', zorder=2, offset=0, **kw)
    ax.contour(x_pdf2, y2, z2, cmap=cmap1, zdir='x', zorder=2, offset=0, **kw)

    # Draw axis lines through origin
    correction = 1
    ax.plot(correction * np.array([-xmax, xmax]), [0, 0], [0, 0],
            color='black', lw=0.8)
    ax.plot([0, 0], correction * np.array([-ymax, ymax]), [0, 0],
            color='black', lw=0.8)
    ax.plot([0, 0], [0, 0], correction * np.array([-zmax, zmax]),
            color='black', lw=0.8)
    
    # Add axis labels
    correction2 = correction+0.05
    axis_label_args = {
        'color': 'k', 'fontsize': font2, 'zorder': 10,
        'horizontalalignment': 'center', 'verticalalignment': 'center'
    }
    ax.text(correction2 * xmax, 0, 0, 'x', **axis_label_args)
    ax.text(0, -1.03 * correction2 * ymax, 0, 'y', **axis_label_args)
    ax.text(0, 0, correction2 * zmax, 'z', **axis_label_args)

    # Arrow scale and tips
    arrow_scale = 1.7
    tip1 = arrow_scale * V_MINPCA
    tip2 = arrow_scale * V_POOL
    tip3 = arrow_scale * V_SEP
    tip4 = arrow_scale * V_REGRET

    # Plot vectors as arrows
    ax.quiver(0, 0, 0, *V_MINPCA, length=arrow_scale, color=arrowcolor1,
              linewidth=1.5, arrow_length_ratio=0.15, zorder=15)
    ax.quiver(0, 0, 0, *V_POOL, length=arrow_scale, color=arrowcolor2,
              linewidth=1.5, arrow_length_ratio=0.15, zorder=15)
    ax.quiver(0, 0, 0, *V_SEP, length=arrow_scale, color=arrowcolor3,
              linewidth=1.5, arrow_length_ratio=0.15, zorder=15)
    ax.quiver(0, 0, 0, *V_REGRET, length=arrow_scale, color=arrowcolor4,
              linewidth=1.5, arrow_length_ratio=0.15, zorder=15)

    # Plot lines for arrows (for better visibility)
    ax.plot([0, tip1[0]], [0, tip1[1]], [0, tip1[2]], zorder=10, color=arrowcolor1)
    ax.plot([0, tip2[0]], [0, tip2[1]], [0, tip2[2]], zorder=10, color=arrowcolor2)
    ax.plot([0, tip3[0]], [0, tip3[1]], [0, tip3[2]], zorder=10, color=arrowcolor3)
    ax.plot([0, tip4[0]], [0, tip4[1]], [0, tip4[2]], zorder=10, color=arrowcolor4)

    # Add labels near arrow tips
    ax.text(*(1.03 * tip1), r'$V^{\text{minPCA}}$ [36\%, 36\%]',
            color=arrowcolor1, fontsize=font1, zorder=10)
    ax.text(*tip2[:2], tip2[2] + 0.2, r'$V^{\text{pool}}$ [45\%, 0\%]',
            color=arrowcolor2, fontsize=font1, zorder=10)
    ax.text(tip3[0] + 0.2, *tip3[1:], r'$V^{\text{sep}}$ [30\%, 0\%]',
            color=arrowcolor3, fontsize=font1, zorder=10)
    text4loc = tip4.copy()
    text4loc[0] += 0.1
    text4loc[2] -= 0.1
    ax.text(*text4loc, r'$V^{\text{maxRegret}}$ [39\%, 24\%]',
            color=arrowcolor4, fontsize=font1, zorder=10)

    # Add dotted lines from tips to axes
    ax.plot([tip1[0], tip1[0]], [tip1[1], tip1[1]], [tip1[2], 0],
            color=arrowcolor1, linestyle='dotted', zorder=10, linewidth=1)
    ax.plot([tip1[0], 0], [tip1[1], tip1[1]], [tip1[2], tip1[2]],
            color=arrowcolor1, linestyle='dotted', zorder=10, linewidth=1)
    ax.plot([tip4[0], tip4[0]], [tip4[1], tip4[1]], [tip4[2], 0],
            color=arrowcolor4, linestyle='dotted', zorder=10, linewidth=1)
    ax.plot([tip4[0], 0], [tip4[1], tip4[1]], [tip4[2], tip4[2]],
            color=arrowcolor4, linestyle='dotted', zorder=10, linewidth=1)

    # Clean up the 3D box
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    # Set plot limits
    lims_adjust = [l * 1.7 for l in lims]
    ax.set(xlim=lims_adjust[0], ylim=lims_adjust[1], zlim=lims_adjust[2])

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('E=', 'domain ') for label in labels]
    ax.legend(
        handles, labels,
        loc=(0.16, 0.36),
        fontsize=font2,
        title='Support of distribution in',
        title_fontsize=str(font2),
        frameon=False
    )

    # Add explanation text
    ax.text(
        tip3[0] + 0.2 + 0.752 + 0.08, tip3[1], tip3[2] + 0.45,
        '[average \% expl. var., worst-case \% expl. var.]',
        color='k', fontsize=font2, zorder=10
    )

    # Set view angle
    ax.view_init(elev=20, azim=-55)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=400)
    plt.close()
    print(f'Saved figure to {FIGURE_FILE}')


if __name__ == '__main__':
    # Ensure directories exist
    Path('figures').mkdir(exist_ok=True)

    # Compute data and make figure
    print('Computing plot data...')
    data = compute_plot_data()
    make_figure(data)
