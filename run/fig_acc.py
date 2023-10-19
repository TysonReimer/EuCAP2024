"""
Tyson Reimer
University of Manitoba
September 23, 2023

Includes contributions by Fatimah Eashour
"""

import os
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

###############################################################################

__O_F_DIR = os.path.join(get_proj_path(), "output/eucap2024/fatimah-sys/")

__O_G_DIR = os.path.join(get_proj_path(), "output/eucap2024/gabbi/")

__O_B_DIR = os.path.join(get_proj_path(), "output/eucap2024/bed/")

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    scan_type = "mono"  # Scan type for Fatimah system

    o_dir = os.path.join(get_proj_path(), "output/eucap2024/figs/")
    verify_path(o_dir)

    # Load Fatimah data
    f_rhos, f_maxes = load_pickle(os.path.join(__O_F_DIR,
                                               "acc-%s/" % scan_type,
                                               "acc.pickle"))
    f_maxes = f_maxes[~np.isnan(f_rhos)]
    f_rhos = f_rhos[~np.isnan(f_rhos)]

    f_fit = linregress(f_rhos, f_maxes / np.max(f_maxes))
    f_fit_xs = np.linspace(np.min(f_rhos), np.max(f_rhos))
    f_fit_ys = f_fit.intercept + f_fit.slope * f_fit_xs
    f_fit_yerr = np.sqrt((f_fit_xs * f_fit.stderr)**2
                         + (f_fit.intercept_stderr)**2)

    # Load Gabbi data
    g_rhos, g_maxes = load_pickle(os.path.join(__O_G_DIR, "acc/acc.pickle"))

    g_fit = linregress(g_rhos, g_maxes / np.max(g_maxes))
    g_fit_xs = np.linspace(np.min(g_rhos), np.max(g_rhos))
    g_fit_ys = g_fit.intercept + g_fit.slope * g_fit_xs
    g_fit_yerr = np.sqrt((g_fit_xs * g_fit.stderr)**2
                         + (g_fit.intercept_stderr)**2)

    # Load Bed data
    b_rhos, b_maxes = load_pickle(os.path.join(__O_B_DIR, "accuracy/acc.pickle"))
    
    b_fit = linregress(b_rhos, b_maxes / np.max(b_maxes))
    b_fit_xs = np.linspace(np.min(b_rhos), np.max(b_rhos))
    b_fit_ys = b_fit.intercept + b_fit.slope * b_fit_xs
    b_fit_yerr = np.sqrt((b_fit_xs * b_fit.stderr)**2
                         + (b_fit.intercept_stderr)**2)

    plt.figure(figsize=(12, 6))
    plt.rc("font", family="Times New Roman")
    plt.tick_params(labelsize=14)

    # Plot bed system data
    plt.scatter(b_rhos, b_maxes / np.max(b_maxes),
                color='k',
                marker="o",
                label="Bed System (p < 0.001)"
                )
    plt.plot(b_fit_xs, b_fit_ys, 'k-')
    plt.fill_between(x=b_fit_xs,
                     y1=b_fit_ys - 3 * b_fit_yerr,
                     y2=b_fit_ys + 3 * b_fit_yerr,
                     color='k',
                     alpha=0.2)

    # Plot Fatimah system data
    plt.scatter(f_rhos, f_maxes / np.max(f_maxes),
                color="g",
                marker="x",
                label="Bench-top System (p < 0.001)")
    plt.plot(f_fit_xs, f_fit_ys, 'g-')
    plt.fill_between(x=f_fit_xs,
                     y1=f_fit_ys - 3 * f_fit_yerr,
                     y2=f_fit_ys + 3 * f_fit_yerr,
                     color='g',
                     alpha=0.2)

    # Plot Gabbi system data
    plt.scatter(g_rhos, g_maxes / np.max(g_maxes),
                color="b",
                marker="+",
                label="Portable System (p = %.3f)" % (g_fit.pvalue)
                )
    plt.plot(g_fit_xs, g_fit_ys, "b-")
    plt.fill_between(x=g_fit_xs,
                     y1=g_fit_ys - 3 * g_fit_yerr,
                     y2=g_fit_ys + 3 * g_fit_yerr,
                     color='b',
                     alpha=0.2)

    plt.legend(fontsize=18, loc='lower left')
    plt.ylim([0.4, 1.05])
    plt.grid(axis='y')
    plt.xlabel(r"Target $\mathdefault{\rho}$ Position (cm)", fontsize=20)
    plt.ylabel("Maximum Image Intensity (Normalized)", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(o_dir, "acc_plt.png"), dpi=300)
