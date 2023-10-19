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

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.hardware import apply_sys_cor


###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/bed/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/bed/")
verify_path(__O_DIR)

###############################################################################


def setup_err_cor(metadata):
    """Correct for positionning setup error

    Parameters
    ----------
    metadata : array_like
        Target x/y positions, format x1, y1, x2, y2, [cm]

    Returns
    -------
    cor_x1 : array_like
        Corrected x-positions of target 1, [cm]
    cor_y1 : array_like
        Corrected y-positions of target 1, [cm]
    cor_x2 : array_like
        Corrected x-positions of target 2, [cm]
    cor_y2 : array_like
        Corrected y-positions of target 2, [cm]
    """

    x1, y1 = metadata[:, 0], metadata[:, 1]
    x2, y2 = metadata[:, 2], metadata[:, 3]

    cor_x1, cor_y1 = apply_sys_cor(x1, y1)
    cor_x2, cor_y2 = apply_sys_cor(x2, y2)

    return cor_x1, cor_y1, cor_x2, cor_y2


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Dir for fd / md
    d_dir = os.path.join(__D_DIR, "psf/")

    o_dir = os.path.join(__O_DIR, "acc/")
    verify_path(o_dir)

    # Load metadata
    md = load_pickle(os.path.join(d_dir, "md.pickle"))

    xs, ys = md[:, 0], md[:, 1]  # Get xs/ys separately

    xs, ys = apply_sys_cor(xs=xs, ys=ys)  # Apply setup error correction

    # Dir where DAS .pickle is
    img_dir = os.path.join(__O_DIR, "psf/")

    # Dir to save figs etc.
    o_dir = os.path.join(__O_DIR, "accuracy/")
    verify_path(o_dir)

    # Load DAS images
    imgs = load_pickle(os.path.join(img_dir, "das.pickle"))

    # Target rho positions, in [cm]
    tar_rhos = np.sqrt(xs**2 + ys**2)

    # Image maximum intensities
    img_maxes = np.max(np.abs(imgs)**2, axis=(1,2))

    plt.figure(figsize=(12, 6))
    plt.rc("font", family="Times New Roman")
    plt.tick_params(labelsize=14)
    plt.scatter(tar_rhos, img_maxes / np.max(img_maxes))
    plt.ylim([0, 1])
    plt.xlabel(r"Target $\mathdefault{\rho}$ Position (cm)", fontsize=20)
    plt.ylabel("Maximum Image Intensity", fontsize=20)
    plt.show()

    save_pickle((tar_rhos, img_maxes), os.path.join(o_dir, "acc.pickle"))
