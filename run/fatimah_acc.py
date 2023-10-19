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


###############################################################################

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/fatimah-sys/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/fatimah-sys/")
verify_path(__O_DIR)

__INI_F = 700e6  # Initial freq: 700 MHz
__FIN_F = 8e9  # Final freq: 8 GHz
__N_FS = 1001  # Number of frequencies
__ANT_T = 9.1e-10  # Antenna 1-way t-delay, [s]
__ANT_RHO = 19.1  # Antenna polar rho coordinate, in [cm]
__INI_ANT_ANG = 7.5  # Initial antenna angle, in [deg]
__N_ANTS = 24  # Number of antennas
__ROI_RHO = 8  # ROI radius, in [cm]

__FS = np.linspace(__INI_F, __FIN_F, __N_FS)  # Scan frequencies

__M_SIZE = int(__ROI_RHO * 2 / 0.1)  # Image size

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    scan_type = "mono"

    # Dir where DAS .pickle is
    img_dir = os.path.join(__O_DIR, "psf/%s/" % scan_type)

    # Dir to save figs etc.
    o_dir = os.path.join(__O_DIR, "acc-%s/" % scan_type)
    verify_path(o_dir)

    # Load DAS images
    imgs = load_pickle(os.path.join(img_dir, "das_%s.pickle" % scan_type))

    md = load_pickle(os.path.join(__D_DIR, "psf_md.pickle"))

    xs = np.array([ii['x '] for ii in md])
    ys = np.array([ii['y '] for ii in md])

    tar_rhos = np.sqrt(xs**2 + ys**2)
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
