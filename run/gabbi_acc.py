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

__D_DIR = os.path.join(get_proj_path(), "data/eucap2024/gabbi/")

__O_DIR = os.path.join(get_proj_path(), "output/eucap2024/gabbi/")
verify_path(__O_DIR)

__INI_F = 600e6  # Initial freq: 700 MHz
__FIN_F = 4.4e9  # Final freq: 8 GHz
__N_FS = 381  # Number of frequencies
# __ANT_T = 9.1e-10  # Antenna 1-way t-delay, [s]
__ANT_RHO = 10  # Antenna polar rho coordinate, in [cm]
__INI_ANT_ANG = -90.0  # Initial antenna angle, in [deg]
__N_ANTS = 24  # Number of antennas
__ROI_RHO = 8  # ROI radius, in [cm]
__SPEED = 299792458  # Speed in [m/s]
__FS = np.linspace(__INI_F, __FIN_F, __N_FS)  # Scan frequencies

__M_SIZE = int(__ROI_RHO * 2 / 0.1)  # Image size

__TWO_TAR_XS = np.array([
    3.314,
    -4.871,
    -9.239,
    -5.985,
    1.918,
    10.407,
    16.602,
    19.142,
    17.846,
    13.265,
    6.305,
    -2.034,
    -10.809,
    -19.22,
    -26.644,
]) / 10 # Convert from [mm] to [cm]

__TWO_TAR_YS = np.array([
    4.264,
    5.57,
    -1.731,
    -9.703,
    -13.262,
    -11.351,
    -5.208,
    3.155,
    11.81,
    19.277,
    24.605,
    27.324,
    27.341,
    24.83,
    20.141,
]) / 10  # Convert from [mm] to [cm]


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Dir where DAS .pickle is
    img_dir = os.path.join(__O_DIR, "psf/")

    o_dir = os.path.join(__O_DIR, "acc/")
    verify_path(o_dir)

    # Load DAS images
    imgs = load_pickle(os.path.join(img_dir, "das.pickle"))

    # Target x/y positions
    xs = load_pickle(os.path.join(__O_DIR, "psf/tar_xs.pickle"))
    ys = load_pickle(os.path.join(__O_DIR, "psf/tar_ys.pickle"))

    tar_rhos = np.sqrt(xs ** 2 + ys ** 2)
    img_maxes = np.max(np.abs(imgs) ** 2, axis=(1, 2))

    plt.figure(figsize=(12, 6))
    plt.rc("font", family="Times New Roman")
    plt.tick_params(labelsize=14)
    plt.scatter(tar_rhos, img_maxes / np.max(img_maxes))
    plt.ylim([0, 1])
    plt.xlabel(r"Target $\mathdefault{\rho}$ Position (cm)", fontsize=20)
    plt.ylabel("Maximum Image Intensity", fontsize=20)
    plt.show()

    save_pickle((tar_rhos, img_maxes), os.path.join(o_dir, "acc.pickle"))
