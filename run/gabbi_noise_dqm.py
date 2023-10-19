"""
Tyson Reimer
University of Manitoba
September 23, 2023

Includes contributions by Fatimah Eashour
"""

import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger


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

###############################################################################


def load_noise_fd():

    d_dir = os.path.join(__D_DIR, "noise/")  # Dir for single rod
    o_dir = os.path.join(__O_DIR, "noise/")
    verify_path(o_dir)

    adi_str = "AdiposeCylinder_"
    ref_str = "OpenChamber"

    all_fs = os.listdir(d_dir)  # All files to load

    fd = np.zeros([len(all_fs) - 2, __N_FS, __N_ANTS], dtype=complex)
    ref_fd = np.load(os.path.join(d_dir, "%s%d.npy" % (ref_str, 1)))

    for ii in range(len(all_fs) - 2):  # For each file

        # Load fd data
        fd[ii, :, :] = (
                np.load(os.path.join(d_dir, "%s%d.npy" % (adi_str, ii + 1)))
                - ref_fd
        )

    return fd


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    o_dir = os.path.join(__O_DIR, "noise/")
    verify_path(o_dir)

    fd = load_noise_fd()

    # Init arr
    all_snrs = np.zeros(np.size(fd, axis=0) - 1)
    cc = 0

    for ii in range(1, np.size(fd, axis=0)):  # For each scan

        fd_here = np.abs(fd[ii, :, :])

        fd_diff = fd_here - np.abs(fd[ii - 1, :, :])

        snr = 10 * np.log10(np.abs(fd_here / fd_diff))

        logger.info('ii = %d\tSNR (dB): %.2f +/- %.2f'
                    % (ii, np.mean(snr), np.std(snr)))

        plt.figure()
        plt.imshow(snr, cmap='inferno', aspect='auto')
        plt.colorbar()
        plt.savefig(os.path.join(o_dir, "temp/snr_fd_%d.png" % ii))
        plt.close()

        all_snrs[cc] = np.mean(snr)
        cc += 1

        plt.figure(figsize=(12, 6))
        plt.title("SNR: %.2f +/- %.2f"
                  % (np.mean(snr), np.std(snr)))
        sns.histplot(fd_diff.flatten(), kde=True)
        plt.savefig(os.path.join(o_dir, "temp/diff_fd_distro_%d.png" % ii))
        plt.close()

    logger.info("SNR (dB): %.3f +/- %.3f"
                % (np.mean(all_snrs), np.std(all_snrs)))
