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

from umbms.loadsave import load_pickle

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

    o_dir = os.path.join(__O_DIR, "noise/%s/" % scan_type)
    verify_path(o_dir)

    fd = load_pickle(os.path.join(__D_DIR, "noise_fd.pickle"))

    # Exclude the index where a new session was started (phantom move
    idxs_to_excl = [4, 5, 9, 10, 14, 15, 19]

    # Init arr
    all_snrs = np.zeros(np.size(fd, axis=0) - len(idxs_to_excl) - 1)
    cc = 0

    for ii in range(1, np.size(fd, axis=0)):  # For each scan

        if not (ii in idxs_to_excl):

            this_fd = fd[ii]


            fd_s11 = np.zeros([__N_FS, __N_ANTS], dtype=complex)
            ref_s11 = np.zeros_like(fd_s11)
            for jj in range(__N_ANTS):
                fd_s11[:, jj] = this_fd["S%d_%d" % (jj, jj)]
                ref_s11[:, jj] = fd[ii - 1]["S%d_%d" % (jj, jj)]

            fd_here = np.abs(fd_s11)

            fd_diff = fd_here - np.abs(ref_s11)

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
